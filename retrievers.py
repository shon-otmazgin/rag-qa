import os
import pickle
from collections import defaultdict
from typing import List
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk


import torch
import faiss

from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
from langchain_community.docstore import InMemoryDocstore
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.stores import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from utils import prepare_kb

MARKDOWN_SEPARATORS = [
    "\n\n",
    "\n",
    " ",
    "",
    "."
]

nltk.download('wordnet')
nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess_func(text: str) -> List[str]:
    tokens = text.lower().split()
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]

    bigrams = ['_'.join(pair) for pair in zip(tokens, tokens[1:])]

    return tokens + bigrams


class DenseRetriever:
    def __init__(self, embedding_model_name, chunk_size, chunk_overlap):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            multi_process=False,
            show_progress=False,
            model_kwargs={"device": device, 'trust_remote_code': True},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.child_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(embedding_model_name),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS
        )

        # Load vector store if it exists
        if os.path.exists('faiss_index'):
            with open('store.pkl', "rb") as f:
                self.store = pickle.load(f)

            self.vector_store = FAISS.load_local(
                'faiss_index', self.embedding_model, allow_dangerous_deserialization=True
            )
            print("Loaded vector store from disk.")
            self._cached_index = True
        else:
            index = faiss.IndexFlatIP(len(self.embedding_model.embed_query("hello world")))
            self.vector_store = FAISS(
                embedding_function=self.embedding_model,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
                distance_strategy=DistanceStrategy.COSINE
            )
            print("Initialized new vector store.")
            self._cached_index = False
            self.store = InMemoryStore()

        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=self.store,
            child_splitter=self.child_splitter,
        )

    def save_vectorstore(self):
        self.vector_store.save_local('faiss_index')
        with open('store.pkl', "wb") as f:
            pickle.dump(self.store, f)
        print("Saved vector store to disk.")

    def add_documents(self, docs: List[Document]):
        if not self._cached_index:
            self.retriever.add_documents(docs)
            self.save_vectorstore()

    def retrieve(self, user_query, k) -> List[Document]:
        # since we store children and we return parents, we may have few children from the same parent.
        # this is quick and dirty solution
        for i in range(1, 5):
            self.retriever.search_kwargs = {'k': k * i}
            retrieved_docs = self.retriever.invoke(user_query)[:k]

            if len(retrieved_docs) == k:
                return retrieved_docs

        # if we dont succed in 5 times return the last.
        return retrieved_docs


class SparseRetriever:
    def __init__(self, docs):
        self.bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=preprocess_func)

    def retrieve(self, user_query, k) -> List[Document]:
        self.bm25_retriever.k = k
        return self.bm25_retriever.invoke(user_query)


class EnsembleRetriever:
    def __init__(self, dense_retriever, sparse_retriever ,dense_w, sparse_w):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.weights = [dense_w, sparse_w]

    def retrieve(self, user_query, k) -> List[Document]:
        c = 5
        dense_docs = self.dense_retriever.retrieve(user_query, k)
        sparse_docs = self.sparse_retriever.retrieve(user_query, k)
        doc_lists = [dense_docs, sparse_docs]

        rrf_score = defaultdict(float)
        for doc_list, weight in zip(doc_lists, self.weights):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score[doc.metadata['id']] += weight / (rank + c)

        all_docs = dense_docs + sparse_docs
        unique_docs = []
        seen_ids = set()
        for doc in all_docs:
            if doc.metadata['id'] not in seen_ids:
                unique_docs.append(doc)
                seen_ids.add(doc.metadata['id'])

        sorted_docs = sorted(
            unique_docs,
            reverse=True,
            key=lambda doc: rrf_score[doc.metadata['id']]
        )
        return sorted_docs[:k]


class EnsembleAndRerankingRetriever:
    def __init__(self, dense_retriever, sparse_retriever ,dense_w, sparse_w):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.weights = [dense_w, sparse_w]

    def retrieve(self, user_query, k) -> List[Document]:
        new_k = k * 2   # we going to rerank so lets increase k :)
        dense_docs = self.dense_retriever.retrieve(user_query, new_k)
        sparse_docs = self.sparse_retriever.retrieve(user_query, new_k)
        all_docs = dense_docs + sparse_docs

        unique_docs = []
        seen_ids = set()
        for doc in all_docs:
            if doc.metadata['id'] not in seen_ids:
                unique_docs.append(doc)
                seen_ids.add(doc.metadata['id'])

        ranker = RankLLMRerank(
            top_n=k,
            model='gpt',
            step_size=10,
            gpt_model='gpt-4o-mini'
        )

        compressed_docs = ranker.compress_documents(unique_docs, user_query)
        return list(compressed_docs)


if __name__ == '__main__':
    EMBEDDING_MODEL_NAME = "thenlper/gte-large"

    docs = prepare_kb('data/Copy of kb_bookings_en_20240614.json')
    retriever = DenseRetriever(
        embedding_model_name=EMBEDDING_MODEL_NAME,
        chunk_size=512,
        chunk_overlap=128,
    )

    # Add documents and save the vector store
    retriever.add_documents(docs)

    query = "Is it possible to adjust the booking form on my website to only show Fridays and Saturdays for booking dates?"
    top_k_docs = retriever.retrieve(query, k=4)
    print(len(top_k_docs))
