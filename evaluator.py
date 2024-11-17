from tqdm.auto import tqdm

from retrievers import DenseRetriever, SparseRetriever, EnsembleRetriever, EnsembleAndRerankingRetriever
from utils import prepare_kb


def format_metrics(metrics):
    formatted = {}
    for k, value in metrics["mrr_at_k"].items():
        formatted[f"MRR@{k}"] = value
    for k, value in metrics["precision_at_k"].items():
        formatted[f"Precision@{k}"] = value
    for k, value in metrics["recall_at_k"].items():
        formatted[f"Recall@{k}"] = value
    formatted["MAP"] = metrics["map"]

    return formatted


def compute_mrr(relevant_flags):
    reciprocal_ranks = [
        1 / (1 + flags.index(1)) if 1 in flags else 0
        for flags in relevant_flags
    ]
    return sum(reciprocal_ranks) / len(relevant_flags)


def compute_ndcg(relevant_flags):
    ndcg_scores = []
    for flags in relevant_flags:
        dcg = sum((rel / np.log2(i + 2)) for i, rel in enumerate(flags))

        ideal_flags = sorted(flags, reverse=True)
        idcg = sum((rel / np.log2(i + 2)) for i, rel in enumerate(ideal_flags))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores)


def mean_average_precision(relevant_flags):
    def average_precision(flags):
        if sum(flags) == 0:
            return 0.0

        ap = 0.0
        for k in range(1, len(flags) + 1):
            p_k = sum(flags[:k]) / k
            rel_k = flags[k-1]
            ap += p_k * rel_k
        return ap / sum(flags)

    return sum(average_precision(flags) for flags in relevant_flags) / len(relevant_flags)


def evaluate(retriever, data):
    k_to_check = [2, 5, 10]

    relevant_flags_at_k = {k: [] for k in k_to_check}
    precision_at_k = {k: [] for k in k_to_check}  # Precision@k
    recall_at_k = {k: [] for k in k_to_check}  # Recall@k
    total_relevant_docs = []

    for query in tqdm(data):
        question = query['question']
        retrieved_docs = retriever.retrieve(
            user_query=question,
            k=max(k_to_check),
        )
        assert len(retrieved_docs) == max(k_to_check), f'Requested {max(k_to_check)} docs retrieve but return {len(retrieved_docs)} docs'

        retrieved_ids = [doc.metadata['id'] for doc in retrieved_docs]
        relevant_ids = set(query['article_ids'])
        total_relevant_docs.append(len(relevant_ids))

        for k in k_to_check:
            top_k_retrieved_ids = retrieved_ids[:k]
            relevant_flags = [1 if doc_id in relevant_ids else 0 for doc_id in top_k_retrieved_ids]
            relevant_flags_at_k[k].append(relevant_flags)

            precision = sum(relevant_flags) / k
            precision_at_k[k].append(precision)

            recall = sum(relevant_flags) / len(relevant_ids) if relevant_ids else 0
            recall_at_k[k].append(recall)

    # Calculate average metrics
    mrr_at_k = {k: compute_mrr(relevant_flags) for k, relevant_flags in relevant_flags_at_k.items()}
    ndcg_at_k = {k: compute_ndcg(relevant_flags) for k, relevant_flags in relevant_flags_at_k.items()}
    precision_at_k_mean = {k: sum(values) / len(values) for k, values in precision_at_k.items()}
    recall_at_k_mean = {k: sum(values) / len(values) for k, values in recall_at_k.items()}
    map_score = mean_average_precision(relevant_flags_at_k[5])

    return {
        "mrr_at_k": mrr_at_k,
        "ndcg_at_k": ndcg_at_k,
        "precision_at_k": precision_at_k_mean,
        "recall_at_k": recall_at_k_mean,
        "map": map_score,
    }


if __name__ == '__main__':
    import json
    import numpy as np
    import pandas as pd

    EMBEDDING_MODEL_NAME = "thenlper/gte-large"

    docs = prepare_kb('data/Copy of kb_bookings_en_20240614.json')
    dense_retriever = DenseRetriever(embedding_model_name=EMBEDDING_MODEL_NAME, chunk_size=512, chunk_overlap=128)
    dense_retriever.add_documents(docs)
    sparse_retriever = SparseRetriever(docs=docs)

    with open('data/Copy of bookings_train.json', 'r') as f:
        qa_data = json.load(f)

    metrics = evaluate(dense_retriever, qa_data)
    print(metrics)

    metrics = evaluate(sparse_retriever, qa_data)
    print(metrics)

    results = {}
    values = np.arange(0, 1.1, 0.1).tolist()
    for dense_w in values:
        ensemble_retriever = EnsembleRetriever(dense_retriever, sparse_retriever, dense_w=dense_w, sparse_w=1 - dense_w)
        metrics = evaluate(ensemble_retriever, qa_data)
        results[(dense_w, 1-dense_w)] = metrics

    data = {k: format_metrics(v) for k, v in results.items()}
    df = pd.DataFrame(data)
    print(df)

    ranker_and_ensemble_retriever = EnsembleAndRerankingRetriever(
        dense_retriever,
        sparse_retriever,
        dense_w=0.6, sparse_w=0.4
    )
    metrics = evaluate(ranker_and_ensemble_retriever, qa_data)
    print(metrics)

