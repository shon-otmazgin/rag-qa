from bs4 import BeautifulSoup
from langchain_core.documents import Document


def extract_text_and_links(element):
    plain_text = element.get_text(separator=" ", strip=True)
    links = element.find_all('a')
    text_urls = []
    for link in links:
        link_url = link.get('href')
        link_text = link.get_text()
        text_urls.append({link_text: link_url})

    return plain_text, text_urls


def parse_article(article):
    html_content = article['html_content']
    soup = BeautifulSoup(html_content, "html.parser")

    texts = [article['title']]
    text_urls = []
    section_title = None
    for element in soup.find_all(['div', 'h1', 'h2', 'h3']):
        if element.name == 'div' and element.get('data-component-type') in ['text', 'informative']:
            section_text, section_urls = extract_text_and_links(element)

            texts.append(f'{section_title}\n{section_text}' if section_title else section_text)
            text_urls += section_urls

            section_title = None

        elif element.name in ['h1', 'h2', 'h3']:
            section_title = element.get_text(separator=" ", strip=True)

    return Document(
        id=article['article_id'],
        page_content='\n\n'.join(texts),
        metadata={
            'id': article['article_id'],
            'title': article['title'],
            'url': article['url'],
            'text_urls': text_urls
        }
    )
