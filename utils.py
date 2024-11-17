import json
from html_parser import parse_article


def prepare_kb(kb_path):
    with open(kb_path, 'r') as f:
        kb = json.load(f)
    return [parse_article(article) for article in kb]
