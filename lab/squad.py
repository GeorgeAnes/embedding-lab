"""SQuAD v1.1 dev as a second, human-written query set: questions against the paragraphs they were written from."""
import json
import re
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json"
PATH = Path(__file__).resolve().parent.parent / "data" / "squad" / "dev-v1.1.json"  # git-ignored, never committed


def fetch():
    if not PATH.exists():
        PATH.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=120) as r:
            PATH.write_bytes(r.read())
    return PATH


def load():
    """Returns (paragraphs, questions, gold) where gold[i] is the index of the paragraph question i was written from."""
    paragraphs, questions, gold = [], [], []
    for article in json.loads(fetch().read_text(encoding="utf8"))["data"]:
        for p in article["paragraphs"]:
            paragraphs.append(p["context"])
            questions += [qa["question"] for qa in p["qas"]]
            gold += [len(paragraphs) - 1] * len(p["qas"])
    return paragraphs, questions, gold


def split_first_sentence(paragraph):
    """(first sentence, the rest): the same summary-and-rest split the docstring pairs use."""
    parts = re.split(r"(?<=[.!?])\s+", paragraph, maxsplit=1)
    return parts[0], parts[1] if len(parts) > 1 else ""
