import json
from pathlib import Path

from lab.text import TIERS, containment, tier, trigrams

ROOT = Path(__file__).resolve().parent.parent


def corpus():
    return [json.loads(line) for line in (ROOT / "data" / "corpus.jsonl").read_text(encoding="utf8").splitlines()]


def test_corpus_records():
    docs = corpus()
    assert len(docs) > 1000 and len({d["id"] for d in docs}) == len(docs)
    assert all(set(d) == {"id", "summary", "full"} and d["full"].startswith(d["summary"]) and len(d["full"]) > len(d["summary"]) for d in docs)


def test_tier_boundaries():
    assert [tier(x) for x in (0.0, 0.49, 0.5, 0.79, 0.8, 1.0)] == ["low", "low", "mid", "mid", "high", "high"]
    assert [t for t, _, _ in TIERS] == ["low", "mid", "high"]


def test_containment_of_a_copied_phrase():
    """Positive control for the leak audit: a phrase copied from a document is fully contained; a paraphrase is not."""
    full = [d["full"] for d in corpus()]
    words = full[100].split()
    grams = set().union(*map(trigrams, full))
    assert containment(" ".join(words[20:30]), grams) == 1.0
    assert containment("zebra xylophone quantum harvest of lunar cheese", grams) == 0.0
