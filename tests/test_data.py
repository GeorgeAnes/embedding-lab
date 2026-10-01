import json
import re
from hashlib import sha256
from pathlib import Path

import pytest

from lab import evaluate
from lab.text import TIERS, containment, tier, trigrams

ROOT = Path(__file__).resolve().parent.parent
STOP = "a an and are as at be by for from in is it of on or that the this to with will".split()


@pytest.fixture(scope="module")
def task():
    return evaluate.docstring_task()


def test_corpus_records():
    docs = [json.loads(line) for line in (ROOT / "data" / "corpus.jsonl").read_text(encoding="utf8").splitlines()]
    assert len(docs) > 1000 and len({d["id"] for d in docs}) == len(docs)
    assert all(set(d) == {"id", "summary", "full"} and d["full"].startswith(d["summary"]) and len(d["full"]) > len(d["summary"]) for d in docs)


def test_query_file_is_frozen_and_well_formed(task):
    recorded = (ROOT / "data" / "queries.sha256").read_text().split()[0]
    assert sha256((ROOT / "data" / "queries.json").read_bytes()).hexdigest() == recorded, "queries changed: log it in docs/method.md and update queries.sha256"
    groups = json.loads((ROOT / "data" / "queries.json").read_text(encoding="utf8"))
    assert len(groups) == 33 and all(len(g["queries"]) == 3 for g in groups)
    assert len(set(task.queries)) == len(task.queries) == 99  # every target id exists, or docstring_task() would have failed


def naive_overlap(query, full):
    words = lambda t: {w for w in re.findall(r"[a-z0-9][a-z0-9-]+", t.lower()) if len(w) > 2 and w not in STOP}
    q = words(query)
    return len(q & words(full)) / len(q)


def test_tiers_follow_their_definition(task):
    """Recompute every query's tier with independent code and compare; also pin the boundaries."""
    for q, gold, got in zip(task.queries, task.gold, task.tier):
        ov = max(naive_overlap(q, task.full[i]) for i in gold)
        want = "low" if ov < 0.5 else "mid" if ov < 0.8 else "high"
        assert got == want, (q, ov)
    assert [tier(x) for x in (0.0, 0.49, 0.5, 0.79, 0.8, 1.0)] == ["low", "low", "mid", "mid", "high", "high"]
    assert [t for t, _, _ in TIERS] == ["low", "mid", "high"]
    assert all((task.tier == t).sum() >= 20 for t, _, _ in TIERS)


def test_leak_audit_flags_a_planted_copy(task):
    """Positive control: a query copied from a document's full text is flagged; a paraphrase is not."""
    words = task.full[100].split()
    grams = set().union(*map(trigrams, task.full))
    assert containment(" ".join(words[20:30]), grams) == 1.0
    assert containment("zebra xylophone quantum harvest of lunar cheese", grams) == 0.0
    flagged = containment(" ".join(words[20:30]), grams) >= evaluate.FLAG
    assert flagged and not task.flagged[task.tier == "low"].any()


def test_verbatim_queries_are_summary_lines(task):
    """Queries that repeat a target's summary line: recount from the corpus file, and the leak audit must flag every one."""
    docs = {d["id"]: d["summary"].lower().strip().rstrip(".") for d in map(json.loads, (ROOT / "data" / "corpus.jsonl").read_text(encoding="utf8").splitlines())}
    groups = json.loads((ROOT / "data" / "queries.json").read_text(encoding="utf8"))
    want = sum(q.lower().strip().rstrip(".") in {docs[t] for t in g["targets"]} for g in groups for q in g["queries"])
    assert task.verbatim.sum() == want > 0
    assert task.flagged[task.verbatim].all()


def test_reported_results_belong_to_the_committed_data():
    path = ROOT / "results" / "results.json"
    if not path.exists():
        pytest.skip("run python -m lab eval first")
    meta = json.loads(path.read_text())["meta"]
    assert meta["queries_sha256"] == sha256((ROOT / "data" / "queries.json").read_bytes()).hexdigest()
    assert meta["corpus_sha256"] == sha256((ROOT / "data" / "corpus.jsonl").read_bytes()).hexdigest()
