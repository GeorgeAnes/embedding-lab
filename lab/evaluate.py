"""Run every method on the benchmarks and compute recall@10 and MRR@10 with cluster-bootstrap intervals.

All methods score the same documents for the same queries.  A query counts as answered at rank r if the best-ranked
acceptable target sits at position r.  Ties in scores are broken by one fixed random order, so a method that scores
everything zero is ranked at chance, not by document order.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib import metadata
from pathlib import Path

import numpy as np

from . import contrastive as C
from . import dense, squad
from .lexical import Index
from .text import TIERS, containment, overlap, tier, trigrams
from .vecops import mean_pairwise_cosine, ranks, rrf, to_int8, truncate

ROOT = Path(__file__).resolve().parent.parent
FLAG = 0.5  # the leak audit flags a query when at least this share of its word trigrams occur in the corpus text
LABELS = {"random": "random", "tfidf_summary": "TF-IDF (auditor), summary text", "tfidf_full": "TF-IDF (auditor), full text",
          "bm25_summary": "BM25, summary text", "bm25_full": "BM25, full text", "lsa64": "LSA-64, full text",
          "static": "static vectors, full text", "own64": "own-64, full text", "hybrid": "hybrid: BM25 full + own-64, RRF k=60",
          "real": "real model, full text"}


# ---- Benchmarks --------------------------------------------------------------------------------------------
class Task:
    """Documents (split into a summary and a rest, as the training pairs are), queries and their acceptable targets."""

    def __init__(self, name, ids, summary, rest, queries, gold, group):
        self.name, self.ids, self.summary, self.rest = name, ids, summary, rest
        self.queries, self.gold, self.group = queries, gold, np.array(group)
        self.full = [s + " " + r for s, r in zip(summary, rest)]
        self.ov = np.array([overlap(q, [self.full[i] for i in g]) for q, g in zip(queries, gold)])
        self.tier = np.array([tier(o) for o in self.ov])
        grams_s, grams_f = set().union(*map(trigrams, summary)), set().union(*map(trigrams, self.full))
        self.tri_s = np.array([containment(q, grams_s) for q in queries])  # leak audit: query trigrams found in any summary
        self.tri_f = np.array([containment(q, grams_f) for q in queries])  # ... or anywhere in the full texts
        self.flagged = np.maximum(self.tri_s, self.tri_f) >= FLAG
        clean = lambda t: t.lower().strip().rstrip(".")
        self.verbatim = np.array([any(clean(q) == clean(summary[i]) for i in g) for q, g in zip(queries, gold)])  # the query IS a target's summary line
        self.masks = {"all": np.ones(len(queries), bool), **{t: self.tier == t for t, _, _ in TIERS}}


def docstring_task():
    docs = [json.loads(line) for line in (ROOT / "data" / "corpus.jsonl").read_text(encoding="utf8").splitlines()]
    pos = {d["id"]: k for k, d in enumerate(docs)}
    queries, gold, group = [], [], []
    for g, item in enumerate(json.loads((ROOT / "data" / "queries.json").read_text(encoding="utf8"))):
        for q in item["queries"]:
            queries.append(q), gold.append([pos[t] for t in item["targets"]]), group.append(g)
    return Task("docstrings", [d["id"] for d in docs], [d["summary"] for d in docs],
                [d["full"][len(d["summary"]):].strip() for d in docs], queries, gold, group)


def squad_task():
    paragraphs, questions, gold = squad.load()
    halves = [squad.split_first_sentence(p) for p in paragraphs]
    return Task("squad", [f"paragraph {i}" for i in range(len(paragraphs))], [h[0] for h in halves], [h[1] for h in halves],
                questions, [[g] for g in gold], gold)


# ---- Training (independent runs in parallel processes) ---------------------------------------------------------
def _fit(args):
    a, b, V, kw = args
    t = time.time()
    WA, WB, history = C.train(a, b, V, **kw)
    return WA, WB, history, time.time() - t


def fit_all(task, jobs):
    """jobs: {name: train() keyword arguments, plus optional 'exclude': document indices left out of training}."""
    vocab = C.build_vocab(task.full)
    a, b = [C.to_ids(s, vocab) for s in task.summary], [C.to_ids(r, vocab) for r in task.rest]
    todo = []
    for kw in jobs.values():
        kw = dict(kw)
        out = set(kw.pop("exclude", ()))
        keep = [i for i in range(len(a)) if len(a[i]) and len(b[i]) and i not in out]
        todo.append(([a[i] for i in keep], [b[i] for i in keep], len(vocab), kw))
    with ProcessPoolExecutor(min(len(todo), os.cpu_count() or 1)) as pool:
        done = list(pool.map(_fit, todo))
    return {n: (C.Model(vocab, r[0], r[1]), r[2], r[3]) for n, r in zip(jobs, done)}


# ---- Scoring and ranking -----------------------------------------------------------------------------------------
def own_fn(task, model, k=64, int8=False):
    """Score function of a trained model: cosine between the query vector and each document vector."""
    D = truncate(model.embed_docs(task.summary, task.rest)[1], k)
    D = to_int8(D) if int8 else D  # documents are quantized, queries stay float
    return lambda qs: truncate(model.embed_queries(qs)[1], k) @ D.T


def method_fns(task, model, extra):
    """The methods, in display order. `extra` is a list of (key, name, embed) for pretrained or remote dense models."""
    full, N, rng = Index(task.full), len(task.ids), np.random.default_rng(0)
    fns = {"random": lambda qs: rng.random((len(qs), N))}
    if task.name == "docstrings":
        summary = Index(task.summary)
        fns |= {"tfidf_summary": summary.tfidf, "tfidf_full": full.tfidf, "bm25_summary": summary.bm25}
    else:
        fns |= {"tfidf_full": full.tfidf}
    fns |= {"bm25_full": full.bm25, "lsa64": full.lsa}
    for key, name, embed in extra:
        D = embed(task.full)
        fns[key] = lambda qs, embed=embed, D=D: embed(qs) @ D.T
        LABELS[key] = name
    fns["own64"] = own_fn(task, model)
    return fns, {"hybrid": ("bm25_full", "own64")}, full


def best_ranks(task, fns, hybrids=None, chunk=1000, top=0):
    """Per method and query: rank of the best acceptable target (1 = first), and optionally the top documents."""
    tiebreak = np.random.default_rng(0).permutation(len(task.ids))
    best, tops, gold_doc = {}, {}, {}
    for lo in range(0, len(task.queries), chunk):
        qs, gold = task.queries[lo:lo + chunk], task.gold[lo:lo + chunk]
        R = {n: ranks(f(qs), tiebreak) for n, f in fns.items()}
        for n, (x, y) in (hybrids or {}).items():
            R[n] = ranks(rrf([R[x], R[y]]), tiebreak)
        for n, r in R.items():
            best.setdefault(n, []).extend(int(r[i, g].min()) for i, g in enumerate(gold))
            if top:
                tops.setdefault(n, []).extend(np.argsort(r, axis=1)[:, :top].tolist())
                gold_doc.setdefault(n, []).extend(int(g[np.argmin(r[i, g])]) for i, g in enumerate(gold))
    return {n: np.array(v) for n, v in best.items()}, tops, gold_doc


def summarize(best, group, ref, masks, B=1000):
    """Metrics per method and subset with 95% intervals from a paired cluster bootstrap.

    Whole groups (the three phrasings of one target, or all questions on one paragraph) are resampled together, and
    every method sees the same resamples, so differences to `ref` carry their own interval.
    Returns {method: {subset: {"n", "r10": [mean, lo, hi], "mrr": [...], "d_r10": [diff to ref, lo, hi], "d_mrr": [...]}}}.
    """
    out = {n: {} for n in best}
    for sub, mask in masks.items():
        if not mask.any():
            continue
        g = np.unique(group[mask], return_inverse=True)[1]
        G = g.max() + 1
        draws = np.random.default_rng(0).integers(0, G, (B, G))
        size = np.bincount(g, minlength=G)[draws].sum(1)
        stat = {}
        for n, r in best.items():
            r = r[mask]
            for m, v in (("r10", (r <= 10) * 1.0), ("mrr", np.where(r <= 10, 1.0 / r, 0.0))):
                stat[n, m] = v.mean(), np.bincount(g, weights=v, minlength=G)[draws].sum(1) / size
        for n in best:
            cell = {"n": int(mask.sum()), "groups": int(G)}
            for m in ("r10", "mrr"):
                cell[m] = [stat[n, m][0], *np.percentile(stat[n, m][1], [2.5, 97.5])]
                cell["d_" + m] = [stat[n, m][0] - stat[ref, m][0], *np.percentile(stat[n, m][1] - stat[ref, m][1], [2.5, 97.5])]
            out[n][sub] = cell
    return out


# ---- Whole run ---------------------------------------------------------------------------------------------------
def audit_rows(task):
    rows = {}
    for t in ["all"] + [t for t, _, _ in TIERS]:
        m = task.masks[t]
        rows[t] = {"n": int(m.sum()), "mean_overlap": float(task.ov[m].mean()), "flagged": int(task.flagged[m].sum()),
                   "tri_summary": float(task.tri_s[m].mean()), "tri_full": float(task.tri_f[m].mean())}
    return rows


def run(real=False, with_squad=True):
    """Train the models, score every method on both benchmarks, return (results, vectors for the page)."""
    started = time.time()
    extra = [("static", *s) for s in [dense.static_vectors()] if s] + ([("real", *dense.real_endpoint())] if real else [])
    task = docstring_task()
    targets = {i for g in task.gold for i in g}
    # Every training run the results tables report: the model used everywhere ("main"), its untrained starting point, the
    # ablations, two more seeds, and a model whose training pairs exclude the benchmark's target documents.
    jobs = {"untrained": {"steps": 0}, "main": {}, "tau_0.02": {"tau": 0.02}, "tau_1.0": {"tau": 1.0}, "no_negatives": {"negatives": False},
            "seed_1": {"seed": 1}, "seed_2": {"seed": 2}, "held_out": {"exclude": targets}}
    fitted = fit_all(task, jobs)
    model = fitted["main"][0]

    # 1. Main comparison: all methods, same documents, same queries.
    fns, hybrids, lex = method_fns(task, model, extra)
    best, tops, gold_doc = best_ranks(task, fns, hybrids, top=3)
    methods = summarize(best, task.group, "bm25_full", task.masks)

    # 2. Ablations, compared with the main model (paired), plus the collapse diagnostic.
    variants = {**{f"model:{n}": own_fn(task, f[0]) for n, f in fitted.items()},
                **{f"dims:{k}:{q}": own_fn(task, model, k, q == "int8") for k in (64, 32, 16, 8) for q in ("float32", "int8")}}
    vbest, _, _ = best_ranks(task, variants)
    ablate = summarize(vbest, task.group, "model:main", task.masks)
    collapse = {n: mean_pairwise_cosine(f[0].embed_docs(task.summary, task.rest)[1]) for n, f in fitted.items()}

    # 3. Everything the tables and the page need.
    results = {
        "meta": {"date": time.strftime("%d %B %Y"), "python": sys.version.split()[0], "numpy": np.__version__,
                 "scipy": metadata.version("scipy"), "wordllama": metadata.version("wordllama") if extra else None,
                 "queries_sha256": sha256((ROOT / "data" / "queries.json").read_bytes()).hexdigest(),
                 "corpus_sha256": sha256((ROOT / "data" / "corpus.jsonl").read_bytes()).hexdigest(),
                 "n_docs": len(task.ids), "n_queries": len(task.queries), "n_targets": len(targets), "vocab": len(model.vocab),
                 "flag_threshold": FLAG, "tiers": [list(t) for t in TIERS], "cpu_count": os.cpu_count()},
        "labels": {k: LABELS[k] for k in methods},
        "docstrings": {"methods": methods, "audit": audit_rows(task)},
        "ablations": {"stats": ablate, "mean_pairwise_cosine": collapse,
                      "loss": {n: f[1] for n, f in fitted.items()}, "train_seconds": {n: f[2] for n, f in fitted.items()}},
        "queries": [{"q": q, "targets": [task.ids[i] for i in task.gold[k]], "group": int(task.group[k]), "overlap": float(task.ov[k]),
                     "tier": str(task.tier[k]), "tri_summary": float(task.tri_s[k]), "tri_full": float(task.tri_f[k]),
                     "flagged": bool(task.flagged[k]), "verbatim": bool(task.verbatim[k]), "rank": {n: int(best[n][k]) for n in best},
                     "top": {n: [task.ids[i] for i in tops[n][k]] for n in best}, "gold_doc": {n: task.ids[gold_doc[n][k]] for n in best}}
                    for k, q in enumerate(task.queries)],
        "squad": None,
    }
    vectors = maths_vectors(task, model, lex, tops, gold_doc)

    if with_squad:  # 4. Second query set: human-written questions, own-64 retrained on the paragraphs.
        try:
            stask = squad_task()
        except OSError as e:
            print("SQuAD not run:", e)
        else:
            sfit = fit_all(stask, {"main": {}})["main"]
            sfns, shyb, _ = method_fns(stask, sfit[0], [e for e in extra if e[0] == "static"])
            sbest, _, _ = best_ranks(stask, sfns, shyb)
            results["squad"] = {"methods": summarize(sbest, stask.group, "bm25_full", stask.masks), "audit": audit_rows(stask),
                                "n_docs": len(stask.ids), "train_seconds": sfit[2], "loss": sfit[1]}
    results["meta"]["wall_seconds"] = round(time.time() - started, 1)
    return results, vectors


def maths_vectors(task, model, lex, tops, gold_doc):
    """Vectors (before scaling to length 1) for the page's 'show the maths' panel: queries and the documents it can show."""
    docs = {n: sorted({i for k in range(len(task.queries)) for i in (tops[n][k][0], gold_doc[n][k])}) for n in ("own64", "lsa64")}
    own_q, own_d = model.embed_queries(task.queries)[0], model.embed_docs(task.summary, task.rest)[0]
    lsa_q, lsa_d = lex.lsa_queries(task.queries)[0], lex.lsa_docs()[0]
    r = lambda M: np.round(M, 4).tolist()
    return {"own64": {"q": r(own_q), "d": {task.ids[i]: r(own_d[i]) for i in docs["own64"]}},
            "lsa64": {"q": r(lsa_q), "d": {task.ids[i]: r(lsa_d[i]) for i in docs["lsa64"]}}}
