"""Turn results into Markdown tables and embed them, with the per-query data, in web/index.html."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBS = ("all", "low", "mid", "high")
ORDER = ("random", "tfidf_summary", "tfidf_full", "bm25_summary", "bm25_full", "lsa64", "static", "own64", "hybrid", "real")


def ci(c, signed=False):
    f = "{:+.2f}" if signed else "{:.2f}"
    return f"{f.format(c[0])} [{f.format(c[1])}, {f.format(c[2])}]"


def md(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]) + "\n"


def sub_header(methods, first):
    n = next(iter(methods.values()))
    return [first] + [f"{s} (n={n[s]['n']})" for s in SUBS if s in n]


def headline(res):
    own, bm = res["docstrings"]["methods"]["own64"]["all"], res["docstrings"]["methods"]["bm25_full"]["all"]
    d = own["d_r10"]
    verdict = ("loses to" if d[2] < 0 else "beats" if d[1] > 0 else
               f"is {'below' if d[0] < 0 else 'above'} the level of, but not reliably separable from,")
    rest = [q for q in res["queries"] if not q["verbatim"]]  # without the queries that repeat a target's summary line
    hit = lambda k: sum(q["rank"][k] <= 10 for q in rest) / len(rest)
    return (f"On the {own['n']} docstring queries own-64 {verdict} BM25 on full text: recall@10 {own['r10'][0]:.2f} against "
            f"{bm['r10'][0]:.2f} (difference {d[0]:+.2f}, 95% interval [{d[1]:+.2f}, {d[2]:+.2f}]) and MRR@10 {own['mrr'][0]:.2f} against "
            f"{bm['mrr'][0]:.2f} (difference {own['d_mrr'][0]:+.2f}, [{own['d_mrr'][1]:+.2f}, {own['d_mrr'][2]:+.2f}]). "
            f"Without the {len(res['queries']) - len(rest)} queries that repeat a target's summary line word for word ({len(rest)} left, no interval): "
            f"recall@10 {hit('own64'):.2f} against {hit('bm25_full'):.2f}.")


def main_tables(methods, labels, ref="BM25, full text"):
    keys = [k for k in ORDER if k in methods]
    out = ""
    for metric, name in (("r10", "recall@10"), ("mrr", "MRR@10")):
        out += f"\n{name}, mean and 95% cluster-bootstrap interval, by overlap tier:\n\n"
        out += md(sub_header(methods, "Method"), [[labels[k]] + [ci(methods[k][s][metric]) for s in SUBS if s in methods[k]] for k in keys])
    out += f"\nPaired difference to {ref} (same resamples for both methods; an interval that excludes 0 is a difference this run can see):\n\n"
    rows = [[f"{labels[k]}: {name}"] + [ci(methods[k][s]["d_" + m], True) for s in SUBS if s in methods[k]]
            for k in keys if k not in ("random", "bm25_full") for m, name in (("r10", "recall@10"), ("mrr", "MRR@10"))]
    return out + md(sub_header(methods, "Method and metric"), rows)


def head_to_head(res):
    qs = res["queries"]
    own, bm = (lambda q: q["rank"]["own64"] <= 10), (lambda q: q["rank"]["bm25_full"] <= 10)
    rows = []
    for t in SUBS:
        sel = [q for q in qs if t == "all" or q["tier"] == t]
        rows.append([t, len(sel), sum(own(q) and not bm(q) for q in sel), sum(bm(q) and not own(q) for q in sel),
                     sum(own(q) and bm(q) for q in sel), sum(not own(q) and not bm(q) for q in sel)])
    out = "Head to head at rank 10, own-64 against BM25 on full text, counted over queries:\n\n"
    out += md(["Tier", "queries", "only own-64 in top 10", "only BM25 full in top 10", "both", "neither"], rows)
    wins = [q for q in qs if own(q) and not bm(q)]
    out += "\nThe queries where own-64 has a target in its top 10 and BM25 on full text does not:\n\n"
    return out + md(["Query", "tier", "own-64 rank", "BM25 full rank"], [[q["q"], q["tier"], q["rank"]["own64"], q["rank"]["bm25_full"]] for q in wins])


def audit_table(audit):
    return md(["Subset", "queries", "mean overlap", "flagged by leak audit", "mean trigram containment, summaries", "... full texts"],
              [[s, a["n"], f"{a['mean_overlap']:.2f}", a["flagged"], f"{a['tri_summary']:.2f}", f"{a['tri_full']:.2f}"] for s, a in audit.items()])


def verbatim_table(res):
    """Means without the queries that repeat a target's summary line word for word (no intervals, see docs/method.md, "Query set")."""
    qs = res["queries"]
    rest = [q for q in qs if not q["verbatim"]]
    mean = lambda sel, k, m: sum((q["rank"][k] <= 10) if m == "r10" else (1 / q["rank"][k] if q["rank"][k] <= 10 else 0) for q in sel) / len(sel)
    rows = [[res["labels"][k], f"{mean(qs, k, 'r10'):.2f}", f"{mean(rest, k, 'r10'):.2f}", f"{mean(qs, k, 'mrr'):.2f}", f"{mean(rest, k, 'mrr'):.2f}"]
            for k in ORDER if k in res["labels"] and k != "random"]
    n = len(qs) - len(rest)
    return (f"Without the {n} queries that repeat one target's summary line word for word (means only, no intervals):\n\n" +
            md(["Method", f"recall@10 all ({len(qs)})", f"recall@10 without them ({len(rest)})", f"MRR@10 all ({len(qs)})", f"MRR@10 without them ({len(rest)})"], rows))


def ablation_tables(res):
    A, st, cos = res["ablations"], res["ablations"]["stats"], res["ablations"]["mean_pairwise_cosine"]
    row = lambda n, name, extra=(): [name, *extra, ci(st[n]["all"]["r10"]), ci(st[n]["all"]["mrr"]), ci(st[n]["all"]["d_r10"], True)]
    last = lambda n: f"{A['loss'][n][-1][1]:.3f}"
    out = "Collapse without negatives (mean cosine over all pairs of document vectors; 1.0 means all vectors are the same):\n\n"
    out += md(["Model", "mean pairwise cosine", "recall@10", "MRR@10", "recall@10 difference to trained model (paired)"],
              [row("model:" + n, name, (f"{cos[n]:.4f}",)) for n, name in
               (("untrained", "untrained (random initial weights)"), ("main", "trained with in-batch negatives"), ("no_negatives", "trained without negatives"))])
    out += "\nTemperature (training loss values are not comparable across temperatures; loss is the mean of the last 100 steps):\n\n"
    out += md(["Temperature", "final loss", "mean pairwise cosine", "recall@10", "MRR@10", "recall@10 difference to tau = 0.1 (paired)", "recall@10 low", "mid", "high"],
              [[f"tau = {t}", last(n), f"{cos[n]:.4f}", ci(st["model:" + n]["all"]["r10"]), ci(st["model:" + n]["all"]["mrr"]), ci(st["model:" + n]["all"]["d_r10"], True)] +
               [f"{st['model:' + n][s]['r10'][0]:.2f}" for s in SUBS[1:]] for n, t in (("tau_0.02", 0.02), ("main", 0.1), ("tau_1.0", 1.0))])
    out += "\nSeeds, and the benchmark's target documents held out of training (interval is for each row alone):\n\n"
    out += md(["Run", "recall@10", "MRR@10", "recall@10 difference to seed 0 (paired)"],
              [row("model:" + n, name) for n, name in (("main", "seed 0 (the model used everywhere else)"), ("seed_1", "seed 1"), ("seed_2", "seed 2"),
                                                       ("held_out", f"seed 0, {res['meta']['n_targets']} target documents excluded from training pairs"))])
    out += "\nVector size (first k of 64 dimensions, renormalized; int8 = one shared scale, documents only):\n\n"
    out += md(["Dimensions", "bytes per vector float32 / int8", "recall@10 float32", "recall@10 int8", "MRR@10 float32", "MRR@10 int8", "recall@10 float32 difference to 64 dimensions (paired)"],
              [[k, f"{4 * k} / {k}"] + [f"{st[f'dims:{k}:{q}']['all'][m][0]:.2f}" for m in ("r10", "mrr") for q in ("float32", "int8")] + [ci(st[f"dims:{k}:float32"]["all"]["d_r10"], True)]
               for k in (64, 32, 16, 8)])
    steps = [s for s, _ in A["loss"]["main"]]
    out += "\nTraining loss (mean over the batches since the previous row):\n\n"
    out += md(["Step", "tau = 0.02", "tau = 0.1", "tau = 1.0"], [[s] + [f"{dict(A['loss'][n])[s]:.3f}" for n in ("tau_0.02", "main", "tau_1.0")] for s in steps])
    return out


def tables(res):
    m, meta = res["docstrings"], res["meta"]
    out = f"{headline(res)}\n\nDocstring benchmark: {meta['n_docs']} documents, {meta['n_queries']} queries in {max(q['group'] for q in res['queries']) + 1} target groups.\n"
    out += main_tables(m["methods"], res["labels"]) + "\nLeak audit and overlap tiers (flagged = at least half of the query's word trigrams occur in the corpus text):\n\n" + audit_table(m["audit"])
    out += "\n" + verbatim_table(res) + "\n" + head_to_head(res) + "\n" + ablation_tables(res)
    if res["squad"]:
        s = res["squad"]
        out += (f"\nSQuAD v1.1 dev check: {s['methods']['random']['all']['n']} questions against {s['n_docs']} paragraphs; own-64 retrained on the paragraphs "
                f"(first sentence paired with the rest). Resampling unit: paragraph.\n" + main_tables(s["methods"], res["labels"]) +
                "\nLeak audit on SQuAD (question trigrams against the paragraph texts):\n\n" + audit_table(s["audit"]))
    else:
        out += "\nSQuAD check: not run (data not fetched).\n"
    return out


def write_all(res, vectors):
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "results.json").write_text(json.dumps(res, indent=1), encoding="utf8")
    text = tables(res)
    (ROOT / "results" / "tables.md").write_text(text, encoding="utf8")
    payload = json.dumps({**res, "vectors": vectors, "headline": headline(res), "tables": text}, separators=(",", ":")).replace("</", "<\\/")
    page = ROOT / "web" / "index.html"
    s = page.read_text(encoding="utf8")
    page.write_text(re.sub(r'(<script id="data" type="application/json">).*?(</script>)', lambda m: m.group(1) + payload + m.group(2), s, flags=re.S), encoding="utf8")
