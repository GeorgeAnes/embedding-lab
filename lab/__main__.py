"""python -m lab eval [--real] [--skip-squad]   |   python -m lab corpus   |   python -m lab fetch-squad"""
import argparse


def main():
    ap = argparse.ArgumentParser(prog="python -m lab", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("eval", help="train, evaluate, write results/ and web/index.html")
    ev.add_argument("--real", action="store_true", help="add a row from an OpenAI-compatible endpoint (LAB_EMBED_BASE_URL, LAB_EMBED_MODEL, LAB_EMBED_KEY)")
    ev.add_argument("--skip-squad", action="store_true", help="do not fetch or run the SQuAD check")
    sub.add_parser("corpus", help="regenerate data/corpus.jsonl from the installed NumPy and SciPy (changes the results)")
    sub.add_parser("fetch-squad", help="download SQuAD v1.1 dev into data/squad/ (git-ignored)")
    args = ap.parse_args()
    if args.cmd == "eval":
        from . import evaluate, report
        results, vectors = evaluate.run(args.real, not args.skip_squad)
        report.write_all(results, vectors)
        print(report.headline(results), f"\nWrote results/results.json, results/tables.md and web/index.html ({results['meta']['wall_seconds']} s).")
    elif args.cmd == "corpus":
        from . import corpus
        corpus.write()
    else:
        from . import squad
        print(squad.fetch())


if __name__ == "__main__":
    main()
