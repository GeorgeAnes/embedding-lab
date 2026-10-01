# Running it

The README has the short version. This page has the command-line options, the timings I measured and the real-model row.

## Commands

```
python -m lab eval [--real] [--skip-squad]
python -m lab corpus
python -m lab fetch-squad
```

- `eval` trains the models, evaluates every method and writes `results/results.json`, `results/tables.md`, the data block of `web/index.html` and, when the files exist, the table in `README.md` and the block in `docs/results.md`. All of these are tracked files, so a run changes them: the date and the timings differ every time, the metrics do not.
- `eval` downloads SQuAD v1.1 dev (about 4.9 MB) from GitHub into `data/squad/` on first use. That folder is git-ignored and the file is never committed. `--skip-squad` skips the SQuAD check, and `fetch-squad` only downloads the file. The files are rewritten without the check: `results/results.json` and `web/index.html` hold no SQuAD data, and `docs/results.md` and `results/tables.md` lose their SQuAD tables. `git checkout -- results web docs README.md` restores the committed versions. The same happens if the download fails.
- `corpus` rebuilds `data/corpus.jsonl` from the installed NumPy and SciPy. Other versions give a different corpus and different numbers (see "Corpus" in [method.md](method.md)).
- The browser test needs Chromium (`python -m playwright install chromium`). Without it the test is skipped.

## Timings

Measured on a 2-core Linux VM (30 September 2026): `pip install -r requirements-dev.txt` into a new venv 21 s; `python -m pytest -q` 5 to 6 s (21 tests); `python -m lab eval` 145 to 188 s in full runs with nothing else running (261 s once, with other jobs running; 179.3 s is the run recorded in `results/results.json`; the spread is unexplained). One training run takes 15 to 41 s (the untrained model takes no time), and the seven trainings run in a pool with one worker per core, so two at a time here, followed by one SQuAD training of 21 to 25 s. The metrics were identical across all runs; only the timings differ.

Training runs execute in parallel worker processes with one BLAS thread each (`lab/__init__.py`, `lab/evaluate.py`). With the default multi-threaded BLAS the same training was 2 to 3 times slower on that VM.

## Real model row

`--real` adds one row to the docstring benchmark, taken from an OpenAI-compatible embeddings endpoint. The SQuAD check does not use it. I have not run it against a real model (see [limitations.md](limitations.md)).

```bash
export LAB_EMBED_BASE_URL=http://localhost:1234/v1     # LM Studio, or another OpenAI-compatible server
export LAB_EMBED_MODEL=<embedding model name>
export LAB_EMBED_KEY=<key, if the server needs one>
python -m lab eval --real
```

```powershell
$env:LAB_EMBED_BASE_URL = "http://localhost:1234/v1"
$env:LAB_EMBED_MODEL = "<embedding model name>"
$env:LAB_EMBED_KEY = "<key, if the server needs one>"
python -m lab eval --real
```

The client (`lab/dense.py`) posts `{"model": ..., "input": [...]}` to `<base URL>/embeddings` with `urllib` and reads `data[i].embedding`. Documents are cut to 6,000 characters before sending. A test runs this client against a local stub server, so the request and response handling is tested, but no real model was called.
