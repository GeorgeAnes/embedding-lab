# Embedding Lab

Embedding Lab trains a 64-dimension two-tower embedding model from scratch in NumPy on the docstrings of NumPy and SciPy. It uses a symmetric contrastive loss with in-batch negatives and an analytic gradient that a test checks against finite differences. I compared the trained model with lexical and static-vector baselines on the same text and the same queries, and the trained model loses to BM25.

On the 99 docstring queries own-64 loses to BM25 on full text: recall@10 0.44 against 0.69 (difference -0.24, 95% interval [-0.33, -0.16]) and MRR@10 0.36 against 0.48 (difference -0.12, [-0.21, -0.04]). Without the 6 queries that repeat a target's summary line word for word (93 left, no interval): recall@10 0.41 against 0.67. Measured on 30 September 2026 with Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1.

The project shows how an embedding model is produced and measured. It is not a competitive retriever. A static page, `web/index.html`, shows the results and lets you open any query to see how each method ranked it.

## Results

<!-- summary:start -->
| Method | recall@10 | MRR@10 | recall@10 without the 6 summary-line queries |
|---|---|---|---|
| random | 0.01 [0.00, 0.03] | 0.01 [0.00, 0.02] | 0.01 |
| TF-IDF (auditor), summary text | 0.45 [0.40, 0.51] | 0.36 [0.30, 0.40] | 0.42 |
| TF-IDF (auditor), full text | 0.55 [0.46, 0.63] | 0.33 [0.26, 0.40] | 0.52 |
| BM25, summary text | 0.45 [0.40, 0.51] | 0.36 [0.31, 0.41] | 0.42 |
| BM25, full text | 0.69 [0.62, 0.76] | 0.48 [0.40, 0.56] | 0.67 |
| LSA-64, full text | 0.35 [0.27, 0.43] | 0.15 [0.10, 0.20] | 0.32 |
| static vectors (WordLlama 256-d), full text | 0.51 [0.39, 0.61] | 0.24 [0.18, 0.30] | 0.47 |
| own-64, full text | 0.44 [0.39, 0.49] | 0.36 [0.31, 0.41] | 0.41 |
| hybrid: BM25 full + own-64, RRF k=60 | 0.58 [0.51, 0.65] | 0.40 [0.35, 0.45] | 0.55 |

<!-- summary:end -->

`own-64` is the trained model. `TF-IDF (auditor)` is the TF-IDF of my [Enterprise AI Document Risk Auditor](https://github.com/GeorgeAnes/enterprise-ai-document-risk-auditor) (raw term counts, smoothed idf). `hybrid` fuses BM25 and own-64 by Reciprocal Rank Fusion. Intervals are 95% percentile intervals from a paired cluster bootstrap over the 33 target groups. The tables by overlap tier, the paired differences, the ablations and the SQuAD check are in [docs/results.md](docs/results.md).

Besides the headline:

- Training helps a little. Recall@10 is 0.44 against 0.35 for the same model before training (a paired difference of -0.09 for the untrained model, interval [-0.14, -0.05]).
- Negatives matter. Without them the vectors collapse (mean pairwise cosine 0.9960 against 0.0001) and recall@10 is 0.33.
- The hybrid scores 0.58, below BM25 alone (difference -0.11, interval [-0.20, -0.04]).
- On SQuAD dev, with own-64 retrained on the paragraphs, BM25 also leads: recall@10 0.94 against 0.44.

## Run It

Tested with Python 3.11 on Linux. The PowerShell commands are the same steps for Windows and have not been run.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m pytest -q
python -m lab eval
```

The same steps in bash:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
python -m lab eval
```

Then open `web/index.html` in a browser (no server needed).

`pytest` runs 20 tests in a few seconds, including the gradient check and a browser test of the page. The browser test needs Chromium (`python -m playwright install chromium`) and is skipped without it. `python -m lab eval` trains the models, evaluates every method and rewrites `results/`, `web/index.html`, the table above and the tables in `docs/results.md`. It takes about three minutes on two cores. On first use it downloads SQuAD v1.1 dev (about 4.9 MB) from GitHub into `data/squad/`, which is git-ignored; `--skip-squad` skips that check. Timings, the options of `python -m lab` and the row for a real embedding model are in [docs/running.md](docs/running.md).

## Repository Structure

```text
embedding-lab/
  lab/
    contrastive.py   The model: a word table per tower, mean pooling, symmetric InfoNCE, analytic gradient, Adam. Read this first.
    lexical.py       TF-IDF as in the auditor, BM25 (k1 1.2, b 0.75, not tuned), LSA-64 by SVD of the TF-IDF matrix
    text.py          The auditor's tokenizer and stop words, overlap tiers, trigram containment for the leak audit
    corpus.py        Builds data/corpus.jsonl from the installed NumPy and SciPy docstrings
    vecops.py        Truncation, int8 quantization, Reciprocal Rank Fusion, ranks with fixed tie-breaking, the collapse diagnostic
    dense.py         WordLlama static vectors (from PyPI) and a client for an OpenAI-compatible embeddings endpoint
    squad.py         Fetches and reads SQuAD v1.1 dev
    evaluate.py      Runs every method: recall@10, MRR@10, tiers, leak audit, paired cluster bootstrap, ablations
    report.py        Writes the tables, results/ and the data block of web/index.html
  data/              corpus.jsonl (1,714 docstrings), queries.json (99 queries, frozen), NOTICE.md
  tests/             Gradient check, collapse test, TF-IDF parity with the auditor's code, BM25 against the formula,
                     tier and leak-audit tests, a stub-server test of the real-model client, a browser test
  results/           results.json and tables.md from the run described above
  web/index.html     One static page with no external assets: results chart, all tables, a per-query inspector
  docs/              Method, results, running, limitations, Azure mapping
  Dockerfile         Runs python -m lab eval in a container
  .github/workflows/ci.yml   Runs the tests and python -m lab eval, and builds the image
```

## Method in Brief

Every word has a learned 64-number vector, and a text is the mean of its words' vectors scaled to length 1. There are two towers, meaning two separate word tables: tower A reads short text (a summary line or a query) and tower B reads long text (the rest of a docstring). Training pairs are (summary line, rest of the same docstring). In a batch of 256 pairs the other 255 are the negatives, and the loss is

```
L = 1/2 ( -1/B sum_i log softmax_j(S[i,:])[i]  +  -1/B sum_j log softmax_i(S[:,j])[j] ),   S = Za Zb^T / tau
```

where `Za` and `Zb` are the unit vectors of the batch and `tau` is the temperature. There is no autodiff library. The gradient, in the order the code computes it (`P` and `Q` are the row-wise and column-wise softmax of `S`, `I` is the identity matrix):

```
dL/dS = ((P - I) + (Q - I)) / (2B)
dL/dZa = dL/dS  Zb / tau,   dL/dZb = (dL/dS)^T Za / tau
dL/du = (dL/dz - z (z . dL/dz)) / |u|           because z = u / |u|
dL/dW = X^T dL/du                               because u = x W
```

`tests/test_contrastive.py` compares it with central finite differences for every entry of both word tables of a small random problem (30 words, 5 dimensions, 6 pairs), with and without negatives, and requires agreement to within 1e-6. Training is Adam for 800 steps with tau 0.1, word dropout 0.2 and seed 0. A document vector is tower A on the summary plus tower B on the rest, scaled to length 1; a query vector is tower A on the query. The derivation is in [docs/method.md](docs/method.md).

The benchmark is 1,714 documents and 99 queries in 33 target groups. A query counts as answered at the rank of its best-ranked acceptable target. I tiered the queries by word overlap with their target (low below 0.5, mid from 0.5 to 0.8, high from 0.8) and ran a trigram leak audit on them. BM25 (k1 1.2, b 0.75) and the Reciprocal Rank Fusion constant (k=60) were fixed in advance, and no method was tuned on the queries.

## Data

- The corpus, `data/corpus.jsonl`, has 1,714 records (id, summary line, full text without Examples sections) from 23 NumPy and SciPy modules. I built it with `python -m lab corpus` from NumPy 2.4.6 and SciPy 1.17.1 on 30 September 2026. Rebuilding on other versions gives a different corpus and different numbers, so the committed file is the one every reported number used. On the same versions a rebuild differs in one record, whose docstring prints a set in hash-seed order (details in [docs/method.md](docs/method.md)).
- The queries, `data/queries.json`, are 33 target groups with three phrasings each. Each group has one to five acceptable target functions. The file is frozen: its sha256 is in `data/queries.sha256` and a test fails if it changes.
- Tiers and the leak audit are described in [docs/method.md](docs/method.md). The audit flags a query when at least half of its word trigrams occur in the corpus text.

## Limitations

- The query set is small: 99 queries in 33 target groups, written by one author, who also chose the acceptable targets. No second person checked them.
- 6 of the 99 queries are word for word the summary line of one of their targets, and the model trained on that exact text. That is why the headline gives the numbers without them (0.41 against 0.67). Removing them lowers the overall recall@10 of every method except random by 0.02 to 0.04 and does not change the order of the methods.
- 11 queries repeat a query from an earlier draft of the query list. The draft is not in this repository; the 11 queries are listed in [docs/method.md](docs/method.md). On the 86 queries left after removing both sets, BM25 on full text has recall@10 0.65 and own-64 0.37, and the order of the methods by recall@10 is again unchanged. This is not part of `python -m lab eval`; a one-line command in [docs/method.md](docs/method.md) repeats it from `results/results.json`.
- The model trains on the documents it searches. The leak audit flagged 32 of the 99 queries (none in the low tier, 4 in mid, 28 in high) and I did not remove them. A run that leaves the 74 target documents out of training scores the same within noise, on one seed.
- The encoder is a bag of word vectors. It ignores word order and drops every word outside its 6,709-word vocabulary, and it trained on about 1,700 pairs. No transformer, hard negatives or larger training set was tried, so the result says nothing about embedding models in general.
- The intervals resample 33 target groups with the percentile method. They are approximate and probably somewhat narrow, and the per-tier intervals are wide (the low tier has 37 queries).
- Not validated: running the Docker image (CI builds it and does not run it), PowerShell and Windows (CI runs on Ubuntu), Python versions other than 3.11, the real-model row against a real endpoint, the [Azure mapping](docs/azure-mapping.md), and rebuilding the corpus on other NumPy or SciPy versions.

The full list is in [docs/limitations.md](docs/limitations.md).

## License

Published as a portfolio artifact, all rights reserved (see `LICENSE`). The corpus is derived from NumPy and SciPy docstrings, which are BSD 3-Clause; their notices are in `data/NOTICE.md`. SQuAD is downloaded by the user and not redistributed. WordLlama is installed from PyPI and not vendored.
