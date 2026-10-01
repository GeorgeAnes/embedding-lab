# Results

Measured on 30 September 2026 with Python 3.11.15, NumPy 2.4.6 and SciPy 1.17.1. The method and the data are described in [method.md](method.md).

Everything between the two markers below is written by `python -m lab eval` (`results/tables.md` has the same text). Intervals are 95% percentile intervals from a paired cluster bootstrap (1,000 resamples, seed 0) that resamples whole target groups (or whole paragraphs on SQuAD), so the three phrasings of one target are not counted as independent. All methods score the same documents for the same queries; ties in scores are broken by one fixed random order. recall@10 is the share of queries that have at least one acceptable target in the top 10 (the hit rate; it is not the share of a query's acceptable targets that were found). MRR@10 is the mean over queries of 1 / rank of the best-ranked acceptable target, counted as 0 when that rank is above 10.

The rows are `random`; `TF-IDF (auditor)`, which is the auditor's method (raw term counts, smoothed idf) on summary text and on full text; `BM25` on summary text and on full text; `LSA-64` (SVD of the TF-IDF matrix of the full text); `static vectors` (WordLlama 256-d, an installed PyPI package with bundled weights, average of token vectors); `own-64`; and `hybrid` (BM25 on full text and own-64 fused by Reciprocal Rank Fusion, k=60). "Full text" is the summary plus the rest of the docstring, and that is the text own-64 sees. The TF-IDF idf values are computed over the whole collection here; the auditor computes them over the sentences of one document. `tests/test_retrieval.py` checks that, given the same collection, the scores equal those of a copy of the auditor's functions (`tests/auditor_reference.py`; type hints removed, logic unchanged).

<!-- results:start -->
On the 99 docstring queries own-64 loses to BM25 on full text: recall@10 0.44 against 0.69 (difference -0.24, 95% interval [-0.33, -0.16]) and MRR@10 0.36 against 0.48 (difference -0.12, [-0.21, -0.04]). Without the 6 queries that repeat a target's summary line word for word (93 left, no interval): recall@10 0.41 against 0.67.

Docstring benchmark: 1714 documents, 99 queries in 33 target groups.

recall@10, mean and 95% cluster-bootstrap interval, by overlap tier:

| Method | all (n=99) | low (n=37) | mid (n=27) | high (n=35) |
|---|---|---|---|---|
| random | 0.01 [0.00, 0.03] | 0.03 [0.00, 0.09] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| TF-IDF (auditor), summary text | 0.45 [0.40, 0.51] | 0.05 [0.00, 0.13] | 0.41 [0.24, 0.60] | 0.91 [0.81, 1.00] |
| TF-IDF (auditor), full text | 0.55 [0.46, 0.63] | 0.14 [0.03, 0.24] | 0.70 [0.54, 0.85] | 0.86 [0.73, 0.97] |
| BM25, summary text | 0.45 [0.40, 0.51] | 0.08 [0.00, 0.17] | 0.41 [0.24, 0.60] | 0.89 [0.78, 0.97] |
| BM25, full text | 0.69 [0.62, 0.76] | 0.24 [0.12, 0.38] | 0.89 [0.78, 1.00] | 1.00 [1.00, 1.00] |
| LSA-64, full text | 0.35 [0.27, 0.43] | 0.08 [0.00, 0.17] | 0.37 [0.18, 0.57] | 0.63 [0.46, 0.78] |
| static vectors (WordLlama 256-d), full text | 0.51 [0.39, 0.61] | 0.27 [0.14, 0.43] | 0.56 [0.37, 0.76] | 0.71 [0.55, 0.86] |
| own-64, full text | 0.44 [0.39, 0.49] | 0.08 [0.00, 0.17] | 0.33 [0.16, 0.54] | 0.91 [0.81, 1.00] |
| hybrid: BM25 full + own-64, RRF k=60 | 0.58 [0.51, 0.65] | 0.14 [0.05, 0.24] | 0.67 [0.46, 0.87] | 0.97 [0.91, 1.00] |

MRR@10, mean and 95% cluster-bootstrap interval, by overlap tier:

| Method | all (n=99) | low (n=37) | mid (n=27) | high (n=35) |
|---|---|---|---|---|
| random | 0.01 [0.00, 0.02] | 0.01 [0.00, 0.04] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| TF-IDF (auditor), summary text | 0.36 [0.30, 0.40] | 0.04 [0.00, 0.10] | 0.17 [0.07, 0.30] | 0.83 [0.73, 0.93] |
| TF-IDF (auditor), full text | 0.33 [0.26, 0.40] | 0.09 [0.02, 0.18] | 0.28 [0.17, 0.40] | 0.62 [0.49, 0.75] |
| BM25, summary text | 0.36 [0.31, 0.41] | 0.04 [0.00, 0.10] | 0.19 [0.09, 0.33] | 0.84 [0.73, 0.94] |
| BM25, full text | 0.48 [0.40, 0.56] | 0.11 [0.04, 0.21] | 0.62 [0.47, 0.79] | 0.76 [0.63, 0.87] |
| LSA-64, full text | 0.15 [0.10, 0.20] | 0.02 [0.00, 0.04] | 0.16 [0.05, 0.28] | 0.29 [0.16, 0.41] |
| static vectors (WordLlama 256-d), full text | 0.24 [0.18, 0.30] | 0.07 [0.03, 0.12] | 0.28 [0.15, 0.44] | 0.39 [0.26, 0.52] |
| own-64, full text | 0.36 [0.31, 0.41] | 0.04 [0.00, 0.09] | 0.21 [0.08, 0.35] | 0.82 [0.72, 0.92] |
| hybrid: BM25 full + own-64, RRF k=60 | 0.40 [0.35, 0.45] | 0.06 [0.01, 0.12] | 0.34 [0.21, 0.50] | 0.81 [0.69, 0.92] |

Paired difference to BM25, full text (same resamples for both methods; an interval that excludes 0 is a difference this run can see):

| Method and metric | all (n=99) | low (n=37) | mid (n=27) | high (n=35) |
|---|---|---|---|---|
| TF-IDF (auditor), summary text: recall@10 | -0.23 [-0.33, -0.15] | -0.19 [-0.34, -0.08] | -0.48 [-0.67, -0.32] | -0.09 [-0.19, +0.00] |
| TF-IDF (auditor), summary text: MRR@10 | -0.12 [-0.23, -0.04] | -0.07 [-0.16, -0.01] | -0.45 [-0.63, -0.26] | +0.08 [-0.05, +0.21] |
| TF-IDF (auditor), full text: recall@10 | -0.14 [-0.20, -0.08] | -0.11 [-0.22, -0.03] | -0.19 [-0.33, -0.07] | -0.14 [-0.27, -0.03] |
| TF-IDF (auditor), full text: MRR@10 | -0.15 [-0.22, -0.09] | -0.02 [-0.08, +0.04] | -0.34 [-0.49, -0.21] | -0.14 [-0.24, -0.03] |
| BM25, summary text: recall@10 | -0.23 [-0.33, -0.15] | -0.16 [-0.33, -0.03] | -0.48 [-0.67, -0.32] | -0.11 [-0.22, -0.03] |
| BM25, summary text: MRR@10 | -0.12 [-0.22, -0.03] | -0.07 [-0.16, -0.00] | -0.43 [-0.60, -0.24] | +0.08 [-0.04, +0.20] |
| LSA-64, full text: recall@10 | -0.33 [-0.44, -0.24] | -0.16 [-0.30, -0.03] | -0.52 [-0.71, -0.32] | -0.37 [-0.54, -0.22] |
| LSA-64, full text: MRR@10 | -0.33 [-0.42, -0.25] | -0.09 [-0.19, -0.02] | -0.46 [-0.67, -0.27] | -0.47 [-0.60, -0.35] |
| static vectors (WordLlama 256-d), full text: recall@10 | -0.18 [-0.30, -0.08] | +0.03 [-0.14, +0.21] | -0.33 [-0.55, -0.10] | -0.29 [-0.45, -0.14] |
| static vectors (WordLlama 256-d), full text: MRR@10 | -0.24 [-0.33, -0.16] | -0.04 [-0.13, +0.04] | -0.34 [-0.54, -0.16] | -0.37 [-0.51, -0.23] |
| own-64, full text: recall@10 | -0.24 [-0.33, -0.16] | -0.16 [-0.32, -0.03] | -0.56 [-0.74, -0.38] | -0.09 [-0.19, +0.00] |
| own-64, full text: MRR@10 | -0.12 [-0.21, -0.04] | -0.08 [-0.16, -0.01] | -0.41 [-0.60, -0.22] | +0.06 [-0.06, +0.19] |
| hybrid: BM25 full + own-64, RRF k=60: recall@10 | -0.11 [-0.20, -0.04] | -0.11 [-0.25, +0.03] | -0.22 [-0.39, -0.08] | -0.03 [-0.09, +0.00] |
| hybrid: BM25 full + own-64, RRF k=60: MRR@10 | -0.08 [-0.15, -0.02] | -0.05 [-0.12, -0.00] | -0.28 [-0.44, -0.10] | +0.06 [-0.05, +0.17] |

Leak audit and overlap tiers (flagged = at least half of the query's word trigrams occur in the corpus text):

| Subset | queries | mean overlap | flagged by leak audit | mean trigram containment, summaries | ... full texts |
|---|---|---|---|---|---|
| all | 99 | 0.60 | 32 | 0.27 | 0.36 |
| low | 37 | 0.25 | 0 | 0.03 | 0.08 |
| mid | 27 | 0.62 | 4 | 0.10 | 0.23 |
| high | 35 | 0.97 | 28 | 0.65 | 0.75 |

Without the 6 queries that repeat one target's summary line word for word (means only, no intervals):

| Method | recall@10 all (99) | recall@10 without them (93) | MRR@10 all (99) | MRR@10 without them (93) |
|---|---|---|---|---|
| TF-IDF (auditor), summary text | 0.45 | 0.42 | 0.36 | 0.31 |
| TF-IDF (auditor), full text | 0.55 | 0.52 | 0.33 | 0.30 |
| BM25, summary text | 0.45 | 0.42 | 0.36 | 0.32 |
| BM25, full text | 0.69 | 0.67 | 0.48 | 0.46 |
| LSA-64, full text | 0.35 | 0.32 | 0.15 | 0.12 |
| static vectors (WordLlama 256-d), full text | 0.51 | 0.47 | 0.24 | 0.22 |
| own-64, full text | 0.44 | 0.41 | 0.36 | 0.32 |
| hybrid: BM25 full + own-64, RRF k=60 | 0.58 | 0.55 | 0.40 | 0.37 |

Head to head at rank 10, own-64 against BM25 on full text, counted over queries:

| Tier | queries | only own-64 in top 10 | only BM25 full in top 10 | both | neither |
|---|---|---|---|---|---|
| all | 99 | 1 | 25 | 43 | 30 |
| low | 37 | 1 | 7 | 2 | 27 |
| mid | 27 | 0 | 15 | 9 | 3 |
| high | 35 | 0 | 3 | 32 | 0 |

The queries where own-64 has a target in its top 10 and BM25 on full text does not:

| Query | tier | own-64 rank | BM25 full rank |
|---|---|---|---|
| fit the coefficients of a linear model to an over-determined set of measurements | low | 5 | 26 |

Collapse without negatives (mean cosine over all pairs of document vectors; 1.0 means all vectors are the same):

| Model | mean pairwise cosine | recall@10 | MRR@10 | recall@10 difference to trained model (paired) |
|---|---|---|---|---|
| untrained (random initial weights) | 0.1265 | 0.35 [0.30, 0.40] | 0.31 [0.26, 0.35] | -0.09 [-0.14, -0.05] |
| trained with in-batch negatives | 0.0001 | 0.44 [0.39, 0.49] | 0.36 [0.31, 0.41] | +0.00 [+0.00, +0.00] |
| trained without negatives | 0.9960 | 0.33 [0.27, 0.39] | 0.25 [0.19, 0.30] | -0.11 [-0.16, -0.06] |

Temperature (training loss values are not comparable across temperatures; loss is the mean of the last 100 steps):

| Temperature | final loss | mean pairwise cosine | recall@10 | MRR@10 | recall@10 difference to tau = 0.1 (paired) | recall@10 low | mid | high |
|---|---|---|---|---|---|---|---|---|
| tau = 0.02 | 0.254 | 0.1236 | 0.44 [0.39, 0.49] | 0.36 [0.31, 0.41] | +0.00 [-0.05, +0.04] | 0.14 | 0.22 | 0.94 |
| tau = 0.1 | 0.280 | 0.0001 | 0.44 [0.39, 0.49] | 0.36 [0.31, 0.41] | +0.00 [+0.00, +0.00] | 0.08 | 0.33 | 0.91 |
| tau = 1.0 | 4.606 | -0.0002 | 0.46 [0.39, 0.54] | 0.32 [0.28, 0.36] | +0.02 [-0.05, +0.09] | 0.11 | 0.37 | 0.91 |

Seeds, and the benchmark's target documents held out of training (interval is for each row alone):

| Run | recall@10 | MRR@10 | recall@10 difference to seed 0 (paired) |
|---|---|---|---|
| seed 0 (the model used everywhere else) | 0.44 [0.39, 0.49] | 0.36 [0.31, 0.41] | +0.00 [+0.00, +0.00] |
| seed 1 | 0.44 [0.39, 0.49] | 0.35 [0.30, 0.39] | +0.00 [-0.04, +0.04] |
| seed 2 | 0.47 [0.42, 0.53] | 0.36 [0.31, 0.40] | +0.03 [-0.03, +0.09] |
| seed 0, 74 target documents excluded from training pairs | 0.46 [0.40, 0.53] | 0.32 [0.27, 0.38] | +0.02 [-0.03, +0.08] |

Vector size (first k of 64 dimensions, renormalized; int8 = one shared scale, documents only):

| Dimensions | bytes per vector float32 / int8 | recall@10 float32 | recall@10 int8 | MRR@10 float32 | MRR@10 int8 | recall@10 float32 difference to 64 dimensions (paired) |
|---|---|---|---|---|---|---|
| 64 | 256 / 64 | 0.44 | 0.44 | 0.36 | 0.36 | +0.00 [+0.00, +0.00] |
| 32 | 128 / 32 | 0.41 | 0.41 | 0.35 | 0.35 | -0.03 [-0.07, +0.01] |
| 16 | 64 / 16 | 0.35 | 0.35 | 0.29 | 0.29 | -0.09 [-0.14, -0.04] |
| 8 | 32 / 8 | 0.26 | 0.26 | 0.19 | 0.19 | -0.18 [-0.26, -0.10] |

Training loss (mean over the batches since the previous row):

| Step | tau = 0.02 | tau = 0.1 | tau = 1.0 |
|---|---|---|---|
| 1 | 17.298 | 6.319 | 5.559 |
| 100 | 1.457 | 0.987 | 4.756 |
| 200 | 0.418 | 0.378 | 4.637 |
| 300 | 0.366 | 0.335 | 4.623 |
| 400 | 0.320 | 0.307 | 4.615 |
| 500 | 0.302 | 0.299 | 4.613 |
| 600 | 0.292 | 0.294 | 4.610 |
| 700 | 0.278 | 0.294 | 4.608 |
| 800 | 0.254 | 0.280 | 4.606 |

SQuAD v1.1 dev check: 10570 questions against 2067 paragraphs; own-64 retrained on the paragraphs (first sentence paired with the rest). Resampling unit: paragraph.

recall@10, mean and 95% cluster-bootstrap interval, by overlap tier:

| Method | all (n=10570) | low (n=3133) | mid (n=5907) | high (n=1530) |
|---|---|---|---|---|
| random | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| TF-IDF (auditor), full text | 0.90 [0.89, 0.91] | 0.75 [0.73, 0.77] | 0.96 [0.95, 0.96] | 0.99 [0.99, 1.00] |
| BM25, full text | 0.94 [0.93, 0.94] | 0.80 [0.78, 0.82] | 0.99 [0.99, 0.99] | 1.00 [1.00, 1.00] |
| LSA-64, full text | 0.53 [0.52, 0.55] | 0.38 [0.36, 0.41] | 0.57 [0.56, 0.59] | 0.68 [0.66, 0.71] |
| static vectors (WordLlama 256-d), full text | 0.85 [0.84, 0.86] | 0.76 [0.74, 0.78] | 0.88 [0.87, 0.89] | 0.90 [0.89, 0.92] |
| own-64, full text | 0.44 [0.43, 0.45] | 0.31 [0.29, 0.32] | 0.49 [0.48, 0.51] | 0.54 [0.51, 0.56] |
| hybrid: BM25 full + own-64, RRF k=60 | 0.65 [0.64, 0.66] | 0.51 [0.49, 0.53] | 0.71 [0.69, 0.72] | 0.72 [0.70, 0.75] |

MRR@10, mean and 95% cluster-bootstrap interval, by overlap tier:

| Method | all (n=10570) | low (n=3133) | mid (n=5907) | high (n=1530) |
|---|---|---|---|---|
| random | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| TF-IDF (auditor), full text | 0.70 [0.69, 0.71] | 0.48 [0.47, 0.50] | 0.77 [0.76, 0.78] | 0.87 [0.86, 0.89] |
| BM25, full text | 0.82 [0.81, 0.83] | 0.57 [0.55, 0.59] | 0.91 [0.90, 0.91] | 0.98 [0.97, 0.98] |
| LSA-64, full text | 0.28 [0.27, 0.29] | 0.17 [0.15, 0.18] | 0.30 [0.29, 0.32] | 0.41 [0.39, 0.44] |
| static vectors (WordLlama 256-d), full text | 0.62 [0.61, 0.63] | 0.51 [0.49, 0.53] | 0.66 [0.65, 0.67] | 0.72 [0.70, 0.74] |
| own-64, full text | 0.31 [0.30, 0.32] | 0.19 [0.17, 0.20] | 0.35 [0.34, 0.37] | 0.42 [0.39, 0.44] |
| hybrid: BM25 full + own-64, RRF k=60 | 0.45 [0.44, 0.46] | 0.31 [0.29, 0.32] | 0.50 [0.49, 0.51] | 0.55 [0.52, 0.57] |

Paired difference to BM25, full text (same resamples for both methods; an interval that excludes 0 is a difference this run can see):

| Method and metric | all (n=10570) | low (n=3133) | mid (n=5907) | high (n=1530) |
|---|---|---|---|---|
| TF-IDF (auditor), full text: recall@10 | -0.04 [-0.04, -0.03] | -0.05 [-0.06, -0.04] | -0.03 [-0.04, -0.03] | -0.01 [-0.01, -0.00] |
| TF-IDF (auditor), full text: MRR@10 | -0.12 [-0.12, -0.11] | -0.09 [-0.10, -0.07] | -0.13 [-0.14, -0.13] | -0.10 [-0.12, -0.09] |
| LSA-64, full text: recall@10 | -0.41 [-0.42, -0.39] | -0.42 [-0.44, -0.40] | -0.42 [-0.44, -0.40] | -0.32 [-0.34, -0.29] |
| LSA-64, full text: MRR@10 | -0.54 [-0.55, -0.53] | -0.40 [-0.42, -0.38] | -0.60 [-0.62, -0.59] | -0.56 [-0.59, -0.54] |
| static vectors (WordLlama 256-d), full text: recall@10 | -0.09 [-0.10, -0.08] | -0.04 [-0.06, -0.02] | -0.11 [-0.12, -0.10] | -0.10 [-0.11, -0.08] |
| static vectors (WordLlama 256-d), full text: MRR@10 | -0.19 [-0.20, -0.18] | -0.06 [-0.08, -0.04] | -0.25 [-0.26, -0.23] | -0.26 [-0.28, -0.24] |
| own-64, full text: recall@10 | -0.49 [-0.51, -0.48] | -0.50 [-0.52, -0.48] | -0.50 [-0.52, -0.49] | -0.46 [-0.49, -0.44] |
| own-64, full text: MRR@10 | -0.50 [-0.51, -0.49] | -0.38 [-0.40, -0.36] | -0.55 [-0.57, -0.54] | -0.56 [-0.59, -0.54] |
| hybrid: BM25 full + own-64, RRF k=60: recall@10 | -0.29 [-0.30, -0.28] | -0.29 [-0.31, -0.27] | -0.29 [-0.30, -0.27] | -0.28 [-0.30, -0.25] |
| hybrid: BM25 full + own-64, RRF k=60: MRR@10 | -0.37 [-0.38, -0.36] | -0.26 [-0.28, -0.25] | -0.41 [-0.42, -0.39] | -0.43 [-0.46, -0.40] |

Leak audit on SQuAD (question trigrams against the paragraph texts):

| Subset | queries | mean overlap | flagged by leak audit | mean trigram containment, summaries | ... full texts |
|---|---|---|---|---|---|
| all | 10570 | 0.57 | 1689 | 0.12 | 0.24 |
| low | 3133 | 0.33 | 105 | 0.07 | 0.14 |
| mid | 5907 | 0.61 | 886 | 0.12 | 0.25 |
| high | 1530 | 0.86 | 698 | 0.19 | 0.43 |

<!-- results:end -->

## What this shows and what it does not

It shows:

- The mechanism works as described. Training lowers the loss (table above), and the trained model retrieves better than the same model before training: recall@10 0.44 against 0.35, a paired difference of -0.09 for the untrained model with interval [-0.14, -0.05].
- Negatives matter. Without them the vectors collapse (mean pairwise cosine 0.9960 against 0.0001) and recall@10 is 0.33, no better than the untrained model's 0.35, and 0.11 below the trained model (paired interval [-0.16, -0.06]).
- On this corpus and these queries, BM25 on the full docstring is the strongest method, and own-64 does not close the gap. Head to head, own-64 has a target in its top 10 for one query where BM25 full text does not, and BM25 full text does so for 25 queries where own-64 does not.
- A random 64-dimension projection of word counts already keeps much of the lexical signal (0.35 recall@10 before any training). The paired interval for the gain from training excludes zero in this run and the gain is modest.
- The hybrid did not beat BM25 alone (difference -0.11 [-0.20, -0.04]). Fusing a stronger ranker with a weaker one by equal-weight RRF pulled the result down here.
- On the low-overlap tier, the static WordLlama vectors scored 0.27 against 0.24 for BM25 full text and 0.08 for own-64, but the intervals overlap heavily (n = 37 queries in 33 groups is too few to separate them).
- int8 quantization changed no reported number at two decimals. Cutting to 32 dimensions changed recall@10 by -0.03 with a paired interval [-0.07, +0.01] that contains 0, so this run cannot separate it from no change; 16 and 8 dimensions lowered it by 0.09 and 0.18 with intervals that exclude 0 (table above). The model was not trained for truncation, so this is the cost without that training.
- On SQuAD (human-written questions, own-64 retrained on the paragraphs), BM25 also leads. The auditor's raw-count TF-IDF is below BM25 there by 0.04 recall@10 and 0.12 MRR@10. On the docstring benchmark the auditor's TF-IDF on full text is 0.14 recall@10 below BM25.

It does not show:

- That embeddings are worse than lexical search in general. This model is a bag of word vectors trained on about 1,700 pairs. The static WordLlama row and the SQuAD numbers already show how much depends on the model and the text.
- Anything about a transformer embedding model, a model trained on far more data, or hard negatives. None was built or run.
- That the temperature, seed or held-out differences mean anything. The three temperatures (0.44, 0.44, 0.46 recall@10), three seeds (0.44, 0.44, 0.47) and the held-out row (0.46) differ by less than their intervals are wide, and every paired recall@10 interval in those two tables contains 0.
- That the trained model generalizes to unseen documents. It trained on the documents it searches. The row that leaves the 74 target documents out of the training pairs scores the same within noise, which is weak evidence that it does not simply memorize them, on one seed. (The vocabulary is still built from all documents, so the words of the left-out documents are in it.)
- Anything about the auditor's claim-to-evidence task. Both benchmarks here are query-to-document retrieval.

The limitations of the data and of the setup are in [limitations.md](limitations.md).
