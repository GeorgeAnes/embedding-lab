# Method

How the model works, where the data came from and how the comparison is set up. The numbers are in [results.md](results.md). "The auditor" below is my [Enterprise AI Document Risk Auditor](https://github.com/GeorgeAnes/enterprise-ai-document-risk-auditor), whose tokenizer and stop words are copied into `lab/text.py` and whose TF-IDF is reimplemented in `lab/lexical.py`.

## The mechanism

The encoder is the smallest thing that shows the idea. Every word has a learned 64-number vector. A text is the mean of its words' vectors, scaled to length 1. There are two towers, meaning two separate word tables: tower A reads short text (a summary line, or a query), tower B reads long text (the rest of a docstring). Training pairs are (summary line, rest of the same docstring). A transformer encoder differs in how a text becomes a vector: layers let each word's vector depend on the words around it and on their order, and a pooling step then turns the word vectors into one vector per text. The contrastive loss keeps the same form. "What a production embedding model does differently" below lists what else changes.

A text is a row `x` of mean-pooling weights over the vocabulary (1/n for each of its n words), `W` is the word table, `u = x W`, and `z = u / |u|` is `u` scaled to length 1, so the dot product of two `z` vectors is their cosine. For a batch of B pairs, `Za` and `Zb` are the B x 64 matrices whose rows are the unit vectors `z_a[i]` and `z_b[j]`, and `S = Za Zb^T / tau`, so `S[i,j] = z_a[i] . z_b[j] / tau`, where `tau` is the temperature. The loss is

```
L = 1/2 ( -1/B sum_i log softmax_j(S[i,:])[i]  +  -1/B sum_j log softmax_i(S[:,j])[j] )
```

Each row asks which b belongs to a_i, each column asks which a belongs to b_j; the other B-1 pairs in the batch are the negatives. The gradient, in the order the code computes it (`I` is the B x B identity matrix, `X` is the B x V matrix with one pooling row `x` per text):

```
P = row-wise softmax(S),  Q = column-wise softmax(S)
dL/dS = ((P - I) + (Q - I)) / (2B)              softmax cross-entropy: probabilities minus one-hot
dL/dZa = dL/dS  Zb / tau,   dL/dZb = (dL/dS)^T Za / tau      because S = Za Zb^T / tau
dL/du = (dL/dz - z (z . dL/dz)) / |u|           because z = u / |u| has Jacobian (I - z z^T) / |u|; row by row
dL/dW = X^T dL/du                               because u = x W, x = mean-pooling weights
```

There is no autodiff library. `tests/test_contrastive.py` compares this gradient with central finite differences for every entry of both word tables of a small random problem (30 words, 5 dimensions, 6 pairs), with and without negatives, and requires agreement to within 1e-6 (relative when the gradient is larger than 1). Training is Adam (learning rate 0.01, betas 0.9 and 0.999, epsilon 1e-8, bias-corrected), batch 256, 800 steps (about 120 passes over the 1,714 training pairs), tau 0.1, word dropout 0.2 (each word of a training text is dropped with probability 0.2; a text with a single word keeps it), seed 0. Both word tables start from independent draws of N(0, 0.1^2). A document vector is the sum of tower A on its summary and tower B on the rest, scaled to length 1; a query vector is tower A on the query. The training pairs come from the same documents that are searched. No query is itself a training pair, but 6 queries are word for word the summary line of one of their targets (see "Query set" below).

Word dropout is the only setting I chose by looking at data. I compared 0, 0.1, 0.2 and 0.3 by how well the model retrieved the rest of a docstring from its summary on 171 held-out docstring pairs, not on the benchmark queries. 0.2 and 0.3 were indistinguishable and I kept 0.2. No command in this repository reproduces that comparison. The other settings were the first values I tried.

## Data

### Corpus

`data/corpus.jsonl` was built by `python -m lab corpus` from NumPy 2.4.6 and SciPy 1.17.1 on 30 September 2026 (Python 3.11.15). It has one record per function, class or ufunc: `id`, `summary` (the docstring's first paragraph, which can run past one line: median 8 words, longest 46; "summary line" elsewhere in these pages means this field) and `full` (the docstring without Examples sections), 1,714 records in all. It covers 23 modules (listed in `lab/corpus.py`), keeps functions, classes and ufuncs, skips aliases and repeated docstrings, and leaves out `numpy.ma`. `data/NOTICE.md` carries the BSD 3-Clause notices of both projects. Regenerating on other versions gives a different corpus and different numbers; the committed file is what every reported number used, and the sha256 in `results/results.json` identifies it. On the same versions `python -m lab corpus` reproduces the file except for one record, `scipy.signal.lombscargle`: its docstring prints a Python set, and the order of the elements follows the hash seed. With `PYTHONHASHSEED=3` the rebuilt file was byte-identical to the committed one, with 0, 1 and 2 it was not. The words are the same and only their order inside that list differs. I did not rerun the evaluation on a rebuilt file.

### Query set

`data/queries.json` has 33 target groups with three phrasings each (99 queries). Each group has one to five acceptable target functions (63 of the 99 queries have more than one), and a query counts as answered at the rank of its best-ranked acceptable target. The queries and the acceptable targets were written on 30 September 2026 by one author, working from the list of function names in the corpus. No second person checked them. The file was frozen before the first full evaluation run (earlier, the rankings of two queries were printed once to check that the lexical code ran) and has not changed since; its hash is below.

- sha256 of `data/queries.json`: `752462f20f2c0e672230b25bdb11fecfb93e62b1b8f213a575c4b9226b7a30be`
- `data/queries.sha256` holds the same value and a test fails if the file changes. If you change a query, update the hash and add a line here saying what changed and why.

Change log: none.

Two properties of the queries weaken the comparison:

- 6 of the 99 queries are word for word the summary line of one of their own targets, and the model trained on that exact text as the short side of a training pair. They are reported separately: the headline in the README and the table "Without the 6 queries that repeat one target's summary line" in [results.md](results.md) give the results without them. All 6 sit in the high tier. Removing them lowers the overall recall@10 of every method except random by 0.02 to 0.04 and does not change the order of the methods.
- 11 of the queries repeat a query from an earlier draft of the query list, which was known when this file was written. The draft is not in this repository, so its comparison with `data/queries.json` (17 of the 33 target groups have the same acceptable functions as in the draft and 26 share at least one) cannot be repeated here. The 11 queries are listed below, numbered by their position in `data/queries.json` (counting from 1, three per group; the same order as `queries` in `results/results.json`). Four of the 11 are among the 6 above, and 10 of the 11 are flagged by the leak audit; both can be read from `results/results.json`. On the 86 queries left after removing both sets, BM25 on full text has recall@10 0.65 and own-64 0.37, and the order of the methods by recall@10 is again unchanged. This calculation is not part of `python -m lab eval`; the following command repeats it from `results/results.json` and prints `86 0.65 0.37` (the number of queries, then recall@10 of BM25 on full text and of own-64):

```
python -c "import json;r=json.load(open('results/results.json'))['queries'];s={31,33,37,40,61,67,76,82,85,91,97};k=[q for i,q in enumerate(r) if not q['verbatim'] and i+1 not in s];print(len(k),*[round(sum(q['rank'][m]<=10 for q in k)/len(k),2) for m in ('bm25_full','own64')])"
```

The 11 queries: 31 solve the continuous-time algebraic Riccati equation; 33 compute the optimal feedback gain for a linear quadratic regulator; 37 t-test for the means of two independent samples; 40 Latin hypercube sampling; 61 compute the one-dimensional discrete Fourier transform; 67 find peaks inside a signal based on peak properties; 76 singular value decomposition of a matrix; 82 perform hierarchical agglomerative clustering; 85 kd-tree for quick nearest-neighbor lookup; 91 solve the sparse linear system Ax=b; 97 least squares polynomial fit.

### Overlap tiers

For each query, overlap is the share of its distinct content words (the auditor's tokenizer) that occur in the full docstring of its best-matching acceptable target. Tiers: low is below 0.5, mid is 0.5 up to 0.8, high is 0.8 and above. The cut points were fixed after computing the overlap distribution of the queries (37, 27 and 35 queries) and before any retrieval method was run. A test recomputes every tier with separate code. Even the low tier is not "no shared words": low means fewer than half of the query's words appear in the docstring, so a docstring of 300 words can still contain some of them.

### Leak audit

For each query, the share of its word trigrams that occur in any summary, and in any full text, of the corpus. A query is flagged when either share is at least 0.5. A test plants a phrase copied from a document and requires it to be flagged. The audit found 32 flagged queries: none in the low tier, 4 in mid, 28 in high. They were not removed or rewritten. The model was trained on that text, so the high tier is largely a test of whether the model retrieves what it saw; the low tier is where a model that learned only word overlap would fail.

## What a production embedding model does differently

- Encoder and pooling. The encoder is a transformer over sub-word tokens. Each token's vector depends on the tokens around it, so word order and context matter, and a word that was never seen whole is still represented by its pieces. A pooling step (the mean of the token vectors, or the vector of one special token) then turns the per-token vectors into one vector per text, usually scaled to length 1. This project's mean of word vectors is the pooling step without a transformer before it. Many production models start from a pretrained language model and are then trained contrastively.
- Loss and data. The training loss is the same kind of InfoNCE loss with a temperature (often one-directional, from query to document). It is trained on far more text pairs, usually in more than one stage: a first stage on very large collections of loosely matched pairs, then a smaller stage on curated retrieval data. Here there are about 1,700 pairs from one small corpus.
- Negatives. Negatives are chosen, not only sampled from the batch. Hard negatives (documents that look relevant but are not) and much larger batches make the task harder, so the vectors learn finer distinctions. Here the negatives are the other 255 pairs in a random batch.
- Truncation. If truncation to fewer dimensions is intended, training includes it (Matryoshka-style: the loss is also applied to the first k dimensions for several k), so the leading dimensions carry the most information. Here it does not.
- Evaluation. Such models are evaluated on public retrieval benchmarks (for example BEIR and MTEB) as well as on the application's own queries.
