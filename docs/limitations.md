# Limitations

## Limits of the data and the setup

- 99 queries in 33 target groups, written by one author, who also chose the acceptable targets. Per-tier intervals are wide, and the low tier has 37 queries. Some queries repeat an earlier draft's queries or a target's own summary line (see "Query set" in [method.md](method.md)); removing the 6 summary-line queries lowers every method's overall recall@10 by 0.02 to 0.04 and does not change their order.
- The acceptable target sets are judgment calls. A different judge would change some ranks. Some groups accept up to five functions (for example the bracketing root finders), which makes those queries easier for every method.
- The intervals come from resampling 33 target groups (fewer within a tier) and use the percentile method. They are approximate and probably somewhat narrow.
- The encoder ignores word order and context, and drops every word outside its vocabulary, which is the 6,709 words that occur in at least two documents.
- Own-64 is trained on the text it searches. The held-out row addresses this only partly.
- The "static vectors" row uses its own tokenizer, not the auditor's. Its weights are derived from other models whose licenses I did not check. The package is MIT licensed. Nothing from it is vendored.
- `requirements.txt` pins NumPy, SciPy and WordLlama; the packages WordLlama itself depends on (for example `tokenizers`) are not pinned, and the static-vector row could change with them.
- `python -m lab corpus` does not reproduce `data/corpus.jsonl` exactly on the same versions: the `scipy.signal.lombscargle` record changes with the hash seed (see "Corpus" in [method.md](method.md)).
- BM25 parameters (k1 1.2, b 0.75) and the RRF constant (k=60) were fixed in advance. No method was tuned on the queries. This favors none of them and may leave every method below what tuning would reach.
- SQuAD questions were written while reading the paragraph, which favors lexical methods. The SQuAD own-64 pairs (first sentence with the rest) were not designed for question answering.
- int8 uses one shared scale for all documents, and only documents are quantized.
- Timings come from a 2-core Linux VM and vary between runs. Ten full runs on the same machine gave identical metrics. A run with multi-threaded BLAS and a run with one BLAS thread also gave identical tables. Other machines may differ in the last digits of the loss values or move a rank.

## Not validated

- The Docker image: CI builds it and does not run it. As far as PyPI shows, WordLlama 0.4.0.post1 has Linux wheels only for x86-64, so a build on an arm64 host (Docker Desktop on Apple silicon, for example) would have to compile it from source, which I did not try.
- The PowerShell commands, Windows, and Python versions other than 3.11. On Windows the training processes start with `spawn`; that path was not run.
- The real-model row: no LM Studio or Azure OpenAI in Foundry Models endpoint was called. Only the local stub test covers the client.
- The Azure mapping in [azure-mapping.md](azure-mapping.md): nothing was run against Azure AI Search or Microsoft Foundry.
- The browser test with any Chromium build other than the one used here.
- Regenerating the corpus on other NumPy or SciPy versions.
