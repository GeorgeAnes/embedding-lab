# Azure mapping (not validated)

Each row relates a knob in this project to a setting of Azure AI Search or Azure OpenAI in Foundry Models. I took the setting names from Microsoft Learn on 30 September 2026 (pages listed below). Nothing was run against a service, and whether a given feature is generally available or in preview, and in which regions and tiers, was not checked.

| Here | Azure AI Search or Foundry Models | Note |
|---|---|---|
| 64 dimensions | The vector field's `dimensions` | The number of dimensions of the stored vectors, set when the field is defined. |
| `truncate` (first k dimensions, renormalized) | `truncationDimension` in a vector compression entry (`compressions`) | The REST reference says to use it only for embeddings trained with Matryoshka Representation Learning, such as OpenAI text-embedding-3. This model was not, and its truncation cost is in the vector-size table in [results.md](results.md). |
| `to_int8` | Vector compression `scalarQuantization`, with `scalarQuantizationParameters.quantizedDataType` set to `int8` | Here: one shared scale, documents only. The service's scheme may differ. |
| Not built (1-bit) | `binaryQuantization` | Learn calls it particularly effective above 1,024 dimensions and recommends testing quality, or trying scalar quantization, for smaller vectors. A 64-dimension model is far below that. |
| BM25 on full text plus vectors, fused with RRF k=60 | A hybrid query: a full-text (BM25) query and a vector query run in parallel and their rankings are merged by Reciprocal Rank Fusion, score `1/(rank + k)` | Learn says the algorithm performs best with a small k such as 60. |
| Scan of all 1,714 documents | The `exhaustiveKnn` algorithm (brute force, exact) or `hnsw` (approximate) | Learn recommends exhaustive KNN for small to medium datasets, or when precision matters more than query speed. |
| `--real` | An embedding deployment in Azure OpenAI in Foundry Models | Learn's v1 API examples point the standard OpenAI client at `https://<resource>.openai.azure.com/openai/v1/` with an API key, and `model` is the deployment name. That client sends its key as a bearer token, as `lab/dense.py` does. Learn recommends Microsoft Entra ID over keys, which this client does not support. Not tested against a resource. |

Pages read: Azure AI Search "vector-search-how-to-quantization", "hybrid-search-ranking" and "vector-search-ranking", the Azure AI Search REST reference "Indexes - Create" (API version 2026-04-01), and the Azure OpenAI "embeddings" and "api-version-lifecycle" pages, all under `https://learn.microsoft.com/`.
