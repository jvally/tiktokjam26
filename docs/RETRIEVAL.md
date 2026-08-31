# Candidate retrieval

Person 2 owns `retrieval/` and the retrieval tests in
`tests/test_search_components.py`. The online boundary is
`CatalogIndex.retrieve(state, limit) -> list[Candidate]`; target IDs and evaluator
labels never enter this call.

## Default lexical path

`CatalogIndex` builds a weighted SQLite FTS5 index over title, category, features,
details, store, and description. It preserves the catalog `parent_asin`, raw BM25
score, retrieval rank, product metadata, and normalized facets. The generated index
is versioned by catalog metadata and schema version, stored under `data/.cache/`, and
copied into memory at startup. A bounded query cache avoids repeated FTS work.

The full agent requests 160 candidates for a vague first turn and up to 300 after a
hard constraint appears. The lexical path is standard-library-only, offline, and is
the production fallback if optional semantic retrieval cannot load or score.

## Optional dense/hybrid path

Install the optional packages and build the frozen-catalog index once:

```sh
python3 -m pip install -r requirements-semantic.txt
python3 -m retrieval.semantic \
  --catalog data/catalog.jsonl \
  --output data/semantic_index
```

The default embedding model is
`sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The index records the model,
revision, catalog SHA-256, product ordering, dimensions, and normalization. Query
embeddings and the 50,000 precomputed product embeddings are L2-normalized, so the
exact matrix scan produces cosine similarity. Generated embeddings stay outside
Git because the current float32 matrix is about 73 MB.

Hybrid retrieval requests a wider pool from both backends (twice the requested
limit, with a minimum of 300 and maximum of 1,000) and applies weighted reciprocal
rank fusion with `k=60`. The tested setting assigns 75% of the fusion weight to
lexical rank and 25% to semantic rank. Returned candidates preserve raw BM25,
semantic similarity, source ranks, and the fusion score. Results are unique and
ties are resolved by `parent_asin`.

Enable the optional path without changing the agent contract:

```sh
export TECHJAM_SEMANTIC_INDEX=data/semantic_index
python3 -m evaluator.local_evaluator --output semantic_results.local.json
```

`TECHJAM_EMBEDDING_MODEL` and `TECHJAM_EMBEDDING_REVISION` may be supplied only
when they match the index metadata. A mismatch fails closed instead of silently
encoding queries with a different model.

## Retrieval-only benchmark

Run the fixed comparison with:

```sh
python3 -m evaluation.retrieval_benchmark \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --semantic-index data/semantic_index \
  --candidate-limit 200 \
  --output retrieval_benchmark.local.json
```

For every public session the benchmark creates two fixed states. `initial` mirrors
the organizer's first customer message. `resolved` contains all target requirements
that would be disclosed, with the obsolete Intent Override preference removed. The
resolved state is an upper-bound diagnostic. Ground truth remains an offline label
and is inspected only after retrieval returns.

The checked-in run used all 200 public sessions (400 probes), the released 50,000
product catalog, and a 200-candidate limit:

| Retriever | Recall@200 | Hit@10 | MRR@10 | Mean latency | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lexical FTS5 | 85.25% | 45.25% | **34.98%** | **48.2 ms** | **99.7 ms** |
| Hybrid 50/50 | 86.00% | 43.00% | 24.71% | 104.2 ms | 174.4 ms |
| **Hybrid 75/25** | **86.00%** | **46.50%** | 30.62% | 102.5 ms | 170.6 ms |

Compared with lexical retrieval, the tested 75/25 hybrid gains 0.75 percentage
points of Recall@200 and 1.25 points of Hit@10. Initial-probe recall rises from
72.5% to 74.0%; resolved recall remains 98.0%. Raw MRR@10 falls by 4.36 points and
mean retrieve-only latency increases by about 54.3 ms. This makes hybrid promising
as a candidate generator, but not an automatic default: the ranker must recover the
better candidate pool, and the end-to-end TechnicalScore must justify the cost.

The subsequent full 200-session evaluator rejected the optional path:

| Full agent | HitRate@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: |
| **Default lexical** | **1.000** | **0.567230** | 2.425 | **0.841669** |
| Hybrid 75/25 | 0.985 | 0.550593 | **2.235** | 0.832978 |

Hybrid finds some targets earlier when it succeeds, but it introduces three misses
and weaker final ordering. The 0.008691 TechnicalScore loss outweighs the MTTC gain.
It therefore remains an experiment; the checked-in default is lexical FTS5.

The compact reproducibility artifact is
[`retrieval_benchmark_results.json`](retrieval_benchmark_results.json). It records
the catalog/session hashes, model revision, fusion settings, measured metrics, and
limitations. Per-query latency excludes index construction and the separately
reported 250 ms model warm-up on the benchmark machine.

## Verification and adoption gate

```sh
python3 -m unittest discover -s tests -v
python3 -m evaluation.pipeline_diagnostics --output pipeline_diagnostics.local.json
python3 -m evaluator.local_evaluator --output semantic_results.local.json
```

Do not enable hybrid by default based on Recall@200 alone. The current end-to-end
result keeps it disabled. For any future model or fusion change, compare public or
target-disjoint HitRate@10, MRR, MTTC, TechnicalScore, startup time, index size,
mean/p95 latency, and peak memory against the lexical agent. Keep lexical FTS5 as the
offline fallback regardless of the result.
