# Optional semantic retrieval and reranking

The official default remains standard-library FTS5/facets plus the checked-in linear
LTR artifact. This neural path is optional because final scoring may disable network
access and impose CPU, memory, storage, or startup limits.

## Build the dense index

Run this during development while the selected model is legally accessible:

```sh
python3 -m pip install -r requirements-semantic.txt
python3 -m retrieval.semantic \
  --catalog data/catalog.jsonl \
  --output data/semantic_index
```

The output contains memory-mappable embeddings, catalog identifiers, and metadata
including the catalog SHA-256, model name, row count, and dimensions. The generated
directory is ignored by Git; package it separately only if submission size and model
licensing rules allow it.

The default is `sentence-transformers/all-MiniLM-L6-v2` pinned to revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Pass `--model` and `--revision`
together for another model, rebuild the index, and record that identity in the
experiment. Runtime model/revision overrides must match the index metadata.

`requirements-semantic.txt` selects PyTorch's CPU wheel index and pins a CPU build so
the experiment does not accidentally download CUDA runtimes. Use a separate virtual
environment; the core agent does not need these packages.

## Enable one or both stages

```sh
export TECHJAM_SEMANTIC_INDEX=data/semantic_index
export TECHJAM_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
python3 -m evaluator.local_evaluator --output semantic_results.local.json
```

- Dense retrieval searches the precomputed catalog embeddings and fuses its result
  with FTS5 using Reciprocal Rank Fusion.
- The CrossEncoder sees at most 30 candidates selected by formula ranking.
- Explicit budget/exclusion penalties remain outside the model bonus.
- Loading or inference failure emits a warning and disables that optional stage.
- Local models report zero API tokens; disclose model storage and compute latency.

## Adoption gate

Compare the optional run with the default on a fixed session-level validation split.
Adopt it only if the improvement survives target-disjoint validation and justifies its
startup time, response latency, memory, storage, and reproducibility cost. Always run:

```sh
python3 -m evaluation.pipeline_diagnostics --output pipeline_diagnostics.local.json
```

If target Recall@300 rises but conditional Top-10 quality does not, tune the ranker.
If recall does not rise, the dense model or fusion weighting is not earning its cost.

## Current benchmark status

The complete MiniLM index contains 50,000 embeddings (384 dimensions, float32) and
is about 73 MB. The fixed 400-probe comparison selected 75/25 lexical/semantic
fusion: Recall@200 improved from `0.8525` to `0.8600` and retrieval Hit@10 from
`0.4525` to `0.4650`, while MRR@10 fell and mean latency increased by about 54 ms.
See `docs/RETRIEVAL.md` and `docs/retrieval_benchmark_results.json` for the full
method and result.

The full public evaluator scored hybrid `0.832978` versus lexical `0.841669`, so the
optional path is not enabled. Before packaging a future index, repeat the complete
target-disjoint agent ablation on the actual submission machine. Record model/index
size, build and startup time, mean/p95 turn latency, peak RSS, Recall@K, Hit@10, MRR,
MTTC, and total score.
