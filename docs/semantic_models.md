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
  --output data/semantic_index \
  --model BAAI/bge-small-en-v1.5
```

The output contains memory-mappable embeddings, catalog identifiers, and metadata
including the catalog SHA-256, model name, row count, and dimensions. The generated
directory is ignored by Git; package it separately only if submission size and model
licensing rules allow it.

`requirements-semantic.txt` selects PyTorch's CPU wheel index and pins a CPU build so
the experiment does not accidentally download CUDA runtimes. Use a separate virtual
environment; the core agent does not need these packages.

## Enable one or both stages

```sh
export TECHJAM_SEMANTIC_INDEX=data/semantic_index
export TECHJAM_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
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

The optional dependencies and model load were verified with `torch 2.10.0+cpu` and
CUDA disabled. On the available WSL CPU, encoding the entire catalog with
`BAAI/bge-small-en-v1.5` at batch size 128 projected about 64 minutes after five of
391 batches. The build was stopped and its incomplete index removed. Accordingly,
neither dense retrieval nor CrossEncoder reranking is enabled in the reported score.

This is a resource result, not a relevance result. Run the complete target-disjoint
ablation on the actual submission machine before deciding whether to package an
index. Record model/index size, build and startup time, mean/p95 turn latency, peak
RSS, Recall@K, Hit@10, MRR, MTTC, and total score.
