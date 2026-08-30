# Search improvement experiments

All results below use the organizer's unchanged evaluator. The full-public numbers
are useful development indicators, not estimates of private-set generalization.
Features are enabled by default only when they improved a fixed comparison or the
target-disjoint validation split.

## Adopted sequence on all 200 public sessions

| Configuration | Hit@10 | MRR | MTTC | Score |
| --- | ---: | ---: | ---: | ---: |
| Previous enhanced agent | 0.800 | 0.474204 | 4.140 | 0.679461 |
| Intent-scoped recommendation rotation | 0.965 | 0.534831 | 3.200 | 0.798949 |
| Normalized facet evidence | 0.965 | 0.551117 | 3.170 | 0.804435 |
| Pairwise LTR/formula blend | 1.000 | 0.567230 | 2.430 | 0.841569 |

Rotation remembers only products returned during the current intent version. A
category-level override starts a fresh version, so products from the abandoned
intent do not suppress valid results later.

Facet extraction stores normalized color, material, size, style, use-case, feature,
category, and brand values beside the FTS index. Ranking uses field-specific evidence
instead of treating any token anywhere in the product text as a match.

## Learned ranking protocol

`evaluation/ltr_experiment.py` creates a deterministic, scenario-stratified split
grouped by target `parent_asin`. With seed `2026`, 160 sessions train the model and 40
target-disjoint sessions validate it. Labels are inspected only after each agent
response. The trainer produced 9,900 positive/hard-negative pairs over 16 target-free
features and saved the weights plus data hash and hyperparameters in
`ranking/models/ltr_v1.json`.

On the 40-session validation split, the formula-only score was `0.844193`. The
selected `0.90` LTR blend scored `0.882851` (Hit@10 `1.0`, MRR `0.694504`, MTTC
`2.275`). The checked-in model can be replaced with `TECHJAM_LTR_WEIGHTS` or disabled
with `Agent(..., use_default_ltr=False)`.

## Implemented but disabled after screening

The retrieval and question-policy screen uses the same deterministic 50-session,
scenario-stratified slice.

| Experiment | Control score | Experiment score | Default |
| --- | ---: | ---: | --- |
| Four-route FTS with reciprocal-rank fusion | 0.758295 | 0.737207 | Off |
| Four-route FTS plus facets | 0.758295 | 0.745564 | Off |
| Counterfactual question utility | 0.774079 | 0.773578 | Off |

These paths remain callable through `enable_multi_route=True` and
`use_counterfactual_questions=True`; keeping them available makes further held-out
calibration possible without silently weakening the default.

Synonym expansion also remains opt-in because its earlier full-public ablation
reduced the composite score.

## Runtime changes

The catalog FTS/facet database is versioned by catalog metadata and persisted under
`data/.cache/`, then copied into memory at agent startup. A bounded 16-entry query
cache avoids repeating identical lexical work within an agent instance. Generated
caches are ignored by Git and are safely rebuilt if the catalog or schema changes.

The final 200-session evaluation completed in `62.39 s` with about `355 MB` peak RSS
in WSL Python 3.12. Across 486 turns, mean post-startup response latency was `117 ms`
and p95 was `220 ms`.

## Optional neural semantic path

Dense Sentence Transformer retrieval and a Top-30 CrossEncoder adapter are complete
but not part of the measured default. A CPU-only BGE index build was trialed in this
workspace; after five of 391 batches it projected roughly 64 minutes, so the run was
stopped and no incomplete index was retained. Model download, index storage, startup
cost, and per-turn latency must be included in a target-disjoint benchmark before
this path is adopted. See `docs/semantic_models.md`.
