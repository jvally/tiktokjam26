# Ranking baseline

This implementation is a transparent starting point, not a fitted ranking model.
No weights, thresholds, or semantic models have been tuned on challenge data.

## Formula

For each candidate:

```text
score = sum(effective_weight[channel] * feature[channel])
        - 0.75 * hard_violation_ratio
        - 1.00 * number_of_negative_attribute_violations
```

| Channel | Buying | Browsing |
| --- | ---: | ---: |
| Normalized BM25 | 0.20 | 0.20 |
| Dense/semantic similarity | 0.20 | 0.35 |
| Metadata compatibility | 0.15 | 0.10 |
| Hard constraint satisfaction | 0.35 | 0.10 |
| Soft preference alignment | 0.05 | 0.20 |
| Profile alignment | 0.05 | 0.05 |

Channels absent for the entire candidate batch are removed and the remaining
weights renormalized to sum to one. A score missing on only one candidate contributes
zero under the same batch-wide weights; it does not get its own favorable rescale.
If no weighted channels are available, the base score is zero. Penalties still apply.
`rank_candidates(..., weights={...})` supports custom nonnegative channel weights.

These are ranking scores, **not purchase probabilities**. They may be negative.
Scores depend on the candidate set; do not compare them across unrelated queries
as if they shared a calibrated scale.

## Retrieval and semantic signals

BM25 uses batch min-max normalization. Identical positive observed BM25 scores map
to 1; identical zero scores map to 0; missing values map to 0. Other supplied scores
must already be in [0, 1]. The bundled retriever computes BM25, not embeddings.

`dense_score` is used directly when supplied. Otherwise, the mean of available
`query_title_similarity`, `query_feature_similarity`, and `use_case_similarity`
is used. If none are supplied, the dense channel is unavailable.

The demo's small lexical expansion table recognizes examples such as
"comfortable" → "cushioned"/"padded" and "travelling" → "travel"/"walking".
It is deliberately inspectable and **not a semantic model**. Real embedding
generation belongs in the retrieval/feature adapter.

## Constraint evidence

Each requested attribute has status `match`, `violation`, or `unknown`.
Default weights are budget/category 3; brand/color/material/size 2; others 1.

```text
constraint = matched_hard_weight / requested_hard_weight
hard_violation_ratio = violated_hard_weight / requested_hard_weight
constraint_coverage = (matched_hard_weight + violated_hard_weight) / requested_hard_weight
```

All three are zero when there are no hard constraints. Unknowns remain in the
requested-weight denominator, so incomplete metadata does not earn a match, but
they incur no violation penalty. Price bounds are inclusive. If both budget bounds
are provided, they are two separately weighted checks.

Metadata values are matched case-insensitively on token/phrase boundaries. Sizes
require equal token sequences, avoiding matches between `8` and `18` or `8.5`.
When structured metadata is absent, product text can provide positive evidence;
absence of a text match stays unknown. Structured metadata takes precedence when
present. There is no full negation, unit-conversion, taxonomy, or variant reasoning.

The metadata channel uses a supplied `metadata_score`, or falls back to weighted
compatibility with merged hard/soft preferences (hard wins on duplicate attributes).
Soft and profile channels use the same matching logic against their own preferences;
a supplied `profile_score` takes precedence. Additional category, brand, color,
material, size, budget, style, feature, use-case, and hybrid retrieval features are
exposed for debugging and later LTR, even when not independent weighted channels.

Negative conditions incur a penalty when any excluded value is positively found.
One forbidden attribute contributes one penalty even if several values match it.
Missing evidence is not treated as a violation. Hard conflicts are **penalized,
not filtered**, to preserve candidates for debugging and potential recall recovery;
the baseline cannot promise every recommendation satisfies every hard constraint.

## Diagnostics and clarification

Score gaps are measured on all deduplicated candidates. Gap 1–10 is null when
fewer than ten candidates exist, not the gap to the last available candidate.
Entropy uses a numerically stable softmax with temperature 0.15; normalized entropy
divides by log(candidate count). Empty/singleton sets have entropy zero, but that
does not establish a confident match. The policy does not treat a singleton alone
as a confident winner.

The starter policy recommends a clear leader only when gap 1–2 >= 0.12, hard
constraint coverage is complete (or none are requested), and the top item has no
known hard/negative violations. Otherwise it tries an unasked differentiating
attribute. These are uncalibrated heuristics for Person 4 to replace.

## Optional experiments

`learning_to_rank.py` exports a stable numeric feature vector and a deterministic
session-grouped split. It does not train a classifier or claim an LTR improvement.
Tiny datasets can produce an empty split: check before fitting. Group further by
user or target when your public-dev protocol requires it.

`reranker.py` accepts at most 20 products and validates returned IDs. It rejects
invented/duplicate IDs and appends omitted candidates in local order. It is not
enabled in the pipeline. If enabled later, keep local-score diagnostics labeled as
pre-reranking: an LLM reorder does not produce calibrated new scores, and local
score gaps/entropy must not silently describe the reordered list.
