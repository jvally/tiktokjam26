# Task 4 technical report

## Starting point

The repository was reset to the organizer participant kit before development. The
untouched starter reproduced the published 200-session baseline: HitRate@10
`0.125`, MRR `0.068034`, MTTC `9.810`, and technical score `0.106710`. The organizer
evaluator and public labels were not edited.

An earlier enhanced implementation scored `0.679461` (HitRate@10 `0.800`, MRR
`0.474204`, MTTC `4.140`). This iteration keeps its multi-slot state, FTS retrieval,
constraint ranking, and candidate-aware questions, then adds measured improvements
to state memory, structured product evidence, learned ranking, and runtime caching.

## Architecture

The required `starter.agent.Agent` delegates to five independently owned stages:

```text
user turn
   -> typed multi-slot state, corrections, intent versions, shown-result memory
   -> cached FTS5 retrieval (160 on a vague first turn, otherwise up to 300)
   -> normalized product facets and optional multi-route/dense fusion
   -> constraint formula plus calibrated pairwise LTR and result rotation
   -> candidate-aware clarification policy
   -> official message / ask_attribute / Top-10 response
```

Conversation state extracts multiple attributes from a clause, normalizes budget
ranges, preserves source turns and confidence, and audits retracted constraints.
Corrections replace only the affected slots. A category-level override retracts stale
product-specific constraints and increments the intent version. Recommendation
history is keyed to that version, allowing subsequent turns to explore new Top-10
items without suppressing products for a replacement intent.

The default retriever is weighted SQLite FTS5. During index construction it also
normalizes color, material, size, style, use-case, feature, category, and brand
facets. Ranking therefore distinguishes evidence in the requested field from a token
that merely occurs somewhere in free text. The persistent database is invalidated by
catalog metadata and schema version, copied into memory at startup, and paired with a
bounded repeated-query cache.

Ranking combines constraint evidence, budget and exclusion penalties, retrieval
signals, rating quality, profile agreement, and prior-result penalties. A 16-feature
pairwise linear model is blended with this safety formula. The model is a small JSON
file loaded by standard-library code; it performs no network call and uses no model
runtime dependency.

The policy returns Top-10 results even while asking. It handles contradictions, empty
retrieval, rejected result sets, Boundary replies, exhausted attributes, and the
turn-10 limit. Candidate entropy/coverage remains the default. A counterfactual
Top-10 movement utility is implemented but disabled after its screen did not improve
the composite score.

## Public development results

| Metric | Organizer baseline | Previous enhanced | Current default | Delta vs previous |
| --- | ---: | ---: | ---: | ---: |
| HitRate@10 | 0.125 | 0.800 | 1.000 | +0.200 |
| MRR | 0.068034 | 0.474204 | 0.567230 | +0.093026 |
| MTTC | 9.810 | 4.140 | 2.430 | -1.710 turns |
| Efficiency | 0.119 | 0.686 | 0.857 | +0.171 |
| Technical score | 0.106710 | 0.679461 | 0.841569 | +0.162108 |

All four public scenarios reached HitRate@10 `1.000`. Buying MRR/MTTC are
`0.576652`/`1.725`, Browsing `0.561741`/`2.375`, Boundary `0.552619`/`2.700`, and
Intent Override `0.561614`/`4.366667`. These are full-public development figures,
not private-test guarantees.

The measured WSL Python 3.12 evaluation took `62.39 s` and about `355 MB` peak RSS.
It made no model/API calls and reported zero prompt and completion tokens.

## LTR validation and ablations

`evaluation/ltr_experiment.py` uses a deterministic scenario-stratified split grouped
by target ID. Seed `2026` assigned 160 sessions to training and 40 target-disjoint
sessions to validation. The trainer gathered 9,900 target/hard-negative pairs while
reading labels only after each `Agent.respond` call. The packaged artifact records
the public-set SHA-256, split, seed, feature order, training parameters, and weights.

On validation, formula-only ranking scored `0.844193`. The selected 0.90 LTR blend
scored `0.882851` with HitRate@10 `1.000`, MRR `0.694504`, and MTTC `2.275`.

Full-public incremental ablations were:

| Addition | Hit@10 | MRR | MTTC | Score |
| --- | ---: | ---: | ---: | ---: |
| Intent-scoped result rotation | 0.965 | 0.534831 | 3.200 | 0.798949 |
| Structured facet evidence | 0.965 | 0.551117 | 3.170 | 0.804435 |
| Pairwise LTR blend | 1.000 | 0.567230 | 2.430 | 0.841569 |

Four-route FTS fusion and counterfactual question choice were also implemented. On a
fixed scenario-stratified 50-session screen they scored `0.737207` versus `0.758295`
and `0.773578` versus `0.774079`, respectively, so both remain off by default.
Detailed experiment commands and adoption rules are in `docs/experiments.md`.

## Pipeline diagnostics

`python3 -m evaluation.pipeline_diagnostics` reads the target only after each agent
response. Across the 486 turns needed by the final default:

- candidate Recall@10 is `0.3642`, Recall@100 `0.7325`, and Recall@300 `0.8498`;
- every target appears in the candidate pool during at least one session turn;
- conditional Top-10 rate when the target is retrieved is `0.5424`;
- average candidate count is `265.4`;
- mean post-startup response latency is `117 ms`, with `220 ms` p95.

The remaining headroom is concentrated in early-turn retrieval and Intent Override
ordering, rather than eventual session recall.

## Model, network, and cost disclosure

The reported default uses the local pairwise linear ranking artifact
`ranking/models/ltr_v1.json`. It requires no external service, API key, embedding
runtime, network access, or paid tokens. Estimated inference cost is zero.

An optional neural path is implemented but not included in the reported score:

- pinned `sentence-transformers/all-MiniLM-L6-v2` catalog/query embeddings;
- `cross-encoder/ms-marco-MiniLM-L6-v2` for at most 30 shortlist candidates;
- weighted Reciprocal Rank Fusion over wider lexical/semantic pools;
- deterministic fallback to FTS5/formula ranking on load or inference failure.

The complete 50,000-product MiniLM index is about 73 MB. On 400 fixed public probes,
75/25 lexical/semantic fusion raised Recall@200 from `0.8525` to `0.8600` and
retrieval-order Hit@10 from `0.4525` to `0.4650`. It lowered MRR@10 from `0.349835`
to `0.306197` and raised mean retrieve-only latency from `48.2 ms` to `102.5 ms`, so
it remains optional. The full public evaluator confirmed that decision: hybrid
scored `0.832978` versus `0.841669` for lexical default, with HitRate@10 `0.985`
versus `1.000`. See `docs/RETRIEVAL.md`; the fixed-probe metrics are not themselves
the interactive TechnicalScore.

## Limitations

- A score of `1.000` HitRate on the public set is not evidence of equivalent private
  performance; the full-public result includes LTR training sessions.
- The 40-session target-disjoint validation set is small. Repeat several grouped
  seeds and preserve a final untouched holdout before further tuning.
- The parser is deterministic and still needs broader testing on unseen paraphrases,
  currencies, alternatives, and compound negation.
- The diagnostic candidate Recall@300 of `0.8498` per turn leaves scope for a faster
  semantic or catalog-aware recall stage.
- FTS5 availability and the generated cache location must be confirmed in the final
  organizer image.

## Reproduction

```sh
python3 -m unittest discover -s tests -v
python3 -m evaluator.local_evaluator --output enhanced_v3_results.local.json
python3 -m evaluation.compare_results baseline_results.local.json enhanced_v3_results.local.json
python3 -m evaluation.pipeline_diagnostics --output pipeline_diagnostics_v3.local.json
python3 -m evaluation.retrieval_benchmark --semantic-index data/semantic_index
python3 -m evaluation.ltr_experiment --output ltr_experiment.local.json
python3 -m examples.demo_session
```

See `TEAM.md` for ownership, `docs/experiments.md` for ablations, and
`docs/RETRIEVAL.md` and `docs/semantic_models.md` for optional-model setup.
