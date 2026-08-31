# Five-person implementation split

This split starts from the organizer's released `starter/agent.py` and official
`reset`/`respond` contract. The public evaluator and labels remain untouched.

## Shared rules

- Preserve `parent_asin` from catalog input through official output.
- Never pass `ground_truth` or public labels into runtime modules.
- Keep model/API credentials out of source control. The default agent is offline.
- Optional models must have a deterministic fallback and measured ablation.
- Coordinate interface changes before merging branches.
- Run `python3 -m unittest discover -s tests -v` before handoff.

## Person 1 — Conversation and search state

Owns `conversation/` and `tests/test_conversation.py`.

Current responsibilities:

- maintain one isolated `ConversationState` per evaluator session;
- extract several slots from one reply into a cumulative retrieval query;
- distinguish hard, soft, and negative constraints;
- normalize upper, lower, approximate, and range budgets;
- retain source turn, confidence, intent version, and retracted constraints;
- retract stale slots on corrections and complete category-level intent overrides;
- track asked/exhausted attributes, boundary replies, and rejected result sets.
- retain returned product IDs by turn and intent version so later turns explore new
  results without contaminating a replacement intent.

Interface: `update_from_message(state, user_message, turn) -> ConversationState`.

Next experiments: broaden paraphrase coverage and test a schema-validated optional
LLM parser behind this state contract. Never make network access mandatory, and
compare the parser on a frozen utterance suite before enabling it.

## Person 2 — Candidate retrieval

Owns `retrieval/` and retrieval tests in `tests/test_search_components.py`.

Current responsibilities:

- build the frozen 50,000-product SQLite FTS5 index;
- persist a catalog/schema-versioned FTS/facet cache and copy it into memory at
  startup, plus retain a bounded per-query cache;
- normalize color, material, size, style, use-case, feature, category, and brand
  facets while retaining every participant-visible metadata field;
- use 160 candidates for the first unconstrained turn and up to 300 afterward;
- optionally fuse wider FTS5 and precomputed Sentence Transformer pools with
  configurable weighted RRF;
- implement broad/core/constraint/exact lexical routes and preserve raw BM25,
  dense, route-specific, and fused ranks in diagnostics.

Interface: `CatalogIndex.retrieve(state, limit=300) -> list[Candidate]`.

The four-route fusion and query expansion paths remain off because their fixed-screen
ablations reduced the composite score. The completed MiniLM retrieval-only benchmark
found 75/25 lexical/semantic fusion improved Recall@200 from `0.8525` to `0.8600`
and Hit@10 from `0.4525` to `0.4650`, while lowering raw MRR@10 and adding latency.
The full public evaluator then scored hybrid `0.832978` versus lexical `0.841669`,
so it remains optional and disabled. See `docs/RETRIEVAL.md` and
`docs/retrieval_benchmark_results.json`.

## Person 3 — Ranking

Owns `ranking/` and ranking tests in `tests/test_search_components.py`.

Current responsibilities:

- preserve retrieval order when no constraints are available;
- rerank using normalized facet and full-token tri-state constraint evidence;
- enforce budget-range and negative-constraint penalties;
- penalize only recommendations already shown during the current intent version;
- blend formula scores with a checked-in 16-feature pairwise linear ranker;
- optionally apply a local CrossEncoder only to the best 30 formula-ranked items;
- keep per-candidate diagnostics separate from official output.

Interface: `rank_candidates(state, candidates, ..., ranking_model=None) -> list[RankedCandidate]`.

The provided experiment uses a seed-2026, scenario-stratified, target-disjoint
160/40 session split and stores the dataset hash, hyperparameters, weights, and
results together. Next, repeat this protocol across several seeds and private-like
holdouts before adding features.

## Person 4 — Clarification policy

Owns `policy/` and policy tests in `tests/test_search_components.py`.

Current responsibilities:

- ask while still returning a Top 10;
- use `other` for its documented multi-constraint response, then estimate question
  value from candidate entropy, field coverage, profile priors, and answerability;
- allow the Boundary scenario one extra collection turn;
- stop repeated questions after a no-additional-preference reply;
- resolve contradictions and change strategy after rejected result sets;
- compute an optional counterfactual utility from the chance that an answer moves a
  tail candidate into the Top 10;
- never ask on turn 10.

Interface: `decide(state, ranked) -> Clarification`.

The counterfactual policy remains off because its fixed 50-session screen scored
`0.773578` versus `0.774079` for the entropy policy. Next, calibrate it on training
sessions and compare scenario-specific conversion turn on held-out sessions.

## Person 5 — Integration and evaluation

Owns `shopping_agent/`, `starter/agent.py`, `evaluation/`, setup/reporting, and
official contract tests.

Current responsibilities:

- keep `starter.agent.Agent` compatible with the organizer harness;
- connect the other four modules without leaking internal diagnostics;
- validate output keys, allowed attributes, unique IDs, and token usage;
- expose target-free development traces without adding official response fields;
- measure Recall@K, conditional Top-10 ranking quality, latency, and question reasons;
- train/evaluate LTR without exposing `ground_truth` to the runtime agent;
- record ablations and enable only improvements that earn their latency/cost;
- preserve the organizer evaluator unchanged and reproduce result comparisons.

Commands:

```sh
python3 -m unittest discover -s tests -v
python3 -m evaluator.local_evaluator --output enhanced_v3_results.local.json
python3 -m evaluation.compare_results baseline_results.local.json enhanced_v3_results.local.json
python3 -m evaluation.pipeline_diagnostics --output pipeline_diagnostics_v3.local.json
python3 -m evaluation.retrieval_benchmark --semantic-index data/semantic_index
python3 -m evaluation.ltr_experiment --output ltr_experiment.local.json
```
