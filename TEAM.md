# Team handoff

Everyone can use the demo pipeline without waiting for another person's model or
dataset work. The ownership below is a proposed five-person split based on the
brief; Person 3 is Ranking and Person 4 consumes ranking diagnostics.

## Shared integration rules

- Read `shared/contracts.py` and `docs/CONTRACTS.md` first. Keep `parent_asin` as the
  catalog identity through every stage; do not replace it with a row index.
- Own your module and its tests. Coordinate changes to `shared/`, public function
  signatures, and response keys with the other roles.
- Runtime paths never receive target labels. Keep public-dev labels in offline
  evaluation/training only; never tune against hidden evaluation targets.
- Missing metadata stays missing. Do not convert an unknown price into zero or
  unavailable semantic scores into invented scores.
- Keep adapters optional so `python3 -m shopping_copilot demo` still runs offline.
- Run `python3 -m unittest discover -s tests -v` before sharing changes.
- Suggested branch names: `codex/conversation`, `codex/retrieval`, `codex/ranking`,
  `codex/clarification`, `codex/integration`. No branches have been created for you.

## Person 1 — Conversation and SearchState

Own `conversation/`; coordinate the state contract in `shared/`.

Available now: explicit slot merge/remove, intent, turn accounting, and preservation
of previous constraints and asked attributes. The current `query` is replaced by
the supplied string, so the caller should provide a retrieval-ready query.

Start with:

1. Map user utterances to validated hard constraints, soft preferences, exclusions,
   and buying/browsing intent. Preserve earlier intent when a user changes one slot.
2. Produce a standalone cumulative retrieval query; don't replace "running shoes"
   with just "blue" after a clarification answer.
3. Add tests for corrections, negation, budget ranges, and contradictory requests.

Smoke test: `python3 -m unittest discover -s tests -p test_contracts.py -v`.
Do not add an LLM client to `shared/`; keep it behind a conversation adapter.

## Person 2 — Retrieval

Own `retrieval/`.

Available now: normalized JSON/JSONL loader, in-memory BM25 index, deterministic
candidate ordering, and `Retriever` protocol. Candidate limit defaults to 200.

Start with:

1. Normalize the frozen catalog and preserve parent-level identities and unknowns.
2. Add a dense/hybrid retriever implementing `retrieve(state, limit=200)`.
3. Return approximately 100–300 unique candidates when possible and preserve raw
   BM25 scores. Supply optional semantic/metadata scores on the documented scale.
4. Measure candidate recall before blaming or tuning the ranker.

Swap your implementation into `ShoppingCopilot(your_retriever)`; the ranker and
policy require no other changes. Keep the lexical retriever as an offline fallback.

## Person 3 — Ranking (this brief)

Own `ranking/`.

Available now: feature extraction, tri-state constraint evidence, intent-specific
weighted scoring, penalties, deduplication, and diagnostics. Optional modules
provide an LTR feature-vector/session-split helper and a shortlist reranker contract.

Start with:

1. Evaluate the current formula on public development sessions, not just fixtures.
2. Tune weights on a train/validation split grouped by session; save the split,
   seed, data version, weights, and results together.
3. Compare Hit@10, MRR@10, retrieval recall, and latency with retrieval order.
4. Only then compare supervised LTR or a maximum-20-item LLM reranker. Track token
   usage/cost if making model calls. Keep an experiment only if it earns its cost.

Smoke test: `python3 -m examples.ranking_only` and
`python3 -m unittest discover -s tests -p test_ranking.py -v`.

## Person 4 — Clarification strategy

Own `policy/`.

Available now: score gaps, entropy, constraint coverage/violations, a conservative
clarify/recommend heuristic, and tracking of already-asked attributes. No question
is asked after turn 10. Top-10 products remain available even during clarification.

Start with:

1. Calibrate ambiguity signals against public sessions; a score is not a probability.
2. Choose questions by expected information gain, not just attribute order.
3. Handle empty retrieval sets, contradictions, and answers that reject all options.
4. Compare interactive early conversion and total turns, not only fixed-state MRR.

Keep `decide` pure: return an action and attribute; orchestration records it in state.

## Person 5 — Integration and evaluation

Own `shopping_copilot/`, `evaluation/`, CLI/API, and end-to-end tests.

Available now: offline demo, localhost HTTP endpoints, explicit-state replay,
retrieval-vs-ranking metrics, and a CI test workflow.

Start with:

1. Connect the official challenge input/output protocol after obtaining its spec.
2. Keep internal diagnostics separate from official submission output; the current
   API response is a development contract, not an assumed judge format.
3. Connect Person 1's parser, Person 2's retriever, and Person 4's policy.
4. Add session replay/simulation and the actual early-conversion scoring formula
   once the official rules and data are available.

Run `python3 -m shopping_copilot serve` for transport integration and
`python3 -m shopping_copilot evaluate` for a fixture-only smoke test.

## First integration milestone

All five modules should run together against the normalized frozen catalog, with
the shared contract unchanged, at most ten unique output IDs per turn, no label
leakage, and an evaluation report distinguishing retrieval misses from ranking
misses. Results on the tiny synthetic fixture do not satisfy that milestone.
