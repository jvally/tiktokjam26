# Shared contracts (baseline v0.1)

Python dataclasses live in `shared/contracts.py`. Use `from_dict`/`to_dict` at JSON
boundaries. The ranker and pipeline also accept state dictionaries directly.

## SearchState

```json
{
  "query": "comfortable black running shoes",
  "intent": "buying",
  "turn": 1,
  "hard_constraints": {"color": "black", "budget_max": 100},
  "soft_preferences": {"feature": ["comfortable"], "use_case": ["running"]},
  "negative_constraints": {"material": ["leather"]},
  "profile": {},
  "asked_attributes": []
}
```

All fields have defaults. `intent` is `buying` or `browsing`; `turn` is 1–10.
`query` should be a standalone cumulative retrieval query. The starter does not
perform natural-language understanding. Unknown state keys are rejected.

The four preference objects accept `category`, `brand`, `color`, `material`, `size`,
`style`, `feature`, `use_case`, `budget_min`, and `budget_max`. Negative constraints
do not accept budget keys; express an allowed price range as hard constraints.
Budget numbers must be finite, nonnegative, and min <= max. Prices and budgets must
be in the same currency/units; this baseline performs no currency conversion.

Text values accept a string or string list, normalized to lists. Lists within an
attribute mean **OR** (black or blue); separate hard attributes are jointly desired.
Negative lists exclude **any** listed value. This includes feature/use-case lists;
if you need AND-within-attribute semantics, coordinate a contract extension.
Plural aliases (`colors`, `materials`, `sizes`, `styles`, `features`, `use_cases`)
are accepted. Empty values are omitted. In `update_state`, a slot set to `null`
removes the previous value. `profile` holds optional preferences, not purchase labels.

## Product and Candidate

```json
{
  "parent_asin": "REAL_CATALOG_ID",
  "title": "Black cushioned running shoe",
  "category": "shoes",
  "brand": "Example",
  "price": 79.0,
  "description": "Lightweight walking and running sneaker",
  "color": ["black"],
  "material": ["mesh"],
  "size": ["8", "9"],
  "style": ["sporty"],
  "features": ["cushioned", "lightweight"],
  "use_cases": ["running", "walking"],
  "bm25_score": 4.2,
  "dense_score": 0.83,
  "metadata_score": 0.90
}
```

Only nonempty string `parent_asin` and `title` are mandatory. Missing price is
`null`, text fields default to `""`, and list fields default to `[]`. List fields
also accept a single string. A category must be a normalized string, not an Amazon
category array; the importer owns flattening. Unknown product fields are ignored,
so normalize raw attributes explicitly rather than expecting automatic mapping.

A `Candidate` contains a `Product` plus optional scores. Flat dictionaries as above
and nested `{"product": {...}, "bm25_score": 4.2}` are both accepted by the ranker.

- `bm25_score`: finite and nonnegative; normalized per candidate batch by the ranker.
- `dense_score`, `metadata_score`, `profile_score`, `query_title_similarity`,
  `query_feature_similarity`, `use_case_similarity`: finite **[0, 1]** values or null.
- If your backend returns cosine similarity in [-1, 1], convert it before calling
  ranking. Use a consistent scale across all candidates and evaluation runs.
- Do not fill an unavailable signal with a fake value. Missing is `None`/`null`.

The ranker ignores unknown product keys, but target labels must never be passed
through the online pipeline. Keep label-bearing evaluation records separate.

## Ranking return value

```python
ranked, diagnostics = rank_candidates(state, candidates, top_n=20)
```

`ranked` is a list of flattened product/score dictionaries, augmented with:

- `rank`, `score`, `penalty` and `score_contributions`.
- `ranking_features`: numeric feature vector as a named dictionary. The original
  product's `features` list remains intact.
- `available`: availability of scoring channels, distinct from their numeric zero.
- `constraint_evidence`: attribute, expected/observed values, match/violation/unknown,
  weight, and whether the condition is negative.

Duplicate IDs are collapsed to their highest-scoring row; ties use ascending
`parent_asin` for deterministic output. Ranking penalizes explicit conflicts rather
than filtering them, so results can still contain violations. Inspect the evidence.

Diagnostics include `top_score`, `score_gap_1_2`, `score_gap_1_10`,
`candidate_entropy`, `normalized_entropy`, candidate/duplicate/returned counts,
effective weights, top constraint coverage, and top hard/negative violations.
They are computed across **all unique candidates before `top_n` truncation**.
Unavailable gaps are null; empty-set `top_score` is null. `calibrated` is false.

## Pipeline and policy

```python
response = ShoppingCopilot(retriever).search(state)
```

Returns `state`, `results` (at most 10 ranked records), `parent_asins` (their IDs in
order), `diagnostics`, and `decision`. The ranker's `returned_count` describes its
internal Top-20 shortlist, not the pipeline's Top-10 presentation; use
`len(response["results"])` for the presentation count.

`decision` has `action` (`clarify`/`recommend`), `question`, `attribute`, `reason`,
and `terminal`. The pipeline adds any proposed attribute to the returned state's
`asked_attributes`. A caller that chooses not to display a question must reconcile
that state itself. No global or server-side conversation state is stored.

`/search` accepts exactly `{"state": {...}}` and does not increment turns.
`/turn` accepts `query`, optional `previous_state`, and optional `updates`. A first
turn has no previous state. Later turns increment it; turn 11 is invalid.
The caller owns the session boundary: these endpoints do not prevent clients from
starting a new state or replaying an old one. Requests are limited to 64 KiB.
