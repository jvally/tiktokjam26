# Evaluation and honest baselines

Run `python3 -m shopping_copilot evaluate` for a smoke test. It replays five fixed
synthetic states against 18 synthetic products. In the initial verified run,
retrieval and ranking both achieved Hit@10 = 1.0 and MRR@10 = 1.0; delta MRR@10 was
0.0. The fixture is intentionally easy. **This demonstrates wiring, not an uplift
or a challenge-quality result.** Timing is measured on each invocation.

## Public-development input

Use a JSON array of records with this shape:

```json
[
  {
    "session_id": "public-dev-session-001",
    "target_parent_asin": "REAL_CATALOG_ID",
    "state": {
      "query": "black running shoes",
      "turn": 2,
      "hard_constraints": {"color": "black", "budget_max": 100}
    }
  }
]
```

Labels are read only by the evaluation runner and compared with returned IDs.
The state passed to retrieval/ranking does not contain the target label. Empty
evaluation sets are rejected. Missing targets remain in metric denominators.

The report contains:

- `retrieval_top10`: Hit@10 and MRR@10 for retriever order.
- `ranked_top10`: Hit@10 and MRR@10 after ranking.
- `delta_mrr@10`: ranked minus retrieval MRR@10.
- `retrieval_recall`: fraction whose target appears anywhere in retrieved candidates.
- `mean_latency_ms` and `p95_latency_ms`: retrieval + ranking only; excludes catalog
  loading/index construction, clarification, HTTP overhead, and any offline fitting.
- Per-case retrieved rank and ranked Top-10 rank; an absent target rank is null.

MRR@10 counts reciprocal rank only for targets within the first ten unique IDs;
otherwise it contributes zero. These are fixed-state, per-record metrics, not
session-averaged metrics or the official interactive challenge score. If sessions
contain unequal numbers of records, they contribute unequal weight here.

## First real comparison

1. Record a version/hash of the frozen catalog and public development sessions.
2. Split sessions before tuning. Keep all candidates and turns of a session in
   one split; keep validation labels out of model fitting and prompt examples.
3. Verify retrieval recall. The ranker cannot recover a target outside its inputs.
4. Compare BM25 order, hybrid retrieval order, and this baseline on identical pools.
5. Tune weights on training data, then report held-out Hit@10/MRR@10 and latency.
6. Separately evaluate the full clarification loop: early conversion, cumulative
   success by turn, and turn limit compliance. Implement the official scoring
   formula only after obtaining the actual specification.
7. For optional LLM reranking record model/version, tokens, latency, cost, failures,
   and benefit on held-out data. No LLM calls occur in the current baseline.

Never report these bundled synthetic sessions as the public or hidden development
set, and never claim the current weights were tuned. Store generated reports under
ignored `artifacts/` and private catalog/session data under `data/private/`.
