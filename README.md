# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is recommended. The starter uses only the Python standard library.

```bash
python3 -m evaluator.local_evaluator
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 x HitRate@10 + 0.30 x MRR + 0.20 x Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer may reimburse model costs through prizes instead of issuing API keys.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Participant release checklist: `docs/participant_release_checklist.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.

## Team implementation built from this starter

The released weak `starter/agent.py` remains the organizer import path, but now
delegates to five team-owned modules:

| Workstream | Directory | Responsibility |
| --- | --- | --- |
| Conversation/state | `conversation/` | typed multi-slot intent, corrections, and overrides |
| Retrieval | `retrieval/` | cached FTS5, normalized facets, measured weighted dense fusion |
| Ranking | `ranking/` | constraint evidence, recommendation rotation, calibrated pairwise LTR |
| Clarification | `policy/` | candidate-aware and experimental counterfactual question value |
| Integration/evaluation | `shopping_agent/`, `evaluation/` | official API and experiments |

The default enhanced agent remains standard-library-only, offline, and zero-token.
It loads a small checked-in pairwise ranking model (JSON weights; no inference
service). On the released 200-session public set it achieves HitRate@10 `1.000`, MRR
`0.567230`, MTTC `2.430`, and technical score `0.841569`. These are development
results, not private-test guarantees. Full methodology, held-out LTR validation,
ablations, cost, latency, and limitations are in `REPORT.md` and
`docs/experiments.md`; the Person 2 benchmark and setup are in
`docs/RETRIEVAL.md`; team ownership is in `TEAM.md`.

```bash
python3 -m unittest discover -s tests -v
python3 -m evaluator.local_evaluator --output enhanced_v3_results.local.json
python3 -m evaluation.compare_results baseline_results.local.json enhanced_v3_results.local.json
python3 -m evaluation.pipeline_diagnostics --output pipeline_diagnostics_v3.local.json
python3 -m evaluation.retrieval_benchmark --semantic-index data/semantic_index
python3 -m evaluation.ltr_experiment --output ltr_experiment.local.json
python3 -m examples.demo_session
```

### Optional local semantic models

The default path does not download a model. Dense retrieval and a maximum-30-item
CrossEncoder are implemented behind optional adapters, but are not enabled without
an ablation. Install the CPU dependencies and build an index while network access is
available:

```bash
python3 -m pip install -r requirements-semantic.txt
python3 -m retrieval.semantic --catalog data/catalog.jsonl --output data/semantic_index
export TECHJAM_SEMANTIC_INDEX=data/semantic_index
export TECHJAM_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
python3 -m evaluator.local_evaluator --output semantic_results.local.json
```

The embedding model and optional `TECHJAM_EMBEDDING_REVISION` must match the model
recorded in the index. If an optional model fails to load or score, the runtime
warns and falls back to FTS5/formula ranking. See `docs/RETRIEVAL.md` and
`docs/semantic_models.md` before adopting model results.
