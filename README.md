# tiktokjam26

Tiktok jam hackathon project: Track 4

## Shopping Copilot team baseline

A runnable starting point for a five-person conversational-search team. Python 3.9+;
**no runtime dependencies, model downloads, API keys, or external services required.**

The ranking brief is implemented as a transparent baseline, with independent work
areas for the other four roles. Role numbering outside Ranking/Clarification is a
proposed split; see [TEAM.md](TEAM.md).

## Start in under a minute

From the repository root:

```sh
python3 -m shopping_copilot demo
python3 -m unittest discover -s tests -v
python3 -m shopping_copilot evaluate
```

The demo returns ten unique product IDs, product details, ranking explanations,
confidence diagnostics, and either a recommendation or clarification decision.
The test suite includes a temporary localhost HTTP server.

Optional isolation (no pip install needed):

```sh
python3 -m venv .venv
source .venv/bin/activate
```

On Windows use `py` instead of `python3` and `.venv\Scripts\activate` to activate.
If you want an installed CLI, `python3 -m pip install -e .` exposes
`shopping-copilot`; this optional packaging step may download setuptools.

## Pick your workstream

| Person | Owns | First integration point |
| --- | --- | --- |
| 1 — Conversation/state | `conversation/` | `update_state(previous, query, updates)` |
| 2 — Retrieval | `retrieval/` | `retrieve(state, limit=200)` |
| 3 — Ranking | `ranking/` | `rank_candidates(state, candidates, top_n=20)` |
| 4 — Clarification | `policy/` | `decide(state, ranked, diagnostics)` |
| 5 — Integration/evaluation | `shopping_copilot/`, `evaluation/` | `ShoppingCopilot.search(state)` |

`shared/` is the common contract. Coordinate changes there; every workstream can
start against the demo data immediately. See [team starter tasks](TEAM.md) and
[exact interface contracts](docs/CONTRACTS.md).

## Run your own search

```sh
python3 -m shopping_copilot search --query "comfortable running shoes"
python3 -m shopping_copilot search --state examples/search_state.json
python3 -m examples.ranking_only
```

Plain queries currently drive lexical retrieval. **The starter does not extract
hard constraints from natural language**: use a state JSON file or explicit slot
updates until Person 1 connects their parser/model.

## Local API for integration

```sh
python3 -m shopping_copilot serve
```

In another terminal:

```sh
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"state":{"query":"running shoes","hard_constraints":{"color":"black","budget_max":100}}}'
curl -X POST http://127.0.0.1:8000/turn \
  -H 'Content-Type: application/json' \
  -d '{"query":"running shoes","updates":{"hard_constraints":{"color":"black"}}}'
```

`/search` ranks an explicit state without incrementing its turn. `/turn` starts a
new state or accepts `previous_state`, increments its turn, and merges `updates`.
Send back the returned **state**, including `asked_attributes`, on the next turn.
No server-side sessions are stored. Turn 10 produces recommendations and no new
question; `/turn` rejects an eleventh turn. This is a local development API, not
an official challenge adapter or a production service. It binds to localhost.

## Connect the real catalog and public development sessions

Place private/local datasets in ignored `data/private/`. Normalize product rows to
the [Product contract](docs/CONTRACTS.md), keeping the real `parent_asin` unchanged.

```sh
python3 -m shopping_copilot search \
  --catalog data/private/catalog.jsonl --state examples/search_state.json
python3 -m shopping_copilot evaluate \
  --catalog data/private/catalog.jsonl --sessions data/private/public_dev.json
```

The loader accepts a JSON array or `.jsonl` file and rejects duplicate IDs. It does
not assume a raw Amazon schema; normalize nested categories, price strings,
variants, and attribute names in a separate importer. It does not fetch a dataset.

The bundled **18 products and five sessions are fictional smoke-test fixtures**.
They are not the frozen 50,000-product challenge catalog. Their Hit@10/MRR scores
do not measure real task quality. The ranker's weights and policy thresholds are
untuned. No official challenge score or early-conversion score is claimed.

## What is ready versus intentionally left to the team

Ready: validated contracts, structured multi-turn state updates, BM25 retrieval,
constraint-aware hybrid ranking, unique Top-10 output, confidence diagnostics,
clarification heuristics, JSON CLI/API, offline metrics, and tests.

Next: actual catalog normalization, conversational NLU, dense retrieval/embeddings,
public-dev tuning, official submission adapter, and interactive-session evaluation.
Optional LTR split/export helpers and a validated shortlist-reranker interface are
provided; no supervised model is trained and no LLM reranker is called by default.

Read [the ranking formula](docs/RANKING.md) and [evaluation notes](docs/EVALUATION.md)
before interpreting scores. `make demo`, `make test`, `make evaluate`, and
`make serve` are equivalent shortcuts where Make is available.
