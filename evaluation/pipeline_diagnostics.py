from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from shopping_agent import Agent


def summarize_turns(turns: list[dict]) -> dict:
    if not turns:
        return {
            "turn_count": 0,
            "session_count": 0,
            "candidate_recall": {},
            "session_candidate_recall": 0.0,
            "ranker_hit_at_10_given_retrieved": 0.0,
            "latency_ms": {"mean": 0.0, "p95": 0.0},
        }
    cutoffs = (10, 50, 100, 200, 300)
    candidate_recall = {
        str(cutoff): round(sum(
            row["target_retrieval_rank"] is not None and row["target_retrieval_rank"] <= cutoff
            for row in turns
        ) / len(turns), 6)
        for cutoff in cutoffs
    }
    sessions: dict[str, list[dict]] = defaultdict(list)
    for row in turns:
        sessions[row["sample_id"]].append(row)
    ever_retrieved = sum(any(row["target_retrieval_rank"] is not None for row in rows)
                         for rows in sessions.values())
    retrieved = [row for row in turns if row["target_retrieval_rank"] is not None]
    conditional_top10 = (sum(
        row["target_ranked_rank"] is not None and row["target_ranked_rank"] <= 10
        for row in retrieved
    ) / len(retrieved)) if retrieved else 0.0
    latencies = sorted(float(row["latency_ms"]) for row in turns)
    p95_index = min(len(latencies) - 1, max(0, int(0.95 * len(latencies)) - 1))
    return {
        "turn_count": len(turns),
        "session_count": len(sessions),
        "candidate_recall": candidate_recall,
        "session_candidate_recall": round(ever_retrieved / len(sessions), 6),
        "ranker_hit_at_10_given_retrieved": round(conditional_top10, 6),
        "average_candidate_count": round(statistics.fmean(row["candidate_count"] for row in turns), 3),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3),
            "p95": round(latencies[p95_index], 3),
        },
        "retrieval_modes": dict(Counter(row["retrieval_mode"] for row in turns)),
        "clarification_reasons": dict(Counter(row["clarification_reason"] for row in turns)),
    }


def evaluate_pipeline(catalog_path: str | Path, dataset_path: str | Path) -> dict:
    """Replay public sessions without ever passing target labels into runtime code."""

    catalog_ids, categories, products = catalog_index(catalog_path)
    samples = load_jsonl(dataset_path)
    agent = Agent(catalog_path)
    turns: list[dict] = []
    try:
        for sample in samples:
            session_id = f"diagnostic_{sample['sample_id']}"
            agent.reset(session_id, sample["user_profile"])
            target = str(sample["ground_truth"]["parent_asin"])
            intent_card, behavior = materialize_hidden_fields(sample, products)
            effective_sample = {**sample, "intent_card": intent_card, "behavior": behavior}
            disclosed: set[str] = set()
            boundary_used = False
            override_applied = sample["scenario_type"] != "intent_override"
            user_message = initial_message(
                effective_sample, coarse_category(categories.get(target, [])), disclosed
            )
            for turn in range(1, MAX_TURNS + 1):
                started = time.perf_counter()
                response = agent.respond(session_id, user_message, turn, TOP_K)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                trace = agent.get_diagnostics(session_id)
                candidate_ids = trace["candidate_ids"]
                ranked_ids = trace["ranked_ids"]
                retrieved_rank = candidate_ids.index(target) + 1 if target in candidate_ids else None
                ranked_rank = ranked_ids.index(target) + 1 if target in ranked_ids else None
                turns.append({
                    "sample_id": sample["sample_id"],
                    "scenario_type": sample["scenario_type"],
                    "turn": turn,
                    "candidate_count": trace["candidate_count"],
                    "target_retrieval_rank": retrieved_rank,
                    "target_ranked_rank": ranked_rank,
                    "retrieval_mode": trace["retrieval_mode"],
                    "clarification_reason": trace["clarification_reason"],
                    "latency_ms": elapsed_ms,
                })
                recommendations = normalize_recommendations(response.get("recommendations"), catalog_ids)
                if override_applied and target in recommendations:
                    break
                if turn == MAX_TURNS:
                    break
                override = effective_sample.get("behavior", {}).get("override") or {}
                if not override_applied and turn + 1 == int(override.get("turn", 3)):
                    override_applied = True
                    new_value = str(override.get("new_value", ""))
                    if new_value:
                        disclosed.add(new_value)
                    user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
                else:
                    user_message, boundary_used = customer_reply(
                        effective_sample, response.get("ask_attribute"), disclosed, boundary_used
                    )
    finally:
        agent.index.close()

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in turns:
        grouped[row["scenario_type"]].append(row)
    return {
        "label_safety": "ground truth is inspected only after Agent.respond returns",
        "overall": summarize_turns(turns),
        "by_scenario": {name: summarize_turns(rows) for name, rows in sorted(grouped.items())},
        "turns": turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure retrieval recall and conditional ranking quality")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="pipeline_diagnostics.local.json")
    args = parser.parse_args()
    result = evaluate_pipeline(args.catalog, args.dataset)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "turns"}, indent=2))


if __name__ == "__main__":
    main()
