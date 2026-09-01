"""Fixed-probe lexical/hybrid candidate-retrieval benchmark.

The target product is retained only as an offline evaluation label. Runtime
``ConversationState`` objects contain customer-visible text and profile data, never
the target identifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from conversation import ConversationState, update_from_message
from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from retrieval import CatalogIndex
from retrieval.semantic import SentenceTransformerIndex


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_probes(catalog_path: str | Path, dataset_path: str | Path) -> list[dict[str, Any]]:
    """Create the same initial probe plus a fully disclosed upper-bound probe."""

    _, categories, products = catalog_index(catalog_path)
    probes: list[dict[str, Any]] = []
    for sample in load_jsonl(dataset_path):
        target = str(sample["ground_truth"]["parent_asin"])
        intent_card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": intent_card, "behavior": behavior}
        category = coarse_category(categories.get(target, []))

        disclosed: set[str] = set()
        first_message = initial_message(effective, category, disclosed)
        initial_state = update_from_message(
            ConversationState(dict(sample["user_profile"])), first_message, 1
        )
        probes.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "probe_stage": "initial",
            "target_parent_asin": target,
            "state": initial_state,
        })

        hard = [str(value) for value in intent_card.get("hard_constraints", [])]
        soft = [str(value) for value in intent_card.get("soft_preferences", [])]
        if sample["scenario_type"] == "intent_override" and soft:
            soft = soft[:-1]
        requirements = list(dict.fromkeys([*hard, *soft]))
        requirement_text = "; ".join(requirements) or str(
            products[target].get("title") or "the target product"
        )
        resolved_message = (
            f"I'm looking for {category}. A key requirement is: {requirement_text}."
        )
        resolved_state = update_from_message(
            ConversationState(dict(sample["user_profile"])), resolved_message, 3
        )
        probes.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "probe_stage": "resolved",
            "target_parent_asin": target,
            "state": resolved_state,
        })
    return probes


def _summarize(rows: list[dict[str, Any]], candidate_limit: int) -> dict[str, Any]:
    if not rows:
        return {
            "cases": 0,
            f"candidate_recall_at_{candidate_limit}": 0.0,
            "retrieval_hit_at_10": 0.0,
            "retrieval_mrr_at_10": 0.0,
            "mean_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }
    latencies = sorted(float(row["latency_ms"]) for row in rows)
    p95_index = min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)
    return {
        "cases": len(rows),
        f"candidate_recall_at_{candidate_limit}": round(statistics.fmean(
            row["retrieval_rank"] is not None for row in rows
        ), 6),
        "retrieval_hit_at_10": round(statistics.fmean(
            row["retrieval_rank"] is not None and row["retrieval_rank"] <= 10
            for row in rows
        ), 6),
        "retrieval_mrr_at_10": round(statistics.fmean(
            1.0 / row["retrieval_rank"]
            if row["retrieval_rank"] is not None and row["retrieval_rank"] <= 10
            else 0.0
            for row in rows
        ), 6),
        "mean_latency_ms": round(statistics.fmean(latencies), 3),
        "p95_latency_ms": round(latencies[p95_index], 3),
        "mean_candidate_count": round(statistics.fmean(
            row["candidate_count"] for row in rows
        ), 3),
    }


def evaluate_index(
    index: CatalogIndex,
    probes: list[dict[str, Any]],
    candidate_limit: int,
    include_cases: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for probe in probes:
        started = time.perf_counter()
        candidates = index.retrieve(probe["state"], limit=candidate_limit)
        latency_ms = (time.perf_counter() - started) * 1000.0
        identifiers = [item.parent_asin for item in candidates]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Retriever returned duplicate parent_asin values")
        if len(identifiers) > candidate_limit:
            raise ValueError("Retriever returned more candidates than requested")
        target = probe["target_parent_asin"]
        rows.append({
            "sample_id": probe["sample_id"],
            "scenario_type": probe["scenario_type"],
            "probe_stage": probe["probe_stage"],
            "target_parent_asin": target,
            "retrieval_rank": identifiers.index(target) + 1 if target in identifiers else None,
            "candidate_count": len(identifiers),
            "latency_ms": latency_ms,
        })

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scenarios: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["probe_stage"]].append(row)
        scenarios[row["scenario_type"]].append(row)
    result = {
        **_summarize(rows, candidate_limit),
        "by_probe_stage": {
            stage: _summarize(stage_rows, candidate_limit)
            for stage, stage_rows in sorted(grouped.items())
        },
        "by_scenario_type": {
            scenario: _summarize(scenario_rows, candidate_limit)
            for scenario, scenario_rows in sorted(scenarios.items())
        },
    }
    if include_cases:
        result["cases_detail"] = rows
    return result


def compare_retrievers(
    catalog_path: str | Path,
    dataset_path: str | Path,
    semantic_index_path: str | Path,
    candidate_limit: int = 200,
    lexical_weights: tuple[float, ...] = (0.5, 0.75),
    include_cases: bool = False,
) -> dict[str, Any]:
    catalog = Path(catalog_path)
    dataset = Path(dataset_path)
    probes = public_probes(catalog, dataset)
    if not probes:
        raise ValueError("Retrieval benchmark requires at least one public session")
    semantic = SentenceTransformerIndex(semantic_index_path, catalog_path=catalog)
    warmup_started = time.perf_counter()
    semantic.retrieve(probes[0]["state"].query, 1)
    warmup_ms = (time.perf_counter() - warmup_started) * 1000.0

    results: dict[str, Any] = {}
    lexical = CatalogIndex(catalog)
    try:
        results["lexical"] = evaluate_index(
            lexical, probes, candidate_limit, include_cases
        )
    finally:
        lexical.close()

    for weight in lexical_weights:
        label = f"hybrid_lexical_{weight:.2f}".replace(".", "_")
        hybrid = CatalogIndex(
            catalog,
            semantic_retriever=semantic,
            semantic_lexical_weight=weight,
            semantic_pool_multiplier=2,
            semantic_minimum_pool=300,
        )
        try:
            results[label] = evaluate_index(
                hybrid, probes, candidate_limit, include_cases
            )
        finally:
            hybrid.close()

    return {
        "evaluation_mode": "fixed_retrieval_probes_offline_labels_never_enter_retriever",
        "source_sessions": len(probes) // 2,
        "probe_cases": len(probes),
        "candidate_limit": candidate_limit,
        "catalog_sha256": _sha256(catalog),
        "sessions_sha256": _sha256(dataset),
        "semantic_index": str(semantic_index_path),
        "semantic_model": (
            f"{semantic.model_name}@{semantic.model_revision}"
            if semantic.model_revision else semantic.model_name
        ),
        "semantic_warmup_ms": round(warmup_ms, 3),
        "fusion": {"rrf_k": 60, "pool_multiplier": 2, "minimum_pool": 300},
        "results": results,
        "limitations": (
            "The resolved probe is generated from public target metadata and is an "
            "upper-bound retrieval diagnostic, not the interactive TechnicalScore."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare lexical and hybrid candidate retrieval")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--semantic-index", default="data/semantic_index")
    parser.add_argument("--candidate-limit", type=int, default=200)
    parser.add_argument("--lexical-weights", nargs="+", type=float, default=[0.5, 0.75])
    parser.add_argument("--include-cases", action="store_true")
    parser.add_argument("--output", default="retrieval_benchmark.local.json")
    args = parser.parse_args()
    if not 10 <= args.candidate_limit <= 1000:
        parser.error("--candidate-limit must be between 10 and 1000")
    if any(not 0.0 <= weight <= 1.0 for weight in args.lexical_weights):
        parser.error("--lexical-weights values must be between zero and one")
    result = compare_retrievers(
        args.catalog,
        args.dataset,
        args.semantic_index,
        args.candidate_limit,
        tuple(args.lexical_weights),
        args.include_cases,
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
