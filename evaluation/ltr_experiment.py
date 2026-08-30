from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    evaluate,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from ranking import LinearLTRModel, train_pairwise
from ranking.ltr import FEATURE_NAMES
from shopping_agent import Agent


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_disjoint_split(
    samples: list[dict],
    validation_fraction: float = 0.20,
    seed: int = 2026,
) -> tuple[list[dict], list[dict]]:
    """Create a deterministic scenario-stratified split grouped by target ID."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    by_target: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        by_target[str(sample["ground_truth"]["parent_asin"])].append(sample)
    by_scenario: dict[str, list[str]] = defaultdict(list)
    for target, rows in by_target.items():
        by_scenario[str(rows[0]["scenario_type"])].append(target)
    validation_targets: set[str] = set()
    for scenario, targets in by_scenario.items():
        ordered = sorted(targets, key=lambda target: hashlib.sha256(
            f"{seed}:{scenario}:{target}".encode()
        ).hexdigest())
        validation_count = max(1, round(len(ordered) * validation_fraction))
        validation_targets.update(ordered[:validation_count])
    training = [sample for sample in samples
                if str(sample["ground_truth"]["parent_asin"]) not in validation_targets]
    validation = [sample for sample in samples
                  if str(sample["ground_truth"]["parent_asin"]) in validation_targets]
    return training, validation


def collect_pairs(
    agent: Agent,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    hard_negatives: int = 25,
) -> list[tuple[dict, dict]]:
    """Collect labels after responses; ground truth never enters Agent methods."""

    pairs: list[tuple[dict, dict]] = []
    for sample in samples:
        session_id = f"ltr_{sample['sample_id']}"
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
            response = agent.respond(session_id, user_message, turn, TOP_K)
            rows = agent.get_rank_features(session_id)
            if override_applied:
                target_row = next((row for row in rows if row["parent_asin"] == target), None)
                if target_row is not None:
                    negatives = [row for row in rows if row["parent_asin"] != target][:hard_negatives]
                    pairs.extend((target_row, negative) for negative in negatives)
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
    return pairs


def _score_subset(
    model: LinearLTRModel | None,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    catalog_path: str | Path,
) -> dict:
    agent = Agent(catalog_path, ranking_model=model, use_default_ltr=False)
    try:
        result = evaluate(agent, samples, catalog_ids, categories, products)
    finally:
        agent.index.close()
    return {key: result[key] for key in (
        "sample_count", "hit_rate_at_10", "mrr", "mttc", "efficiency",
        "recommended_technical_score", "scenario_metrics",
    )}


def run_experiment(
    catalog_path: str | Path,
    dataset_path: str | Path,
    weights_path: str | Path,
    *,
    seed: int = 2026,
    validation_fraction: float = 0.20,
) -> dict:
    samples = load_jsonl(dataset_path)
    training, validation = target_disjoint_split(samples, validation_fraction, seed)
    catalog_ids, categories, products = catalog_index(catalog_path)
    collector = Agent(catalog_path, use_default_ltr=False)
    try:
        pairs = collect_pairs(collector, training, catalog_ids, categories, products)
    finally:
        collector.index.close()
    metadata = {
        "dataset_sha256": file_sha256(dataset_path),
        "catalog_size": Path(catalog_path).stat().st_size,
        "split_seed": seed,
        "validation_fraction": validation_fraction,
        "training_sample_ids": [sample["sample_id"] for sample in training],
        "validation_sample_ids": [sample["sample_id"] for sample in validation],
    }
    model = train_pairwise(pairs, seed=seed, metadata=metadata)
    model.save(weights_path)
    baseline = _score_subset(None, validation, catalog_ids, categories, products, catalog_path)
    learned = _score_subset(model, validation, catalog_ids, categories, products, catalog_path)
    return {
        "configuration": metadata,
        "pair_count": len(pairs),
        "weights_path": str(weights_path),
        "weights": dict(zip(FEATURE_NAMES, model.weights)),
        "validation_baseline": baseline,
        "validation_ltr": learned,
        "validation_score_delta": round(
            learned["recommended_technical_score"] - baseline["recommended_technical_score"], 6
        ),
        "adopted": learned["recommended_technical_score"] > baseline["recommended_technical_score"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and validate the lightweight pairwise ranker")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--weights", default="ranking/ltr_weights.local.json")
    parser.add_argument("--output", default="ltr_experiment.local.json")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    args = parser.parse_args()
    result = run_experiment(
        args.catalog, args.dataset, args.weights,
        seed=args.seed, validation_fraction=args.validation_fraction,
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
