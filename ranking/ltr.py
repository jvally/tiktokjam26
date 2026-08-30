from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence


FEATURE_NAMES = (
    "base_formula",
    "lexical",
    "hybrid_fusion",
    "hard_constraint",
    "soft_constraint",
    "constraint_floor",
    "exact_constraint_ratio",
    "category_title",
    "profile",
    "rating_quality",
    "facet_coverage",
    "retrieval_reciprocal",
    "price_known",
    "hard_violations",
    "negative_hits",
    "already_shown",
)


def _numeric(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return 0.0


def feature_vector(diagnostics: Mapping[str, object]) -> tuple[float, ...]:
    values = {name: _numeric(diagnostics.get(name)) for name in FEATURE_NAMES}
    values["hard_violations"] = min(1.0, values["hard_violations"] / 3.0)
    values["negative_hits"] = min(1.0, values["negative_hits"] / 3.0)
    values["base_formula"] = max(-2.0, min(1.5, values["base_formula"])) / 2.0 + 0.5
    return tuple(values[name] for name in FEATURE_NAMES)


class RankingModel(Protocol):
    def score(self, diagnostics: Sequence[Mapping[str, object]]) -> Sequence[float]: ...


@dataclass(frozen=True)
class LinearLTRModel:
    weights: tuple[float, ...]
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if len(self.weights) != len(FEATURE_NAMES):
            raise ValueError("LTR weight count does not match feature schema")

    def score(self, diagnostics: Sequence[Mapping[str, object]]) -> list[float]:
        return [
            sum(weight * value for weight, value in zip(self.weights, feature_vector(row)))
            for row in diagnostics
        ]

    def save(self, path: str | Path) -> None:
        payload = {
            "format_version": 1,
            "feature_names": list(FEATURE_NAMES),
            "weights": list(self.weights),
            "metadata": self.metadata,
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> LinearLTRModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("LTR model uses an incompatible feature schema")
        return cls(tuple(float(value) for value in payload["weights"]), dict(payload.get("metadata", {})))


def train_pairwise(
    pairs: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
    *,
    epochs: int = 24,
    learning_rate: float = 0.04,
    l2: float = 0.001,
    seed: int = 2026,
    metadata: Mapping[str, object] | None = None,
) -> LinearLTRModel:
    """Fit pairwise logistic ranking weights using deterministic SGD."""

    if not pairs:
        raise ValueError("At least one positive/negative pair is required")
    differences = [
        tuple(positive - negative for positive, negative in zip(
            feature_vector(target), feature_vector(non_target)
        ))
        for target, non_target in pairs
    ]
    weights = [0.0] * len(FEATURE_NAMES)
    order = list(range(len(differences)))
    rng = random.Random(seed)
    for epoch in range(epochs):
        rng.shuffle(order)
        rate = learning_rate / (1.0 + 0.08 * epoch)
        for index in order:
            difference = differences[index]
            margin = sum(weight * value for weight, value in zip(weights, difference))
            probability_error = 1.0 / (1.0 + math.exp(min(30.0, max(-30.0, margin))))
            for feature_index, value in enumerate(difference):
                weights[feature_index] += rate * (
                    probability_error * value - l2 * weights[feature_index]
                )
    training_metadata = {
        "algorithm": "pairwise_logistic_sgd",
        "pair_count": len(pairs),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "l2": l2,
        "seed": seed,
        **dict(metadata or {}),
    }
    return LinearLTRModel(tuple(weights), training_metadata)
