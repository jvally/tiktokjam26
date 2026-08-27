"""Offline experiment helpers; no model is trained or used in the baseline."""

import hashlib
from typing import Any, Sequence

FEATURE_NAMES = ("bm25", "dense", "metadata", "constraint", "preference", "profile",
                 "category_match", "brand_match", "color_match", "material_match",
                 "size_match", "style_match", "feature_match", "use_case_match",
                 "budget_compatibility", "hard_violation_ratio", "negative_violations",
                 "constraint_coverage")


def feature_vector(ranked_candidate: dict[str, Any]) -> list[float]:
    return [float(ranked_candidate["ranking_features"][key]) for key in FEATURE_NAMES]


def split_by_session(rows: Sequence[dict[str, Any]], validation_fraction: float = .2,
                     seed: str = "techjam-baseline") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split entire public-dev sessions together, not individual candidates.

    This stable hash split may yield an empty side for tiny data; callers must check.
    Never pass hidden evaluation labels here or into online ranking.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    train, validation = [], []
    for row in rows:
        session_id = row.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Training rows require a nonempty session_id")
        bucket = int(hashlib.sha256(f"{seed}:{session_id}".encode()).hexdigest()[:8], 16) / 2**32
        (validation if bucket < validation_fraction else train).append(row)
    return train, validation
