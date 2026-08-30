"""Person 5: offline result comparison helpers."""

from typing import Any, Mapping


def compare_results(baseline: Mapping[str, Any], enhanced: Mapping[str, Any]) -> dict[str, float]:
    from .compare_results import compare_results as implementation
    return implementation(baseline, enhanced)


def summarize_turns(turns: list[dict]) -> dict:
    from .pipeline_diagnostics import summarize_turns as implementation
    return implementation(turns)


__all__ = ["compare_results", "summarize_turns"]
