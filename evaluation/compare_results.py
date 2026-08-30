from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


METRICS = (
    "hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score",
)


def compare_results(baseline: Mapping[str, Any], enhanced: Mapping[str, Any]) -> dict[str, float]:
    comparison: dict[str, float] = {}
    for metric in METRICS:
        before, after = baseline.get(metric), enhanced.get(metric)
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            raise ValueError(f"Both result files need numeric {metric}")
        comparison[f"baseline_{metric}"] = float(before)
        comparison[f"enhanced_{metric}"] = float(after)
        comparison[f"delta_{metric}"] = round(float(after) - float(before), 6)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare organizer evaluator result files")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("enhanced", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    enhanced = json.loads(args.enhanced.read_text(encoding="utf-8"))
    print(json.dumps(compare_results(baseline, enhanced), indent=2))


if __name__ == "__main__":
    main()
