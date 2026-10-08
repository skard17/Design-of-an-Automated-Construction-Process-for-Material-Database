"""Measure label and confidence repeatability across independent CARE-IE runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(stress_root: Path, run_paths: list[Path], output_path: Path) -> dict[str, Any]:
    if len(run_paths) < 2:
        raise ValueError("repeatability evaluation requires at least two runs")
    gold_payload = read_json(stress_root / "evaluator" / "stress_gold.json")
    gold = {row["case_id"]: tuple(row["responsible_components"]) for row in gold_payload["records"]}
    runs = [read_json(path) for path in run_paths]
    model_maps = [
        {row["case_id"]: row for row in run.get("model_results", run["results"])}
        for run in runs
    ]
    final_maps = [{row["case_id"]: row for row in run["results"]} for run in runs]
    shared = sorted(set.intersection(*(set(rows) for rows in model_maps)))
    rows = []
    for case_id in shared:
        model_sets = [tuple(mapping[case_id]["responsible_components"]) for mapping in model_maps]
        minimal_sets = [tuple(mapping[case_id]["minimal_sufficient_set"]) for mapping in model_maps]
        final_sets = [tuple(mapping[case_id]["responsible_components"]) for mapping in final_maps]
        confidences = [float(mapping[case_id].get("confidence") or 0.0) for mapping in model_maps]
        rows.append(
            {
                "case_id": case_id,
                "model_label_agreement": len(set(model_sets)) == 1,
                "model_minimal_set_agreement": len(set(minimal_sets)) == 1,
                "final_label_agreement": len(set(final_sets)) == 1,
                "all_model_runs_correct": all(row == gold[case_id] for row in model_sets),
                "all_final_runs_correct": all(row == gold[case_id] for row in final_sets),
                "confidence_mean": statistics.fmean(confidences),
                "confidence_range": max(confidences) - min(confidences),
                "model_sets": [list(row) for row in model_sets],
            }
        )
    expected_cases = min(int(run["case_count"]) for run in runs)
    report = {
        "schema_version": "care-ie.repeatability_evaluation.v1",
        "status": "complete" if len(shared) == expected_cases else "incomplete",
        "run_count": len(runs),
        "run_paths": [str(path) for path in run_paths],
        "expected_case_count": expected_cases,
        "shared_case_count": len(shared),
        "model_label_agreement_rate": sum(row["model_label_agreement"] for row in rows) / len(rows) if rows else 0.0,
        "model_minimal_set_agreement_rate": sum(row["model_minimal_set_agreement"] for row in rows) / len(rows) if rows else 0.0,
        "final_label_agreement_rate": sum(row["final_label_agreement"] for row in rows) / len(rows) if rows else 0.0,
        "all_model_runs_correct_rate": sum(row["all_model_runs_correct"] for row in rows) / len(rows) if rows else 0.0,
        "all_final_runs_correct_rate": sum(row["all_final_runs_correct"] for row in rows) / len(rows) if rows else 0.0,
        "mean_confidence_range": statistics.fmean(row["confidence_range"] for row in rows) if rows else 0.0,
        "max_confidence_range": max((row["confidence_range"] for row in rows), default=0.0),
        "disagreements": [row for row in rows if not row["model_label_agreement"] or not row["model_minimal_set_agreement"]],
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress-root", required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate(Path(args.stress_root), [Path(path) for path in args.run], Path(args.output))
    print(json.dumps({key: report[key] for key in (
        "status", "run_count", "shared_case_count", "model_label_agreement_rate", "all_model_runs_correct_rate", "max_confidence_range"
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
