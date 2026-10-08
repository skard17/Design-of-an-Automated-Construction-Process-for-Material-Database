"""Independently score controlled K3 responsibility predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from care_counterfactual import COMPONENTS


def score(gold: list[str], predicted: list[str], label: str) -> dict[str, Any]:
    tp = sum(g == label and p == label for g, p in zip(gold, predicted))
    fp = sum(g != label and p == label for g, p in zip(gold, predicted))
    fn = sum(g == label and p != label for g, p in zip(gold, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"support": tp + fn, "precision": precision, "recall": recall, "f1": f1}


def calibration(rows: list[tuple[bool, float]], bins: int = 10) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "mean_confidence": 0.0, "brier_score": 0.0, "ece": 0.0, "bins": []}
    bucket_rows: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            (correct, confidence)
            for correct, confidence in rows
            if lower <= confidence < upper or (index == bins - 1 and confidence == 1.0)
        ]
        if not members:
            continue
        accuracy = sum(correct for correct, _ in members) / len(members)
        mean_confidence = sum(confidence for _, confidence in members) / len(members)
        ece += len(members) / len(rows) * abs(accuracy - mean_confidence)
        bucket_rows.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
            }
        )
    return {
        "count": len(rows),
        "mean_confidence": sum(confidence for _, confidence in rows) / len(rows),
        "brier_score": sum((confidence - float(correct)) ** 2 for correct, confidence in rows) / len(rows),
        "ece": ece,
        "bins": bucket_rows,
    }


def evaluation_scope(schema_version: str) -> str:
    if "frozen_real" in schema_version:
        return (
            "Frozen in-domain real-record controlled perturbations across untouched papers; "
            "responsibility gold comes from injected faults, while captured Step9 records are not "
            "independently validated scientific gold."
        )
    return "Balanced synthetic single-fault executable traces; not natural-paper generalization."


def evaluate(controlled_root: Path, predictions_path: Path, output_path: Path) -> dict[str, Any]:
    evaluator = json.loads((controlled_root / "evaluator" / "controlled_gold.json").read_text(encoding="utf-8"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    gold_by_case = {item["case_id"]: item["responsible_component"] for item in evaluator["records"]}
    predicted_by_case = {item["case_id"]: item for item in predictions["results"]}
    shared = sorted(set(gold_by_case) & set(predicted_by_case))
    missing = sorted(set(gold_by_case) - set(predicted_by_case))
    unexpected = sorted(set(predicted_by_case) - set(gold_by_case))
    gold = [gold_by_case[case_id] for case_id in shared]
    predicted = [predicted_by_case[case_id]["primary_component"] for case_id in shared]
    per_class = {label: score(gold, predicted, label) for label in COMPONENTS if label in set(gold)}
    accuracy = sum(g == p for g, p in zip(gold, predicted)) / len(shared) if shared else 0.0
    macro_f1 = sum(item["f1"] for item in per_class.values()) / len(per_class) if per_class else 0.0
    errors = [
        {
            "case_id": case_id,
            "gold_component": gold_by_case[case_id],
            "predicted_component": predicted_by_case[case_id]["primary_component"],
            "confidence": predicted_by_case[case_id].get("confidence"),
            "rationale": predicted_by_case[case_id].get("rationale"),
        }
        for case_id in shared
        if gold_by_case[case_id] != predicted_by_case[case_id]["primary_component"]
    ]
    calibration_rows = [
        (
            gold_by_case[case_id] == predicted_by_case[case_id]["primary_component"],
            max(0.0, min(1.0, float(predicted_by_case[case_id].get("confidence") or 0.0))),
        )
        for case_id in shared
    ]
    report = {
        "schema_version": "care-ie.controlled_attribution_evaluation.v1",
        "model": predictions["model"],
        "gold_case_count": len(gold_by_case),
        "prediction_case_count": len(predicted_by_case),
        "evaluated_cases": len(shared),
        "coverage": len(shared) / len(gold_by_case) if gold_by_case else 0.0,
        "missing_predictions": missing,
        "unexpected_predictions": unexpected,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "gold_distribution": dict(Counter(gold)),
        "prediction_distribution": dict(Counter(predicted)),
        "confusion": dict(Counter(f"{g}->{p}" for g, p in zip(gold, predicted))),
        "error_count": len(errors),
        "errors": errors,
        "calibration": calibration(calibration_rows),
        "scope": evaluation_scope(str(evaluator.get("schema_version") or "")),
        "status": "complete" if not missing and not unexpected else "incomplete",
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlled-root", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate(Path(args.controlled_root), Path(args.predictions), Path(args.output))
    print(json.dumps({key: report[key] for key in ("evaluated_cases", "accuracy", "macro_f1", "error_count")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
