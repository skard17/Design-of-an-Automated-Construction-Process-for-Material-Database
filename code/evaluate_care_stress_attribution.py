"""Independently evaluate zero/single/multi-fault CARE-IE predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from care_counterfactual import COMPONENTS
from evaluate_controlled_attribution import calibration


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(stress_root: Path, predictions_path: Path, output_path: Path) -> dict[str, Any]:
    gold = read_json(stress_root / "evaluator" / "stress_gold.json")
    predictions = read_json(predictions_path)
    gold_by_case = {row["case_id"]: row for row in gold["records"]}
    pred_by_case = {row["case_id"]: row for row in predictions["results"]}
    shared = sorted(set(gold_by_case) & set(pred_by_case))
    missing = sorted(set(gold_by_case) - set(pred_by_case))
    unexpected = sorted(set(pred_by_case) - set(gold_by_case))
    rows = []
    for case_id in shared:
        expected = set(gold_by_case[case_id]["responsible_components"])
        predicted = set(pred_by_case[case_id]["responsible_components"])
        minimal = set(pred_by_case[case_id]["minimal_sufficient_set"])
        union = expected | predicted
        rows.append(
            {
                "case_id": case_id,
                "kind": gold_by_case[case_id]["kind"],
                "expected": [name for name in COMPONENTS if name in expected],
                "predicted": [name for name in COMPONENTS if name in predicted],
                "minimal_sufficient_set": [name for name in COMPONENTS if name in minimal],
                "exact": expected == predicted,
                "minimal_exact": expected == minimal,
                "jaccard": len(expected & predicted) / len(union) if union else 1.0,
                "confidence": float(pred_by_case[case_id].get("confidence") or 0.0),
            }
        )
    per_kind = {}
    for kind in sorted({row["kind"] for row in rows}):
        subset = [row for row in rows if row["kind"] == kind]
        per_kind[kind] = {
            "support": len(subset),
            "exact_set_accuracy": sum(row["exact"] for row in subset) / len(subset),
            "minimal_set_accuracy": sum(row["minimal_exact"] for row in subset) / len(subset),
            "mean_jaccard": sum(row["jaccard"] for row in subset) / len(subset),
        }
    per_component = {}
    for component in COMPONENTS:
        tp = sum(component in row["expected"] and component in row["predicted"] for row in rows)
        fp = sum(component not in row["expected"] and component in row["predicted"] for row in rows)
        fn = sum(component in row["expected"] and component not in row["predicted"] for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_component[component] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    no_fault = [row for row in rows if row["kind"] == "no_fault"]
    exact_accuracy = sum(row["exact"] for row in rows) / len(rows) if rows else 0.0
    model_only = None
    if predictions.get("model_results") is not None:
        raw_by_case = {row["case_id"]: row for row in predictions["model_results"]}
        raw_rows = []
        for case_id in shared:
            expected = set(gold_by_case[case_id]["responsible_components"])
            predicted = set(raw_by_case[case_id]["responsible_components"])
            minimal = set(raw_by_case[case_id]["minimal_sufficient_set"])
            union = expected | predicted
            raw_rows.append(
                {
                    "case_id": case_id,
                    "kind": gold_by_case[case_id]["kind"],
                    "expected": [name for name in COMPONENTS if name in expected],
                    "predicted": [name for name in COMPONENTS if name in predicted],
                    "minimal_sufficient_set": [name for name in COMPONENTS if name in minimal],
                    "exact": expected == predicted,
                    "minimal_exact": expected == minimal,
                    "jaccard": len(expected & predicted) / len(union) if union else 1.0,
                }
            )
        model_only = {
            "exact_set_accuracy": sum(row["exact"] for row in raw_rows) / len(raw_rows) if raw_rows else 0.0,
            "minimal_set_accuracy": sum(row["minimal_exact"] for row in raw_rows) / len(raw_rows) if raw_rows else 0.0,
            "mean_jaccard": sum(row["jaccard"] for row in raw_rows) / len(raw_rows) if raw_rows else 0.0,
            "error_count": sum(not row["exact"] for row in raw_rows),
            "errors": [row for row in raw_rows if not row["exact"]],
        }
    report = {
        "schema_version": "care-ie.stress_attribution_evaluation.v1",
        "status": "complete" if not missing and not unexpected else "incomplete",
        "model": predictions["model"],
        "gold_case_count": len(gold_by_case),
        "prediction_case_count": len(pred_by_case),
        "evaluated_cases": len(rows),
        "coverage": len(rows) / len(gold_by_case) if gold_by_case else 0.0,
        "missing_predictions": missing,
        "unexpected_predictions": unexpected,
        "exact_set_accuracy": exact_accuracy,
        "minimal_set_accuracy": sum(row["minimal_exact"] for row in rows) / len(rows) if rows else 0.0,
        "mean_jaccard": sum(row["jaccard"] for row in rows) / len(rows) if rows else 0.0,
        "no_fault_false_positive_rate": sum(bool(row["predicted"]) for row in no_fault) / len(no_fault) if no_fault else 0.0,
        "per_kind": per_kind,
        "per_component": per_component,
        "macro_component_f1": sum(row["f1"] for row in per_component.values()) / len(per_component),
        "calibration": calibration([(row["exact"], max(0.0, min(1.0, row["confidence"]))) for row in rows]),
        "error_count": sum(not row["exact"] for row in rows),
        "errors": [row for row in rows if not row["exact"]],
        "kind_distribution": dict(Counter(row["kind"] for row in rows)),
        "model_only": model_only,
        "final_verdict_source": "deterministic_minimal_sufficient_set_from_counterfactual_replay",
        "scope": "Frozen superconductivity records with controlled no-fault, two-fault, and observable-invariance stress cases.",
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress-root", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate(Path(args.stress_root), Path(args.predictions), Path(args.output))
    print(json.dumps({key: report[key] for key in (
        "status", "evaluated_cases", "exact_set_accuracy", "minimal_set_accuracy", "no_fault_false_positive_rate", "error_count"
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
