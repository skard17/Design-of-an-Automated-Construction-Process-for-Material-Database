"""Evaluate answer-free attribution predictions against evaluator-only annotations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


COMPONENTS = ("schema", "evidence", "extraction", "binding", "normalization")


def precision_recall_f1(gold: list[str], predicted: list[str], label: str) -> dict[str, Any]:
    tp = sum(g == label and p == label for g, p in zip(gold, predicted))
    fp = sum(g != label and p == label for g, p in zip(gold, predicted))
    fn = sum(g == label and p != label for g, p in zip(gold, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"support": tp + fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate(eval_root: Path, prediction_path: Path, output_path: Path) -> dict[str, Any]:
    annotations = [
        json.loads(line)
        for line in (eval_root / "annotation" / "pilot_annotations_codex.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    gold_by_case = {
        item["case_id"]: item["adjudication"]["primary_component"] for item in annotations
    }
    prediction_payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    predictions_by_case = {
        item["case_id"]: item["primary_component"] for item in prediction_payload["results"]
    }
    shared = sorted(set(gold_by_case) & set(predictions_by_case))
    gold = [gold_by_case[case_id] for case_id in shared]
    predicted = [predictions_by_case[case_id] for case_id in shared]
    per_class = {
        label: precision_recall_f1(gold, predicted, label)
        for label in COMPONENTS
        if label in set(gold)
    }
    macro_f1 = sum(item["f1"] for item in per_class.values()) / len(per_class) if per_class else 0.0
    accuracy = sum(g == p for g, p in zip(gold, predicted)) / len(shared) if shared else 0.0
    majority = Counter(gold).most_common(1)[0][1] / len(gold) if gold else 0.0
    confusion = Counter(f"{g}->{p}" for g, p in zip(gold, predicted))
    observed_gold_classes = [label for label in COMPONENTS if label in set(gold)]
    usage = prediction_payload.get("api_usage") or []
    total_tokens = sum(
        int(((item.get("usage") or {}).get("total_tokens") or 0)) for item in usage
    )
    report = {
        "schema_version": "care-ie.attribution_evaluation.v1",
        "mode": prediction_payload["mode"],
        "model": prediction_payload["model"],
        "evaluated_cases": len(shared),
        "missing_predictions": sorted(set(gold_by_case) - set(predictions_by_case)),
        "accuracy": accuracy,
        "macro_f1_observed_classes": macro_f1,
        "majority_baseline_accuracy": majority,
        "accuracy_gain_over_majority": accuracy - majority,
        "per_class": per_class,
        "gold_distribution": dict(Counter(gold)),
        "prediction_distribution": dict(Counter(predicted)),
        "confusion": dict(confusion),
        "api_total_tokens": total_tokens,
        "annotation_reference": "Codex simulated dual expert pass with documented expert reinspection; not independent human gold.",
        "scope_limit": (
            "Natural superconductivity targeted failures expose only: "
            + ", ".join(observed_gold_classes)
            + ". Unobserved components require controlled cases or broader corpora."
        ),
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate(Path(args.eval_root), Path(args.predictions), Path(args.output))
    print(
        json.dumps(
            {key: report[key] for key in ("mode", "model", "evaluated_cases", "accuracy", "macro_f1_observed_classes", "majority_baseline_accuracy")},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
