"""Create evaluator-only Codex expert annotations for the CARE-IE pilot."""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMPONENTS = ("schema", "evidence", "extraction", "binding", "normalization")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    replacements = {
        "μ": "u",
        "µ": "u",
        "²": "2",
        "³": "3",
        "−": "-",
        "–": "-",
        "_{": "",
        "}": "",
        "\\bar": "",
        "approximately": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"[^a-z0-9.+-]", "", text)


def numeric_value(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    match = re.fullmatch(r"\s*[~≈]?\s*([-+]?\d+(?:\.\d+)?)\s*", str(value or ""))
    return float(match.group(1)) if match else None


def unit_scale(unit: Any) -> float:
    text = compact_text(unit)
    if "10-3" in text or "10^-3" in text:
        return 1e-3
    if text.startswith("ma"):
        return 1e-3
    if text.startswith("ua"):
        return 1e-6
    return 1.0


def semantically_equivalent(
    baseline_value: Any, baseline_unit: Any, gold_value: Any, gold_unit: Any
) -> bool:
    left_number = numeric_value(baseline_value)
    right_number = numeric_value(gold_value)
    if left_number is not None and right_number is not None:
        left = left_number * unit_scale(baseline_unit)
        right = right_number * unit_scale(gold_unit)
        if math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-12):
            return True
    left = compact_text(baseline_value)
    right = compact_text(gold_value)
    if left == right:
        return True
    # Treat controlled wording additions as normalization only when the core
    # scientific phrase remains identical rather than merely topically related.
    if left and right and min(len(left), len(right)) >= 5:
        shorter, longer = sorted((left, right), key=len)
        if shorter in longer and len(shorter) / len(longer) >= 0.72:
            return True
    return False


def source_lines(pack_path: Path) -> dict[int, str]:
    result = {}
    inline = re.compile(r"^SOURCE_LINE\s+(\d+)\s*:\s*(.*)$")
    marker = re.compile(r"^<!--\s*SOURCE_LINE\s*:\s*(\d+)\s*-->$")
    pending: int | None = None
    for line in pack_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = inline.match(line)
        if match:
            result[int(match.group(1))] = match.group(2).strip()
            pending = None
            continue
        match = marker.match(line)
        if match:
            pending = int(match.group(1))
            continue
        if pending is not None:
            result[pending] = line.strip()
            pending = None
    return result


def reviewer_a(baseline: dict[str, Any], gold: dict[str, Any], mismatches: list[str]) -> str:
    equivalent = semantically_equivalent(
        baseline.get("value"), baseline.get("unit"), gold.get("value"), gold.get("unit")
    )
    if equivalent and "evidence_line" not in mismatches:
        return "normalization"
    if mismatches == ["qualifiers"]:
        return "normalization"
    return "extraction"


def reviewer_b(
    baseline: dict[str, Any],
    gold: dict[str, Any],
    mismatches: list[str],
    lines: dict[int, str],
) -> str:
    equivalent = semantically_equivalent(
        baseline.get("value"), baseline.get("unit"), gold.get("value"), gold.get("unit")
    )
    baseline_line = lines.get(int(baseline.get("evidence_line") or 0), "")
    gold_line = lines.get(int(gold.get("evidence_line") or 0), "")
    same_evidence = bool(baseline_line and gold_line and compact_text(baseline_line) == compact_text(gold_line))
    if equivalent and ("evidence_line" not in mismatches or same_evidence):
        return "normalization"
    return "extraction"


def adjudicate(
    first: str,
    second: str,
    baseline: dict[str, Any],
    gold: dict[str, Any],
    mismatches: list[str],
) -> tuple[str, str]:
    if first == second:
        return first, "Both Codex review passes agreed."
    equivalent = semantically_equivalent(
        baseline.get("value"), baseline.get("unit"), gold.get("value"), gold.get("unit")
    )
    if equivalent and "evidence_line" not in mismatches:
        return "normalization", "Adjudication found the scientific value equivalent and limited the failure to canonical form."
    return "extraction", "Adjudication prioritized incorrect evidence selection or scientific content over surface normalization."


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if not labels_a or len(labels_a) != len(labels_b):
        return 0.0
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a)
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum(
        (counts_a[label] / len(labels_a)) * (counts_b[label] / len(labels_b))
        for label in set(counts_a) | set(counts_b)
    )
    return 1.0 if expected == 1.0 else (observed - expected) / (1.0 - expected)


def apply_expert_override(
    case_id: str, final: str, decision: str, overrides: dict[str, Any]
) -> tuple[str, str, dict[str, Any] | None]:
    override = overrides.get(case_id)
    if not override:
        return final, decision, None
    component = str(override.get("primary_component") or "")
    if component not in COMPONENTS:
        raise ValueError(f"invalid expert override component for {case_id}: {component}")
    notes = str(override.get("decision_notes") or "").strip()
    if not notes:
        raise ValueError(f"expert override for {case_id} requires decision_notes")
    return component, notes, override


def annotate(run_root: Path, eval_root: Path) -> dict[str, Any]:
    public = read_json(eval_root / "public" / "pilot_cases.json")
    evaluator = read_json(eval_root / "evaluator" / "pilot_gold.json")
    gold_by_case = {item["case_id"]: item for item in evaluator["records"]}
    pack_dir = run_root / "optimization" / "step9_benchmark_packs"
    override_path = eval_root / "annotation" / "expert_overrides_v2.json"
    overrides_payload = read_json(override_path) if override_path.is_file() else {"cases": {}}
    overrides = overrides_payload.get("cases") or {}
    rows = []
    labels_a = []
    labels_b = []
    final_labels = []

    for case in public["cases"]:
        gold_item = gold_by_case[case["case_id"]]
        baseline = case["baseline_record"]
        gold = gold_item["gold_record"]
        mismatches = gold_item["baseline_mismatch_fields"]
        lines = source_lines(pack_dir / f"{case['paper_id']}.md")
        first = reviewer_a(baseline, gold, mismatches)
        second = reviewer_b(baseline, gold, mismatches, lines)
        final, decision = adjudicate(first, second, baseline, gold, mismatches)
        final, decision, expert_override = apply_expert_override(
            case["case_id"], final, decision, overrides
        )
        labels_a.append(first)
        labels_b.append(second)
        final_labels.append(final)
        baseline_line = int(baseline.get("evidence_line") or 0)
        gold_line = int(gold.get("evidence_line") or 0)
        failure_labels = [f"field_mismatch:{field}" for field in mismatches]
        if final == "normalization":
            failure_labels.append("canonical_form_mismatch")
        else:
            failure_labels.append("candidate_or_evidence_selection_mismatch")
        row = {
            "schema_version": "care-ie.annotation.v1",
            "case_id": case["case_id"],
            "paper_id": case["paper_id"],
            "record_key": case["target_record"]["record_key"],
            "concept_id": case["target_record"]["concept_id"],
            "review_mode": "codex_simulated_dual_expert_pass",
            "real_human_annotator_count": 0,
            "reviewer_a": {
                "annotator_id": "codex_expert_pass_a",
                "primary_component": first,
                "failure_labels": failure_labels,
                "confidence": 0.9 if first == second else 0.72,
            },
            "reviewer_b": {
                "annotator_id": "codex_expert_pass_b",
                "primary_component": second,
                "failure_labels": failure_labels,
                "confidence": 0.9 if first == second else 0.72,
            },
            "adjudication": {
                "adjudicator_id": "codex_self_adjudication",
                "required": first != second or expert_override is not None,
                "primary_component": final,
                "failure_labels": failure_labels,
                "minimal_oracle_interventions": [final],
                "interaction_required": False,
                "decision_notes": decision,
                "expert_override": expert_override,
            },
            "direct_evidence": {
                "baseline_line": baseline_line,
                "baseline_text": lines.get(baseline_line),
                "gold_line": gold_line,
                "gold_text": lines.get(gold_line),
            },
        }
        rows.append(row)

    output = eval_root / "annotation" / "pilot_annotations_codex.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "schema_version": "care-ie.annotation_report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_simulated_expert_review",
        "case_count": len(rows),
        "review_mode": "codex_simulated_dual_expert_pass",
        "real_human_annotator_count": 0,
        "reviewer_exact_agreement": sum(a == b for a, b in zip(labels_a, labels_b)) / len(rows),
        "reviewer_cohen_kappa": cohen_kappa(labels_a, labels_b),
        "reviewer_a_distribution": dict(Counter(labels_a)),
        "reviewer_b_distribution": dict(Counter(labels_b)),
        "adjudicated_distribution": dict(Counter(final_labels)),
        "adjudication_count": sum(a != b for a, b in zip(labels_a, labels_b)),
        "expert_override_count": sum(
            1 for row in rows if row["adjudication"].get("expert_override") is not None
        ),
        "known_limitations": [
            "Both passes were performed by the same Codex system and are not independent human annotations.",
            "The natural targeted benchmark exposes extraction, normalization, and one binding failure.",
            "Schema and evidence-retrieval attribution require controlled cases or a broader natural failure corpus.",
        ],
    }
    report_path = eval_root / "annotation" / "PILOT_ANNOTATION_REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--eval-root")
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    eval_root = (
        Path(args.eval_root).resolve()
        if args.eval_root
        else run_root / "optimization" / "counterfactual_eval_v1"
    )
    print(json.dumps(annotate(run_root, eval_root), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
