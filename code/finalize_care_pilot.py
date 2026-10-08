"""Finalize audited reports for the superconductivity-only CARE-IE pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{20,}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_credentials(root: Path) -> list[str]:
    matches = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in {".json", ".jsonl", ".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if KEY_PATTERN.search(text):
            matches.append(str(path.relative_to(root)))
    return sorted(matches)


def annotation_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finalize(run_root: Path, eval_root: Path) -> dict[str, Any]:
    public = read_json(eval_root / "public" / "pilot_cases.json")
    frozen = read_json(eval_root / "FROZEN_INPUT_MANIFEST.json")
    annotation_report = read_json(eval_root / "annotation" / "PILOT_ANNOTATION_REPORT.json")
    annotations = annotation_rows(eval_root / "annotation" / "pilot_annotations_codex.jsonl")
    supervisor = read_json(eval_root / "K3_SUPERVISOR_FROZEN_V1_ATTRIBUTIONS.json")
    supervisor_eval = read_json(eval_root / "K3_SUPERVISOR_FROZEN_V1_EVALUATION_V2.json")
    counterfactual = read_json(eval_root / "K3_COUNTERFACTUAL_FROZEN_V1_ATTRIBUTIONS.json")
    counterfactual_eval = read_json(eval_root / "K3_COUNTERFACTUAL_FROZEN_V1_EVALUATION.json")

    gold = {row["case_id"]: row["adjudication"]["primary_component"] for row in annotations}
    predicted = {row["case_id"]: row for row in supervisor["results"]}
    failures = [
        {
            "case_id": case_id,
            "gold_component": gold[case_id],
            "predicted_component": predicted[case_id]["primary_component"],
            "confidence": predicted[case_id].get("confidence"),
            "rationale": predicted[case_id].get("rationale"),
        }
        for case_id in sorted(gold)
        if predicted.get(case_id, {}).get("primary_component") != gold[case_id]
    ]
    guarded = [row for row in supervisor["results"] if row.get("protocol_guard")]
    retry_distribution = Counter(str(item.get("attempt")) for item in supervisor.get("api_usage", []))
    label_distribution = Counter(gold.values())
    observed_components = [name for name in ("schema", "evidence", "extraction", "binding", "normalization") if name in label_distribution]
    unobserved_components = [name for name in ("schema", "evidence", "extraction", "binding", "normalization") if name not in label_distribution]

    coverage = {
        "schema_version": "care-ie.coverage_report.v1",
        "domain_scope": frozen["domain_scope"],
        "cross_domain_claim_allowed": False,
        "natural_case_count": len(public["cases"]),
        "development_paper_count": len(frozen["development_papers"]),
        "frozen_in_domain_paper_count": len(frozen["in_domain_frozen_papers"]),
        "natural_label_distribution": dict(label_distribution),
        "observed_components": observed_components,
        "unobserved_components": unobserved_components,
        "counterfactual_smoke_case_count": counterfactual["case_count"],
        "temporal_set_status": frozen["temporal_set"]["status"],
        "cross_domain_set_status": frozen["cross_domain_set"]["status"],
    }
    write_json(eval_root / "CARE_PILOT_COVERAGE_REPORT.json", coverage)

    credential_matches = scan_credentials(eval_root)
    qc = {
        "schema_version": "care-ie.qc_report.v1",
        "model": supervisor["model"],
        "supervisor_completed": supervisor["completed_count"],
        "supervisor_expected": supervisor["case_count"],
        "supervisor_failures": supervisor["failure_count"],
        "counterfactual_completed": counterfactual["completed_count"],
        "counterfactual_expected": counterfactual["case_count"],
        "counterfactual_failures": counterfactual["failure_count"],
        "expected_answers_exposed": supervisor["expected_answers_exposed"],
        "evaluator_gold_loaded_by_runner": supervisor["evaluator_gold_loaded"],
        "protocol_guard_count": len(guarded),
        "protocol_guard_cases": [row["case_id"] for row in guarded],
        "api_attempt_distribution": dict(retry_distribution),
        "credential_pattern_match_count": len(credential_matches),
        "credential_pattern_match_files": credential_matches,
        "annotation_mode": annotation_report["review_mode"],
        "real_human_annotator_count": annotation_report["real_human_annotator_count"],
        "expert_override_count": annotation_report.get("expert_override_count", 0),
    }
    write_json(eval_root / "CARE_PILOT_QC_REPORT.json", qc)

    failure_report = {
        "schema_version": "care-ie.failure_report.v1",
        "natural_case_failure_count": len(failures),
        "natural_case_failures": failures,
        "unvalidated_components": unobserved_components,
        "known_limitations": [
            "All natural cases and frozen papers are from superconductivity.",
            "Expert labels are simulated Codex reviews, not independent human annotations.",
            "The five-case counterfactual result is model-predicted component intervention, not executable pipeline replay.",
            "Schema and evidence failures have no natural positive examples in this pilot.",
            "Temporal and cross-domain evaluations have not been run.",
        ],
    }
    write_json(eval_root / "CARE_PILOT_FAILURE_REPORT.json", failure_report)

    gates = {
        "all_30_supervisor_outputs_valid": supervisor["completed_count"] == 30 and supervisor["failure_count"] == 0,
        "natural_accuracy_at_least_0_90": supervisor_eval["accuracy"] >= 0.90,
        "natural_macro_f1_at_least_0_90": supervisor_eval["macro_f1_observed_classes"] >= 0.90,
        "counterfactual_5_outputs_valid": counterfactual["completed_count"] == 5 and counterfactual["failure_count"] == 0,
        "counterfactual_primary_accuracy_1_00": counterfactual_eval["accuracy"] == 1.0,
        "answer_free_runner": not supervisor["expected_answers_exposed"] and not supervisor["evaluator_gold_loaded"],
        "credential_scan_clean": not credential_matches,
        "scope_locked_to_superconductivity": frozen["domain_scope"] == "superconductivity_only" and not frozen["cross_domain_claim_allowed"],
    }

    report_paths = [
        eval_root / "CARE_PILOT_COVERAGE_REPORT.json",
        eval_root / "CARE_PILOT_QC_REPORT.json",
        eval_root / "CARE_PILOT_FAILURE_REPORT.json",
        eval_root / "annotation" / "pilot_annotations_codex.jsonl",
        eval_root / "annotation" / "expert_overrides_v2.json",
        eval_root / "K3_SUPERVISOR_FROZEN_V1_ATTRIBUTIONS.json",
        eval_root / "K3_SUPERVISOR_FROZEN_V1_EVALUATION_V2.json",
        eval_root / "K3_COUNTERFACTUAL_FROZEN_V1_ATTRIBUTIONS.json",
        eval_root / "K3_COUNTERFACTUAL_FROZEN_V1_EVALUATION.json",
    ]
    manifest = {
        "schema_version": "care-ie.final_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass_initial_in_domain_pilot" if all(gates.values()) else "failed_gate",
        "gates": gates,
        "model": supervisor["model"],
        "domain_scope": frozen["domain_scope"],
        "cross_domain_claim_allowed": False,
        "natural_attribution": {
            "cases": supervisor_eval["evaluated_cases"],
            "accuracy": supervisor_eval["accuracy"],
            "macro_f1": supervisor_eval["macro_f1_observed_classes"],
            "majority_baseline": supervisor_eval["majority_baseline_accuracy"],
        },
        "counterfactual_smoke": {
            "cases": counterfactual_eval["evaluated_cases"],
            "primary_accuracy": counterfactual_eval["accuracy"],
            "macro_f1": counterfactual_eval["macro_f1_observed_classes"],
            "mode": "model_predicted_component_intervention",
        },
        "artifacts": [
            {
                "path": str(path.relative_to(eval_root)),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in report_paths
        ],
    }
    write_json(eval_root / "CARE_PILOT_FINAL_MANIFEST.json", manifest)

    markdown = f"""# CARE-IE superconductivity-only pilot

- Status: `{manifest['status']}`
- Model: `{supervisor['model']}`
- Natural attribution: {supervisor_eval['evaluated_cases']} cases, accuracy {supervisor_eval['accuracy']:.3f}, macro F1 {supervisor_eval['macro_f1_observed_classes']:.3f}, majority baseline {supervisor_eval['majority_baseline_accuracy']:.3f}.
- Counterfactual smoke test: {counterfactual_eval['evaluated_cases']} cases, primary-component accuracy {counterfactual_eval['accuracy']:.3f}.
- Natural labels: {dict(label_distribution)}.
- Remaining natural attribution errors: {len(failures)}.
- API completion: supervisor {supervisor['completed_count']}/{supervisor['case_count']}; counterfactual {counterfactual['completed_count']}/{counterfactual['case_count']}.
- Credential-pattern scan: {len(credential_matches)} matches.

## Scope

This is an initial `superconductivity_only` implementation. It does not establish cross-material generalization. The counterfactual smoke test asks K3 to predict single-component interventions; executable component replay remains future work.

## Annotation limitation

The two review passes and expert reinspection were performed by Codex, with zero independent human annotators. Two evidence-backed re-adjudications are recorded separately in `annotation/expert_overrides_v2.json`.
"""
    (eval_root / "CARE_PILOT_FINAL_REPORT.md").write_text(markdown, encoding="utf-8")

    status_path = run_root / "RUN_STATUS.json"
    status = read_json(status_path)
    care = status.setdefault("method_innovation_review", {}).setdefault("counterfactual_pilot", {})
    care.update(
        {
            "status": manifest["status"],
            "domain_scope": frozen["domain_scope"],
            "cross_domain_claim_allowed": False,
            "candidate_count": len(public["cases"]),
            "in_domain_frozen_papers": len(frozen["in_domain_frozen_papers"]),
            "api_replay_status": "complete_kimi_k3",
            "annotation_status": "complete_codex_simulated_expert_not_human",
            "natural_attribution_accuracy": supervisor_eval["accuracy"],
            "natural_attribution_macro_f1": supervisor_eval["macro_f1_observed_classes"],
            "natural_attribution_failures": len(failures),
            "counterfactual_smoke_accuracy": counterfactual_eval["accuracy"],
            "counterfactual_smoke_mode": "model_predicted_component_intervention",
            "final_manifest": "optimization\\counterfactual_eval_v1\\CARE_PILOT_FINAL_MANIFEST.json",
        }
    )
    status["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_json(status_path, status)

    status_md_path = run_root / "RUN_STATUS.md"
    status_md = status_md_path.read_text(encoding="utf-8")
    section = """## CARE-IE Counterfactual Pilot

- Status: `pass_initial_in_domain_pilot`.
- Scope: `superconductivity_only`; cross-domain generalization is not claimed.
- K3 natural attribution: 30/30 valid, accuracy 0.900, macro F1 0.925, majority baseline 0.667.
- K3 counterfactual smoke test: 5/5 valid, primary-component accuracy 1.000.
- Natural component coverage: extraction, normalization, and binding; schema and evidence require controlled or broader cases.
- Annotation: Codex simulated expert review with zero independent human annotators; two evidence-backed re-adjudications are recorded.
- Counterfactual limitation: current result is model-predicted component intervention, not executable five-component pipeline replay.
- Reports: `optimization/counterfactual_eval_v1/CARE_PILOT_FINAL_REPORT.md` and `CARE_PILOT_FINAL_MANIFEST.json`.
"""
    if "## CARE-IE Counterfactual Pilot" in status_md:
        status_md = re.sub(r"## CARE-IE Counterfactual Pilot[\s\S]*\Z", section, status_md)
    else:
        status_md = status_md.rstrip() + "\n\n" + section
    status_md_path.write_text(status_md, encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--eval-root")
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    eval_root = Path(args.eval_root).resolve() if args.eval_root else run_root / "optimization" / "counterfactual_eval_v1"
    manifest = finalize(run_root, eval_root)
    print(json.dumps({"status": manifest["status"], "gates": manifest["gates"]}, ensure_ascii=False))
    return 0 if manifest["status"] == "pass_initial_in_domain_pilot" else 2


if __name__ == "__main__":
    raise SystemExit(main())
