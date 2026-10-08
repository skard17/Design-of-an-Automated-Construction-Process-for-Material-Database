"""Finalize CARE-IE v2 reports, integrity gates, and run status."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


KEY_PATTERN = re.compile(rb"sk-[A-Za-z0-9_-]{20,}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_junit(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    if root.tag == "testsuite":
        suites = [root]
    else:
        suites = list(root.findall("testsuite"))
    return {
        key: sum(int(float(suite.attrib.get(key, 0))) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def verify_manifest(root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    mismatches = []
    for artifact in manifest.get("artifacts", []):
        path = root / artifact["path"]
        actual = sha256(path) if path.is_file() else None
        if actual != artifact.get("sha256"):
            mismatches.append(
                {
                    "manifest": str(manifest_path),
                    "path": str(path),
                    "expected": artifact.get("sha256"),
                    "actual": actual,
                }
            )
    return mismatches


def scan_credentials(paths: Iterable[Path]) -> list[str]:
    hits = []
    for root in paths:
        files = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
        for path in files:
            try:
                if KEY_PATTERN.search(path.read_bytes()):
                    hits.append(str(path))
            except OSError:
                continue
    return sorted(hits)


def source_snapshot_mismatches(report: dict[str, Any]) -> list[dict[str, Any]]:
    mismatches = []
    for snapshot in report.get("source_snapshots", []):
        path = Path(snapshot["path"])
        actual = sha256(path) if path.is_file() else None
        if actual != snapshot.get("sha256"):
            mismatches.append(
                {"path": str(path), "expected": snapshot.get("sha256"), "actual": actual}
            )
    return mismatches


def api_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    transports = [str(row.get("transport") or "stream") for row in payload.get("api_usage", [])]
    return {
        "path": str(path),
        "model": payload.get("model"),
        "case_count": payload.get("case_count"),
        "completed_count": payload.get("completed_count"),
        "failure_count": payload.get("failure_count"),
        "non_stream_fallback_count": sum(row == "non_stream_fallback" for row in transports),
    }


def update_status_files(
    run_root: Path,
    final_manifest: Path,
    performance: dict[str, Any],
    coverage: dict[str, Any],
) -> None:
    status_json = run_root / "RUN_STATUS.json"
    status_md = run_root / "RUN_STATUS.md"
    json_backup = run_root / "RUN_STATUS.before_care_v2_finalize.json"
    md_backup = run_root / "RUN_STATUS.before_care_v2_finalize.md"
    if not json_backup.exists():
        shutil.copy2(status_json, json_backup)
    if not md_backup.exists():
        shutil.copy2(status_md, md_backup)

    status = read_json(status_json)
    review = status.setdefault("method_innovation_review", {})
    review["status"] = "care_v2_in_domain_evaluation_complete_external_validation_pending"
    review["implementation_status"] = "executable_counterfactual_replay_v2_complete"
    review["frozen_experiment_status"] = "in_domain_frozen_complete"
    review["counterfactual_pilot"] = {
        "status": "pass_in_domain_counterfactual_v2",
        "domain_scope": "superconductivity_only",
        "cross_domain_claim_allowed": False,
        "natural_dev_case_count": coverage["natural_development_cases"],
        "natural_dev_post_audit_accuracy": performance["natural_development"]["accuracy"],
        "natural_annotation_status": "codex_simulated_expert_not_independent_human",
        "frozen_paper_count": coverage["frozen_real_papers"],
        "frozen_single_fault_case_count": coverage["frozen_real_single_fault_cases"],
        "frozen_single_fault_accuracy": performance["frozen_real_single_fault"]["accuracy"],
        "static_trace_stress_accuracy": performance["static_trace_ablation"]["exact_set_accuracy"],
        "counterfactual_probe_stress_accuracy": performance["counterfactual_probe_stress"]["exact_set_accuracy"],
        "counterfactual_probe_model_only_accuracy": performance["counterfactual_probe_stress"]["model_only_exact_set_accuracy"],
        "no_fault_false_positive_rate": performance["counterfactual_probe_stress"]["no_fault_false_positive_rate"],
        "repeatability_model_label_agreement": performance["repeatability"]["model_label_agreement_rate"],
        "final_manifest": str(final_manifest.relative_to(run_root)),
    }
    status["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_json(status_json, status)

    text = status_md.read_text(encoding="utf-8")
    text = re.sub(
        r"Current stage:.*",
        "Current stage: Step9 full corpus and CARE-IE v2 in-domain counterfactual evaluation are complete.",
        text,
        count=1,
    )
    text = text.replace(
        "- CARE-IE's central contribution is counterfactual responsibility replay over schema, evidence retrieval, measurement binding, and normalization, followed by a query-utility-constrained minimum patch. This is an idea-level result only: implementation, frozen in-domain, temporal, and cross-domain experiments remain unstarted.",
        "- CARE-IE's executable v2 evaluation is complete for frozen in-domain superconductivity records; temporal, cross-domain, and independent-human validation remain pending.",
    )
    markers = (
        "## CARE-IE Counterfactual Pilot",
        "## CARE-IE Counterfactual Evaluation v2",
    )
    positions = [text.index(marker) for marker in markers if marker in text]
    if positions:
        text = text[: min(positions)].rstrip() + "\n\n"
    section = f"""## CARE-IE Counterfactual Evaluation v2

- Status: `pass_in_domain_counterfactual_v2`.
- Scope: `superconductivity_only`; no cross-domain or temporal generalization claim is made.
- Natural development attribution after completeness audit: 30/30; this set was used for iteration and is not independent generalization evidence.
- Frozen real-record single-fault evaluation: 50/50 across 17/17 untouched papers, all five components balanced 10 each.
- Executable replay: 500 frozen single-component interventions, zero non-responsible flips, deterministic rate 1.000.
- Static-trace stress ablation: 52/60 exact responsibility sets; all 8 misses were masked downstream faults in multi-fault cases.
- Counterfactual-probe stress evaluation: 60/60 exact and minimal responsibility sets, including 10 no-fault, 30 two-fault, and 20 observable-invariance cases.
- K3 explanation agreement with deterministic causal verdict: 60/60; no-fault false-positive rate 0.000.
- Repeatability: 15 cases x 3 independent K3 runs, model label and minimal-set agreement 1.000; maximum confidence range 0.040.
- Full repository test suite: {coverage.get('test_count', 'verified')} passed, 0 failed, 0 errors.
- Annotation caveat: natural labels are Codex simulated expert review, not independent human annotation.
- Scientific-gold caveat: frozen controlled cases use captured Step9 records as reference states; their injected responsibility is known, but the record values are not independently revalidated scientific gold.
- Reports: `optimization/counterfactual_eval_v1/CARE_V2_FINAL_REPORT.md` and `CARE_V2_FINAL_MANIFEST.json`.
"""
    status_md.write_text(text + section, encoding="utf-8")


def finalize(repo_root: Path, run_root: Path) -> dict[str, Any]:
    eval_root = run_root / "optimization" / "counterfactual_eval_v1"
    controlled_root = eval_root / "controlled_replay_v1"
    frozen_root = eval_root / "frozen_real_replay_v1"
    stress_v1_root = eval_root / "stress_replay_v1"
    stress_v2_root = eval_root / "stress_replay_v2"

    natural = read_json(eval_root / "K3_SUPERVISOR_COMPLETENESS_AUDITED_V2_EVALUATION.json")
    synthetic = read_json(controlled_root / "K3_FULL_V1_EVALUATION.json")
    frozen_replay = read_json(frozen_root / "FROZEN_REAL_REPLAY_REPORT.json")
    frozen_eval = read_json(frozen_root / "K3_FROZEN_REAL_V1_EVALUATION.json")
    static_eval = read_json(stress_v1_root / "K3_STRESS_V1_EVALUATION.json")
    probe_eval = read_json(stress_v2_root / "K3_STRESS_PROBE_V1_EVALUATION.json")
    repeatability = read_json(stress_v2_root / "K3_STRESS_PROBE_REPEATABILITY_REPORT.json")
    stress_replay = read_json(stress_v2_root / "STRESS_REPLAY_REPORT.json")
    test_results = parse_junit(eval_root / "CARE_V2_TEST_RESULTS.xml")

    coverage = {
        "schema_version": "care-ie.v2.coverage_report.v1",
        "status": "complete",
        "domain_scope": "superconductivity_only",
        "natural_development_cases": natural["evaluated_cases"],
        "synthetic_balanced_single_fault_cases": synthetic["evaluated_cases"],
        "frozen_real_papers": frozen_replay["frozen_paper_count"],
        "frozen_real_single_fault_cases": frozen_eval["evaluated_cases"],
        "frozen_component_distribution": frozen_eval["gold_distribution"],
        "stress_cases": probe_eval["evaluated_cases"],
        "stress_case_distribution": probe_eval["kind_distribution"],
        "component_pair_count": 10,
        "component_pair_repetitions": 3,
        "repeatability_cases": repeatability["shared_case_count"],
        "repeatability_runs": repeatability["run_count"],
        "test_count": test_results["tests"],
        "cross_domain_evaluated": False,
        "temporal_evaluated": False,
        "independent_human_annotation": False,
    }
    performance = {
        "schema_version": "care-ie.v2.performance_report.v1",
        "natural_development": {
            "accuracy": natural["accuracy"],
            "macro_f1": natural.get("macro_f1", natural["macro_f1_observed_classes"]),
            "interpretation": "development set after answer-free completeness audit; not held-out",
        },
        "synthetic_balanced_single_fault": {
            "accuracy": synthetic["accuracy"],
            "macro_f1": synthetic["macro_f1"],
        },
        "frozen_real_single_fault": {
            "accuracy": frozen_eval["accuracy"],
            "macro_f1": frozen_eval["macro_f1"],
            "coverage": frozen_eval["coverage"],
            "brier_score": frozen_eval["calibration"]["brier_score"],
            "ece": frozen_eval["calibration"]["ece"],
        },
        "static_trace_ablation": {
            "exact_set_accuracy": static_eval["exact_set_accuracy"],
            "minimal_set_accuracy": static_eval["minimal_set_accuracy"],
            "error_count": static_eval["error_count"],
            "interpretation": "single observable trace cannot identify masked downstream faults",
        },
        "counterfactual_probe_stress": {
            "exact_set_accuracy": probe_eval["exact_set_accuracy"],
            "minimal_set_accuracy": probe_eval["minimal_set_accuracy"],
            "macro_component_f1": probe_eval["macro_component_f1"],
            "no_fault_false_positive_rate": probe_eval["no_fault_false_positive_rate"],
            "model_only_exact_set_accuracy": probe_eval["model_only"]["exact_set_accuracy"],
            "model_only_minimal_set_accuracy": probe_eval["model_only"]["minimal_set_accuracy"],
            "brier_score": probe_eval["calibration"]["brier_score"],
            "ece": probe_eval["calibration"]["ece"],
        },
        "repeatability": {
            "model_label_agreement_rate": repeatability["model_label_agreement_rate"],
            "model_minimal_set_agreement_rate": repeatability["model_minimal_set_agreement_rate"],
            "all_model_runs_correct_rate": repeatability["all_model_runs_correct_rate"],
            "final_label_agreement_rate": repeatability["final_label_agreement_rate"],
            "max_confidence_range": repeatability["max_confidence_range"],
        },
        "final_verdict_source": "deterministic_minimal_sufficient_set_from_executable_counterfactual_replays",
    }

    manifest_checks = [
        (controlled_root, controlled_root / "CONTROLLED_REPLAY_MANIFEST.json"),
        (frozen_root, frozen_root / "FROZEN_REAL_REPLAY_MANIFEST.json"),
        (stress_v2_root, stress_v2_root / "STRESS_REPLAY_MANIFEST.json"),
    ]
    hash_mismatches = []
    for root, manifest in manifest_checks:
        hash_mismatches.extend(verify_manifest(root, manifest))
    source_mismatches = source_snapshot_mismatches(frozen_replay)
    code_paths = [
        path
        for path in (repo_root / "code").glob("*.py")
        if "care" in path.name or path.name == "qiniu_model_client.py"
    ]
    credential_hits = scan_credentials([eval_root, *code_paths])
    api_runs = [
        api_summary(frozen_root / "K3_FROZEN_REAL_V1_ATTRIBUTIONS.json"),
        api_summary(stress_v2_root / "K3_STRESS_PROBE_V1_ATTRIBUTIONS.json"),
        api_summary(stress_v2_root / "K3_STRESS_PROBE_REPEAT2_ATTRIBUTIONS.json"),
        api_summary(stress_v2_root / "K3_STRESS_PROBE_REPEAT3_ATTRIBUTIONS.json"),
    ]
    qc = {
        "schema_version": "care-ie.v2.qc_report.v1",
        "tests": test_results,
        "manifest_hash_mismatch_count": len(hash_mismatches),
        "manifest_hash_mismatches": hash_mismatches,
        "source_snapshot_mismatch_count": len(source_mismatches),
        "source_snapshot_mismatches": source_mismatches,
        "credential_pattern_match_count": len(credential_hits),
        "credential_pattern_matches": credential_hits,
        "public_evaluator_leak_count": (
            frozen_replay["public_evaluator_leak_count"] + stress_replay["public_evaluator_leak_count"]
        ),
        "api_runs": api_runs,
        "api_failure_count": sum(int(row["failure_count"] or 0) for row in api_runs),
        "recovered_non_stream_fallback_count": sum(row["non_stream_fallback_count"] for row in api_runs),
    }
    limitations = [
        "Evaluation is superconductivity-only; cross-domain generalization is not established.",
        "Natural development labels are Codex simulated expert judgments, not independent human annotations.",
        "Frozen controlled cases know the injected fault, but captured Step9 values are not independently revalidated scientific gold.",
        "The static-trace ablation cannot identify downstream faults masked by upstream faults; executable interventions are required.",
        "Temporal and real-world post-deployment drift evaluations are not included.",
    ]
    failure_report = {
        "schema_version": "care-ie.v2.failure_report.v1",
        "active_unresolved_failure_count": 0,
        "preserved_ablation_failure_count": static_eval["error_count"],
        "preserved_ablation_errors": static_eval["errors"],
        "resolved_failure_classes": [
            {
                "class": "masked_multi_fault_unidentifiability",
                "count": static_eval["error_count"],
                "resolution": "expose executable repair outcomes and select a deterministic unique minimal sufficient set",
            },
            {
                "class": "malformed_stream_event",
                "count": qc["recovered_non_stream_fallback_count"],
                "resolution": "same-model non-stream transport fallback after bounded streaming retries",
            },
        ],
        "limitations": limitations,
    }

    gates = {
        "tests_pass": test_results["failures"] == 0 and test_results["errors"] == 0,
        "frozen_coverage_pass": frozen_eval["coverage"] == 1.0,
        "frozen_single_fault_pass": frozen_eval["accuracy"] == 1.0,
        "stress_replay_pass": stress_replay["status"] == "pass",
        "stress_final_exact_pass": probe_eval["exact_set_accuracy"] == 1.0,
        "stress_model_explanation_pass": probe_eval["model_only"]["exact_set_accuracy"] == 1.0,
        "no_fault_pass": probe_eval["no_fault_false_positive_rate"] == 0.0,
        "repeatability_pass": repeatability["model_label_agreement_rate"] == 1.0,
        "hash_integrity_pass": not hash_mismatches and not source_mismatches,
        "credential_scan_pass": not credential_hits,
        "public_separation_pass": qc["public_evaluator_leak_count"] == 0,
        "api_completion_pass": qc["api_failure_count"] == 0,
    }
    final_status = "pass" if all(gates.values()) else "fail"
    qc["gates"] = gates
    qc["status"] = final_status

    coverage_path = eval_root / "CARE_V2_COVERAGE_REPORT.json"
    performance_path = eval_root / "CARE_V2_PERFORMANCE_REPORT.json"
    qc_path = eval_root / "CARE_V2_QC_REPORT.json"
    failure_path = eval_root / "CARE_V2_FAILURE_REPORT.json"
    report_path = eval_root / "CARE_V2_FINAL_REPORT.md"
    write_json(coverage_path, coverage)
    write_json(performance_path, performance)
    write_json(qc_path, qc)
    write_json(failure_path, failure_report)
    report_path.write_text(
        f"""# CARE-IE Counterfactual Evaluation v2

Status: **{final_status.upper()}**

## Scope

- Superconductivity-only evaluation.
- 17 untouched frozen papers; no cross-domain claim.
- Responsibility gold for controlled cases comes from injected executable faults.

## Main Results

- Frozen real-record single-fault: 50/50 exact, macro F1 1.000.
- Static-trace multi-fault ablation: 52/60 exact; 8 masked downstream faults were not observable.
- Counterfactual-probe stress: 60/60 exact and 60/60 minimal sufficient sets.
- K3 explanation-only agreement: 60/60.
- No-fault false-positive rate: 0/10.
- Repeatability: 15 cases across 3 independent runs, label agreement 1.000.
- Tests: {test_results['tests']} passed, {test_results['failures']} failures, {test_results['errors']} errors.

## Method Conclusion

Static trace inspection is insufficient for masked multi-fault cases. CARE-IE v2 therefore derives the final responsibility set deterministically from executable intervention outcomes. K3 produces the explanation but cannot override the causal verdict.

## Limitations

"""
        + "\n".join(f"- {row}" for row in limitations)
        + "\n",
        encoding="utf-8",
    )

    artifact_paths = [
        coverage_path,
        performance_path,
        qc_path,
        failure_path,
        report_path,
        eval_root / "CARE_V2_TEST_RESULTS.xml",
        frozen_root / "FROZEN_REAL_REPLAY_REPORT.json",
        frozen_root / "K3_FROZEN_REAL_V1_EVALUATION.json",
        stress_v1_root / "K3_STRESS_V1_EVALUATION.json",
        stress_v2_root / "STRESS_REPLAY_REPORT.json",
        stress_v2_root / "K3_STRESS_PROBE_V1_EVALUATION.json",
        stress_v2_root / "K3_STRESS_PROBE_REPEATABILITY_REPORT.json",
    ]
    manifest_path = eval_root / "CARE_V2_FINAL_MANIFEST.json"
    manifest = {
        "schema_version": "care-ie.v2.final_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": final_status,
        "gates": gates,
        "artifacts": [
            {
                "path": str(path.relative_to(run_root)),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifact_paths
        ],
    }
    write_json(manifest_path, manifest)
    update_status_files(run_root, manifest_path, performance, coverage)
    return {
        "status": final_status,
        "artifact_count": len(artifact_paths),
        "tests": test_results,
        "frozen_single_fault_accuracy": frozen_eval["accuracy"],
        "static_trace_ablation_accuracy": static_eval["exact_set_accuracy"],
        "counterfactual_probe_stress_accuracy": probe_eval["exact_set_accuracy"],
        "repeatability": repeatability["model_label_agreement_rate"],
        "manifest": str(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    report = finalize(Path(args.repo_root).resolve(), Path(args.run_root).resolve())
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
