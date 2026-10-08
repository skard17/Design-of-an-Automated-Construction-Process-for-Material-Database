"""Prepare a leak-separated CARE-IE superconductivity pilot dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from care_counterfactual import (
    COMPONENTS,
    content_sha256,
    file_sha256,
    forbidden_keys,
    strict_record_mismatches,
)


PUBLIC_BANNED_KEYS = {
    "gold",
    "gold_record",
    "expected",
    "expected_value",
    "expected_unit",
    "evidence_tokens",
    "failure_labels",
    "minimal_oracle_interventions",
    "oracle_payload",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def select_diverse(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    remaining = sorted(candidates, key=lambda item: item["case_id"])
    selected: list[dict[str, Any]] = []
    concept_counts: Counter[str] = Counter()
    paper_counts: Counter[str] = Counter()
    while remaining and len(selected) < limit:
        best = min(
            remaining,
            key=lambda item: (
                concept_counts[item["concept_id"]],
                paper_counts[item["paper_id"]],
                item["case_id"],
            ),
        )
        remaining.remove(best)
        selected.append(best)
        concept_counts[best["concept_id"]] += 1
        paper_counts[best["paper_id"]] += 1
    return selected


def build_candidates(
    gold_facts: list[dict[str, Any]], prediction_facts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    predictions = {item["target_id"]: item for item in prediction_facts}
    candidates = []
    for gold in gold_facts:
        case_id = gold["fact_id"]
        prediction = predictions.get(case_id)
        if prediction is None:
            prediction = {
                "target_id": case_id,
                "paper_id": gold["paper_id"],
                "record_key": gold["record_key"],
                "concept_id": gold["concept_id"],
                "value": None,
                "unit": None,
                "qualifiers": {},
                "evidence_line": None,
            }
        mismatch_fields = strict_record_mismatches(gold, prediction)
        if mismatch_fields:
            candidates.append(
                {
                    "case_id": case_id,
                    "paper_id": gold["paper_id"],
                    "record_key": gold["record_key"],
                    "concept_id": gold["concept_id"],
                    "baseline_record": prediction,
                    "gold_record": gold,
                    "mismatch_fields": mismatch_fields,
                }
            )
    return candidates


def source_snapshot(repo_root: Path, run_root: Path) -> list[dict[str, Any]]:
    paths = [
        run_root / "optimization" / "step9_gold_v3.json",
        run_root / "optimization" / "step9_targeted_predictions.json",
        run_root / "optimization" / "step9_benchmark_targets.json",
        run_root / "relevance_audit" / "relevance_manifest.json",
        run_root / "optimization" / "step8_superconductor_28paper_verified.json",
        repo_root / "code" / "step9_extraction_build_graph.py",
        repo_root / "code" / "downstream_extraction_runner.py",
        repo_root / "code" / "prompt_quality_code_agent.py",
        repo_root / "code" / "care_counterfactual.py",
        repo_root / "code" / "build_care_counterfactual_pilot.py",
    ]
    records = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": file_sha256(str(path)),
            }
        )
    return records


def annotation_row(case: dict[str, Any]) -> dict[str, Any]:
    empty_review = {
        "annotator_id": None,
        "primary_component": None,
        "failure_labels": [],
        "evidence_subtype": None,
        "minimal_oracle_interventions": [],
        "interaction_required": None,
        "direct_evidence": [],
        "confidence": None,
        "notes": None,
    }
    return {
        "schema_version": "care-ie.annotation.v1",
        "case_id": case["case_id"],
        "paper_id": case["paper_id"],
        "record_key": case["record_key"],
        "concept_id": case["concept_id"],
        "allowed_primary_components": list(COMPONENTS),
        "reviewer_a": dict(empty_review),
        "reviewer_b": dict(empty_review),
        "adjudication": {
            "adjudicator_id": None,
            "required": None,
            "primary_component": None,
            "failure_labels": [],
            "minimal_oracle_interventions": [],
            "decision_notes": None,
        },
    }


def update_run_status(run_root: Path, report: dict[str, Any]) -> None:
    status_path = run_root / "RUN_STATUS.json"
    status = read_json(status_path)
    review = status.setdefault("method_innovation_review", {})
    review["implementation_status"] = "initial_counterfactual_pilot_scaffold_complete"
    review["frozen_experiment_status"] = "dev_pilot_prepared_main_not_started"
    review["counterfactual_pilot"] = {
        "status": "prepared",
        "domain_scope": "superconductivity_only",
        "cross_domain_claim_allowed": False,
        "candidate_count": report["selected_candidate_count"],
        "in_domain_frozen_papers": report["in_domain_frozen_paper_count"],
        "temporal_target_papers": 8,
        "output_dir": "optimization\\counterfactual_eval_v1",
        "api_replay_status": "not_started",
        "human_annotation_status": "template_ready_not_started",
    }
    status["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_json(status_path, status)

    md_path = run_root / "RUN_STATUS.md"
    marker = "## CARE-IE Counterfactual Pilot"
    block = (
        f"\n{marker}\n\n"
        "- Scope is explicitly limited to superconductivity; cross-domain generalization is not claimed.\n"
        f"- Prepared {report['selected_candidate_count']} development pilot candidates from pre-canonicalization predictions.\n"
        f"- Frozen {report['in_domain_frozen_paper_count']} unannotated in-domain papers for later evaluation.\n"
        "- Counterfactual contracts, isolation checks, evaluator, and dual-review annotation template are ready.\n"
        "- API replay, human annotation, temporal-paper collection, and main experiments have not started.\n"
    )
    current = md_path.read_text(encoding="utf-8")
    if marker in current:
        current = current.split(marker, 1)[0].rstrip() + "\n"
    md_path.write_text(current.rstrip() + "\n" + block, encoding="utf-8")


def build_pilot(
    run_root: Path, repo_root: Path, output_dir: Path, pilot_size: int
) -> dict[str, Any]:
    optimization = run_root / "optimization"
    gold = read_json(optimization / "step9_gold_v3.json")
    predictions = read_json(optimization / "step9_targeted_predictions.json")
    relevance = read_json(run_root / "relevance_audit" / "relevance_manifest.json")

    all_candidates = build_candidates(gold["facts"], predictions["facts"])
    selected = select_diverse(all_candidates, pilot_size)
    dev_papers = sorted({item["paper_id"] for item in gold["facts"]})
    included_papers = sorted(
        item["paper_id"]
        for item in relevance["papers"]
        if item.get("include_in_optimization") is True
    )
    frozen_papers = [paper_id for paper_id in included_papers if paper_id not in dev_papers]

    public_cases = {
        "schema_version": "care-ie.public_case_set.v1",
        "domain": "superconductivity",
        "domain_scope": "superconductivity_only",
        "cross_domain_claim_allowed": False,
        "split": "dev_pilot",
        "baseline_source": "step9_targeted_predictions.json",
        "canonicalization_allowed": False,
        "case_count": len(selected),
        "cases": [
            {
                "schema_version": "care-ie.pilot_candidate.v1",
                "case_id": item["case_id"],
                "domain": "superconductivity",
                "split": "dev_pilot",
                "paper_id": item["paper_id"],
                "target_record": {
                    "record_key": item["record_key"],
                    "concept_id": item["concept_id"],
                },
                "baseline_record": item["baseline_record"],
                "responsibility_annotation_status": "pending_double_review",
            }
            for item in selected
        ],
    }
    leaks = forbidden_keys(public_cases, PUBLIC_BANNED_KEYS)
    if leaks:
        raise RuntimeError(f"public case set leaks evaluator-only keys: {leaks}")

    evaluator_gold = {
        "schema_version": "care-ie.evaluator_gold.v1",
        "access": "evaluator_only",
        "case_count": len(selected),
        "records": [
            {
                "case_id": item["case_id"],
                "gold_record": item["gold_record"],
                "baseline_mismatch_fields": item["mismatch_fields"],
            }
            for item in selected
        ],
    }
    annotations = [annotation_row(item) for item in selected]
    failure_signals = {
        "schema_version": "care-ie.answer_free_failure_signals.v1",
        "source": "independent_evaluator_field_names_only",
        "expected_answers_exposed": False,
        "case_count": len(selected),
        "signals": [
            {
                "case_id": item["case_id"],
                "mismatch_fields": item["mismatch_fields"],
            }
            for item in selected
        ],
    }
    snapshots = source_snapshot(repo_root, run_root)

    manifest = {
        "schema_version": "care-ie.frozen_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_mode": "sha256_of_current_dirty_worktree_and_run_artifacts",
        "domain_scope": "superconductivity_only",
        "cross_domain_claim_allowed": False,
        "development_papers": dev_papers,
        "in_domain_frozen_papers": frozen_papers,
        "temporal_set": {
            "status": "not_collected",
            "target_count": 8,
            "release_after": "2026-07-16",
        },
        "cross_domain_set": {"status": "deferred_after_initial_implementation"},
        "prohibited_runtime_inputs": [
            "step9_gold_v3.json",
            "canonicalize_step9_targeted_predictions.py",
            "step9_targeted_canonical_predictions*.json",
        ],
        "source_artifacts": snapshots,
        "source_snapshot_sha256": content_sha256(snapshots),
    }

    report = {
        "schema_version": "care-ie.pilot_preparation_report.v1",
        "status": "pass",
        "domain_scope": "superconductivity_only",
        "cross_domain_claim_allowed": False,
        "available_precanonicalization_failures": len(all_candidates),
        "selected_candidate_count": len(selected),
        "development_paper_count": len(dev_papers),
        "in_domain_frozen_paper_count": len(frozen_papers),
        "temporal_target_paper_count": 8,
        "public_evaluator_key_leaks": leaks,
        "human_annotation_status": "template_ready_not_started",
        "api_replay_status": "not_started",
        "canonicalization_allowed": False,
        "candidate_concepts": dict(Counter(item["concept_id"] for item in selected)),
        "candidate_papers": dict(Counter(item["paper_id"] for item in selected)),
    }

    write_json(output_dir / "public" / "pilot_cases.json", public_cases)
    write_json(output_dir / "public" / "pilot_failure_signals.json", failure_signals)
    write_json(output_dir / "evaluator" / "pilot_gold.json", evaluator_gold)
    annotation_path = output_dir / "annotation" / "pilot_annotation_template.jsonl"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in annotations),
        encoding="utf-8",
    )
    write_json(output_dir / "FROZEN_INPUT_MANIFEST.json", manifest)
    write_json(output_dir / "PILOT_PREPARATION_REPORT.json", report)
    (output_dir / "PILOT_PREPARATION_REPORT.md").write_text(
        "# CARE-IE Counterfactual Pilot Preparation\n\n"
        "- Status: pass\n"
        "- Domain: superconductivity only\n"
        "- Cross-domain claim allowed: no\n"
        f"- Pre-canonicalization failures available: {len(all_candidates)}\n"
        f"- Pilot candidates selected: {len(selected)}\n"
        f"- In-domain papers frozen for later evaluation: {len(frozen_papers)}\n"
        "- Temporal papers: 0/8 collected\n"
        "- Human annotation: not started\n"
        "- API counterfactual replay: not started\n"
        "- Old canonicalization: prohibited from runtime and evaluation inputs\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# CARE-IE Counterfactual Evaluation v1\n\n"
        "This is an initial superconductivity-only implementation. It validates data "
        "separation, component-isolated interventions, strict record scoring, and "
        "responsibility accounting. It does not establish cross-material-domain "
        "generalization.\n\n"
        "## Layout\n\n"
        "- `public/pilot_cases.json`: pre-canonicalization baseline candidates available "
        "to the replay side.\n"
        "- `evaluator/pilot_gold.json`: evaluator-only records; never load this file in "
        "baseline extraction or repair code.\n"
        "- `annotation/pilot_annotation_template.jsonl`: two independent reviews plus "
        "adjudication for each case.\n"
        "- `FROZEN_INPUT_MANIFEST.json`: development/frozen splits and source hashes.\n"
        "- `PILOT_PREPARATION_REPORT.json`: preparation gates and current progress.\n\n"
        "## Current boundary\n\n"
        "The offline replay contracts are implemented in `code/care_counterfactual.py`. "
        "Human responsibility annotation, executable Step9 trace adapters, API replay, "
        "the eight-paper temporal set, and cross-domain experiments remain future work.\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-dir")
    parser.add_argument("--pilot-size", type=int, default=30)
    parser.add_argument("--update-run-status", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_root / "optimization" / "counterfactual_eval_v1"
    )
    report = build_pilot(run_root, repo_root, output_dir, args.pilot_size)
    if args.update_run_status:
        update_run_status(run_root, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
