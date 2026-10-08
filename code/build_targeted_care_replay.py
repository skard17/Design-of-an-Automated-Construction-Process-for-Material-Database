"""Build deterministic CARE replay cases from targeted extraction outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from care_counterfactual import COMPONENTS, forbidden_keys
from care_executable_replay import (
    answer_free_observable,
    build_controlled_case_from_oracle,
    oracle_trace_from_fixture,
    run_controlled_case,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)


def _same_value_type(left: Any, right: Any) -> bool:
    left_numeric = isinstance(left, (int, float)) and not isinstance(left, bool)
    right_numeric = isinstance(right, (int, float)) and not isinstance(right, bool)
    return left_numeric == right_numeric


def _source_lines(frozen_manifest: dict[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    papers = frozen_manifest.get("papers", [])
    has_gold_markers = any("in_extraction_gold" in paper for paper in papers)
    for paper in papers:
        if has_gold_markers and not paper.get("in_extraction_gold"):
            continue
        paper_id = str(paper.get("paper_id") or "")
        markdown_path = Path(str(paper.get("markdown_path") or paper.get("path") or ""))
        if not paper_id or not markdown_path.is_file():
            raise ValueError(f"missing frozen Markdown for CARE replay paper: {paper_id}")
        result[paper_id] = markdown_path.read_text(encoding="utf-8").splitlines()
    return result


def eligible_records(
    prediction_payload: dict[str, Any], frozen_manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    lines_by_paper = _source_lines(frozen_manifest)
    records: list[dict[str, Any]] = []
    for source_index, fact in enumerate(prediction_payload.get("facts", [])):
        paper_id = str(fact.get("paper_id") or "")
        value = fact.get("value")
        line_number = fact.get("evidence_line")
        if paper_id not in lines_by_paper or not _scalar(value):
            continue
        if not isinstance(line_number, int) or not 1 <= line_number <= len(lines_by_paper[paper_id]):
            continue
        evidence = lines_by_paper[paper_id][line_number - 1].strip()
        if not evidence:
            continue
        records.append(
            {
                "paper_id": paper_id,
                "source_index": source_index,
                "target_id": str(fact.get("target_id") or f"prediction-{source_index}"),
                "record_key": str(fact.get("record_key") or paper_id),
                "concept_id": str(fact.get("concept_id") or "unresolved_concept"),
                "value": value,
                "unit": fact.get("unit"),
                "qualifiers": dict(fact.get("qualifiers") or {}),
                "evidence_line": line_number,
                "evidence": evidence,
            }
        )
    return records


def select_round_robin(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    by_paper: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for record in sorted(records, key=lambda item: (item["paper_id"], item["source_index"])):
        by_paper[record["paper_id"]].append(record)
    selected: list[dict[str, Any]] = []
    paper_ids = sorted(by_paper)
    while len(selected) < count:
        progressed = False
        for paper_id in paper_ids:
            if len(selected) >= count:
                break
            if by_paper[paper_id]:
                selected.append(by_paper[paper_id].popleft())
                progressed = True
        if not progressed:
            raise ValueError(f"only {len(selected)} eligible targeted records available for {count} cases")
    return selected


def find_distractor(record: dict[str, Any], records: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    independent = [
        other
        for other in records
        if other["paper_id"] != record["paper_id"]
        and other["record_key"] != record["record_key"]
        and (other["value"], other["unit"], other["evidence"])
        != (record["value"], record["unit"], record["evidence"])
        and _same_value_type(record["value"], other["value"])
    ]
    same_concept = [other for other in independent if other["concept_id"] == record["concept_id"]]
    pool = same_concept or independent
    if not pool:
        raise ValueError(f"no independent distractor for targeted record: {record['target_id']}")
    mode = "same_concept_cross_paper" if same_concept else "same_value_type_cross_paper"
    return sorted(pool, key=lambda item: (item["paper_id"], item["source_index"]))[0], mode


def _candidate(record: dict[str, Any], *, concept_id: str | None = None) -> dict[str, Any]:
    return {
        "paper_id": record["paper_id"],
        "record_key": record["record_key"],
        "concept_id": concept_id or record["concept_id"],
        "value": record["value"],
        "unit": record["unit"],
        "qualifiers": record["qualifiers"],
        "evidence_line": record["evidence_line"],
        "evidence": record["evidence"],
    }


def build_and_run(
    *,
    predictions_path: Path,
    frozen_manifest_path: Path,
    output_root: Path,
    per_component: int,
    repeats: int,
    domain_label: str,
) -> dict[str, Any]:
    prediction_payload = read_json(predictions_path)
    frozen_manifest = read_json(frozen_manifest_path)
    records = eligible_records(prediction_payload, frozen_manifest)
    case_count = per_component * len(COMPONENTS)
    selected = select_round_robin(records, case_count)

    bundles = []
    distractor_modes: Counter[str] = Counter()
    for index, record in enumerate(selected):
        component = COMPONENTS[index % len(COMPONENTS)]
        distractor, distractor_mode = find_distractor(record, records)
        distractor_modes[distractor_mode] += 1
        fixture = {
            "paper_id": record["paper_id"],
            "candidates": [
                _candidate(record),
                _candidate(distractor, concept_id=record["concept_id"]),
            ],
        }
        oracle = oracle_trace_from_fixture(fixture)
        bundles.append(
            build_controlled_case_from_oracle(
                case_id=f"targeted-{index + 1:03d}",
                component=component,
                oracle=oracle,
                fault_index=index,
                domain=domain_label,
                split="in_domain_frozen",
                provenance={
                    "source_target_id": record["target_id"],
                    "source_prediction_index": record["source_index"],
                    "distractor_target_id": distractor["target_id"],
                    "distractor_mode": distractor_mode,
                    "reference_record_status": "captured_targeted_output_not_independent_scientific_gold",
                },
            )
        )

    public_cases = {
        "schema_version": "care-ie.targeted_controlled_public_set.v1",
        "answer_free_responsibility_labels": True,
        "case_count": len(bundles),
        "cases": [answer_free_observable(bundle) for bundle in bundles],
    }
    banned = {"gold_record", "responsible_component", "interventions", "oracle"}
    leaks = forbidden_keys(public_cases, banned)
    if leaks:
        raise RuntimeError(f"targeted public set leaks evaluator keys: {leaks}")

    evaluator = {
        "schema_version": "care-ie.targeted_controlled_evaluator_set.v1",
        "access": "evaluator_only",
        "case_count": len(bundles),
        "records": [
            {
                "case_id": bundle["case"]["case_id"],
                "case": bundle["case"],
                "responsible_component": bundle["responsible_component"],
                "gold_record": bundle["gold_record"],
                "interventions": bundle["interventions"],
            }
            for bundle in bundles
        ],
    }
    results = [run_controlled_case(bundle, repeats=repeats) for bundle in bundles]
    exact = sum(item["exact_unique_responsibility"] for item in results)
    deterministic = sum(item["deterministic_repeats"] for item in results)
    false_flips = sum(
        len([name for name in item["successful_components"] if name != item["responsible_component"]])
        for item in results
    )
    component_distribution = Counter(bundle["responsible_component"] for bundle in bundles)
    paper_distribution = Counter(bundle["case"]["paper_id"] for bundle in bundles)
    expected_papers = {str(fact.get("paper_id")) for fact in prediction_payload.get("facts", [])}
    source_snapshots = [
        {"path": str(predictions_path), "size": predictions_path.stat().st_size, "sha256": sha256(predictions_path)},
        {"path": str(frozen_manifest_path), "size": frozen_manifest_path.stat().st_size, "sha256": sha256(frozen_manifest_path)},
    ]
    status = "pass" if (
        exact == len(results)
        and deterministic == len(results)
        and false_flips == 0
        and set(paper_distribution) == expected_papers
        and all(component_distribution[name] == per_component for name in COMPONENTS)
        and not leaks
    ) else "fail"
    report = {
        "schema_version": "care-ie.targeted_controlled_replay_report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": "Controlled responsibility replay over captured targeted outputs; this is not scientific extraction accuracy.",
        "case_count": len(results),
        "eligible_source_record_count": len(records),
        "source_paper_count": len(paper_distribution),
        "source_papers_expected": len(expected_papers),
        "paper_distribution": dict(paper_distribution),
        "gold_distribution": dict(component_distribution),
        "distractor_modes": dict(distractor_modes),
        "repeat_count_per_intervention": repeats,
        "interventions_executed": len(results) * len(COMPONENTS) * repeats,
        "exact_unique_responsibility_count": exact,
        "exact_unique_responsibility_accuracy": exact / len(results),
        "deterministic_case_count": deterministic,
        "deterministic_rate": deterministic / len(results),
        "non_responsible_false_flip_count": false_flips,
        "public_evaluator_leak_count": len(leaks),
        "source_snapshots": source_snapshots,
    }

    write_json(output_root / "public" / "controlled_cases.json", public_cases)
    write_json(output_root / "evaluator" / "controlled_gold.json", evaluator)
    write_json(output_root / "EXECUTABLE_REPLAY_RESULTS.json", {"results": results})
    write_json(output_root / "TARGETED_REPLAY_REPORT.json", report)
    artifacts = [
        output_root / "public" / "controlled_cases.json",
        output_root / "evaluator" / "controlled_gold.json",
        output_root / "EXECUTABLE_REPLAY_RESULTS.json",
        output_root / "TARGETED_REPLAY_REPORT.json",
    ]
    write_json(
        output_root / "TARGETED_REPLAY_MANIFEST.json",
        {
            "schema_version": "care-ie.targeted_controlled_replay_manifest.v1",
            "status": status,
            "source_snapshots": source_snapshots,
            "artifacts": [
                {"path": str(path.relative_to(output_root)), "size": path.stat().st_size, "sha256": sha256(path)}
                for path in artifacts
            ],
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--frozen-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--per-component", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--domain-label", default="materials_literature")
    args = parser.parse_args()
    report = build_and_run(
        predictions_path=Path(args.predictions).resolve(),
        frozen_manifest_path=Path(args.frozen_manifest).resolve(),
        output_root=Path(args.output_root).resolve(),
        per_component=args.per_component,
        repeats=args.repeats,
        domain_label=args.domain_label,
    )
    print(json.dumps({key: report[key] for key in (
        "status",
        "case_count",
        "source_paper_count",
        "exact_unique_responsibility_accuracy",
        "deterministic_rate",
    )}, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
