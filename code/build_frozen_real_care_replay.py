"""Build executable CARE-IE cases from untouched frozen-paper records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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


EXCLUDED_STAGES = {"paper_info", "figure_classification"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line_number(source_hint: Any, fallback: int) -> int:
    match = re.search(r"(?:line|SOURCE_LINE:)\s*(\d+)", str(source_hint or ""), re.IGNORECASE)
    return int(match.group(1)) if match else fallback


def _scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)


def eligible_records(full_corpus: dict[str, Any], frozen_ids: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for document in full_corpus.get("documents", []):
        paper_id = str(document.get("paper_id") or "")
        if paper_id not in frozen_ids:
            continue
        for source_index, field in enumerate(document.get("extracted_fields", [])):
            evidence = str(field.get("evidence_text") or "").strip()
            value = field.get("value")
            if field.get("stage") in EXCLUDED_STAGES:
                continue
            if not _scalar(value) or len(str(value)) > 300 or len(evidence) < 40:
                continue
            records.append(
                {
                    "paper_id": paper_id,
                    "source_index": source_index,
                    "stage": str(field.get("stage") or ""),
                    "field_path": str(field.get("field_path") or ""),
                    "value": value,
                    "unit": field.get("unit"),
                    "material_system": str(field.get("material_system") or paper_id),
                    "evidence_text": evidence,
                    "source_hint": field.get("source_hint"),
                    "source_type": field.get("source_type"),
                    "confidence": field.get("confidence"),
                }
            )
    by_field_papers: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_field_papers[record["field_path"]].add(record["paper_id"])
    return [record for record in records if len(by_field_papers[record["field_path"]]) >= 2]


def select_round_robin(records: list[dict[str, Any]], frozen_ids: list[str], count: int) -> list[dict[str, Any]]:
    by_paper: dict[str, deque[dict[str, Any]]] = {}
    for paper_id in frozen_ids:
        paper_records = sorted(
            (record for record in records if record["paper_id"] == paper_id),
            key=lambda record: (record["field_path"], record["source_index"]),
        )
        if not paper_records:
            raise ValueError(f"frozen paper has no eligible real record: {paper_id}")
        by_paper[paper_id] = deque(paper_records)

    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for paper_id in frozen_ids:
            if len(selected) >= count:
                break
            if by_paper[paper_id]:
                selected.append(by_paper[paper_id].popleft())
                progressed = True
        if not progressed:
            raise ValueError(f"only {len(selected)} eligible records available for {count} cases")
    return selected


def _record_key(record: dict[str, Any]) -> str:
    return record["material_system"] or record["paper_id"]


def find_distractor(record: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        other
        for other in records
        if other["field_path"] == record["field_path"]
        and other["paper_id"] != record["paper_id"]
        and _record_key(other) != _record_key(record)
        and (other["value"], other["unit"], other["evidence_text"])
        != (record["value"], record["unit"], record["evidence_text"])
    ]
    if not candidates:
        raise ValueError(
            f"no independent same-field distractor for {record['paper_id']}:{record['source_index']}"
        )
    return sorted(candidates, key=lambda item: (item["paper_id"], item["source_index"]))[0]


def to_candidate(record: dict[str, Any], fallback_line: int) -> dict[str, Any]:
    return {
        "paper_id": record["paper_id"],
        "record_key": _record_key(record),
        "concept_id": record["field_path"],
        "value": record["value"],
        "unit": record["unit"],
        "qualifiers": {
            "stage": record["stage"],
            "source_hint": record["source_hint"],
        },
        "evidence_line": _line_number(record["source_hint"], fallback_line),
        "evidence": record["evidence_text"],
        "source_hint": record["source_hint"],
        "source_type": record["source_type"],
        "confidence": record["confidence"],
    }


def build_fixture(record: dict[str, Any], distractor: dict[str, Any], index: int) -> dict[str, Any]:
    target = to_candidate(record, 1000 + index * 2)
    other = to_candidate(distractor, 1001 + index * 2)
    # Evidence selection must not implicitly become a schema error.
    other["concept_id"] = target["concept_id"]
    return {"paper_id": record["paper_id"], "candidates": [target, other]}


def build_and_run(
    *,
    frozen_manifest_path: Path,
    full_corpus_path: Path,
    output_root: Path,
    per_component: int,
    repeats: int,
) -> dict[str, Any]:
    frozen_manifest = read_json(frozen_manifest_path)
    frozen_ids = list(frozen_manifest["in_domain_frozen_papers"])
    full_corpus = read_json(full_corpus_path)
    records = eligible_records(full_corpus, set(frozen_ids))
    case_count = per_component * len(COMPONENTS)
    selected = select_round_robin(records, frozen_ids, case_count)

    bundles = []
    for index, record in enumerate(selected):
        component = COMPONENTS[index % len(COMPONENTS)]
        distractor = find_distractor(record, records)
        fixture = build_fixture(record, distractor, index)
        oracle = oracle_trace_from_fixture(fixture)
        bundles.append(
            build_controlled_case_from_oracle(
                case_id=f"frozen-real-{index + 1:03d}",
                component=component,
                oracle=oracle,
                fault_index=index,
                domain="superconductivity",
                split="in_domain_frozen",
                provenance={
                    "source_paper_id": record["paper_id"],
                    "source_stage": record["stage"],
                    "source_field_path": record["field_path"],
                    "source_field_index": record["source_index"],
                    "distractor_paper_id": distractor["paper_id"],
                    "reference_record_status": "captured_pipeline_output_not_independent_scientific_gold",
                },
            )
        )

    public_cases = {
        "schema_version": "care-ie.frozen_real_controlled_public_set.v1",
        "answer_free": True,
        "case_count": len(bundles),
        "frozen_paper_count": len(set(bundle["case"]["paper_id"] for bundle in bundles)),
        "cases": [answer_free_observable(bundle) for bundle in bundles],
    }
    banned = {"gold_record", "responsible_component", "interventions", "oracle"}
    leaks = forbidden_keys(public_cases, banned)
    if leaks:
        raise RuntimeError(f"frozen public set leaks evaluator keys: {leaks}")

    evaluator = {
        "schema_version": "care-ie.frozen_real_controlled_evaluator_set.v1",
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
    paper_distribution = Counter(bundle["case"]["paper_id"] for bundle in bundles)
    component_distribution = Counter(bundle["responsible_component"] for bundle in bundles)
    source_snapshots = [
        {"path": str(frozen_manifest_path), "size": frozen_manifest_path.stat().st_size, "sha256": sha256(frozen_manifest_path)},
        {"path": str(full_corpus_path), "size": full_corpus_path.stat().st_size, "sha256": sha256(full_corpus_path)},
    ]
    status = "pass" if (
        exact == len(results)
        and deterministic == len(results)
        and false_flips == 0
        and set(paper_distribution) == set(frozen_ids)
        and all(component_distribution[name] == per_component for name in COMPONENTS)
        and not leaks
    ) else "fail"
    report = {
        "schema_version": "care-ie.frozen_real_controlled_replay_report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": "Frozen in-domain real-record controlled perturbations; reference records are captured Step9 outputs, not independently validated scientific gold.",
        "case_count": len(results),
        "eligible_source_record_count": len(records),
        "frozen_paper_count": len(paper_distribution),
        "frozen_papers_expected": len(frozen_ids),
        "paper_distribution": dict(paper_distribution),
        "gold_distribution": dict(component_distribution),
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
    write_json(output_root / "FROZEN_REAL_REPLAY_REPORT.json", report)
    artifact_paths = [
        output_root / "public" / "controlled_cases.json",
        output_root / "evaluator" / "controlled_gold.json",
        output_root / "EXECUTABLE_REPLAY_RESULTS.json",
        output_root / "FROZEN_REAL_REPLAY_REPORT.json",
    ]
    write_json(
        output_root / "FROZEN_REAL_REPLAY_MANIFEST.json",
        {
            "schema_version": "care-ie.frozen_real_controlled_replay_manifest.v1",
            "status": status,
            "source_snapshots": source_snapshots,
            "artifacts": [
                {"path": str(path.relative_to(output_root)), "size": path.stat().st_size, "sha256": sha256(path)}
                for path in artifact_paths
            ],
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-manifest", required=True)
    parser.add_argument("--full-corpus", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--per-component", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    report = build_and_run(
        frozen_manifest_path=Path(args.frozen_manifest).resolve(),
        full_corpus_path=Path(args.full_corpus).resolve(),
        output_root=Path(args.output_root).resolve(),
        per_component=args.per_component,
        repeats=args.repeats,
    )
    print(json.dumps({key: report[key] for key in (
        "status",
        "case_count",
        "frozen_paper_count",
        "exact_unique_responsibility_accuracy",
        "deterministic_rate",
    )}, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
