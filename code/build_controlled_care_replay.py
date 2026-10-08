"""Build and execute a balanced controlled CARE-IE replay benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from care_counterfactual import COMPONENTS, forbidden_keys
from care_executable_replay import answer_free_observable, build_controlled_case, run_controlled_case


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_and_run(output_root: Path, per_component: int, repeats: int) -> dict[str, Any]:
    bundles = []
    case_number = 1
    for fixture_index in range(per_component):
        for component in COMPONENTS:
            bundles.append(build_controlled_case(case_number, component, fixture_index))
            case_number += 1

    public_cases = {
        "schema_version": "care-ie.controlled_public_set.v1",
        "answer_free": True,
        "case_count": len(bundles),
        "cases": [answer_free_observable(bundle) for bundle in bundles],
    }
    banned = {"gold_record", "responsible_component", "interventions", "oracle"}
    leaks = forbidden_keys(public_cases, banned)
    if leaks:
        raise RuntimeError(f"controlled public set leaks evaluator keys: {leaks}")

    evaluator = {
        "schema_version": "care-ie.controlled_evaluator_set.v1",
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
    false_flip_count = sum(
        len([name for name in item["successful_components"] if name != item["responsible_component"]])
        for item in results
    )
    report = {
        "schema_version": "care-ie.controlled_replay_report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(results),
        "per_component": per_component,
        "repeat_count_per_intervention": repeats,
        "interventions_executed": len(results) * len(COMPONENTS) * repeats,
        "gold_distribution": dict(Counter(item["responsible_component"] for item in results)),
        "exact_unique_responsibility_count": exact,
        "exact_unique_responsibility_accuracy": exact / len(results),
        "deterministic_case_count": deterministic,
        "deterministic_rate": deterministic / len(results),
        "non_responsible_false_flip_count": false_flip_count,
        "public_evaluator_leak_count": len(leaks),
        "status": "pass" if exact == len(results) and deterministic == len(results) and false_flip_count == 0 else "fail",
    }

    write_json(output_root / "public" / "controlled_cases.json", public_cases)
    write_json(output_root / "evaluator" / "controlled_gold.json", evaluator)
    write_json(output_root / "EXECUTABLE_REPLAY_RESULTS.json", {"results": results})
    write_json(output_root / "CONTROLLED_REPLAY_REPORT.json", report)
    artifacts = [
        output_root / "public" / "controlled_cases.json",
        output_root / "evaluator" / "controlled_gold.json",
        output_root / "EXECUTABLE_REPLAY_RESULTS.json",
        output_root / "CONTROLLED_REPLAY_REPORT.json",
    ]
    manifest = {
        "schema_version": "care-ie.controlled_replay_manifest.v1",
        "status": report["status"],
        "artifacts": [
            {"path": str(path.relative_to(output_root)), "size": path.stat().st_size, "sha256": sha256(path)}
            for path in artifacts
        ],
    }
    write_json(output_root / "CONTROLLED_REPLAY_MANIFEST.json", manifest)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--per-component", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    report = build_and_run(Path(args.output_root).resolve(), args.per_component, args.repeats)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
