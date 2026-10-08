"""Build no-fault, multi-fault, and observable-invariance CARE-IE cases."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from care_counterfactual import CASE_SCHEMA, COMPONENTS, forbidden_keys, strict_record_match
from care_executable_replay import (
    CONTROLLED_CASE_SCHEMA,
    answer_free_observable,
    build_controlled_case_from_oracle,
    execute_trace_adapter,
    inject_single_fault,
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


def inject_faults(oracle: dict[str, Any], components: tuple[str, ...], fault_index: int) -> dict[str, Any]:
    faulty = copy.deepcopy(oracle)
    for offset, component in enumerate(components):
        faulty = inject_single_fault(faulty, component, fault_index + offset)
    return faulty


def make_case(
    *,
    case_id: str,
    oracle: dict[str, Any],
    baseline: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CASE_SCHEMA,
        "controlled_schema_version": CONTROLLED_CASE_SCHEMA,
        "case_id": case_id,
        "domain": "superconductivity",
        "split": "in_domain_frozen",
        "paper_id": oracle["fixture"]["paper_id"],
        "baseline_trace": baseline,
        "provenance": provenance,
    }


def replay_subset(
    baseline: dict[str, Any], oracle: dict[str, Any], components: tuple[str, ...]
) -> dict[str, Any]:
    patched = copy.deepcopy(baseline)
    for component in components:
        patched[component] = copy.deepcopy(oracle[component])
    return execute_trace_adapter("normalization", patched)


def evaluate_multi_bundle(bundle: dict[str, Any], repeats: int) -> dict[str, Any]:
    case = bundle["case"]
    oracle = bundle["oracle"]
    gold = bundle["gold_record"]
    expected = tuple(bundle["responsible_components"])
    evaluated = []
    for size in (1, 2):
        for subset in itertools.combinations(COMPONENTS, size):
            hashes = []
            outcomes = []
            for repeat_id in range(1, repeats + 1):
                final = replay_subset(case["baseline_trace"], oracle, subset)
                encoded = json.dumps(final, ensure_ascii=False, sort_keys=True).encode("utf-8")
                hashes.append(hashlib.sha256(encoded).hexdigest())
                outcomes.append(strict_record_match(gold, final))
            evaluated.append(
                {
                    "components": list(subset),
                    "repeat_count": repeats,
                    "strict_matches": outcomes,
                    "deterministic": len(set(hashes)) == 1,
                }
            )
    successful = [row["components"] for row in evaluated if all(row["strict_matches"])]
    expected_list = list(expected)
    return {
        "case_id": case["case_id"],
        "responsible_components": expected_list,
        "baseline_matches": strict_record_match(gold, case["baseline_trace"]["final_record"]),
        "successful_subsets_up_to_size_2": successful,
        "single_component_successes": [row for row in successful if len(row) == 1],
        "expected_pair_succeeds": expected_list in successful,
        "unexpected_pair_successes": [row for row in successful if row != expected_list],
        "deterministic": all(row["deterministic"] for row in evaluated),
        "evaluated_repairs": evaluated,
    }


def source_oracles(source_root: Path) -> list[dict[str, Any]]:
    source = read_json(source_root / "evaluator" / "controlled_gold.json")
    rows = []
    for record in source["records"]:
        rows.append(
            {
                "source_case_id": record["case_id"],
                "source_component": record["responsible_component"],
                "oracle": oracle_trace_from_fixture(record["case"]["baseline_trace"]["fixture"]),
            }
        )
    return rows


def interleave(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    queues = [deque(group) for group in groups]
    rows: list[dict[str, Any]] = []
    while any(queues):
        for queue in queues:
            if queue:
                rows.append(queue.popleft())
    return rows


def build_specs(
    sources: list[dict[str, Any]],
    *,
    no_fault_count: int,
    per_pair: int,
    robust_count: int,
) -> list[dict[str, Any]]:
    required = max(no_fault_count, len(list(itertools.combinations(COMPONENTS, 2))) * per_pair, robust_count)
    if len(sources) < required:
        raise ValueError(f"stress benchmark needs at least {required} source fixtures")
    no_fault = [
        {"kind": "no_fault", "source": sources[index], "components": tuple(), "variant": "identity"}
        for index in range(no_fault_count)
    ]
    multi = []
    source_index = 0
    for pair in itertools.combinations(COMPONENTS, 2):
        for _ in range(per_pair):
            multi.append(
                {"kind": "multi_fault", "source": sources[source_index], "components": pair, "variant": "identity"}
            )
            source_index += 1
    robust = []
    for index in range(robust_count):
        source = sources[-(index + 1)]
        robust.append(
            {
                "kind": "single_fault_robustness",
                "source": source,
                "components": (source["source_component"],),
                "variant": "candidate_order_reversed" if index % 2 == 0 else "duplicate_candidate_added",
            }
        )
    return interleave([no_fault, multi, robust])


def build_bundle(spec: dict[str, Any], case_id: str, fault_index: int) -> dict[str, Any]:
    oracle = copy.deepcopy(spec["source"]["oracle"])
    components = tuple(spec["components"])
    provenance = {
        "source_case_id": spec["source"]["source_case_id"],
        "stress_kind": spec["kind"],
        "observable_variant": spec["variant"],
    }
    if len(components) == 1:
        bundle = build_controlled_case_from_oracle(
            case_id=case_id,
            component=components[0],
            oracle=oracle,
            fault_index=fault_index,
            domain="superconductivity",
            split="in_domain_frozen",
            provenance=provenance,
        )
        bundle["responsible_components"] = list(components)
        bundle["kind"] = spec["kind"]
        bundle["oracle"] = oracle
        return bundle
    baseline = oracle if not components else inject_faults(oracle, components, fault_index)
    case = make_case(case_id=case_id, oracle=oracle, baseline=baseline, provenance=provenance)
    return {
        "case": case,
        "gold_record": oracle["final_record"],
        "responsible_components": list(components),
        "kind": spec["kind"],
        "oracle": oracle,
    }


def counterfactual_probe_payload(result: dict[str, Any]) -> dict[str, Any]:
    kind = result["kind"]
    if kind == "no_fault":
        return {"baseline_contract_match": True, "probes": []}
    if kind == "multi_fault":
        probes = [
            {
                "repair_components": row["components"],
                "contract_match": all(row["strict_matches"]),
                "deterministic": row["deterministic"],
                "repeat_count": row["repeat_count"],
            }
            for row in result["evaluated_repairs"]
        ]
        return {"baseline_contract_match": result["baseline_matches"], "probes": probes}
    grouped: dict[str, list[dict[str, Any]]] = {component: [] for component in COMPONENTS}
    for replay in result["evaluated_replays"]:
        grouped[replay["component"]].append(replay)
    probes = []
    for component in COMPONENTS:
        rows = grouped[component]
        probes.append(
            {
                "repair_components": [component],
                "contract_match": all(row["strict_match"] for row in rows),
                "deterministic": len({row["final_record_sha256"] for row in rows}) == 1,
                "repeat_count": len(rows),
            }
        )
    return {"baseline_contract_match": False, "probes": probes}


def public_observable(bundle: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    observable = answer_free_observable(bundle)
    variant = bundle["case"]["provenance"]["observable_variant"]
    if variant == "candidate_order_reversed":
        observable["source_candidates"] = list(reversed(observable["source_candidates"]))
    elif variant == "duplicate_candidate_added":
        duplicate = copy.deepcopy(observable["source_candidates"][-1])
        duplicate["candidate_copy"] = "semantically_duplicate"
        observable["source_candidates"].append(duplicate)
    observable["counterfactual_replay"] = counterfactual_probe_payload(result)
    return observable


def build_and_run(
    *,
    source_root: Path,
    output_root: Path,
    no_fault_count: int = 10,
    per_pair: int = 3,
    robust_count: int = 20,
    repeats: int = 2,
) -> dict[str, Any]:
    sources = source_oracles(source_root)
    specs = build_specs(
        sources,
        no_fault_count=no_fault_count,
        per_pair=per_pair,
        robust_count=robust_count,
    )
    bundles = [build_bundle(spec, f"stress-{index + 1:03d}", index) for index, spec in enumerate(specs)]
    local_results = []
    for bundle in bundles:
        if bundle["kind"] == "no_fault":
            local_results.append(
                {
                    "case_id": bundle["case"]["case_id"],
                    "kind": bundle["kind"],
                    "baseline_matches": strict_record_match(
                        bundle["gold_record"], bundle["case"]["baseline_trace"]["final_record"]
                    ),
                    "minimal_sufficient_set": [],
                    "deterministic": True,
                }
            )
        elif bundle["kind"] == "multi_fault":
            local_results.append({"kind": bundle["kind"], **evaluate_multi_bundle(bundle, repeats)})
        else:
            result = run_controlled_case(bundle, repeats=repeats)
            local_results.append({"kind": bundle["kind"], **result})

    public = {
        "schema_version": "care-ie.stress_public_set.v2",
        "answer_free": True,
        "counterfactual_probe_results_exposed": True,
        "case_count": len(bundles),
        "cases": [
            public_observable(bundle, result)
            for bundle, result in zip(bundles, local_results)
        ],
    }
    banned = {"responsible_component", "responsible_components", "fault_status", "case_type", "gold_record", "oracle", "interventions"}
    leaks = forbidden_keys(public, banned)
    if leaks:
        raise RuntimeError(f"stress public set leaks evaluator keys: {leaks}")

    evaluator = {
        "schema_version": "care-ie.stress_evaluator_set.v2",
        "access": "evaluator_only",
        "case_count": len(bundles),
        "records": [
            {
                "case_id": bundle["case"]["case_id"],
                "kind": bundle["kind"],
                "responsible_components": bundle["responsible_components"],
                "gold_record": bundle["gold_record"],
                "provenance": bundle["case"]["provenance"],
            }
            for bundle in bundles
        ],
    }

    no_fault_ok = all(row["baseline_matches"] for row in local_results if row["kind"] == "no_fault")
    multi_rows = [row for row in local_results if row["kind"] == "multi_fault"]
    multi_ok = all(
        not row["baseline_matches"]
        and not row["single_component_successes"]
        and row["expected_pair_succeeds"]
        and not row["unexpected_pair_successes"]
        and row["deterministic"]
        for row in multi_rows
    )
    robust_rows = [row for row in local_results if row["kind"] == "single_fault_robustness"]
    robust_ok = all(row["exact_unique_responsibility"] and row["deterministic_repeats"] for row in robust_rows)
    distribution = Counter(row["kind"] for row in local_results)
    status = "pass" if no_fault_ok and multi_ok and robust_ok and not leaks else "fail"
    source_evaluator = source_root / "evaluator" / "controlled_gold.json"
    report = {
        "schema_version": "care-ie.stress_replay_report.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "case_count": len(bundles),
        "case_distribution": dict(distribution),
        "no_fault_baseline_match_rate": sum(
            row["baseline_matches"] for row in local_results if row["kind"] == "no_fault"
        ) / max(1, distribution["no_fault"]),
        "multi_fault_exact_minimal_set_rate": sum(
            not row["single_component_successes"]
            and row["expected_pair_succeeds"]
            and not row["unexpected_pair_successes"]
            for row in multi_rows
        ) / max(1, len(multi_rows)),
        "robust_single_fault_unique_responsibility_rate": sum(
            row["exact_unique_responsibility"] for row in robust_rows
        ) / max(1, len(robust_rows)),
        "public_evaluator_leak_count": len(leaks),
        "counterfactual_probe_results_exposed": True,
        "source_snapshot": {
            "path": str(source_evaluator),
            "size": source_evaluator.stat().st_size,
            "sha256": sha256(source_evaluator),
        },
    }
    write_json(output_root / "public" / "stress_cases.json", public)
    write_json(output_root / "evaluator" / "stress_gold.json", evaluator)
    write_json(output_root / "STRESS_REPLAY_RESULTS.json", {"results": local_results})
    write_json(output_root / "STRESS_REPLAY_REPORT.json", report)
    artifacts = [
        output_root / "public" / "stress_cases.json",
        output_root / "evaluator" / "stress_gold.json",
        output_root / "STRESS_REPLAY_RESULTS.json",
        output_root / "STRESS_REPLAY_REPORT.json",
    ]
    write_json(
        output_root / "STRESS_REPLAY_MANIFEST.json",
        {
            "schema_version": "care-ie.stress_replay_manifest.v2",
            "status": status,
            "source_snapshot": report["source_snapshot"],
            "artifacts": [
                {"path": str(path.relative_to(output_root)), "size": path.stat().st_size, "sha256": sha256(path)}
                for path in artifacts
            ],
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--no-fault-count", type=int, default=10)
    parser.add_argument("--per-pair", type=int, default=3)
    parser.add_argument("--robust-count", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    report = build_and_run(
        source_root=Path(args.source_root).resolve(),
        output_root=Path(args.output_root).resolve(),
        no_fault_count=args.no_fault_count,
        per_pair=args.per_pair,
        robust_count=args.robust_count,
        repeats=args.repeats,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
