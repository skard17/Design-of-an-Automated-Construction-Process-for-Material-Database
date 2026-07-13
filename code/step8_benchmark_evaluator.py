import argparse
import json
from pathlib import Path
from typing import Any


def unwrap_result(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result")
    if isinstance(result, dict) and "schema_definition" in result:
        return result
    module_outputs = payload.get("module_outputs")
    if isinstance(module_outputs, dict):
        aggregation = module_outputs.get("aggregation")
        if isinstance(aggregation, dict) and "schema_definition" in aggregation:
            return aggregation
        schema_design = module_outputs.get("schema_design_module")
        if isinstance(schema_design, dict) and isinstance(schema_design.get("field_registry"), list):
            return {
                "schema_definition": {
                    "top_level_keys": schema_design.get("top_level_keys", []),
                    "field_registry": schema_design.get("field_registry", []),
                }
            }
    return payload


def field_paths_from_result(result: dict[str, Any]) -> list[str]:
    fields = ((result.get("schema_definition") or {}).get("field_registry")) or []
    return [
        str(field.get("field_path") or "")
        for field in fields
        if isinstance(field, dict) and field.get("field_path")
    ]


def evaluate_result(result: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    result = unwrap_result(result)
    field_paths = field_paths_from_result(result)
    lowered_paths = [path.casefold() for path in field_paths]
    required_concepts = benchmark.get("required_concepts") or {}
    covered = []
    missing = []
    evidence = {}
    for concept, aliases in required_concepts.items():
        matched_paths = [
            field_paths[index]
            for index, lowered_path in enumerate(lowered_paths)
            if any(str(alias).casefold() in lowered_path for alias in aliases or [])
        ]
        evidence[concept] = matched_paths
        if matched_paths:
            covered.append(concept)
        else:
            missing.append(concept)

    forbidden_patterns = [str(pattern).casefold() for pattern in benchmark.get("forbidden_patterns", [])]
    forbidden_fields = [
        path for path in field_paths if any(pattern in path.casefold() for pattern in forbidden_patterns)
    ]
    total = len(required_concepts)
    return {
        "required_concepts": total,
        "covered_concepts": len(covered),
        "required_concept_recall": 1.0 if not total else len(covered) / total,
        "covered_concept_ids": sorted(covered),
        "missing_concepts": sorted(missing),
        "concept_evidence": evidence,
        "forbidden_fields": sorted(forbidden_fields),
        "unrelated_domain_field_count": len(forbidden_fields),
        "field_count": len(field_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Step8 outputs against test-only domain benchmarks.")
    parser.add_argument("outputs", nargs="+", help="Step8 JSON outputs to evaluate")
    parser.add_argument("--benchmark", required=True, help="Benchmark JSON file")
    parser.add_argument("--domain", required=True, help="Benchmark domain key")
    parser.add_argument("--output", required=True, help="Report JSON path")
    args = parser.parse_args()

    benchmarks = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    benchmark = benchmarks[args.domain]
    reports = []
    for output in args.outputs:
        path = Path(output)
        payload = json.loads(path.read_text(encoding="utf-8"))
        reports.append({"output": str(path), "evaluation": evaluate_result(payload, benchmark)})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"domain": args.domain, "reports": reports}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
