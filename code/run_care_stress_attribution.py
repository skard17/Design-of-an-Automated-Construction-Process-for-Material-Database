"""Run answer-free K3 attribution where zero, one, or multiple faults may exist."""

from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from care_counterfactual import COMPONENTS
from qiniu_model_client import QiniuModelClient, load_api_key


WRITE_LOCK = threading.Lock()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    definitions = {
        "schema": "The schema stage emits a wrong concept or cannot represent the target contract.",
        "evidence": "The evidence stage selects the wrong source candidate before extraction.",
        "extraction": "Evidence is correct but extraction emits a wrong value, unit, or qualifier.",
        "binding": "The extracted fact is correct but attached to the wrong record key or entity.",
        "normalization": "Pre-normalization content is correct but type, unit, notation, or canonical form changes incorrectly.",
    }
    instruction = (
        "Diagnose the observable execution without assuming a fault count. There may be zero, one, or multiple "
        "responsible components. Identify the earliest independent faults; do not blame later propagation. "
        "counterfactual_replay contains actual repair experiments. A repair set is sufficient only when "
        "contract_match is true and deterministic. Select the smallest sufficient repair set; when the baseline "
        "already matches, report no_fault. "
        "Return JSON with case_id, fault_status (no_fault|single_fault|multi_fault), primary_component "
        "(null only for no_fault), responsible_components, confidence (0..1), component_scores (all five keys, "
        "each containing responsibility_probability and rationale), minimal_sufficient_set, rationale, and "
        "recommended_intervention. responsible_components and minimal_sufficient_set must contain only component "
        "names. Do not invent expected values. Return only JSON."
    )
    payload = {
        "case": case,
        "component_definitions": definitions,
        "protocol": {
            "fault_count_unknown": True,
            "stage_outputs_are_chronological": True,
            "expected_answers_exposed": False,
        },
    }
    return [
        {"role": "system", "content": "You are a rigorous scientific pipeline failure analyst. Return only valid JSON."},
        {"role": "user", "content": instruction + "\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False)},
    ]


def derive_probe_verdict(case: dict[str, Any]) -> dict[str, Any]:
    replay = case.get("counterfactual_replay")
    if not isinstance(replay, dict):
        raise ValueError("counterfactual_replay is required")
    if replay.get("baseline_contract_match") is True:
        return {
            "status": "identified",
            "fault_status": "no_fault",
            "responsible_components": [],
            "minimal_sufficient_set": [],
            "minimal_cardinality": 0,
        }
    passing = []
    for probe in replay.get("probes", []):
        components = _component_list(probe.get("repair_components"), "repair_components")
        if probe.get("contract_match") is True and probe.get("deterministic") is True:
            passing.append(tuple(components))
    if not passing:
        return {
            "status": "indeterminate",
            "fault_status": "multi_fault",
            "responsible_components": [],
            "minimal_sufficient_set": [],
            "minimal_cardinality": None,
        }
    minimum = min(len(row) for row in passing)
    minimal = sorted({row for row in passing if len(row) == minimum})
    if len(minimal) != 1:
        return {
            "status": "ambiguous",
            "fault_status": "multi_fault" if minimum > 1 else "single_fault",
            "responsible_components": [],
            "minimal_sufficient_set": [],
            "minimal_cardinality": minimum,
            "alternative_minimal_sets": [list(row) for row in minimal],
        }
    components = list(minimal[0])
    return {
        "status": "identified",
        "fault_status": "single_fault" if len(components) == 1 else "multi_fault",
        "responsible_components": components,
        "minimal_sufficient_set": components,
        "minimal_cardinality": len(components),
    }


def _component_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    rows = [str(item) for item in value]
    if len(rows) != len(set(rows)) or any(item not in COMPONENTS for item in rows):
        raise ValueError(f"{field} contains invalid or duplicate components")
    return [item for item in COMPONENTS if item in rows]


def validate_result(payload: Any, case_id: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("model result must be an object")
    result = dict(payload)
    components = _component_list(result.get("responsible_components"), "responsible_components")
    minimal = _component_list(result.get("minimal_sufficient_set"), "minimal_sufficient_set")
    expected_status = "no_fault" if not components else "single_fault" if len(components) == 1 else "multi_fault"
    if result.get("fault_status") != expected_status:
        raise ValueError("fault_status is inconsistent with responsible_components")
    primary = result.get("primary_component")
    if not components:
        if primary is not None:
            raise ValueError("primary_component must be null for no_fault")
    elif primary not in components:
        raise ValueError("primary_component must be one of responsible_components")
    scores = result.get("component_scores")
    if not isinstance(scores, dict) or set(scores) != set(COMPONENTS):
        raise ValueError("component_scores must contain all five components")
    result["case_id"] = case_id
    result["responsible_components"] = components
    result["minimal_sufficient_set"] = minimal
    confidence = result.get("confidence")
    result["confidence"] = max(0.0, min(1.0, float(confidence))) if isinstance(confidence, (int, float)) else 0.0
    return result


def run(
    *,
    stress_root: Path,
    credential_path: Path,
    run_version: str,
    max_workers: int,
    limit: int | None,
    max_tokens: int,
) -> dict[str, Any]:
    public = read_json(stress_root / "public" / "stress_cases.json")
    cases = public["cases"][:limit] if limit else public["cases"]
    safe_version = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_version)
    run_dir = stress_root / f"k3_{safe_version}_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    client = QiniuModelClient(api_key=load_api_key(credential_path))
    model = client.select_available_model()

    def process(case: dict[str, Any]) -> dict[str, Any]:
        path = run_dir / f"{case['case_id']}.json"
        if path.is_file():
            existing = read_json(path)
            validate_result(existing["model_result"], case["case_id"])
            return existing
        causal_verdict = derive_probe_verdict(case)
        if causal_verdict["status"] != "identified":
            raise ValueError(f"counterfactual probes are {causal_verdict['status']}")
        payload, metadata = client.chat_json(build_messages(case), max_tokens=max_tokens)
        model_result = validate_result(payload, case["case_id"])
        result = dict(model_result)
        result["model_responsible_components"] = model_result["responsible_components"]
        result["model_minimal_sufficient_set"] = model_result["minimal_sufficient_set"]
        result["model_fault_status"] = model_result["fault_status"]
        result["responsible_components"] = causal_verdict["responsible_components"]
        result["minimal_sufficient_set"] = causal_verdict["minimal_sufficient_set"]
        result["fault_status"] = causal_verdict["fault_status"]
        result["primary_component"] = (
            causal_verdict["responsible_components"][0]
            if causal_verdict["responsible_components"]
            else None
        )
        result["model_agrees_with_causal_verdict"] = (
            model_result["responsible_components"] == causal_verdict["responsible_components"]
            and model_result["minimal_sufficient_set"] == causal_verdict["minimal_sufficient_set"]
        )
        wrapper = {
            "schema_version": "care-ie.stress_model_run.v2",
            "case_id": case["case_id"],
            "model": model,
            "result": result,
            "model_result": model_result,
            "causal_verdict": causal_verdict,
            "api_metadata": metadata,
            "expected_answers_exposed": False,
            "evaluator_gold_loaded": False,
        }
        with WRITE_LOCK:
            path.write_text(json.dumps(wrapper, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return wrapper

    completed = []
    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process, case): case for case in cases}
        for future in as_completed(futures):
            case = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:
                failures.append({"case_id": case["case_id"], "error_type": type(exc).__name__, "message": str(exc)})
    completed.sort(key=lambda row: row["case_id"])
    summary = {
        "schema_version": "care-ie.stress_model_summary.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "run_version": safe_version,
        "case_count": len(cases),
        "completed_count": len(completed),
        "failure_count": len(failures),
        "failures": failures,
        "expected_answers_exposed": False,
        "evaluator_gold_loaded": False,
        "results": [row["result"] for row in completed],
        "model_results": [row["model_result"] for row in completed],
        "causal_verdicts": [row["causal_verdict"] for row in completed],
        "api_usage": [row["api_metadata"] for row in completed],
    }
    output = stress_root / f"K3_{safe_version.upper()}_ATTRIBUTIONS.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress-root", required=True)
    parser.add_argument("--credential-path", required=True)
    parser.add_argument("--run-version", default="stress_v1")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=384000)
    args = parser.parse_args()
    summary = run(
        stress_root=Path(args.stress_root).resolve(),
        credential_path=Path(args.credential_path),
        run_version=args.run_version,
        max_workers=args.max_workers,
        limit=args.limit,
        max_tokens=args.max_tokens,
    )
    print(json.dumps({key: summary[key] for key in ("model", "case_count", "completed_count", "failure_count")}, ensure_ascii=False))
    return 0 if summary["failure_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
