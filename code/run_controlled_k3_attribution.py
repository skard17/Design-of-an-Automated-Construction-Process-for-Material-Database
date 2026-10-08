"""Run answer-free K3 responsibility localization on controlled stage traces."""

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
        "schema": "The schema stage emits the wrong concept or cannot represent the target contract.",
        "evidence": "The evidence stage selects the wrong source candidate before extraction.",
        "extraction": "Evidence is correct but the extraction stage emits an incorrect value, unit, or qualifier.",
        "binding": "The extracted fact is correct but the binding stage attaches it to the wrong record key or entity.",
        "normalization": "Pre-normalization content is correct but the normalization stage changes type, unit, notation, or canonical form incorrectly.",
    }
    instruction = (
        "Locate the single responsible component by tracing the first stage whose output becomes inconsistent "
        "with the target contract and source candidates. Later stages may faithfully propagate an earlier error; "
        "do not blame propagation. Assess all five components. Return JSON with case_id, primary_component, "
        "confidence (0..1), component_scores (all five keys, each with responsibility_probability 0..1 and rationale), "
        "minimal_sufficient_set, rationale, and recommended_intervention. Do not invent expected values."
    )
    payload = {
        "case": case,
        "component_definitions": definitions,
        "protocol": {
            "single_fault_assumption": True,
            "stage_outputs_are_chronological": True,
            "expected_answers_exposed": False,
        },
    }
    return [
        {
            "role": "system",
            "content": "You are a rigorous scientific pipeline failure analyst. Return only valid JSON without markdown.",
        },
        {"role": "user", "content": instruction + "\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False)},
    ]


def validate_result(payload: Any, case_id: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("model result must be an object")
    primary = str(payload.get("primary_component") or "")
    if primary not in COMPONENTS:
        raise ValueError(f"invalid primary_component: {primary}")
    scores = payload.get("component_scores")
    if not isinstance(scores, dict) or set(scores) != set(COMPONENTS):
        raise ValueError("component_scores must contain all five components")
    result = dict(payload)
    result["case_id"] = case_id
    result["primary_component"] = primary
    confidence = result.get("confidence")
    result["confidence"] = max(0.0, min(1.0, float(confidence))) if isinstance(confidence, (int, float)) else 0.0
    return result


def run(
    controlled_root: Path,
    credential_path: Path,
    run_version: str,
    max_workers: int,
    limit: int | None,
    max_tokens: int,
) -> dict[str, Any]:
    public = read_json(controlled_root / "public" / "controlled_cases.json")
    cases = public["cases"][:limit] if limit else public["cases"]
    safe_version = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_version)
    run_dir = controlled_root / f"k3_{safe_version}_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    client = QiniuModelClient(api_key=load_api_key(credential_path))
    model = client.select_available_model()

    def process(case: dict[str, Any]) -> dict[str, Any]:
        path = run_dir / f"{case['case_id']}.json"
        if path.is_file():
            existing = read_json(path)
            validate_result(existing["result"], case["case_id"])
            return existing
        payload, metadata = client.chat_json(build_messages(case), max_tokens=max_tokens)
        result = validate_result(payload, case["case_id"])
        wrapper = {
            "schema_version": "care-ie.controlled_model_run.v1",
            "case_id": case["case_id"],
            "model": model,
            "result": result,
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
    completed.sort(key=lambda item: item["case_id"])
    summary = {
        "schema_version": "care-ie.controlled_model_summary.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "run_version": safe_version,
        "case_count": len(cases),
        "completed_count": len(completed),
        "failure_count": len(failures),
        "failures": failures,
        "expected_answers_exposed": False,
        "evaluator_gold_loaded": False,
        "results": [item["result"] for item in completed],
        "api_usage": [item["api_metadata"] for item in completed],
    }
    output = controlled_root / f"K3_{safe_version.upper()}_ATTRIBUTIONS.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlled-root", required=True)
    parser.add_argument("--credential-path", required=True)
    parser.add_argument("--run-version", default="pilot_v1")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=384000)
    args = parser.parse_args()
    summary = run(
        Path(args.controlled_root).resolve(),
        Path(args.credential_path),
        args.run_version,
        args.max_workers,
        args.limit,
        args.max_tokens,
    )
    print(json.dumps({key: summary[key] for key in ("model", "case_count", "completed_count", "failure_count")}, ensure_ascii=False))
    return 0 if summary["failure_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
