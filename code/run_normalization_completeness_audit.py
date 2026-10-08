"""Answer-free K3 audit for normalization versus incomplete extraction."""

from __future__ import annotations

import argparse
import copy
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qiniu_model_client import QiniuModelClient, load_api_key
from run_care_attribution_pilot import evidence_excerpt, parse_source_lines


WRITE_LOCK = threading.Lock()
ALLOWED = {"extraction", "normalization"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_messages(
    case: dict[str, Any], mismatch_fields: list[str], excerpt: list[dict[str, Any]], initial: dict[str, Any]
) -> list[dict[str, str]]:
    instruction = (
        "Audit whether the initial normalization attribution is scientifically complete. Choose normalization only "
        "when the baseline preserves the complete target-relevant proposition and differs solely by type, notation, "
        "unit representation, formula spelling, identifier formatting, or controlled synonymous wording. Choose extraction "
        "when it omits a target-relevant method/software identifier, measurement range or bound, magnitude, condition, "
        "or qualifier; adds an unsupported qualifier; or broadens/narrows the scientific claim. Exact overlap of a short "
        "phrase is not sufficient when the direct evidence contains required detail. Follow target_contract_policy: "
        "approximation or relation markers remain normalization unless an explicit requested qualifier key requires them. "
        "Return JSON with case_id, decision "
        "(extraction or normalization), confidence 0..1, checks containing complete_value, complete_qualifiers, "
        "surface_equivalent booleans, rationale, and recommended_action. Do not invent or output an expected answer."
    )
    payload = {
        "case_id": case["case_id"],
        "target_record": case["target_record"],
        "baseline_record": case["baseline_record"],
        "answer_free_failure_signal": {"mismatch_fields": mismatch_fields},
        "evidence_excerpt": excerpt,
        "initial_attribution": {
            "primary_component": initial["primary_component"],
            "confidence": initial.get("confidence"),
            "rationale": initial.get("rationale"),
        },
        "target_contract_policy": {
            "approximation_and_relation_markers": "normalization_only_unless_explicitly_requested_as_a_qualifier_key",
            "omitted_software_range_condition_or_supported_qualifier": "extraction",
        },
    }
    return [
        {
            "role": "system",
            "content": "You are a strict scientific record-completeness auditor. Return only valid JSON without markdown.",
        },
        {"role": "user", "content": instruction + "\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False)},
    ]


def validate_audit(payload: Any, case_id: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("audit result must be an object")
    decision = str(payload.get("decision") or "")
    if decision not in ALLOWED:
        raise ValueError(f"invalid completeness audit decision: {decision}")
    checks = payload.get("checks")
    required = {"complete_value", "complete_qualifiers", "surface_equivalent"}
    if not isinstance(checks, dict) or not required.issubset(checks):
        raise ValueError("audit checks are incomplete")
    result = dict(payload)
    result["case_id"] = case_id
    result["decision"] = decision
    confidence = result.get("confidence")
    result["confidence"] = max(0.0, min(1.0, float(confidence))) if isinstance(confidence, (int, float)) else 0.0
    return result


def run(
    run_root: Path,
    eval_root: Path,
    credential_path: Path,
    run_version: str,
    max_workers: int,
    max_tokens: int,
) -> dict[str, Any]:
    public = read_json(eval_root / "public" / "pilot_cases.json")
    cases = {item["case_id"]: item for item in public["cases"]}
    signals_payload = read_json(eval_root / "public" / "pilot_failure_signals.json")
    signals = {item["case_id"]: item["mismatch_fields"] for item in signals_payload["signals"]}
    initial_payload = read_json(eval_root / "K3_SUPERVISOR_FROZEN_V1_ATTRIBUTIONS.json")
    initial = {item["case_id"]: item for item in initial_payload["results"]}
    audit_ids = sorted(case_id for case_id, item in initial.items() if item["primary_component"] == "normalization")
    pack_dir = run_root / "optimization" / "step9_benchmark_packs"
    safe_version = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_version)
    run_dir = eval_root / f"k3_normalization_completeness_audit_{safe_version}_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    client = QiniuModelClient(api_key=load_api_key(credential_path))
    model = client.select_available_model()

    def process(case_id: str) -> dict[str, Any]:
        path = run_dir / f"{case_id}.json"
        if path.is_file():
            existing = read_json(path)
            validate_audit(existing["audit"], case_id)
            return existing
        case = cases[case_id]
        lines = parse_source_lines(pack_dir / f"{case['paper_id']}.md")
        excerpt = evidence_excerpt(case, lines)
        payload, metadata = client.chat_json(
            build_messages(case, signals[case_id], excerpt, initial[case_id]),
            max_tokens=max_tokens,
        )
        audit = validate_audit(payload, case_id)
        wrapper = {
            "schema_version": "care-ie.normalization_completeness_audit.v1",
            "case_id": case_id,
            "model": model,
            "audit": audit,
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
        futures = {pool.submit(process, case_id): case_id for case_id in audit_ids}
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:
                failures.append({"case_id": case_id, "error_type": type(exc).__name__, "message": str(exc)})
    audits = {item["case_id"]: item for item in completed}
    final_results = []
    for case_id in sorted(initial):
        result = copy.deepcopy(initial[case_id])
        if case_id in audits:
            audit = audits[case_id]["audit"]
            result["pre_audit_primary_component"] = result["primary_component"]
            result["primary_component"] = audit["decision"]
            result["normalization_completeness_audit"] = audit
        final_results.append(result)
    summary = {
        "schema_version": "care-ie.completeness_audited_attribution_summary.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "supervisor_with_normalization_completeness_audit",
        "model": model,
        "run_version": safe_version,
        "case_count": len(final_results),
        "audit_case_count": len(audit_ids),
        "completed_count": len(completed),
        "failure_count": len(failures),
        "failures": failures,
        "expected_answers_exposed": False,
        "evaluator_gold_loaded": False,
        "results": final_results,
        "api_usage": [item["api_metadata"] for item in completed],
    }
    output = eval_root / f"K3_SUPERVISOR_COMPLETENESS_AUDITED_{safe_version.upper()}_ATTRIBUTIONS.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--eval-root")
    parser.add_argument("--credential-path", required=True)
    parser.add_argument("--run-version", default="v2")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=384000)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    eval_root = Path(args.eval_root).resolve() if args.eval_root else run_root / "optimization" / "counterfactual_eval_v1"
    summary = run(run_root, eval_root, Path(args.credential_path), args.run_version, args.max_workers, args.max_tokens)
    print(json.dumps({key: summary[key] for key in ("model", "case_count", "audit_case_count", "completed_count", "failure_count")}, ensure_ascii=False))
    return 0 if summary["failure_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
