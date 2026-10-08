"""Run answer-free Kimi K3 attribution over CARE-IE pilot cases."""

from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qiniu_model_client import QiniuModelClient, load_api_key


COMPONENTS = ("schema", "evidence", "extraction", "binding", "normalization")
WRITE_LOCK = threading.Lock()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_source_lines(path: Path) -> dict[int, str]:
    inline = re.compile(r"^SOURCE_LINE\s+(\d+)\s*:\s*(.*)$")
    marker = re.compile(r"^<!--\s*SOURCE_LINE\s*:\s*(\d+)\s*-->$")
    result = {}
    pending: int | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = inline.match(raw)
        if match:
            result[int(match.group(1))] = match.group(2).strip()
            pending = None
            continue
        match = marker.match(raw)
        if match:
            pending = int(match.group(1))
            continue
        if pending is not None:
            result[pending] = raw.strip()
            pending = None
    return result


def concept_terms(concept: str) -> list[str]:
    aliases = {
        "transition_temperature": ["tc", "transition", "superconduct"],
        "upper_critical_field": ["hc2", "critical field"],
        "lower_critical_field": ["hc1", "critical field"],
        "critical_current": ["critical current", "ic"],
        "critical_current_density": ["current density", "jc"],
        "penetration_depth": ["penetration depth", "lambda"],
        "coherence_length": ["coherence length", "xi"],
        "pairing_symmetry": ["pairing", "wave", "gap"],
        "superconducting_gap": ["gap", "delta"],
        "resistance_temperature_curve": ["resistance", "temperature", "r-t"],
        "resistance_field_curve": ["resistance", "field", "r-h"],
        "current_voltage_curve": ["current", "voltage", "i-v"],
    }
    return aliases.get(concept, concept.replace("_", " ").split())


def evidence_excerpt(case: dict[str, Any], lines: dict[int, str], *, max_lines: int = 24) -> list[dict[str, Any]]:
    baseline = case["baseline_record"]
    baseline_line = int(baseline.get("evidence_line") or 0)
    selected = set(range(max(1, baseline_line - 3), baseline_line + 4))
    target = case["target_record"]
    terms = [
        term.casefold()
        for term in [str(target.get("record_key") or ""), *concept_terms(str(target.get("concept_id") or ""))]
        if len(term.strip()) >= 2
    ]
    ranked = []
    for number, text in lines.items():
        folded = text.casefold()
        score = sum(3 if term == str(target.get("record_key") or "").casefold() else 1 for term in terms if term in folded)
        if score:
            ranked.append((-score, abs(number - baseline_line), number))
    for _, _, number in sorted(ranked):
        selected.add(number)
        if len(selected) >= max_lines:
            break
    return [
        {
            "source_line": number,
            "text": lines[number][:3600 if number == baseline_line else 900],
        }
        for number in sorted(selected)
        if number in lines
    ][:max_lines]


def build_prompt(
    case: dict[str, Any], mismatch_fields: list[str], excerpt: list[dict[str, Any]], mode: str
) -> list[dict[str, str]]:
    definitions = {
        "schema": "The target contract lacks the field, type, multiplicity, or required qualifier needed to represent the answer.",
        "evidence": "The direct source evidence was unavailable because of OCR, figure alignment, retrieval, or context truncation.",
        "extraction": "The schema and evidence are available, but the system selected or interpreted the wrong value, qualifier, or supporting line.",
        "binding": "A correct fact was attached to the wrong material, sample, process, measurement, or result record.",
        "normalization": "The scientific content is correct but its unit, scale, formula, symbol, identifier, or controlled wording is non-canonical.",
    }
    shared = {
        "case_id": case["case_id"],
        "target_record": case["target_record"],
        "baseline_record": case["baseline_record"],
        "answer_free_failure_signal": {"mismatch_fields": mismatch_fields},
        "evaluation_protocol_facts": {
            "target_contract_complete": True,
            "record_key_forced_by_case": True,
            "full_line_numbered_evidence_pack_available": True,
        },
        "retrieved_evidence_excerpt": excerpt,
        "component_definitions": definitions,
    }
    if mode == "supervisor":
        instruction = (
            "Choose the single most likely responsible component. Use only the supplied baseline, "
            "answer-free mismatch field names, and source excerpts. Do not invent the expected answer. "
            "The failure signal is diagnostic: if evidence_line is mismatched, normalization alone "
            "cannot repair the record because normalization does not change evidence provenance. "
            "Binding requires explicit evidence that the fact belongs to a different entity or condition. "
            "Prefer an explicit subject-value-condition statement over approximate values or ordering in "
            "nearby narrative, and do not re-pair values merely because a paragraph mentions multiple cases. "
            "For superconducting critical fields, do not confuse an approximate applied field that nearly "
            "suppresses a transition with a fitted or extrapolated Hc2(0); they are distinct quantities. "
            "If a baseline value is visibly identical to the direct source value even though value is flagged "
            "as mismatched, treat the mismatch as JSON type or canonical-form normalization unless another "
            "scientific qualifier or the evidence provenance is substantively wrong. A sentence elsewhere "
            "in the paper that repeats the baseline wording does not prove the wording answers this target. "
            "Return JSON with case_id, primary_component, confidence from 0 to 1, rationale, and recommended_action. "
            "Keep rationale and recommended_action under 50 words each."
        )
    else:
        instruction = (
            "Perform CARE-IE counterfactual responsibility analysis. For every component, decide whether "
            "changing only that component while holding the other four fixed is likely to make the complete "
            "record correct. A wrong evidence line selected despite available evidence is extraction, not "
            "retrieval. Surface-equivalent units, symbols, scales, formulas, identifiers, or controlled labels "
            "are normalization. Binding must be supported by an explicit entity/condition association conflict; "
            "nearby multi-case narrative is not enough to infer one. If the baseline value is visibly identical to direct evidence, a value mismatch "
            "For critical-field records, distinguish a transition-suppression field from a fitted or extrapolated Hc2(0). "
            "signal usually means JSON type/canonicalization rather than extraction, but only when evidence "
            "provenance is not also mismatched. Return JSON with case_id, "
            "assessments (an object containing all five component "
            "keys, each with predicted_flip boolean, effect 0..1, rationale), primary_component, confidence, "
            "minimal_sufficient_set, and recommended_action. Do not invent or output an expected answer."
        )
    return [
        {
            "role": "system",
            "content": "You are a rigorous scientific information-extraction failure analyst. Return only valid JSON without markdown.",
        },
        {
            "role": "user",
            "content": instruction + "\n\nINPUT:\n" + json.dumps(shared, ensure_ascii=False),
        },
    ]


def validate_model_result(payload: Any, case_id: str, mode: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("model result must be an object")
    component = str(payload.get("primary_component") or "")
    if component not in COMPONENTS:
        raise ValueError(f"invalid primary_component: {component}")
    result = dict(payload)
    result["case_id"] = case_id
    result["primary_component"] = component
    confidence = result.get("confidence")
    if not isinstance(confidence, (int, float)):
        result["confidence"] = 0.0
    else:
        result["confidence"] = max(0.0, min(1.0, float(confidence)))
    if mode == "counterfactual":
        assessments = result.get("assessments")
        if not isinstance(assessments, dict) or set(assessments) != set(COMPONENTS):
            raise ValueError("counterfactual result must assess all five components")
    return result


def normalize_surface(value: Any) -> str:
    return re.sub(r"[^a-z0-9.+-]+", "", str(value).casefold())


def baseline_value_on_recorded_evidence(
    case: dict[str, Any], excerpt: list[dict[str, Any]]
) -> bool:
    baseline = case["baseline_record"]
    line_number = baseline.get("evidence_line")
    value = normalize_surface(baseline.get("value"))
    if not value or line_number is None:
        return False
    for item in excerpt:
        if item.get("source_line") == line_number:
            return value in normalize_surface(item.get("text") or "")
    return False


def apply_answer_free_guards(
    result: dict[str, Any],
    mismatch_fields: list[str],
    case: dict[str, Any] | None = None,
    excerpt: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    guarded = dict(result)
    original = guarded["primary_component"]
    if original == "normalization" and "evidence_line" in mismatch_fields:
        guarded["model_primary_component"] = original
        guarded["primary_component"] = "extraction"
        guarded["protocol_guard"] = {
            "rule": "normalization_cannot_repair_evidence_provenance",
            "inputs": ["mismatch_fields:evidence_line"],
        }
    elif (
        original in {"evidence", "binding"}
        and "evidence_line" not in mismatch_fields
        and set(mismatch_fields).issubset({"value", "unit", "qualifiers"})
        and case is not None
        and baseline_value_on_recorded_evidence(case, excerpt or [])
    ):
        guarded["model_primary_component"] = original
        guarded["primary_component"] = "normalization"
        guarded["protocol_guard"] = {
            "rule": "recorded_evidence_directly_supports_baseline_surface_value",
            "inputs": [
                "full_evidence_pack_available",
                "recorded_evidence_line_not_mismatched",
                "baseline_value_present_on_recorded_evidence_line",
            ],
        }
    return guarded


def run_pilot(
    run_root: Path,
    eval_root: Path,
    credential_path: Path,
    mode: str,
    run_version: str,
    max_workers: int,
    limit: int | None,
    max_tokens: int,
) -> dict[str, Any]:
    public = read_json(eval_root / "public" / "pilot_cases.json")
    signals_payload = read_json(eval_root / "public" / "pilot_failure_signals.json")
    signals = {item["case_id"]: item["mismatch_fields"] for item in signals_payload["signals"]}
    cases = public["cases"][:limit] if limit else public["cases"]
    pack_dir = run_root / "optimization" / "step9_benchmark_packs"
    safe_version = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_version)
    run_dir = eval_root / f"k3_{mode}_{safe_version}_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    api_key = load_api_key(credential_path)
    client = QiniuModelClient(api_key=api_key)
    selected_model = client.select_available_model()

    def process(case: dict[str, Any]) -> dict[str, Any]:
        output_path = run_dir / f"{case['case_id']}.json"
        if output_path.is_file():
            existing = read_json(output_path)
            validate_model_result(existing["result"], case["case_id"], mode)
            return existing
        lines = parse_source_lines(pack_dir / f"{case['paper_id']}.md")
        excerpt = evidence_excerpt(case, lines)
        payload, metadata = client.chat_json(
            build_prompt(case, signals[case["case_id"]], excerpt, mode),
            max_tokens=max_tokens,
        )
        result = validate_model_result(payload, case["case_id"], mode)
        result = apply_answer_free_guards(
            result, signals[case["case_id"]], case, excerpt
        )
        wrapper = {
            "schema_version": "care-ie.model_attribution_run.v1",
            "mode": mode,
            "case_id": case["case_id"],
            "model": selected_model,
            "result": result,
            "api_metadata": metadata,
            "expected_answer_exposed": False,
            "evaluator_gold_loaded": False,
        }
        with WRITE_LOCK:
            output_path.write_text(
                json.dumps(wrapper, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
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
                failures.append(
                    {"case_id": case["case_id"], "error_type": type(exc).__name__, "message": str(exc)}
                )
    completed.sort(key=lambda item: item["case_id"])
    summary = {
        "schema_version": "care-ie.model_attribution_summary.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "model": selected_model,
        "case_count": len(cases),
        "completed_count": len(completed),
        "failure_count": len(failures),
        "initial_max_tokens": max_tokens,
        "failures": failures,
        "expected_answers_exposed": False,
        "evaluator_gold_loaded": False,
        "results": [item["result"] for item in completed],
        "api_usage": [item["api_metadata"] for item in completed],
    }
    output = eval_root / f"K3_{mode.upper()}_{safe_version.upper()}_ATTRIBUTIONS.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--credential-path", required=True)
    parser.add_argument("--eval-root")
    parser.add_argument("--mode", choices=("supervisor", "counterfactual"), required=True)
    parser.add_argument("--run-version", default="v1")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=384000)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    eval_root = (
        Path(args.eval_root).resolve()
        if args.eval_root
        else run_root / "optimization" / "counterfactual_eval_v1"
    )
    summary = run_pilot(
        run_root,
        eval_root,
        Path(args.credential_path),
        args.mode,
        args.run_version,
        args.max_workers,
        args.limit,
        args.max_tokens,
    )
    print(
        json.dumps(
            {key: summary[key] for key in ("mode", "model", "case_count", "completed_count", "failure_count")},
            ensure_ascii=False,
        )
    )
    return 0 if summary["failure_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
