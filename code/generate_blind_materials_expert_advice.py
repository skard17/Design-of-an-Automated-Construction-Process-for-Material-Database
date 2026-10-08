"""Generate blind expert advice for a materials-literature database run.

Only the task contract and the current version's optional artifact are exposed to
the expert. Evaluator-only gold references are deliberately never loaded into the
prompt payload.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import materials_agent_protocol
import run_artifact_guard
import section_design_agent


ALLOWED_TASK_KEYS = (
    "objective",
    "source_scope",
    "record_scope",
    "discipline",
    "query_requirements",
)
PIPELINE = "blind_materials_expert_advice_fresh"


def load_blind_task_contract(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    task = payload.get("task") or {}
    return {key: task[key] for key in ALLOWED_TASK_KEYS if key in task}


def build_prompt(
    task: dict,
    round_number: int,
    current_artifact: dict | None = None,
    structured_protocol: bool = False,
) -> str:
    artifact_block = "No current-version artifact is available in this round."
    if current_artifact is not None:
        artifact_block = json.dumps(current_artifact, ensure_ascii=False, indent=2)

    protocol_instruction = ""
    if structured_protocol:
        protocol_instruction = """
This version uses the structured materials-agent protocol. Also return `schema_concepts`, an
array of atomic cross-domain database concepts justified by the visible task contract. Each item
must contain concept_id, label, entity_id, owner_key, object_kind, required,
condition_requirements, and evidence_types. Keep these concepts generic enough for materials
science; do not invent a superconductivity field catalog. Domain-specific quantities must still
be derived by the downstream agents from the current literature corpus.
There is no target concept count. Keep the concept contract compact: each concept must add a
distinct query, entity-binding, evidence, or provenance capability; merge aliases, unit variants,
and semantically duplicate concepts.
"""

    top_level_keys = (
        "round, verdict, blocking_issues, recommendations, schema_concepts, "
        "forbidden_information_used"
        if structured_protocol
        else "round, verdict, blocking_issues, recommendations, forbidden_information_used"
    )

    return f"""Act as GPT-5.6 Sol, the blind expert reviewer for an automated materials-database
construction workflow whose input is scientific literature Markdown.

You know only the task contract below and, when supplied, the current version artifact. You do
not know any manually authored target schema, core-field mapping, extraction gold, evaluation
score, or output from another version. Never guess or reconstruct such hidden references.

Task contract:
{json.dumps(task, ensure_ascii=False, indent=2)}

Current-version artifact:
{artifact_block}

This is expert round {round_number} of at most three. Give task-level, evidence-backed design
advice that remains valid for materials-database construction in general while specializing the
actual field inventory from this task and its literature. Focus on entity boundaries, sample and
condition binding, atomic executable fields, cardinality, missingness, uncertainty, evidence and
provenance, experimental-versus-theoretical ownership, figure/table ownership, and acceptance
checks. Do not provide a copied field catalog and do not optimize against hidden metrics.
{protocol_instruction}

Return one JSON object with exactly these top-level keys: {top_level_keys}.
- round: integer
- verdict: "accept" or "revision_required"
- blocking_issues: array of concise strings
- recommendations: array of objects with id, priority, requirement, rationale, acceptance_check
- forbidden_information_used: false

Use verdict="accept" only if there is no blocking task-level issue. Otherwise provide concrete,
testable recommendations. Output JSON only."""


def build_structured_protocol_advice(advice: dict, task: dict) -> dict:
    payload = deepcopy(advice)
    payload["structured_protocol_enabled"] = True
    payload.setdefault("schema_concepts", [])
    message = materials_agent_protocol.make_message(
        sender="blind_materials_expert",
        receiver="step8_supervisor",
        phase=f"expert_round_{payload.get('round', 1)}",
        status="accepted" if payload.get("verdict") == "accept" else "revision_required",
        payload_refs={
            "task_contract": "blind_task_contract",
            "schema_concepts": "schema_concepts",
            "recommendations": "recommendations",
        },
        decision={
            "verdict": payload.get("verdict"),
            "forbidden_information_used": payload.get("forbidden_information_used"),
        },
        requested_actions=deepcopy(payload.get("recommendations") or []),
        produced_artifacts=[
            {
                "ref": "schema_concepts",
                "count": len(payload.get("schema_concepts") or []),
            }
        ],
        next_route="step8_prepare" if payload.get("verdict") == "accept" else "step8_supervisor",
        evidence=deepcopy(payload.get("blocking_issues") or []),
    )
    payload["protocol_messages"] = [message]
    payload["protocol_validation"] = materials_agent_protocol.validate_message_list([message])
    payload["blind_task_contract"] = deepcopy(task)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--round", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--current-artifact", default="")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--llm-backend", choices=("openai", "langchain", "qiniu"), default="openai")
    parser.add_argument("--request-timeout", type=float, default=1200)
    parser.add_argument("--structured-protocol", action="store_true")
    args = parser.parse_args()

    task = load_blind_task_contract(args.task_spec)
    current_artifact = None
    if args.current_artifact:
        current_artifact = json.loads(Path(args.current_artifact).read_text(encoding="utf-8"))
    output = Path(args.output).resolve()
    input_paths = [args.task_spec]
    if args.current_artifact:
        input_paths.append(args.current_artifact)
    input_identity = run_artifact_guard.build_input_identity(
        PIPELINE,
        {
            "round": args.round,
            "model": args.model,
            "base_url": args.base_url,
            "llm_backend": args.llm_backend,
            "request_timeout": args.request_timeout,
            "structured_protocol": bool(args.structured_protocol),
            "expert_gold_visibility": "none",
            "maximum_expert_rounds": 3,
        },
        input_paths,
    )
    run_identity = run_artifact_guard.reserve_fresh_run(
        pipeline=PIPELINE,
        output_path=output,
        input_identity=input_identity,
        artifact_paths=[output],
    )
    try:
        client = section_design_agent.get_client(
            base_url=args.base_url,
            api_key=args.api_key,
            backend=args.llm_backend,
            timeout=args.request_timeout,
        )
        raw = section_design_agent.chat(
            client,
            args.model,
            build_prompt(task, args.round, current_artifact, args.structured_protocol),
            system_prompt=(
                "You are a blind expert reviewer for automated materials-database construction from "
                "scientific literature. Never use or infer evaluator-only gold information."
            ),
            temperature=0,
            max_tokens=384000,
        )
        parsed = section_design_agent.parse_json_response(raw)
        if parsed.get("forbidden_information_used") is not False:
            raise ValueError(
                "blind expert response did not explicitly set forbidden_information_used=false"
            )
        if parsed.get("round") != args.round:
            raise ValueError("blind expert response round does not match requested round")
        if parsed.get("verdict") not in {"accept", "revision_required"}:
            raise ValueError("blind expert response has an invalid verdict")
        if args.structured_protocol:
            parsed = build_structured_protocol_advice(parsed, task)
            if not parsed["protocol_validation"]["valid"]:
                raise ValueError("blind expert protocol message is invalid")
        parsed["run_identity"] = run_identity
        run_artifact_guard.atomic_write_json(
            output,
            parsed,
            run_identity=run_identity,
        )
        run_artifact_guard.update_run_status(
            run_identity,
            "completed",
            output_status="success",
            expert_round=args.round,
            verdict=parsed["verdict"],
        )
    except BaseException as exc:
        run_artifact_guard.update_run_status(
            run_identity,
            "interrupted",
            error_type=type(exc).__name__,
            error_message=str(exc)[:1000],
        )
        raise
    print(json.dumps({"output": str(output), "round": args.round, "verdict": parsed["verdict"]}))


if __name__ == "__main__":
    main()
