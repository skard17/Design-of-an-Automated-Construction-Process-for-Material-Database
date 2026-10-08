import hashlib
import json
from copy import deepcopy
from typing import Any


PROTOCOL_VERSION = "materials-db-agent-protocol-v1"
TASK_CONTRACT_VERSION = "materials-literature-v1"

_TASK_CONTRACT = {
    "task_type": "automated_materials_database_construction",
    "source_scope": "scientific_literature_only",
    "source_artifact": "parsed_literature_markdown",
    "metadata_artifact": "download_time_bibliographic_metadata",
    "record_scope": "queryable material, sample, process, measurement, and mechanism records",
    "excluded_source_classes": [
        "laboratory_notebook",
        "textbook",
        "lecture_note",
        "general_web_text",
        "arbitrary_scientific_text",
    ],
    "scope_version": TASK_CONTRACT_VERSION,
}


def materials_literature_task_contract() -> dict[str, Any]:
    return deepcopy(_TASK_CONTRACT)


def stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest().upper()


def make_message(
    *,
    sender: str,
    receiver: str,
    phase: str,
    status: str,
    task_contract: dict[str, Any] | None = None,
    payload_refs: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
    requested_actions: list[Any] | None = None,
    produced_artifacts: list[Any] | None = None,
    next_route: str = "",
    evidence: list[Any] | None = None,
) -> dict[str, Any]:
    contract = deepcopy(task_contract or materials_literature_task_contract())
    body = {
        "protocol_version": PROTOCOL_VERSION,
        "task_contract": contract,
        "sender": str(sender),
        "receiver": str(receiver),
        "phase": str(phase),
        "status": str(status),
        "payload_refs": deepcopy(payload_refs or {}),
        "decision": deepcopy(decision or {}),
        "requested_actions": deepcopy(requested_actions or []),
        "produced_artifacts": deepcopy(produced_artifacts or []),
        "next_route": str(next_route or ""),
        "evidence": deepcopy(evidence or []),
    }
    body["message_id"] = "message:" + stable_digest(body).split(":", 1)[1][:24]
    return body


def validate_message(message: Any) -> list[str]:
    if not isinstance(message, dict):
        return ["protocol message must be an object"]
    errors = []
    if message.get("protocol_version") != PROTOCOL_VERSION:
        errors.append(f"protocol_version must be {PROTOCOL_VERSION}")
    for key in ("message_id", "sender", "receiver", "phase", "status"):
        if not str(message.get(key) or "").strip():
            errors.append(f"{key} is required")
    contract = message.get("task_contract") or {}
    if contract.get("task_type") != _TASK_CONTRACT["task_type"]:
        errors.append("task_contract.task_type is outside the materials-database task")
    if contract.get("source_scope") != _TASK_CONTRACT["source_scope"]:
        errors.append("task_contract.source_scope must be scientific_literature_only")
    for key, expected_type in (
        ("payload_refs", dict),
        ("decision", dict),
        ("requested_actions", list),
        ("produced_artifacts", list),
        ("evidence", list),
    ):
        if not isinstance(message.get(key), expected_type):
            errors.append(f"{key} must be {expected_type.__name__}")
    return errors


def validate_message_list(messages: Any) -> dict[str, Any]:
    if not isinstance(messages, list):
        return {"valid": False, "message_count": 0, "errors": ["protocol_messages must be a list"]}
    errors = []
    for index, message in enumerate(messages):
        errors.extend(f"message[{index}]: {error}" for error in validate_message(message))
    return {"valid": not errors, "message_count": len(messages), "errors": errors}
