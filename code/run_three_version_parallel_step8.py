"""Prepare and run three isolated Step8 benchmark variants in parallel.

The campaign copies the frozen literature corpus and executable workspace into
each version namespace.  Generation processes receive no manual schema, gold
records, prior checkpoints, or sibling-version artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import generate_blind_materials_expert_advice as expert_advice
import run_artifact_guard


PIPELINE = "superconductivity_three_version_parallel_step8"
VERSIONS = (
    {
        "version_id": "v1_baseline",
        "care": False,
        "structured_protocol": False,
        "human_gate": "plain_task_only",
    },
    {
        "version_id": "v2_care",
        "care": True,
        "structured_protocol": False,
        "human_gate": "plain_task_only",
    },
    {
        "version_id": "v3_care_protocol",
        "care": True,
        "structured_protocol": True,
        "human_gate": "structured_task_only",
    },
)
VERSION_BY_ID = {item["version_id"]: item for item in VERSIONS}
TASK = {
    "objective": (
        "Automatically construct an evidence-backed, queryable materials database "
        "from scientific literature Markdown for the supplied superconductivity task."
    ),
    "source_scope": "scientific_literature_markdown_only",
    "record_scope": (
        "materials, samples, preparation, measurements, properties, mechanisms, "
        "evidence, and provenance"
    ),
    "discipline": "Materials science and condensed matter physics",
    "query_requirements": (
        "Support reliable comparison and retrieval across material systems while "
        "preserving entity and sample identity, experimental or computational "
        "conditions, uncertainty, evidence, provenance, and explicit missingness. "
        "Derive the exact field inventory from the task and literature rather than "
        "from a supplied superconductivity schema."
    ),
}


def build_care_counterfactual_queries() -> list[dict[str, Any]]:
    """Return gold-blind, domain-general counterfactual schema probes."""
    return [
        {
            "query_id": "care_neighbor_distinction",
            "category": "semantic_separation",
            "query": (
                "If two neighboring scientific quantities share units or vocabulary but "
                "have different physical definitions, can they be queried independently?"
            ),
            "required_distinctions": [
                "physical definition",
                "measurement criterion",
                "scientific quantity identity",
            ],
            "failure_exposed": "umbrella fields that merge independently queryable quantities",
        },
        {
            "query_id": "care_entity_condition_swap",
            "category": "binding",
            "query": (
                "If the same value is reported for different samples or conditions, would "
                "swapping the owner or condition change the database record and query result?"
            ),
            "required_distinctions": [
                "entity or sample owner",
                "measurement conditions",
                "separate observation instances",
            ],
            "failure_exposed": "values detached from their sample, entity, or conditions",
        },
        {
            "query_id": "care_provenance_swap",
            "category": "provenance",
            "query": (
                "If a reported value changes from measured to fitted, calculated, simulated, "
                "assumed, or author-interpreted, can the database preserve that distinction?"
            ),
            "required_distinctions": [
                "source or derivation type",
                "method or model provenance",
                "claim attribution",
            ],
            "failure_exposed": "mixing measured observations with derived or interpreted claims",
        },
        {
            "query_id": "care_evidence_removal",
            "category": "evidence",
            "query": (
                "If supporting text, table, or figure evidence is removed or contradicted, can "
                "the value or inferred label be withheld or downgraded without changing unrelated records?"
            ),
            "required_distinctions": [
                "evidence locator",
                "support type",
                "confidence or unresolved status",
            ],
            "failure_exposed": "claims that survive without auditable literature evidence",
        },
        {
            "query_id": "care_normalization_alias",
            "category": "normalization",
            "query": (
                "If notation, spelling, symbol, or unit representation changes while scientific "
                "meaning stays fixed, is the value normalized without creating duplicate fields?"
            ),
            "required_distinctions": [
                "canonical value",
                "reported representation",
                "alias or unit normalization rule",
            ],
            "failure_exposed": "duplicate fields for aliases or loss of the reported representation",
        },
        {
            "query_id": "care_missingness_status",
            "category": "missingness",
            "query": (
                "If a field is not reported, not applicable, experimentally unresolved, or "
                "explicitly absent, can the database distinguish those states without inference?"
            ),
            "required_distinctions": [
                "not reported",
                "not applicable",
                "unresolved",
                "explicitly absent",
            ],
            "failure_exposed": "conflating null states or inventing unsupported values",
        },
    ]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def discover_corpus(corpus_dir: Path, expected_count: int) -> list[Path]:
    papers = sorted(path for path in corpus_dir.rglob("*.md") if path.is_file())
    if len(papers) != expected_count:
        raise ValueError(
            f"expected exactly {expected_count} Markdown papers, found {len(papers)} "
            f"under {corpus_dir}"
        )
    paper_ids = [path.stem for path in papers]
    if len(set(paper_ids)) != expected_count:
        raise ValueError("paper stems are not unique; a flat frozen corpus would collide")
    empty = [str(path) for path in papers if path.stat().st_size < 1024]
    if empty:
        raise ValueError(f"unusable Markdown files below 1 KiB: {empty}")
    return papers


def discover_metadata(metadata_dir: Path, expected_count: int) -> list[Path]:
    records = sorted(path for path in metadata_dir.rglob("*.json") if path.is_file())
    if len(records) != expected_count:
        raise ValueError(
            f"expected exactly {expected_count} metadata records, found {len(records)} "
            f"under {metadata_dir}"
        )
    return records


def copy_workspace(repo_root: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"workspace destination already exists: {destination}")
    ignore = shutil.ignore_patterns(
        "__pycache__",
        "*.pyc",
        ".pytest_cache",
        "_tmp*",
        "artifacts",
        "checkpoints",
        "logs",
        "outputs",
        "results",
        "runs",
        "*.log",
    )
    shutil.copytree(repo_root / "code", destination / "code", ignore=ignore)
    shutil.copytree(
        repo_root / "reference_pipelines" / "pure_extraction_pipeline",
        destination / "reference_pipelines" / "pure_extraction_pipeline",
        ignore=ignore,
    )
    fixture_dir = destination / "tests" / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(
        repo_root / "tests" / "fixtures" / "no_reference_field_catalog.txt",
        fixture_dir / "no_reference_field_catalog.txt",
    )


def select_versions(version_ids: list[str] | None) -> tuple[dict[str, Any], ...]:
    requested = list(version_ids or VERSION_BY_ID)
    if len(requested) != len(set(requested)):
        raise ValueError("version selection contains duplicates")
    unknown = [version_id for version_id in requested if version_id not in VERSION_BY_ID]
    if unknown:
        raise ValueError(f"unknown version ids: {unknown}")
    if not requested:
        raise ValueError("at least one version must be selected")
    return tuple(VERSION_BY_ID[version_id] for version_id in requested)


def copy_inputs(
    papers: list[Path], metadata: list[Path], destination: Path
) -> tuple[list[Path], list[Path]]:
    corpus_out = destination / "corpus"
    metadata_out = destination / "metadata"
    corpus_out.mkdir(parents=True, exist_ok=False)
    metadata_out.mkdir(parents=True, exist_ok=False)
    copied_papers = []
    for source in papers:
        target = corpus_out / f"{source.stem}.md"
        shutil.copy2(source, target)
        copied_papers.append(target)
    copied_metadata = []
    for source in metadata:
        target = metadata_out / source.name
        shutil.copy2(source, target)
        copied_metadata.append(target)
    return copied_papers, copied_metadata


def corpus_inventory(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "paper_id": path.stem,
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    ]


def blind_advice_payload(structured: bool, care: bool = False) -> dict[str, Any]:
    advice = {
        "round": 1,
        "verdict": "revision_required",
        "blocking_issues": [
            (
                "The generated schema must demonstrate one independent database use for "
                "every retained leaf and must reject aliases, unit variants, deterministic "
                "derivatives, and mechanical repetitions of shared contracts."
            ),
            (
                "Every process, measurement, property, and interpretation must bind to its "
                "owning entity or sample and to the conditions under which it is valid."
            ),
            (
                "Evidence, provenance, uncertainty, and null versus not-reported versus "
                "not-applicable semantics must survive extraction and supervisor review."
            ),
        ],
        "recommendations": [
            {
                "id": "entity-boundaries",
                "priority": "blocking",
                "requirement": (
                    "Define explicit document, material, sample, process, measurement, "
                    "property-observation, and interpretation ownership boundaries."
                ),
                "rationale": "Values from distinct entities or conditions cannot share a record.",
                "acceptance_check": (
                    "Every non-document leaf declares an owner and repeated cardinality where needed."
                ),
            },
            {
                "id": "atomic-query-fields",
                "priority": "blocking",
                "requirement": (
                    "Keep independently queried values, units, criteria, directions, methods, "
                    "conditions, and uncertainty explicit while using reusable object contracts "
                    "for shared mechanics."
                ),
                "rationale": "Umbrella fields lose query semantics; Cartesian expansion adds noise.",
                "acceptance_check": (
                    "The critic can justify each leaf and finds no umbrella compression or repeated boilerplate."
                ),
            },
            {
                "id": "evidence-round-trip",
                "priority": "blocking",
                "requirement": (
                    "Attach source document, evidence text and location, attribution, and "
                    "condition context to every extracted value without inferring absent values."
                ),
                "rationale": "Database records must remain auditable against literature evidence.",
                "acceptance_check": (
                    "A supervisor can round-trip every value to explicit text, table, or figure evidence."
                ),
            },
            {
                "id": "scope-and-stopping-rule",
                "priority": "high",
                "requirement": (
                    "Derive domain quantities from the supplied task and corpus, and stop only "
                    "when remaining candidates are unsupported, redundant, aliases, or derivatives."
                ),
                "rationale": "A fixed field count is neither a completeness nor quality criterion.",
                "acceptance_check": (
                    "No numeric field target is used and omitted candidates have explicit reasons."
                ),
            },
        ],
        "forbidden_information_used": False,
        "blind_task_contract": dict(TASK),
        "expert_provenance": {
            "role": "GPT-5.6 Sol blind expert",
            "visible_information": ["task objective", "source scope", "record scope", "query requirements"],
            "hidden_information": [
                "manual superconductivity schema",
                "core-field mapping",
                "extraction gold",
                "prior or sibling version outputs",
            ],
            "maximum_rounds": 3,
        },
        "human_gate_policy": {
            "progress_owner": "in_graph_supervisor",
            "round_1_role": "raise a small set of material task-level issues before design",
            "later_round_default": (
                "accept once the first-round issues are repaired and the internal supervisor "
                "and critic have accepted the schema"
            ),
            "new_rejection_threshold": (
                "only a newly discovered severe entity-binding error, unsupported evidence "
                "policy, or structurally non-executable schema"
            ),
            "maximum_rounds": 3,
        },
    }
    if care:
        advice["care_counterfactual_enabled"] = True
        advice["counterfactual_queries"] = build_care_counterfactual_queries()
        advice["care_component_contract"] = {
            "gold_visibility": "none",
            "task_scope": "materials_database_from_scientific_literature",
            "requirement": (
                "Treat every counterfactual query as an independently traceable design "
                "requirement; do not satisfy it only with a broad umbrella field."
            ),
        }
    if not structured:
        return advice

    advice["schema_concepts"] = [
        {
            "concept_id": "document_provenance",
            "label": "document provenance",
            "entity_id": "document",
            "owner_key": "paper_info",
            "object_kind": "provenance",
            "required": True,
            "condition_requirements": [],
            "evidence_types": ["metadata", "text"],
        },
        {
            "concept_id": "material_identity",
            "label": "material identity and composition",
            "entity_id": "material",
            "owner_key": "material_info",
            "object_kind": "entity",
            "required": True,
            "condition_requirements": [],
            "evidence_types": ["text", "table", "figure"],
        },
        {
            "concept_id": "sample_identity",
            "label": "sample identity and variant binding",
            "entity_id": "sample",
            "owner_key": "samples",
            "object_kind": "entity",
            "required": True,
            "condition_requirements": [],
            "evidence_types": ["text", "table", "figure"],
        },
        {
            "concept_id": "process_history",
            "label": "ordered synthesis and processing history",
            "entity_id": "process",
            "owner_key": "processes",
            "object_kind": "process",
            "required": True,
            "condition_requirements": ["sequence", "inputs", "conditions", "outputs"],
            "evidence_types": ["text", "table"],
        },
        {
            "concept_id": "measurement_context",
            "label": "measurement method and conditions",
            "entity_id": "measurement",
            "owner_key": "measurements",
            "object_kind": "measurement",
            "required": True,
            "condition_requirements": ["method", "instrument", "conditions", "criterion"],
            "evidence_types": ["text", "table", "figure"],
        },
        {
            "concept_id": "property_observation",
            "label": "property observation bound to sample and conditions",
            "entity_id": "observation",
            "owner_key": "observations",
            "object_kind": "measurement",
            "required": True,
            "condition_requirements": ["sample_ref", "measurement_ref", "conditions", "uncertainty"],
            "evidence_types": ["text", "table", "figure"],
        },
        {
            "concept_id": "scientific_interpretation",
            "label": "reported mechanism or interpretation",
            "entity_id": "interpretation",
            "owner_key": "interpretations",
            "object_kind": "interpretation",
            "required": False,
            "condition_requirements": ["attribution", "supporting_observation_refs"],
            "evidence_types": ["text", "figure"],
        },
        {
            "concept_id": "evidence_provenance",
            "label": "evidence and provenance link",
            "entity_id": "evidence",
            "owner_key": "evidence_items",
            "object_kind": "evidence",
            "required": True,
            "condition_requirements": ["source", "location", "entity_ref"],
            "evidence_types": ["metadata", "text", "table", "figure"],
        },
        {
            "concept_id": "missingness_uncertainty",
            "label": "missingness and uncertainty semantics",
            "entity_id": "observation",
            "owner_key": "observations",
            "object_kind": "quality",
            "required": True,
            "condition_requirements": ["reported_status", "uncertainty", "qualifier"],
            "evidence_types": ["text", "table", "figure"],
        },
    ]
    return expert_advice.build_structured_protocol_advice(advice, TASK)


def write_expert_artifact(path: Path, structured: bool, care: bool = False) -> dict[str, Any]:
    payload = blind_advice_payload(structured, care=care)
    identity_input = run_artifact_guard.build_input_identity(
        "gpt56_sol_blind_expert_advice",
        {
            "round": 1,
            "structured_protocol": structured,
            "care_counterfactual": care,
            "gold_visibility": "none",
            "maximum_rounds": 3,
            "task": TASK,
        },
    )
    identity = run_artifact_guard.reserve_fresh_run(
        pipeline="gpt56_sol_blind_expert_advice",
        output_path=path,
        input_identity=identity_input,
        artifact_paths=[path],
    )
    payload["run_identity"] = identity
    run_artifact_guard.atomic_write_json(path, payload, run_identity=identity)
    run_artifact_guard.update_run_status(
        identity,
        "completed",
        output_status="success",
        expert_round=1,
        forbidden_information_used=False,
    )
    return identity


def load_credentials(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    required = ("CODE_AGENT_BASE_URL", "CODE_AGENT_API_KEY", "CODE_AGENT_MODEL")
    missing = [name for name in required if not values.get(name)]
    if missing:
        raise ValueError(f"credential file is missing required settings: {missing}")
    return values


def load_named_api_key(path: Path, label: str) -> str:
    raw = path.read_bytes()
    text = ""
    for encoding in ("utf-8-sig", "gb18030", "utf-16"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if not text:
        raise ValueError(f"could not decode API key document: {path}")
    matches = re.findall(
        rf"(?im)^\s*{re.escape(label)}\s*=\s*(\S+)\s*$",
        text,
    )
    if len(matches) != 1:
        raise ValueError(
            f"API key document must contain exactly one {label}=... entry; found {len(matches)}"
        )
    return matches[0]


def load_runtime_credentials(
    args: argparse.Namespace,
    campaign: dict[str, Any],
) -> dict[str, str]:
    contract = campaign.get("execution_contract") or {}
    if args.credential_key_file:
        if not args.credential_key_label:
            raise ValueError("--credential-key-label is required with --credential-key-file")
        api_key = load_named_api_key(
            Path(args.credential_key_file).resolve(strict=True),
            args.credential_key_label,
        )
        values = {
            "CODE_AGENT_API_KEY": api_key,
            "CODE_AGENT_BASE_URL": str(
                args.base_url or contract.get("base_url") or ""
            ).strip(),
            "CODE_AGENT_MODEL": str(args.model or contract.get("model") or "").strip(),
        }
    else:
        if not args.credential_file:
            raise ValueError(
                "provide either --credential-file or --credential-key-file with a key label"
            )
        values = load_credentials(Path(args.credential_file).resolve(strict=True))
        if args.base_url:
            values["CODE_AGENT_BASE_URL"] = args.base_url
        if args.model:
            values["CODE_AGENT_MODEL"] = args.model
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(f"runtime credentials are missing settings: {missing}")
    return values


def prepare_campaign(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve(strict=False)
    if output_root.exists():
        raise FileExistsError(
            f"fresh campaign refused because output root already exists: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=False)

    repo_root = Path(args.repo_root).resolve(strict=True)
    papers = discover_corpus(Path(args.corpus_dir).resolve(strict=True), args.expected_count)
    metadata = discover_metadata(
        Path(args.metadata_dir).resolve(strict=True), args.expected_count
    )
    campaign_started_at = utc_now()
    versions = select_versions(args.versions)
    task_spec = {
        "schema_version": "materials-db.three-version-parallel-step8.v1",
        "campaign_id": output_root.name,
        "campaign_started_at": campaign_started_at,
        "task": TASK,
        "expected_paper_count": args.expected_count,
        "versions": list(versions),
        "isolation_contract": {
            "fresh_process_per_version": True,
            "version_local_code_snapshot": True,
            "version_local_corpus_copy": True,
            "version_local_metadata_copy": True,
            "resume_from_prior_checkpoint": False,
            "reuse_prior_module_output": False,
            "reuse_prior_schema": False,
            "reuse_cross_version_artifact": False,
            "generation_gold_visibility": "none",
            "parallel_execution": len(versions) > 1,
        },
        "execution_contract": {
            "base_url": str(args.base_url or "").strip() or None,
            "model": str(args.model or "").strip() or None,
            "backend": "qiniu",
            "max_supervisor_retries": args.max_supervisor_retries,
            "max_total_supervisor_repairs": args.max_total_supervisor_repairs,
            "schema_inspection_interval": args.schema_inspection_interval,
            "repeated_blocker_inspection_threshold": args.repeated_blocker_inspection_threshold,
            "credential_values_persisted": False,
        },
    }
    write_json_exclusive(output_root / "CAMPAIGN_SPEC.json", task_spec)

    source_inventory = corpus_inventory(papers)
    source_manifest = {
        "paper_count": len(source_inventory),
        "aggregate_sha256": stable_digest(source_inventory),
        "papers": source_inventory,
        "metadata_count": len(metadata),
        "source_role": "immutable literature input only; no prior generated output",
    }
    write_json_exclusive(output_root / "SOURCE_CORPUS_MANIFEST.json", source_manifest)

    version_records = []
    aggregate_hashes = set()
    for version in versions:
        version_root = output_root / version["version_id"]
        version_root.mkdir(parents=False, exist_ok=False)
        workspace = version_root / "workspace"
        copy_workspace(repo_root, workspace)
        copied_papers, copied_metadata = copy_inputs(
            papers, metadata, version_root / "inputs"
        )
        inventory = corpus_inventory(copied_papers)
        aggregate = stable_digest(
            [
                {"paper_id": item["paper_id"], "bytes": item["bytes"], "sha256": item["sha256"]}
                for item in inventory
            ]
        )
        aggregate_hashes.add(aggregate)
        advice_path = version_root / "inputs" / "expert_round1.json"
        advice_identity = write_expert_artifact(
            advice_path,
            bool(version["structured_protocol"]),
            care=bool(version["care"]),
        )
        version_manifest = {
            "version": version,
            "campaign_started_at": campaign_started_at,
            "workspace": run_artifact_guard.path_digest(workspace),
            "corpus_count": len(inventory),
            "corpus_aggregate_sha256": aggregate,
            "corpus": inventory,
            "metadata_count": len(copied_metadata),
            "expert_advice": str(advice_path),
            "expert_run_identity": advice_identity,
            "forbidden_generation_inputs": [
                "manual schema",
                "core mapping",
                "extraction gold",
                "prior checkpoints",
                "sibling version directories",
            ],
        }
        write_json_exclusive(version_root / "INPUT_MANIFEST.json", version_manifest)
        version_records.append(
            {
                "version_id": version["version_id"],
                "root": str(version_root),
                "workspace": str(workspace),
                "corpus_paths": [str(path) for path in copied_papers],
                "metadata_paths": [str(path) for path in copied_metadata],
                "advice_path": str(advice_path),
                "structured_protocol": bool(version["structured_protocol"]),
                "care": bool(version["care"]),
            }
        )

    if len(aggregate_hashes) != 1:
        raise RuntimeError("version-local corpus copies do not have one identical digest")
    campaign = {
        "schema_version": "materials-db.parallel-campaign-status.v1",
        "campaign_id": output_root.name,
        "campaign_started_at": campaign_started_at,
        "pipeline": PIPELINE,
        "status": "prepared",
        "paper_count": args.expected_count,
        "requested_version_ids": [item["version_id"] for item in versions],
        "execution_contract": task_spec["execution_contract"],
        "corpus_aggregate_sha256": next(iter(aggregate_hashes)),
        "versions": version_records,
        "events": [
            {
                "at": utc_now(),
                "event": "campaign_prepared",
                "detail": (
                    f"{len(versions)} isolated namespace(s) and identical "
                    f"{args.expected_count}-paper copies created"
                ),
            }
        ],
    }
    atomic_write_json(output_root / "RUN_STATUS.json", campaign)
    return campaign


def build_step8_command(
    python_executable: str,
    record: dict[str, Any],
    model: str,
    base_url: str,
    max_supervisor_retries: int = 4,
    max_total_supervisor_repairs: int = 24,
    schema_inspection_interval: int = 10,
    repeated_blocker_inspection_threshold: int = 3,
) -> list[str]:
    version_root = Path(record["root"])
    workspace = Path(record["workspace"])
    artifacts = version_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=False)
    command = [
        python_executable,
        str(workspace / "code" / "section_design_langgraph_human_gate.py"),
        "--database-goal",
        TASK["objective"],
        "--discipline",
        TASK["discipline"],
        "--query-requirements",
        TASK["query_requirements"],
        "--key-description-path",
        str(workspace / "tests" / "fixtures" / "no_reference_field_catalog.txt"),
        "--reference-papers",
        *record["corpus_paths"],
        "--output",
        str(artifacts / "step8.json"),
        "--checkpoint-output",
        str(artifacts / "step8.state.json"),
        "--model",
        model,
        "--base-url",
        base_url,
        "--llm-backend",
        "qiniu",
        "--request-timeout",
        "1200",
        "--max-retries",
        "10",
        "--max-supervisor-retries",
        str(max_supervisor_retries),
        "--max-total-supervisor-repairs",
        str(max_total_supervisor_repairs),
        "--schema-inspection-interval",
        str(schema_inspection_interval),
        "--repeated-blocker-inspection-threshold",
        str(repeated_blocker_inspection_threshold),
        "--thread-id",
        f"{record['version_id']}-parallel-step8",
    ]
    command.extend(["--human-advice-path", record["advice_path"]])
    if record["structured_protocol"]:
        command.append("--structured-protocol")
    return command


def launch_and_wait(
    campaign: dict[str, Any],
    output_root: Path,
    credentials: dict[str, str],
    python_executable: str,
) -> int:
    execution_contract = campaign.setdefault("execution_contract", {})
    for key, credential_key in (
        ("base_url", "CODE_AGENT_BASE_URL"),
        ("model", "CODE_AGENT_MODEL"),
    ):
        frozen = str(execution_contract.get(key) or "").strip()
        actual = credentials[credential_key]
        if frozen and frozen != actual:
            raise ValueError(
                f"prepared campaign {key} does not match launch credentials"
            )
        execution_contract[key] = actual
    execution_contract.setdefault("backend", "qiniu")
    execution_contract.setdefault("max_supervisor_retries", 4)
    execution_contract.setdefault("max_total_supervisor_repairs", 24)
    execution_contract.setdefault("schema_inspection_interval", 10)
    execution_contract.setdefault("repeated_blocker_inspection_threshold", 3)
    execution_contract["credential_values_persisted"] = False
    atomic_write_json(output_root / "RUN_STATUS.json", campaign)

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "CODE_AGENT_BASE_URL": credentials["CODE_AGENT_BASE_URL"],
            "CODE_AGENT_MODEL": credentials["CODE_AGENT_MODEL"],
            "CODE_AGENT_API_KEY": credentials["CODE_AGENT_API_KEY"],
            "SECTION_AGENT_BASE_URL": credentials["CODE_AGENT_BASE_URL"],
            "SECTION_AGENT_MODEL": credentials["CODE_AGENT_MODEL"],
            "SECTION_AGENT_API_KEY": credentials["CODE_AGENT_API_KEY"],
            "SECTION_AGENT_LLM_BACKEND": "qiniu",
        }
    )
    processes: dict[str, dict[str, Any]] = {}
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    for record in campaign["versions"]:
        version_id = record["version_id"]
        version_root = Path(record["root"])
        log_dir = version_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=False)
        stdout_handle = (log_dir / "step8.stdout.log").open(
            "x", encoding="utf-8", buffering=1
        )
        stderr_handle = (log_dir / "step8.stderr.log").open(
            "x", encoding="utf-8", buffering=1
        )
        command = build_step8_command(
            python_executable,
            record,
            credentials["CODE_AGENT_MODEL"],
            credentials["CODE_AGENT_BASE_URL"],
            int(execution_contract["max_supervisor_retries"]),
            int(execution_contract["max_total_supervisor_repairs"]),
            int(execution_contract["schema_inspection_interval"]),
            int(execution_contract["repeated_blocker_inspection_threshold"]),
        )
        process = subprocess.Popen(
            command,
            cwd=record["workspace"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=creationflags,
        )
        process_record = {
            "pid": process.pid,
            "started_at": utc_now(),
            "status": "running",
            "command_contract": {
                "script": "code/section_design_langgraph_human_gate.py",
                "model": credentials["CODE_AGENT_MODEL"],
                "backend": "qiniu",
                "paper_count": len(record["corpus_paths"]),
                "structured_protocol": record["structured_protocol"],
                "expert_advice_round": 1,
                "resume": False,
                "max_supervisor_retries": int(
                    execution_contract["max_supervisor_retries"]
                ),
                "max_total_supervisor_repairs": int(
                    execution_contract["max_total_supervisor_repairs"]
                ),
                "schema_inspection_interval": int(
                    execution_contract["schema_inspection_interval"]
                ),
                "repeated_blocker_inspection_threshold": int(
                    execution_contract["repeated_blocker_inspection_threshold"]
                ),
            },
        }
        write_json_exclusive(version_root / "PROCESS.json", process_record)
        processes[version_id] = {
            "process": process,
            "stdout": stdout_handle,
            "stderr": stderr_handle,
            "record": record,
            "process_record": process_record,
        }

    campaign["status"] = (
        "running_step8_parallel" if len(processes) > 1 else "running_step8_single"
    )
    campaign["events"].append(
        {
            "at": utc_now(),
            "event": "step8_processes_started",
            "pids": {
                key: value["process"].pid for key, value in processes.items()
            },
        }
    )
    atomic_write_json(output_root / "RUN_STATUS.json", campaign)
    print(
        json.dumps(
            {
                "event": "started",
                "campaign": str(output_root),
                "versions": {
                    key: value["process"].pid for key, value in processes.items()
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    last_report = 0.0
    while any(item["process"].poll() is None for item in processes.values()):
        now = time.monotonic()
        if now - last_report >= 30:
            live = {
                key: (
                    "running"
                    if item["process"].poll() is None
                    else f"exit_{item['process'].returncode}"
                )
                for key, item in processes.items()
            }
            print(json.dumps({"event": "progress", "versions": live}), flush=True)
            campaign["live_process_status"] = live
            atomic_write_json(output_root / "RUN_STATUS.json", campaign)
            last_report = now
        time.sleep(2)

    failures = 0
    terminal = {}
    for version_id, item in processes.items():
        item["stdout"].close()
        item["stderr"].close()
        process = item["process"]
        record = item["record"]
        output_path = Path(record["root"]) / "artifacts" / "step8.json"
        result_status = "missing_output"
        run_id = ""
        if output_path.exists():
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
                result_status = str(payload.get("status") or "unknown")
                run_id = str((payload.get("run_identity") or {}).get("run_id") or "")
            except json.JSONDecodeError:
                result_status = "invalid_output_json"
        passed = process.returncode == 0 and result_status == "success" and bool(run_id)
        if not passed:
            failures += 1
        process_record = item["process_record"]
        process_record.update(
            {
                "finished_at": utc_now(),
                "status": "passed" if passed else "needs_review",
                "exit_code": process.returncode,
                "step8_result_status": result_status,
                "step8_run_id": run_id,
            }
        )
        atomic_write_json(Path(record["root"]) / "PROCESS.json", process_record)
        terminal[version_id] = process_record

    campaign["status"] = (
        "step8_all_passed" if failures == 0 else "step8_terminal_with_failures"
    )
    campaign["step8_terminal"] = terminal
    campaign["events"].append(
        {
            "at": utc_now(),
            "event": campaign["status"],
            "failure_count": failures,
        }
    )
    atomic_write_json(output_root / "RUN_STATUS.json", campaign)
    print(
        json.dumps(
            {
                "event": "finished",
                "status": campaign["status"],
                "failure_count": failures,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if failures == 0 else 2


def load_prepared_campaign(output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve(strict=True)
    status_path = output_root / "RUN_STATUS.json"
    campaign = json.loads(status_path.read_text(encoding="utf-8"))
    if campaign.get("status") != "prepared":
        raise ValueError(
            f"prepared launch requires status=prepared, found {campaign.get('status')}"
        )
    records = campaign.get("versions") or []
    requested_ids = campaign.get("requested_version_ids")
    expected_ids = set(requested_ids or VERSION_BY_ID)
    if not expected_ids or not expected_ids.issubset(VERSION_BY_ID):
        raise ValueError("prepared campaign contains an invalid version selection")
    actual_ids = {str(item.get("version_id") or "") for item in records}
    if actual_ids != expected_ids or len(records) != len(expected_ids):
        raise ValueError("prepared campaign version records do not match its frozen selection")

    corpus_hashes = set()
    for record in records:
        version_root = Path(record["root"]).resolve(strict=True)
        workspace = Path(record["workspace"]).resolve(strict=True)
        try:
            version_root.relative_to(output_root)
            workspace.relative_to(version_root)
        except ValueError as exc:
            raise ValueError("prepared version path escapes its campaign namespace") from exc
        forbidden_existing = [
            version_root / "artifacts",
            version_root / "logs",
            version_root / "PROCESS.json",
        ]
        stale = [str(path) for path in forbidden_existing if path.exists()]
        if stale:
            raise run_artifact_guard.CleanRunRequiredError(
                f"prepared version already has launch artifacts: {stale}"
            )
        paper_paths = [Path(path).resolve(strict=True) for path in record["corpus_paths"]]
        if len(paper_paths) != int(campaign["paper_count"]):
            raise ValueError("prepared version paper count changed after preparation")
        for path in paper_paths:
            try:
                path.relative_to(version_root / "inputs" / "corpus")
            except ValueError as exc:
                raise ValueError("prepared corpus path escapes its version namespace") from exc
        inventory = corpus_inventory(paper_paths)
        corpus_hashes.add(
            stable_digest(
                [
                    {
                        "paper_id": item["paper_id"],
                        "bytes": item["bytes"],
                        "sha256": item["sha256"],
                    }
                    for item in inventory
                ]
            )
        )
        advice_path = Path(record["advice_path"]).resolve(strict=True)
        try:
            advice_path.relative_to(version_root / "inputs")
        except ValueError as exc:
            raise ValueError("prepared expert advice escapes its version namespace") from exc
    if corpus_hashes != {campaign["corpus_aggregate_sha256"]}:
        raise ValueError("prepared corpus hash no longer matches the frozen campaign hash")
    return campaign


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--credential-file", default="")
    parser.add_argument("--credential-key-file", default="")
    parser.add_argument("--credential-key-label", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=tuple(VERSION_BY_ID),
        default=None,
    )
    parser.add_argument("--expected-count", type=int, default=30)
    parser.add_argument("--max-supervisor-retries", type=int, default=4)
    parser.add_argument("--max-total-supervisor-repairs", type=int, default=24)
    parser.add_argument("--schema-inspection-interval", type=int, default=10)
    parser.add_argument("--repeated-blocker-inspection-threshold", type=int, default=3)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--launch-prepared", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.prepare_only and args.launch_prepared:
        raise ValueError("--prepare-only and --launch-prepared are mutually exclusive")
    output_root = Path(args.output_root).resolve(strict=False)
    if args.launch_prepared:
        campaign = load_prepared_campaign(output_root)
    else:
        try:
            campaign = prepare_campaign(args)
        except BaseException as exc:
            if output_root.exists():
                atomic_write_json(
                    output_root / "PREPARATION_FAILURE.json",
                    {
                        "status": "preparation_failed_not_launchable",
                        "failed_at": utc_now(),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:2000],
                        "policy": "preserve this partial namespace and use a new output root",
                    },
                )
            raise
    output_root = output_root.resolve(strict=True)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "output_root": str(output_root),
                    "paper_count": campaign["paper_count"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    credentials = load_runtime_credentials(args, campaign)
    return launch_and_wait(campaign, output_root, credentials, args.python)


if __name__ == "__main__":
    raise SystemExit(main())
