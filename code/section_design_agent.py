import argparse
import hashlib
import json
from task_domain_knowledge import attach_to_context
import os
import re
from copy import deepcopy
from pathlib import Path

from openai import OpenAI

import materials_agent_protocol as agent_protocol
import run_artifact_guard
from field_definition_contract import validate_definitions
from section_design_agent_prompt import (
    FINAL_OUTPUT_SCHEMA_DESCRIPTION,
    STEP8_FRAMEWORK,
    build_aggregation_prompt,
    build_evidence_model_prompt,
    build_field_planning_prompt,
    build_figure_classification_prompt,
    build_locating_prompt,
    build_mechanism_requirement_prompt,
    build_module_redo_prompt,
    build_query_semantics_prompt,
    build_redo_prompt,
    build_schema_design_prompt,
    build_section_partition_prompt,
    build_shared_context_block,
    build_specialization_critic_prompt,
    build_supervisor_prompt,
    build_subjective_supervisor_prompt,
    build_topic_adaptation_prompt,
)


DEFAULT_SYSTEM_PROMPT = "You are an expert in scientific database schema design."
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_LLM_BACKEND = "openai"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 1200
DEFAULT_MODULE_MAX_TOKENS = 384000
MATERIAL_LITERATURE_TASK_CONTRACT = agent_protocol.materials_literature_task_contract()
MODULE_MAX_TOKENS = {
    "locating_module": 384000,
    "mechanism_requirement_module": 384000,
    "query_semantics_module": 384000,
    "evidence_model_module": 384000,
    "subjective_supervisor_module": 384000,
    "topic_adaptation_module": 384000,
    "section_partition_module": 384000,
    "field_planning_module": 384000,
    "supervisor_module": 384000,
    "figure_classification_module": 384000,
    "schema_design_module": 384000,
    "specialization_critic_module": 384000,
    "aggregation": 384000,
}

REQUIRED_PIPELINE_MODULES = tuple(MODULE_MAX_TOKENS)

MATERIAL_SCHEMA_ROOTS = {
    "paper_info",
    "primary_signature",
    "primary_signature_normalized",
    "material_info",
    "section5",
    "normalization_aliases",
    "material_name_aliases",
    "formula_aliases",
    "sample_id_aliases",
}

SUPERVISOR_ALLOWED_EXTENSION_ROOTS = {
    "device_info",
    "reaction_info",
    "dataset_info",
    "interface_info",
}


def is_safe_entity_owner_key(value):
    value = str(value or "")
    return bool(re.fullmatch(r"[a-z][a-z0-9_]{1,63}", value)) and (
        value.endswith("_info") or value.endswith("s")
    )

MODULE_SCHEMAS = {
    "locating_module": {
        "required_keys": [
            "database_goal",
            "discipline",
            "query_requirements",
            "retrieval_unit",
            "organization_focus",
            "design_rationale",
        ],
        "schema_text": """
        {
          "database_goal": "string",
          "discipline": "string",
          "query_requirements": ["string"],
          "retrieval_unit": "string",
          "organization_focus": "string",
          "design_rationale": "string"
        }
        """.strip(),
    },
    "mechanism_requirement_module": {
        "required_keys": [
            "domain_focus",
            "must_have_concepts",
            "recommended_objects",
            "red_flag_patterns",
        ],
        "schema_text": """
        {
          "domain_focus": "string",
          "must_have_concepts": ["string"],
          "recommended_objects": ["string"],
          "red_flag_patterns": ["string"]
        }
        """.strip(),
    },
    "query_semantics_module": {
        "required_keys": ["query_objects"],
        "schema_text": """
        {
          "query_objects": [
            {
              "query_requirement": "string",
              "object_type": "string",
              "recommended_field_groups": ["string"],
              "comparison_axes": ["string"],
              "anti_generic_warning": "string"
            }
          ]
        }
        """.strip(),
    },
    "evidence_model_module": {
        "required_keys": [
            "evidence_layers",
            "figure_ownership_principles",
            "anti_generic_evidence_patterns",
        ],
        "schema_text": """
        {
          "evidence_layers": [
            {
              "layer_name": "string",
              "description": "string",
              "typical_methods": ["string"],
              "schema_implication": "string"
            }
          ],
          "figure_ownership_principles": ["string"],
          "anti_generic_evidence_patterns": ["string"]
        }
        """.strip(),
    },
    "subjective_supervisor_module": {
        "required_keys": [
            "database_nature",
            "modeling_position",
            "must_have_concepts",
            "requirement_contract",
            "entity_registry",
            "must_not_become",
            "red_flags",
            "approved_section_strategy",
            "redo_directives",
        ],
        "schema_text": """
        {
          "database_nature": "string",
          "modeling_position": "string",
          "must_have_concepts": ["string"],
          "requirement_contract": {"concepts": [{"concept_id": "string", "label": "string"}]},
          "entity_registry": [{"entity_id": "material", "owner_key": "material_info"}],
          "must_not_become": ["string"],
          "red_flags": ["string"],
          "approved_section_strategy": ["string"],
          "redo_directives": ["string"]
        }
        """.strip(),
    },
    "topic_adaptation_module": {
        "required_keys": [
            "topic_type",
            "adaptation_principles",
            "avoid_generic_template",
            "topic_specific_adjustments",
        ],
        "schema_text": """
        {
          "topic_type": "string",
          "adaptation_principles": ["string"],
          "avoid_generic_template": ["string"],
          "topic_specific_adjustments": ["string"]
        }
        """.strip(),
    },
    "section_partition_module": {
        "required_keys": ["core_sections", "non_core_sections"],
        "schema_text": """
        {
          "core_sections": [{"section_id": "section0"}],
          "non_core_sections": [{"section_id": "sectionX"}]
        }
        """.strip(),
    },
    "field_planning_module": {
        "required_keys": ["field_groups", "red_flag_fixes"],
        "schema_text": """
        {
          "field_groups": [
            {
              "section_id": "string",
              "group_name": "string",
              "purpose": "string",
              "recommended_fields": ["string"],
              "evidence_strategy": "string"
            }
          ],
          "reference_field_audit": {
            "policy": "task_adaptive",
            "catalog_leaf_count": 0,
            "included_leaf_paths": ["exact.reference.leaf_path"],
            "adapted_leaf_mappings": [
              {"source_path": "reference.leaf", "target_path": "target.leaf", "reason": "string"}
            ],
            "excluded_path_prefixes": [
              {"path_prefix": "reference.family", "reason": "string"}
            ],
            "unresolved_leaf_paths": []
          },
          "red_flag_fixes": ["string"]
        }
        """.strip(),
    },
    "supervisor_module": {
        "required_keys": [
            "review_stage",
            "enable_figure_classification",
            "routing_decision",
            "decision_summary",
            "risk_signals",
            "risk_assessment",
            "trigger_sections",
            "expected_figure_fields",
            "skip_reason",
        ],
        "schema_text": """
        {
          "review_stage": "post_section_partition",
          "enable_figure_classification": true,
          "routing_decision": "figure_classification_path",
          "decision_summary": "string",
          "risk_signals": ["string"],
          "risk_assessment": "string",
          "trigger_sections": ["section4", "section5"],
          "expected_figure_fields": ["material_info.section4.domain_specific_curve.figure"],
          "skip_reason": ""
        }
        """.strip(),
    },
    "figure_classification_module": {
        "required_keys": [
            "enable_figure_classification",
            "routing_status",
            "classification_strategy",
            "section_figure_plan",
            "skip_reason",
        ],
        "schema_text": """
        {
          "enable_figure_classification": true,
          "routing_status": "active",
          "classification_strategy": ["string"],
          "section_figure_plan": [
            {
              "section_id": "section4",
              "section_name": "string",
              "figure_scope": "string",
              "allowed_figure_categories": ["domain_specific_curve"],
              "blocked_neighbor_sections": ["section5"],
              "blocked_figure_categories": ["band_structure"],
              "routing_rule": "string"
            }
          ],
          "skip_reason": ""
        }
        """.strip(),
    },
    "schema_design_module": {
        "required_keys": ["top_level_keys", "field_registry"],
        "schema_text": """
        {
          "top_level_keys": [{"key": "string", "description": "string"}],
          "field_registry": [
            {
              "field_path": "string",
              "section_id": "string",
              "field_name": "string",
              "description": "string",
              "extraction_notes": "precise inclusion, exclusion, condition-binding, and normalization rules",
              "data_type": "string",
              "required": true,
              "source_basis": ["text", "table", "figure"],
              "concept_ids": ["ascii_snake_case_id"],
              "object_contract": {
                "object_kind": "measurement",
                "required_subfields": ["value", "conditions", "entity_ref", "source_type", "evidence", "extraction_confidence"]
              },
              "figure_constraint": {
                "uses_figure_classification": true,
                "allowed_sections": ["section4"],
                "allowed_figure_categories": ["domain_specific_curve"],
                "why_needed": "string"
              },
              "reason": "string"
            }
          ]
        }
        """.strip(),
    },
    "specialization_critic_module": {
        "required_keys": [
            "specialization_status",
            "is_generic",
            "missing_concepts",
            "structural_weaknesses",
            "redo_needed",
            "redo_directives",
            "field_utility_audit",
            "patch_operations",
        ],
        "schema_text": """
        {
          "specialization_status": "pass",
          "is_generic": false,
          "missing_concepts": ["string"],
          "structural_weaknesses": ["string"],
          "redo_needed": false,
          "redo_directives": ["string"],
          "field_utility_audit": {
            "reviewed_field_count": 0,
            "decision": "pass | needs_pruning",
            "unsupported_fields": ["exact.field.path"],
            "redundant_fields": ["exact.field.path"],
            "alias_or_unit_variant_fields": ["exact.field.path"],
            "derivable_duplicate_fields": ["exact.field.path"],
            "rationale": "string"
          },
          "patch_operations": [
            {
              "op": "add_field | update_field | remove_field | move_field",
              "field_path": "canonical dotted field path",
              "target_path": "canonical dotted field path for move_field only",
              "field": {},
              "changes": {"object_contract.required_subfields": ["value", "unit", "evidence"]},
              "reason": "string"
            }
          ]
        }
        """.strip(),
    },
    "aggregation": {
        "required_keys": [
            "database_positioning",
            "section_design",
            "schema_definition",
            "quality_check",
        ],
        "schema_text": FINAL_OUTPUT_SCHEMA_DESCRIPTION,
    },
}


def get_client(base_url=None, api_key=None, backend=DEFAULT_LLM_BACKEND, timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS):
    if not api_key or api_key == "[REDACTED]":
        api_key = os.getenv("SECTION_AGENT_API_KEY") or os.getenv("CODE_AGENT_API_KEY")
    if backend == "langchain":
        return get_langchain_client(base_url=base_url, api_key=api_key, timeout=timeout)
    if backend == "qiniu":
        from qiniu_model_client import QiniuModelClient

        if not api_key:
            raise ValueError("Qiniu backend requires an API key")
        return {
            "backend": "qiniu",
            "client": QiniuModelClient(
                api_key=api_key,
                base_url=base_url or "https://api.qnaigc.com/v1",
                connect_timeout=min(float(timeout or 30), 30),
                read_timeout=float(timeout or DEFAULT_REQUEST_TIMEOUT_SECONDS),
                max_retries=0,
                max_reasoning_seconds=float(timeout or DEFAULT_REQUEST_TIMEOUT_SECONDS),
                max_wall_seconds=float(timeout or DEFAULT_REQUEST_TIMEOUT_SECONDS),
            ),
        }
    if backend != "openai":
        raise ValueError(f"Unsupported LLM backend: {backend}")

    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    if timeout:
        kwargs["timeout"] = timeout
    # Module-level retries already validate and retry malformed responses. Disable
    # the SDK's hidden transport retries so --request-timeout remains observable.
    kwargs["max_retries"] = 0
    return {"backend": "openai", "client": OpenAI(**kwargs)}


def get_langchain_client(base_url=None, api_key=None, timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "LangChain backend requires the optional dependency `langchain-openai`. "
            "Install it with `pip install langchain-openai` or use `--llm-backend openai`."
        ) from exc

    return {
        "backend": "langchain",
        "chat_model_cls": ChatOpenAI,
        "base_url": base_url,
        "api_key": api_key,
        "timeout": timeout,
    }


def build_langchain_chat_model(client, model, temperature, max_tokens=None):
    chat_model_cls = client["chat_model_cls"]
    common_kwargs = {
        "model": model,
        "temperature": temperature,
    }
    if client.get("api_key"):
        common_kwargs["api_key"] = client["api_key"]
    if client.get("base_url"):
        common_kwargs["base_url"] = client["base_url"]
    if client.get("timeout"):
        common_kwargs["request_timeout"] = client["timeout"]
    if max_tokens:
        common_kwargs["max_tokens"] = max_tokens

    try:
        return chat_model_cls(**common_kwargs)
    except TypeError:
        # Older langchain-openai versions used OpenAI-prefixed parameter names.
        fallback_kwargs = dict(common_kwargs)
        if "api_key" in fallback_kwargs:
            fallback_kwargs["openai_api_key"] = fallback_kwargs.pop("api_key")
        if "base_url" in fallback_kwargs:
            fallback_kwargs["openai_api_base"] = fallback_kwargs.pop("base_url")
        return chat_model_cls(**fallback_kwargs)


def chat(
    client,
    model,
    prompt,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    temperature=0,
    max_tokens=None,
):
    if isinstance(client, dict) and client.get("backend") == "qiniu":
        qiniu_client = client["client"]
        qiniu_client.model = model
        parsed, _metadata = qiniu_client.chat_json(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens or DEFAULT_MODULE_MAX_TOKENS,
            temperature=temperature,
        )
        return json.dumps(parsed, ensure_ascii=False)

    if isinstance(client, dict) and client.get("backend") == "langchain":
        chat_model = build_langchain_chat_model(
            client,
            model,
            temperature,
            max_tokens=max_tokens,
        )
        response = chat_model.invoke(
            [
                ("system", system_prompt),
                ("user", prompt),
            ]
        )
        return response.content

    openai_client = client["client"] if isinstance(client, dict) else client
    request_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "stream": True,
    }
    if max_tokens:
        request_kwargs["max_tokens"] = max_tokens
    response = openai_client.chat.completions.create(**request_kwargs)
    content_parts = []
    reasoning_chars = 0
    finish_reason = None
    for chunk in response:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None) or finish_reason
        delta = getattr(choice, "delta", None)
        reasoning_chars += len(getattr(delta, "reasoning_content", None) or "")
        piece = getattr(delta, "content", None)
        if piece:
            content_parts.append(piece)
    content = "".join(content_parts).strip()
    if not content:
        raise RuntimeError(
            "model stream returned no final content "
            f"(reasoning_chars={reasoning_chars}, "
            f"finish_reason={finish_reason or 'unknown'})"
        )
    return content


def load_query_requirements(query_input):
    path = Path(query_input)
    if path.exists():
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return [item.strip() for item in query_input.split("||") if item.strip()]


def load_key_description_text(path):
    return Path(path).read_text(encoding="utf-8")


REFERENCE_FIELD_PATH_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)


def _reference_data_type(description, fallback=""):
    match = re.search(
        r"Data format:\s*(.*?)(?:\.\s+(?:Concept|Notes):|$)",
        str(description or ""),
        flags=re.IGNORECASE,
    )
    return (match.group(1).strip() if match else str(fallback or "").strip()) or "unspecified"


def _reference_description_hint(description, limit=180):
    text = str(description or "").strip()
    concept_match = re.search(
        r"Concept:\s*(.*?)(?:\.\s+Notes:|$)",
        text,
        flags=re.IGNORECASE,
    )
    if concept_match:
        text = concept_match.group(1).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _reference_extraction_notes(description, limit=700):
    text = str(description or "").strip()
    notes_match = re.search(r"Notes:\s*(.*)$", text, flags=re.IGNORECASE)
    if not notes_match:
        return ""
    notes = re.sub(r"\s+", " ", notes_match.group(1).strip())
    if len(notes) <= limit:
        return notes
    return notes[: limit - 3].rstrip() + "..."


def _reference_group(path):
    parts = str(path or "").split(".")
    if len(parts) >= 2 and parts[0] in {"material_info", "paper_info"}:
        return ".".join(parts[:2])
    return parts[0] if parts else "reference"


def _load_reference_mapping(text, source_path=""):
    suffix = Path(source_path).suffix.lower() if source_path else ""
    if suffix == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        if isinstance(loaded, (dict, list)):
            return loaded
    except (ImportError, ValueError, TypeError):
        pass

    mapping = {}
    for line in text.splitlines():
        match = re.match(
            r"^([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*):\s*(.*)$",
            line,
        )
        if not match:
            continue
        value = match.group(2).strip()
        if value.startswith('"') and value.endswith('"'):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = value[1:-1]
        mapping[match.group(1)] = value
    return mapping or None


def build_reference_field_contract(key_description_text, source_path=""):
    """Build a complete, domain-neutral leaf-field catalog from YAML or JSON input."""
    parsed = _load_reference_mapping(key_description_text, source_path)
    if parsed is None:
        return {
            "available": False,
            "source_path": str(source_path or ""),
            "all_path_count": 0,
            "leaf_count": 0,
            "leaf_fields": [],
            "groups": [],
        }

    entries = {}
    is_flat_mapping = isinstance(parsed, dict) and parsed and all(
        isinstance(key, str)
        and REFERENCE_FIELD_PATH_PATTERN.match(key)
        and not isinstance(value, (dict, list))
        for key, value in parsed.items()
    )

    if is_flat_mapping:
        for path, description in parsed.items():
            entries[path] = {
                "path": path,
                "data_type": _reference_data_type(description),
                "description_hint": _reference_description_hint(description),
                "extraction_notes": _reference_extraction_notes(description),
            }
    else:
        def visit(value, path=""):
            if path:
                if isinstance(value, dict):
                    data_type = "object"
                elif isinstance(value, list):
                    data_type = "array"
                elif value is None:
                    data_type = "unspecified"
                else:
                    data_type = type(value).__name__
                entries[path] = {
                    "path": path,
                    "data_type": data_type,
                    "description_hint": "",
                    "extraction_notes": "",
                }
            if isinstance(value, dict):
                for key, child in value.items():
                    child_path = f"{path}.{key}" if path else str(key)
                    visit(child, child_path)
            elif isinstance(value, list) and value:
                sample = value[0]
                if isinstance(sample, dict):
                    for key, child in sample.items():
                        visit(child, f"{path}.{key}")

        visit(parsed)

    design_directive_keys = {
        "schema_policy",
        "field_design_policy",
        "field_generation_policy",
        "generation_policy",
    }
    design_directives = []
    for path in list(entries):
        if path.lower() not in design_directive_keys:
            continue
        design_directives.append(
            {
                "path": path,
                "instruction": entries[path].get("description_hint") or "",
            }
        )
        entries.pop(path)

    paths = list(entries)
    array_paths = {
        path
        for path, item in entries.items()
        if "array" in str(item.get("data_type") or "").lower()
    }
    leaf_paths = [
        path
        for path in paths
        if not any(candidate.startswith(path + ".") for candidate in paths)
    ]

    def path_with_arrays(path):
        rendered = []
        prefix = []
        for part in path.split("."):
            prefix.append(part)
            current = ".".join(prefix)
            rendered.append(part + ("[]" if current in array_paths else ""))
        return ".".join(rendered)

    group_counts = {}
    leaf_fields = []
    for path in leaf_paths:
        item = entries[path]
        group = _reference_group(path)
        group_counts[group] = group_counts.get(group, 0) + 1
        leaf_fields.append(
            {
                "path": path,
                "path_with_arrays": path_with_arrays(path),
                "data_type": item.get("data_type") or "unspecified",
                "description_hint": item.get("description_hint") or "",
                "extraction_notes": item.get("extraction_notes") or "",
                "group": group,
            }
        )

    source = Path(source_path) if source_path else None
    source_bytes = (
        source.read_bytes()
        if source is not None and source.exists()
        else key_description_text.encode("utf-8")
    )
    return {
        "available": bool(leaf_fields),
        "source_path": str(source_path or ""),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest().upper(),
        "policy": "task_adaptive",
        "all_path_count": len(paths),
        "leaf_count": len(leaf_fields),
        "leaf_fields": leaf_fields,
        "design_directives": design_directives,
        "groups": [
            {"group": group, "leaf_count": count}
            for group, count in group_counts.items()
        ],
    }


def load_optional_text(path):
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8").strip()


def _truncate_text(text, max_chars):
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[:max_chars].rstrip() + "\n...[truncated]"


def _balanced_reference_excerpt(text, max_chars):
    """Keep evidence from every dotted-key family instead of only the file prefix."""
    if len(text.strip()) <= max_chars:
        return text.strip()
    buckets = {}
    preamble = []
    for line in text.splitlines():
        match = re.match(
            r"^([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*):",
            line,
        )
        if not match:
            if len(preamble) < 12 and line.strip():
                preamble.append(line.strip())
            continue
        buckets.setdefault(_reference_group(match.group(1)), []).append(line.strip())
    if not buckets:
        return _truncate_text(text, max_chars)

    header = "\n".join(preamble)
    remaining = max(max_chars - len(header) - 256, 1)
    per_group = max(remaining // len(buckets), 300)
    rendered = [header] if header else []
    for group, lines in buckets.items():
        block = []
        used = 0
        for line in lines:
            concise = line if len(line) <= 700 else line[:697].rstrip() + "..."
            if block and used + len(concise) + 1 > per_group:
                break
            block.append(concise)
            used += len(concise) + 1
        rendered.append(f"# Reference group: {group}\n" + "\n".join(block))
    excerpt = "\n\n".join(rendered)
    return _truncate_text(excerpt, max_chars)


def load_reference_paper_context(reference_papers, max_chars_per_paper=150000):
    if not reference_papers:
        return ""

    paper_blocks = []
    for raw_path in reference_papers:
        path = Path(raw_path)
        if not path.exists():
            paper_blocks.append(f"[Missing paper] {raw_path}")
            continue
        text = path.read_text(encoding="utf-8")
        paper_blocks.append(
            f"Paper: {path.name}\n{_truncate_text(text, max_chars_per_paper)}"
        )
    return "\n\n".join(paper_blocks)


def strip_markdown_json_fence(raw_response):
    text = (raw_response or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    return text


def parse_json_response(raw_response):
    cleaned_response = strip_markdown_json_fence(raw_response)
    try:
        return json.loads(cleaned_response), []
    except json.JSONDecodeError as exc:
        return None, [f"Invalid JSON response: {exc}"]


def normalize_module_result(module_name, result):
    if not isinstance(result, dict):
        return result

    if module_name == "specialization_critic_module":
        status = str(result.get("specialization_status", "")).strip().lower()
        status_aliases = {
            "pass": "pass",
            "ok": "pass",
            "success": "pass",
            "approved": "pass",
            "needs_redesign": "needs_redesign",
            "need_redesign": "needs_redesign",
            "redesign": "needs_redesign",
            "redo": "needs_redesign",
            "redo_needed": "needs_redesign",
            "fail": "needs_redesign",
            "failed": "needs_redesign",
            "reject": "needs_redesign",
            "rejected": "needs_redesign",
        }
        normalized_status = status_aliases.get(status)
        if normalized_status:
            result["specialization_status"] = normalized_status

        if result.get("specialization_status") == "pass":
            result["redo_needed"] = False
            if not isinstance(result.get("is_generic"), bool):
                result["is_generic"] = False
        elif result.get("specialization_status") == "needs_redesign":
            result["redo_needed"] = True
            if not isinstance(result.get("is_generic"), bool):
                result["is_generic"] = True

    return result


def validate_module_result(module_name, result):
    errors = []
    schema = MODULE_SCHEMAS[module_name]
    if not isinstance(result, dict):
        return [f"{module_name} output must be a JSON object"]
    for key in schema["required_keys"]:
        if key not in result:
            errors.append(f"{module_name} missing key: {key}")
    if module_name == "section_partition_module":
        if not isinstance(result.get("core_sections"), list) or not result.get("core_sections"):
            errors.append("section_partition_module.core_sections must be a non-empty list")
        if not isinstance(result.get("non_core_sections"), list):
            errors.append("section_partition_module.non_core_sections must be a list")
    if module_name == "mechanism_requirement_module":
        if not isinstance(result.get("must_have_concepts"), list) or not result.get("must_have_concepts"):
            errors.append("mechanism_requirement_module.must_have_concepts must be a non-empty list")
    if module_name == "query_semantics_module":
        if not isinstance(result.get("query_objects"), list) or not result.get("query_objects"):
            errors.append("query_semantics_module.query_objects must be a non-empty list")
    if module_name == "evidence_model_module":
        if not isinstance(result.get("evidence_layers"), list) or not result.get("evidence_layers"):
            errors.append("evidence_model_module.evidence_layers must be a non-empty list")
    if module_name == "subjective_supervisor_module":
        if not isinstance(result.get("must_have_concepts"), list) or not result.get("must_have_concepts"):
            errors.append("subjective_supervisor_module.must_have_concepts must be a non-empty list")
        if not isinstance(result.get("must_not_become"), list) or not result.get("must_not_become"):
            errors.append("subjective_supervisor_module.must_not_become must be a non-empty list")
        requirement_contract = result.get("requirement_contract")
        if not isinstance(requirement_contract, dict) or not isinstance(requirement_contract.get("concepts"), list) or not requirement_contract.get("concepts"):
            errors.append("subjective_supervisor_module.requirement_contract.concepts must be a non-empty list")
        else:
            concept_ids = []
            for index, concept in enumerate(requirement_contract.get("concepts") or []):
                if not isinstance(concept, dict):
                    errors.append(
                        f"subjective_supervisor_module.requirement_contract.concepts[{index}] must be an object"
                    )
                    continue
                concept_id = str(concept.get("concept_id") or "").strip()
                concept_ids.append(concept_id)
                if not concept_id:
                    errors.append(
                        f"subjective_supervisor_module.requirement_contract.concepts[{index}] missing concept_id"
                    )
                if not list(concept.get("source_requirement_ids") or []):
                    errors.append(
                        f"subjective_supervisor_module.requirement_contract.concepts[{index}] missing source_requirement_ids"
                    )
                if not str(concept.get("why_needed") or "").strip():
                    errors.append(
                        f"subjective_supervisor_module.requirement_contract.concepts[{index}] missing why_needed"
                    )
            duplicate_concept_ids = sorted(
                concept_id
                for concept_id in set(concept_ids)
                if concept_id and concept_ids.count(concept_id) > 1
            )
            if duplicate_concept_ids:
                errors.append(
                    "subjective_supervisor_module.requirement_contract contains duplicate concept_ids: "
                    + ",".join(duplicate_concept_ids)
                )
        if not isinstance(result.get("entity_registry"), list) or not result.get("entity_registry"):
            errors.append("subjective_supervisor_module.entity_registry must be a non-empty list")
    if module_name == "field_planning_module":
        errors.extend(validate_definitions(result))
        if not isinstance(result.get("field_groups"), list) or not result.get("field_groups"):
            errors.append("field_planning_module.field_groups must be a non-empty list")
        if "reference_field_audit" in result and not isinstance(result.get("reference_field_audit"), dict):
            errors.append("field_planning_module.reference_field_audit must be an object")
    if module_name == "supervisor_module":
        if not isinstance(result.get("enable_figure_classification"), bool):
            errors.append("supervisor_module.enable_figure_classification must be a boolean")
        if result.get("routing_decision") not in {"figure_classification_path", "direct_schema_path"}:
            errors.append(
                "supervisor_module.routing_decision must be figure_classification_path or direct_schema_path"
            )
    if module_name == "figure_classification_module":
        if not isinstance(result.get("enable_figure_classification"), bool):
            errors.append("figure_classification_module.enable_figure_classification must be a boolean")
        if result.get("routing_status") not in {"active", "skipped"}:
            errors.append("figure_classification_module.routing_status must be active or skipped")
        if not isinstance(result.get("classification_strategy"), list):
            errors.append("figure_classification_module.classification_strategy must be a list")
        if not isinstance(result.get("section_figure_plan"), list):
            errors.append("figure_classification_module.section_figure_plan must be a list")
        if result.get("enable_figure_classification") and result.get("routing_status") != "active":
            errors.append("figure_classification_module must be active when enable_figure_classification is true")
        if (not result.get("enable_figure_classification")) and result.get("routing_status") != "skipped":
            errors.append("figure_classification_module must be skipped when enable_figure_classification is false")
    if module_name == "schema_design_module":
        if not isinstance(result.get("field_registry"), list) or not result.get("field_registry"):
            errors.append("schema_design_module.field_registry must be a non-empty list")
        else:
            for index, item in enumerate(result.get("field_registry", [])):
                if not isinstance(item, dict):
                    continue
                if not str(item.get("reason") or "").strip():
                    errors.append(
                        f"schema_design_module.field_registry[{index}] missing field utility reason"
                    )
            field_paths = [
                str(item.get("field_path", ""))
                for item in result.get("field_registry", [])
                if isinstance(item, dict)
            ]
            top_keys = [
                str(item.get("key", ""))
                for item in result.get("top_level_keys", [])
                if isinstance(item, dict)
            ]
            declared_entity_roots = {
                key for key in top_keys if is_safe_entity_owner_key(key)
            }
            unapproved_roots = sorted(
                {
                    path.split(".", 1)[0]
                    for path in [*field_paths, *top_keys]
                    if path.split(".", 1)[0]
                    and path.split(".", 1)[0] not in MATERIAL_SCHEMA_ROOTS
                    and path.split(".", 1)[0] not in SUPERVISOR_ALLOWED_EXTENSION_ROOTS
                    and path.split(".", 1)[0] not in declared_entity_roots
                }
            )
            material_hits = [
                path
                for path in field_paths
                if path.split(".", 1)[0] in MATERIAL_SCHEMA_ROOTS
                or path.startswith("material_info.section")
                or path.split(".", 1)[0] in declared_entity_roots
            ]
            if unapproved_roots:
                errors.append(
                    "schema_design_module contains schema roots not approved by the Step 8 section plan "
                    f"{unapproved_roots}; the supervisor/checker must discard or justify them"
                )
            if not material_hits:
                errors.append(
                    "schema_design_module has no materials-database fields under "
                    "the declared material, publication, sample, observation, or evidence owners"
                )
    if module_name == "specialization_critic_module":
        if result.get("specialization_status") not in {"pass", "needs_redesign"}:
            errors.append("specialization_critic_module.specialization_status must be pass or needs_redesign")
        if not isinstance(result.get("is_generic"), bool):
            errors.append("specialization_critic_module.is_generic must be a boolean")
        if not isinstance(result.get("redo_needed"), bool):
            errors.append("specialization_critic_module.redo_needed must be a boolean")
        patch_operations = result.get("patch_operations")
        if not isinstance(patch_operations, list):
            errors.append("specialization_critic_module.patch_operations must be a list")
        elif result.get("redo_needed") and not patch_operations:
            errors.append(
                "specialization_critic_module.patch_operations must be non-empty when redo_needed is true"
            )
        utility_audit = result.get("field_utility_audit")
        utility_issue_keys = (
            "unsupported_fields",
            "redundant_fields",
            "alias_or_unit_variant_fields",
            "derivable_duplicate_fields",
        )
        if not isinstance(utility_audit, dict):
            errors.append("specialization_critic_module.field_utility_audit must be an object")
        else:
            reviewed_count = utility_audit.get("reviewed_field_count")
            if not isinstance(reviewed_count, int) or isinstance(reviewed_count, bool) or reviewed_count < 0:
                errors.append(
                    "specialization_critic_module.field_utility_audit.reviewed_field_count must be a non-negative integer"
                )
            if utility_audit.get("decision") not in {"pass", "needs_pruning"}:
                errors.append(
                    "specialization_critic_module.field_utility_audit.decision must be pass or needs_pruning"
                )
            utility_issues = []
            for key in utility_issue_keys:
                values = utility_audit.get(key)
                if not isinstance(values, list):
                    errors.append(
                        f"specialization_critic_module.field_utility_audit.{key} must be a list"
                    )
                else:
                    utility_issues.extend(str(item) for item in values if str(item).strip())
            if not str(utility_audit.get("rationale") or "").strip():
                errors.append(
                    "specialization_critic_module.field_utility_audit.rationale must be non-empty"
                )
            if utility_issues and utility_audit.get("decision") != "needs_pruning":
                errors.append(
                    "specialization_critic_module.field_utility_audit with issues must use needs_pruning"
                )
            if utility_issues and not result.get("redo_needed"):
                errors.append(
                    "specialization_critic_module utility issues require redo_needed=true"
                )
            if not utility_issues and utility_audit.get("decision") == "needs_pruning":
                errors.append(
                    "specialization_critic_module needs_pruning requires at least one utility issue"
                )
    return errors


CRITIC_UTILITY_ISSUE_KEYS = (
    "unsupported_fields",
    "redundant_fields",
    "alias_or_unit_variant_fields",
    "derivable_duplicate_fields",
)


def validate_critic_field_utility(critic_result, schema_result):
    """Validate that the critic reviewed this exact schema and made executable findings."""
    if not isinstance(critic_result, dict) or not isinstance(schema_result, dict):
        return ["specialization_critic utility validation requires critic and schema objects"]
    registry = [
        item
        for item in schema_result.get("field_registry", []) or []
        if isinstance(item, dict) and str(item.get("field_path") or "").strip()
    ]
    field_paths = [str(item["field_path"]) for item in registry]
    field_path_set = set(field_paths)
    audit = critic_result.get("field_utility_audit")
    if not isinstance(audit, dict):
        return ["specialization_critic did not return field_utility_audit"]

    errors = []
    if audit.get("reviewed_field_count") != len(registry):
        errors.append(
            "specialization_critic field_utility_audit reviewed_field_count does not match "
            f"the current schema ({audit.get('reviewed_field_count')} != {len(registry)})"
        )

    issue_paths = []
    for key in CRITIC_UTILITY_ISSUE_KEYS:
        values = audit.get(key) or []
        issue_paths.extend(str(item) for item in values if str(item).strip())
    unknown_issue_paths = sorted(set(issue_paths) - field_path_set)
    if unknown_issue_paths:
        errors.append(
            "specialization_critic field_utility_audit references unknown fields: "
            + ",".join(unknown_issue_paths)
        )

    executable_paths = {
        str(operation.get("field_path") or "")
        for operation in critic_result.get("patch_operations", []) or []
        if isinstance(operation, dict)
        and operation.get("op") in {"remove_field", "update_field", "move_field"}
    }
    uncovered_issues = sorted(set(issue_paths) - executable_paths)
    if uncovered_issues:
        errors.append(
            "specialization_critic utility findings lack executable patch operations: "
            + ",".join(uncovered_issues)
        )
    return errors


def summarize_field_paths(field_registry, owner_id, limit=6):
    matches = []
    for field in field_registry or []:
        if isinstance(field, dict) and field.get("section_id") == owner_id and field.get("field_path"):
            matches.append(field["field_path"])
    return matches[:limit]


def build_agent_flow_trace(inputs, modules, result, validation_errors):
    if not isinstance(result, dict):
        return {
            "user_input": inputs,
            "section_design_agent": {
                "subjective_supervisor_module": modules.get("subjective_supervisor_module", {"status": "unavailable"}),
                "locating_module": modules.get("locating_module", {"status": "unavailable"}),
                "mechanism_requirement_module": modules.get("mechanism_requirement_module", {"status": "unavailable"}),
                "query_semantics_module": modules.get("query_semantics_module", {"status": "unavailable"}),
                "evidence_model_module": modules.get("evidence_model_module", {"status": "unavailable"}),
                "topic_adaptation_module": modules.get("topic_adaptation_module", {"status": "unavailable"}),
                "section_partition_module": modules.get("section_partition_module", {"status": "unavailable"}),
                "supervisor_module": modules.get("supervisor_module", {"status": "unavailable"}),
                "figure_classification_module": modules.get("figure_classification_module", {"status": "unavailable"}),
                "field_planning_module": modules.get("field_planning_module", {"status": "unavailable"}),
                "schema_design_module": modules.get("schema_design_module", {"status": "unavailable"}),
                "specialization_critic_module": modules.get("specialization_critic_module", {"status": "unavailable"}),
                "aggregation": modules.get("aggregation", {"status": "unavailable"}),
            },
            "validator": {"status": "failed", "errors": validation_errors},
            "redo_agent": {"triggered": bool(validation_errors), "reason": validation_errors},
        }

    section_design = result.get("section_design", {})
    schema_definition = result.get("schema_definition", {})
    quality_check = result.get("quality_check", {})
    field_registry = schema_definition.get("field_registry", [])
    subjective_result = modules.get("subjective_supervisor_module", {})
    supervisor_result = modules.get("supervisor_module", {})
    figure_result = modules.get("figure_classification_module", {})
    critic_result = modules.get("specialization_critic_module", {})

    section_samples = []
    for section in section_design.get("core_sections", [])[:3]:
        if isinstance(section, dict):
            section_samples.append(
                {
                    "section_id": section.get("section_id"),
                    "section_name": section.get("section_name"),
                    "sample_fields": summarize_field_paths(field_registry, section.get("section_id")),
                }
            )

    figure_summary = {"text": 0, "table": 0, "figure": 0, "figure_classified": 0}
    for field in field_registry or []:
        if not isinstance(field, dict):
            continue
        source_basis = field.get("source_basis") or []
        if "text" in source_basis:
            figure_summary["text"] += 1
        if "table" in source_basis:
            figure_summary["table"] += 1
        if "figure" in source_basis:
            figure_summary["figure"] += 1
        figure_constraint = field.get("figure_constraint") or {}
        if figure_constraint.get("uses_figure_classification"):
            figure_summary["figure_classified"] += 1

    return {
        "user_input": inputs,
        "section_design_agent": {
            "subjective_supervisor_module": {
                **subjective_result,
                "must_have_count": len(subjective_result.get("must_have_concepts", [])),
            },
            "locating_module": modules.get("locating_module", {}),
            "mechanism_requirement_module": modules.get("mechanism_requirement_module", {}),
            "query_semantics_module": modules.get("query_semantics_module", {}),
            "evidence_model_module": modules.get("evidence_model_module", {}),
            "topic_adaptation_module": modules.get("topic_adaptation_module", {}),
            "section_partition_module": modules.get("section_partition_module", {}),
            "field_planning_module": modules.get("field_planning_module", {}),
            "supervisor_review": {
                **supervisor_result,
                "decision": (
                    "figure_classification_path"
                    if supervisor_result.get("enable_figure_classification")
                    else "direct_schema_path"
                ),
            },
            "risk_assessment": {
                "summary": supervisor_result.get("risk_assessment"),
                "risk_signals": supervisor_result.get("risk_signals", []),
                "trigger_sections": supervisor_result.get("trigger_sections", []),
                "expected_figure_fields": supervisor_result.get("expected_figure_fields", []),
            },
            "figure_classification_branch": {
                **figure_result,
                "planned_sections": [item.get("section_id") for item in figure_result.get("section_figure_plan", [])],
            },
            "schema_design_module": {
                **modules.get("schema_design_module", {}),
                "section_field_samples": section_samples,
                "figure_basis_summary": figure_summary,
            },
            "specialization_critic": critic_result,
            "aggregation": {
                **modules.get("aggregation", {}),
                "assembled_result_keys": list(result.keys()),
                "coverage_check": quality_check.get("coverage_check", []),
            },
        },
        "validator": {"status": "passed" if not validation_errors else "failed", "errors": validation_errors},
        "redo_agent": {
            "triggered": bool(validation_errors),
            "reason": validation_errors,
            "returns_to": "supervisor_agent",
        },
    }


def validate_result(result):
    errors = []
    required_top_keys = [
        "database_positioning",
        "section_design",
        "schema_definition",
        "quality_check",
    ]
    for key in required_top_keys:
        if key not in result:
            errors.append(f"Missing top-level key: {key}")
    if errors:
        return errors

    task_contract = result.get("task_contract") or {}
    if task_contract.get("task_type") != "automated_materials_database_construction":
        errors.append("task_contract.task_type must remain automated_materials_database_construction")
    if task_contract.get("source_scope") != "scientific_literature_only":
        errors.append("task_contract.source_scope must remain scientific_literature_only")

    section_design = result.get("section_design", {})
    core_sections = section_design.get("core_sections")
    non_core_sections = section_design.get("non_core_sections")
    if not isinstance(core_sections, list) or not core_sections:
        errors.append("core_sections must be a non-empty list")
    if not isinstance(non_core_sections, list):
        errors.append("non_core_sections must be a list")

    schema_definition = result.get("schema_definition", {})
    top_level_keys = schema_definition.get("top_level_keys")
    field_registry = schema_definition.get("field_registry")
    if not isinstance(top_level_keys, list) or not top_level_keys:
        errors.append("top_level_keys must be a non-empty list")
    if not isinstance(field_registry, list) or not field_registry:
        errors.append("field_registry must be a non-empty list")

    valid_basis = {"text", "table", "figure"}
    has_core_field = False
    has_text_field = False
    has_figure_or_table_field = False
    has_figure_constraint = False
    section_ids = set()
    top_level_key_names = set()
    core_section_ids = set()

    for section in core_sections or []:
        if isinstance(section, dict) and section.get("section_id"):
            section_ids.add(section["section_id"])
            core_section_ids.add(section["section_id"])
    for section in non_core_sections or []:
        if isinstance(section, dict) and section.get("section_id"):
            section_ids.add(section["section_id"])
    for item in top_level_keys or []:
        if isinstance(item, dict) and item.get("key"):
            top_level_key_names.add(item["key"])

    for field in field_registry or []:
        if not isinstance(field, dict):
            errors.append("Each field_registry item must be an object")
            continue
        field_path = field.get("field_path")
        section_id = field.get("section_id")
        source_basis = field.get("source_basis")
        if not field_path:
            errors.append("field_registry contains an item without field_path")
        if not str(field.get("field_rule_id") or "").startswith("field."):
            errors.append(f"field {field_path or '<unknown>'} missing stable field_rule_id")
        if not str(field.get("field_rule_version") or "").startswith("sha256:"):
            errors.append(f"field {field_path or '<unknown>'} missing field_rule_version")
        if not str(field.get("inclusion_rule") or "").strip():
            errors.append(f"field {field_path or '<unknown>'} missing inclusion_rule")
        if not str(field.get("absence_rule") or "").strip():
            errors.append(f"field {field_path or '<unknown>'} missing absence_rule")
        if not isinstance(field.get("evidence_requirements"), dict):
            errors.append(f"field {field_path or '<unknown>'} missing evidence_requirements")
        if not isinstance(field.get("relation_constraints"), dict):
            errors.append(f"field {field_path or '<unknown>'} missing relation_constraints")
        if not section_id:
            errors.append(f"field_registry item {field_path or '<unknown>'} missing section_id")
        elif section_id in core_section_ids:
            has_core_field = True
        inferred_top_level_key = field_path.split(".", 1)[0] if field_path else None
        valid_field_owner_ids = section_ids | top_level_key_names
        if section_id and section_id not in valid_field_owner_ids:
            errors.append(f"field {field_path} references unknown section_id {section_id}")
        if section_id in top_level_key_names and inferred_top_level_key != section_id:
            errors.append(
                f"field {field_path} uses top-level owner {section_id} but path starts with {inferred_top_level_key}"
            )
        if not isinstance(source_basis, list) or not source_basis:
            errors.append(f"field {field_path} must have a non-empty source_basis list")
        else:
            invalid = [item for item in source_basis if item not in valid_basis]
            if invalid:
                errors.append(f"field {field_path} has invalid source_basis values: {invalid}")
            if "text" in source_basis:
                has_text_field = True
            if "table" in source_basis or "figure" in source_basis:
                has_figure_or_table_field = True
        figure_constraint = field.get("figure_constraint")
        if "figure" in (source_basis or []):
            if not isinstance(figure_constraint, dict):
                errors.append(f"field {field_path} is figure-based but missing figure_constraint")
            else:
                if not isinstance(figure_constraint.get("uses_figure_classification"), bool):
                    errors.append(f"field {field_path} figure_constraint.uses_figure_classification must be boolean")
                allowed_sections = figure_constraint.get("allowed_sections")
                allowed_categories = figure_constraint.get("allowed_figure_categories")
                why_needed = figure_constraint.get("why_needed")
                if not isinstance(allowed_sections, list) or not allowed_sections:
                    errors.append(f"field {field_path} figure_constraint.allowed_sections must be a non-empty list")
                if not isinstance(allowed_categories, list) or not allowed_categories:
                    errors.append(
                        f"field {field_path} figure_constraint.allowed_figure_categories must be a non-empty list"
                    )
                if not isinstance(why_needed, str) or not why_needed.strip():
                    errors.append(f"field {field_path} figure_constraint.why_needed must be a non-empty string")
                has_figure_constraint = True

    quality_check = result.get("quality_check", {})
    adjustments = quality_check.get("topic_specific_adjustments")
    coverage = quality_check.get("coverage_check")
    if not isinstance(adjustments, list) or not adjustments:
        errors.append("quality_check.topic_specific_adjustments must be a non-empty list")
    if not isinstance(coverage, list) or not coverage:
        errors.append("quality_check.coverage_check must be a non-empty list")

    if not has_core_field:
        errors.append("No field_registry item is mapped to a core section")
    if not has_text_field:
        errors.append("Schema must include at least one text-based field")
    if len(core_sections or []) < 2:
        errors.append("At least two core sections are required for a useful design")
    return errors


OBJECT_KIND_REQUIRED_SLOTS = {
    "measurement": {"value", "conditions", "entity_ref", "source_type", "evidence", "extraction_confidence"},
    "classification": {"label", "assignment_basis", "source_type", "evidence", "extraction_confidence"},
    "entity_descriptor": {"identity", "evidence"},
    "process": {"method", "conditions", "entity_ref", "evidence"},
    "evidence_collection": {"evidence"},
    "freeform_object": set(),
}


def stable_concept_id(value):
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return text[:96] or "concept"


def is_non_extractable_requirement(label):
    """Return True for workflow constraints that must not become data fields."""
    lowered = re.sub(r"\s+", " ", str(label or "").strip().lower())
    if not lowered:
        return True
    if any(token in lowered for token in ("schema", "database design")) and any(
        token in lowered
        for token in ("treat this", "general", "generic", "derived", "not a", "not an")
    ):
        return True
    patterns = (
        r"\b(?:schema|database design)\b.*\b(?:not|derived|general|generic)\b",
        r"\b(?:not|rather than)\b.*\b(?:schema|database design)\b",
        r"\b(?:current|test)\s+(?:documents?|corpus)\b",
        r"\bconverted from\b",
        r"\b(?:without|no)\s+(?:figure images?|vlm|ocr)\b",
        r"\bmust remain unavailable\b",
        r"^(?:the\s+)?(?:schema|system|model|pipeline|extractor|database|users?|records?)\b.*\b(?:must|should|can|cannot|needs?|permits?|remains?|does not|never)\b",
        r"^(?:every|all)\b.*\b(?:references?|includes?|exists?|stored|queryable|validated)\b",
        r"\b(?:blocking issues?|recommendations?|acceptance criteria|issue type|forbidden information used|verdict)\b",
        r"\b(?:support reliable comparison|retrieval across|derive the exact field inventory)\b",
        r"^(?:literature|rationale|recommendation)\b",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def parse_structured_human_advice(human_advice):
    """Return a structured expert protocol object, or None for ordinary prose."""
    text = str(human_advice or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def explicit_human_advice_schema_concepts(human_advice):
    """Return only field concepts explicitly declared in a structured expert protocol."""
    payload = parse_structured_human_advice(human_advice)
    if payload is None:
        return []
    raw_concepts = payload.get("schema_concepts")
    if not isinstance(raw_concepts, list):
        return []
    concepts = []
    for raw in raw_concepts:
        item = dict(raw) if isinstance(raw, dict) else {"label": str(raw)}
        label = str(item.get("label") or item.get("name") or item.get("concept_id") or "").strip()
        if not label or is_non_extractable_requirement(label):
            continue
        item["label"] = label
        concepts.append(item)
    return concepts


def explicit_human_advice_counterfactual_queries(human_advice):
    """Return validated CARE queries declared by a structured expert protocol."""
    payload = parse_structured_human_advice(human_advice)
    if payload is None or not payload.get("care_counterfactual_enabled"):
        return []
    raw_queries = payload.get("counterfactual_queries")
    if not isinstance(raw_queries, list):
        return []
    queries = []
    seen_ids = set()
    for raw in raw_queries:
        if not isinstance(raw, dict):
            continue
        query_id = stable_concept_id(raw.get("query_id") or "")
        query = str(raw.get("query") or "").strip()
        if not query_id or not query or query_id in seen_ids:
            continue
        distinctions = [
            str(item).strip()
            for item in raw.get("required_distinctions") or []
            if str(item).strip()
        ]
        queries.append(
            {
                "query_id": query_id,
                "category": str(raw.get("category") or "schema").strip(),
                "query": query,
                "required_distinctions": list(dict.fromkeys(distinctions)),
                "failure_exposed": str(raw.get("failure_exposed") or "").strip(),
            }
        )
        seen_ids.add(query_id)
    return queries


def atomize_query_requirement(requirement):
    text = str(requirement or "").strip()
    text = re.sub(
        r"^(?:search\s+by|compare|retrieve|record|cover|include|separate|extract|identify|represent)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    parts = re.split(
        r"(?:\s*(?:,|;|/|\band\b)\s*|(?<=[.!?])\s+)",
        text,
        flags=re.IGNORECASE,
    )
    labels = []
    for part in parts:
        label = re.sub(r"^(?:and|or)\s+", "", part.strip(), flags=re.IGNORECASE)
        label = re.sub(
            r"^(?:search\s+by|compare|retrieve|record|cover|include|separate|extract|identify|represent)\s+",
            "",
            label,
            flags=re.IGNORECASE,
        )
        label = re.sub(r"\s+", " ", label).strip(" .:-")
        if label and label.lower() not in {"and", "or"}:
            labels.append(label)
    return list(dict.fromkeys(labels))


def infer_requirement_entity(label):
    lowered = str(label or "").lower()
    if any(token in lowered for token in ("device", "junction", "electrode")):
        return "device", "device_info"
    if any(token in lowered for token in ("interface", "heterostructure", "boundary")):
        return "interface", "interface_info"
    if any(token in lowered for token in ("reaction", "catalysis")):
        return "reaction", "reaction_info"
    return "material", "material_info"


def append_atomic_requirement_concepts(
    concepts,
    seen_ids,
    requirement_id,
    requirement,
    skip_semantic_duplicates=False,
):
    for label in atomize_query_requirement(requirement):
        if is_non_extractable_requirement(label):
            continue
        concept_id = stable_concept_id(label)
        if concept_id in seen_ids:
            existing = next(item for item in concepts if item["concept_id"] == concept_id)
            existing["source_requirement_ids"] = list(
                dict.fromkeys([*existing.get("source_requirement_ids", []), requirement_id])
            )
            continue
        if skip_semantic_duplicates:
            stop_tokens = {"a", "an", "and", "include", "keep", "of", "record", "the", "with"}
            label_tokens = set(stable_concept_id(label).split("_")) - stop_tokens
            ranked = []
            for existing in concepts:
                existing_tokens = set(
                    stable_concept_id(
                        f"{existing.get('concept_id', '')} {existing.get('label', '')}"
                    ).split("_")
                ) - stop_tokens
                ranked.append((len(label_tokens & existing_tokens), existing))
            overlap, existing = max(ranked, default=(0, None), key=lambda item: item[0])
            if existing is not None and overlap >= min(2, max(len(label_tokens), 1)):
                existing["source_requirement_ids"] = list(
                    dict.fromkeys(
                        [*existing.get("source_requirement_ids", []), requirement_id]
                    )
                )
                continue
        entity_id, owner_key = infer_requirement_entity(label)
        lowered = label.lower()
        if any(token in lowered for token in ("synthesis", "fabrication", "growth", "processing")):
            object_kind = "process"
        elif any(token in lowered for token in ("evidence", "figure", "table", "spectrum", "curve")):
            object_kind = "evidence_collection"
        elif re.fullmatch(r"[a-z0-9]{1,2}\s*[-_]\s*[a-z0-9]{1,2}", lowered):
            object_kind = "evidence_collection"
        elif any(token in lowered for token in ("label", "classification", "symmetry", "polarity", "state")):
            object_kind = "classification"
        elif any(
            token in lowered
            for token in (
                "temperature",
                "current",
                "field",
                "pressure",
                "range",
                "depth",
                "length",
                "response",
                "efficiency",
                "resistance",
                "resistivity",
                "susceptibility",
            )
        ):
            object_kind = "measurement"
        else:
            object_kind = "scalar"
        concepts.append(
            {
                "concept_id": concept_id,
                "label": label,
                "required": True,
                "entity_id": entity_id,
                "owner_key": owner_key,
                "object_kind": object_kind,
                "condition_requirements": [],
                "evidence_types": ["text"],
                "source_requirement_ids": [requirement_id],
            }
        )
        seen_ids.add(concept_id)


def normalize_requirement_contract(shared_context, subjective_result):
    raw_contract = subjective_result.get("requirement_contract") if isinstance(subjective_result, dict) else None
    raw_contract = raw_contract if isinstance(raw_contract, dict) else {}
    raw_concepts = raw_contract.get("concepts")
    if not isinstance(raw_concepts, list) or not raw_concepts:
        raw_concepts = subjective_result.get("must_have_concepts", []) if isinstance(subjective_result, dict) else []

    query_requirements = list(shared_context.get("query_requirements", []) or [])
    human_advice = str(shared_context.get("human_advice") or "").strip()
    source_tokens = set(
        stable_concept_id(" ".join([*map(str, query_requirements), human_advice])).split("_")
    )
    concepts = []
    seen_ids = set()
    for raw in raw_concepts or []:
        item = dict(raw) if isinstance(raw, dict) else {"label": str(raw)}
        label = str(item.get("label") or item.get("name") or item.get("concept_id") or "").strip()
        concept_id = stable_concept_id(item.get("concept_id") or label)
        if not label or concept_id in seen_ids or is_non_extractable_requirement(label):
            continue
        item_sources = list(item.get("source_requirement_ids") or [])
        concept_tokens = set(stable_concept_id(f"{concept_id} {label}").split("_"))
        if not item_sources and not (concept_tokens & source_tokens):
            continue
        seen_ids.add(concept_id)
        concepts.append(
            {
                "concept_id": concept_id,
                "label": label,
                "required": bool(item.get("required", True)),
                "entity_id": str(item.get("entity_id") or "material"),
                "owner_key": str(item.get("owner_key") or "material_info"),
                "object_kind": str(item.get("object_kind") or "scalar"),
                "condition_requirements": list(item.get("condition_requirements") or []),
                "evidence_types": list(item.get("evidence_types") or ["text"]),
                "data_type": str(item.get("data_type") or item.get("value_kind") or ""),
                "definition": str(item.get("definition") or item.get("description") or ""),
                "evidence_expectation": str(item.get("evidence_expectation") or ""),
                "why_needed": str(item.get("why_needed") or ""),
                "object_contract": deepcopy(item.get("object_contract"))
                if isinstance(item.get("object_contract"), dict)
                else None,
                "source_requirement_ids": item_sources,
            }
        )

    structured_concept_seeding_only = bool(
        shared_context.get("structured_concept_seeding_only")
    )
    care_requirement_ids = set(
        map(str, shared_context.get("care_counterfactual_requirement_ids") or [])
    )
    structured_human_advice = parse_structured_human_advice(human_advice)
    for index, requirement in enumerate(query_requirements, start=1):
        requirement_id = f"query_{index}"
        if not structured_concept_seeding_only or requirement_id in care_requirement_ids:
            append_atomic_requirement_concepts(
                concepts,
                seen_ids,
                requirement_id,
                requirement,
            )
    if not structured_concept_seeding_only:
        if human_advice and structured_human_advice is None:
            append_atomic_requirement_concepts(
                concepts,
                seen_ids,
                "human_advice",
                human_advice,
                skip_semantic_duplicates=True,
            )
    for explicit_concept in explicit_human_advice_schema_concepts(human_advice):
        label = explicit_concept["label"]
        concept_id = stable_concept_id(explicit_concept.get("concept_id") or label)
        if concept_id in seen_ids:
            existing = next(item for item in concepts if item["concept_id"] == concept_id)
            existing["source_requirement_ids"] = list(
                dict.fromkeys([*existing.get("source_requirement_ids", []), "human_advice"])
            )
            continue
        entity_id, owner_key = infer_requirement_entity(label)
        concepts.append(
            {
                "concept_id": concept_id,
                "label": label,
                "required": bool(explicit_concept.get("required", True)),
                "entity_id": str(explicit_concept.get("entity_id") or entity_id),
                "owner_key": str(explicit_concept.get("owner_key") or owner_key),
                "object_kind": str(explicit_concept.get("object_kind") or "scalar"),
                "condition_requirements": list(explicit_concept.get("condition_requirements") or []),
                "evidence_types": list(explicit_concept.get("evidence_types") or ["text"]),
                "data_type": str(explicit_concept.get("data_type") or ""),
                "definition": str(explicit_concept.get("definition") or ""),
                "source_requirement_ids": ["human_advice"],
            }
        )
        seen_ids.add(concept_id)

    source_requirements = [
        {
            "requirement_id": f"query_{index}",
            "source": (
                "care_counterfactual_query"
                if f"query_{index}" in care_requirement_ids
                else "query_requirement"
            ),
            "text": str(requirement),
        }
        for index, requirement in enumerate(query_requirements, start=1)
    ]
    if human_advice:
        source_requirements.append(
            {"requirement_id": "human_advice", "source": "human_expert", "text": human_advice}
        )

    traced_ids = {
        str(requirement_id)
        for concept in concepts
        for requirement_id in concept.get("source_requirement_ids", [])
        if requirement_id
    }
    for source_requirement in source_requirements:
        requirement_id = source_requirement["requirement_id"]
        if requirement_id in traced_ids or not concepts:
            continue
        requirement_tokens = set(stable_concept_id(source_requirement.get("text")).split("_"))
        best_concept = max(
            concepts,
            key=lambda concept: len(
                requirement_tokens
                & set(
                    stable_concept_id(
                        f"{concept.get('concept_id', '')} {concept.get('label', '')}"
                    ).split("_")
                )
            ),
        )
        best_concept["source_requirement_ids"] = list(
            dict.fromkeys([*best_concept.get("source_requirement_ids", []), requirement_id])
        )
        traced_ids.add(requirement_id)

    reference_count = int(shared_context.get("reference_paper_count", 0) or 0)
    broad_goal = any(
        token in str(shared_context.get("database_goal", "")).lower()
        for token in ("general", "broad", "across", "covering", "通用", "广泛", "覆盖")
    )
    return {
        "concepts": concepts,
        "source_requirements": source_requirements,
        "reference_representativeness": {
            "reference_count": reference_count,
            "broad_goal": broad_goal,
            "risk": "broad_goal_with_single_reference" if broad_goal and reference_count <= 1 else "none",
        },
    }


def build_step8_protocol_messages(inputs, modules, result, validation_errors, module_errors=None):
    """Represent legacy Step8 module handoffs as supervisor-visible protocol messages."""
    messages = []
    module_errors = module_errors or {}
    for module_name, module_output in (modules or {}).items():
        errors = module_errors.get(module_name) or []
        messages.append(
            agent_protocol.make_message(
                sender=module_name,
                receiver="step8_supervisor",
                phase=module_name,
                status="failed" if errors else "completed",
                task_contract=(inputs or {}).get("task_contract"),
                payload_refs={"input": "inputs", "output": f"module_outputs.{module_name}"},
                decision={"accepted_by_supervisor": not bool(errors)},
                requested_actions=[{"action": "repair_module", "errors": errors}] if errors else [],
                produced_artifacts=[{"ref": f"module_outputs.{module_name}", "present": module_output is not None}],
                next_route="step8_supervisor",
                evidence=errors,
            )
        )
    messages.append(
        agent_protocol.make_message(
            sender="step8_supervisor",
            receiver="step9_supervisor" if result is not None and not validation_errors else "step8_repair_or_human_gate",
            phase="step8_final_supervision",
            status="accepted" if result is not None and not validation_errors else "needs_review",
            task_contract=(inputs or {}).get("task_contract"),
            payload_refs={"schema": "result", "validation_errors": "validation_errors"},
            decision={
                "action": "release_to_step9" if result is not None and not validation_errors else "hold_for_repair_or_review",
                "validation_error_count": len(validation_errors or []),
            },
            requested_actions=list(validation_errors or []),
            produced_artifacts=[{"ref": "result", "present": result is not None}],
            next_route="step9_workflow_plan" if result is not None and not validation_errors else "step8_repair_or_human_gate",
            evidence=list(validation_errors or []),
        )
    )
    return messages


def normalize_entity_registry(subjective_result, requirement_contract):
    raw_entities = subjective_result.get("entity_registry", []) if isinstance(subjective_result, dict) else []
    concept_entities = {
        item.get("entity_id")
        for item in requirement_contract.get("concepts", [])
        if item.get("entity_id")
    }
    source_tokens = set(
        stable_concept_id(
            " ".join(
                str(item.get("text") or "")
                for item in requirement_contract.get("source_requirements", []) or []
            )
        ).split("_")
    )
    entity_stop_tokens = {
        "collection",
        "data",
        "entity",
        "info",
        "record",
        "records",
    }
    source_tokens -= entity_stop_tokens
    entities = []
    seen = set()
    preserve_structured_registry = bool(
        isinstance(subjective_result, dict)
        and subjective_result.get("structured_entity_registry")
    )
    for raw in raw_entities or []:
        if not isinstance(raw, dict):
            continue
        entity_id = stable_concept_id(raw.get("entity_id") or raw.get("label"))
        if entity_id in seen:
            continue
        entity_tokens = set(
            stable_concept_id(f"{entity_id} {raw.get('label') or ''}").split("_")
        ) - entity_stop_tokens
        if (
            not preserve_structured_registry
            and
            entity_id not in concept_entities
            and entity_id not in {"material", "sample", "paper"}
            and not (entity_tokens & source_tokens)
        ):
            continue
        seen.add(entity_id)
        owner_key = str(
            raw.get("owner_key")
            or ("material_info" if entity_id == "material" else f"{entity_id}_info")
        )
        independent_owner = bool(raw.get("independent_owner", entity_id != "material"))
        if entity_id in {"evidence", "provenance", "source"}:
            owner_key = "material_info"
            independent_owner = False
        if (
            owner_key not in (MATERIAL_SCHEMA_ROOTS | SUPERVISOR_ALLOWED_EXTENSION_ROOTS)
            and not is_safe_entity_owner_key(owner_key)
        ):
            owner_key = "material_info" if entity_id == "material" else f"{entity_id}_info"
        entities.append(
            {
                "entity_id": entity_id,
                "label": str(raw.get("label") or entity_id),
                "required": bool(raw.get("required", True)),
                "owner_key": owner_key,
                "independent_owner": independent_owner,
                "primary_key": str(raw.get("primary_key") or ""),
                "record_type": str(raw.get("record_type") or raw.get("type") or ""),
                "role": str(raw.get("role") or ""),
                "independent_query_role": bool(
                    raw.get("independent_query_role", independent_owner)
                ),
            }
        )
    for entity_id in sorted(concept_entities - seen):
        entities.append(
            {
                "entity_id": entity_id,
                "label": entity_id.replace("_", " ").title(),
                "required": True,
                "owner_key": "material_info" if entity_id == "material" else f"{entity_id}_info",
                "independent_owner": entity_id != "material",
            }
        )
    return entities


def object_contract_missing_slots(field):
    if "object" not in str(field.get("data_type") or "").lower():
        return set()
    contract = field.get("object_contract")
    if not isinstance(contract, dict):
        return {"object_contract"}
    object_kind = str(contract.get("object_kind") or field.get("object_kind") or "").strip().lower()
    required_subfields = {
        str(item).strip() for item in contract.get("required_subfields", []) if str(item).strip()
    }
    if not object_kind or (not required_subfields and object_kind != "freeform_object"):
        return {"object_kind", "required_subfields"}
    expected = OBJECT_KIND_REQUIRED_SLOTS.get(object_kind, {"evidence"})
    present_slots = {str(item).rsplit(".", 1)[-1] for item in required_subfields}
    return expected - present_slots


def normalize_reference_field_path(path):
    return str(path or "").replace("[]", "").strip().strip(".")


def normalize_planned_field_path(path, fallback_section_id="material_info.section1"):
    """Canonicalize one field-plan path without importing the graph implementation."""
    field_path = re.sub(r"\s*\([^)]*\)", "", str(path or "").strip().lstrip("."))
    field_path = field_path.replace("[]", "")
    field_path = field_path.replace(
        "material_info.section1.core_parameter.",
        "material_info.section1.core_parameters.",
    )
    if field_path.startswith("material_info.section5."):
        field_path = "section5." + field_path[len("material_info.section5.") :]
    fallback_section_id = canonical_section_id(
        fallback_section_id or "material_info.section1"
    )
    root = field_path.split(".", 1)[0]
    if root in MATERIAL_SCHEMA_ROOTS or root in SUPERVISOR_ALLOWED_EXTENSION_ROOTS:
        return field_path
    if is_safe_entity_owner_key(root):
        return field_path
    if str(fallback_section_id).startswith("material_info."):
        return f"{fallback_section_id}.{field_path}"
    if fallback_section_id in {"section5", "paper_info"}:
        return f"{fallback_section_id}.{field_path}"
    return field_path


def build_field_plan_traceability(field_plan_result):
    """Build the exact path allow-list justified by the current upstream field plan."""
    groups = (
        field_plan_result.get("field_groups", [])
        if isinstance(field_plan_result, dict)
        else []
    )
    entries = []
    seen = set()
    for group_index, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        fallback_section_id = str(
            group.get("section_id") or "material_info.section1"
        )
        for raw_path in group.get("recommended_fields", []) or []:
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            canonical_path = normalize_planned_field_path(
                raw_path,
                fallback_section_id,
            )
            if not canonical_path or canonical_path in seen:
                continue
            seen.add(canonical_path)
            entries.append(
                {
                    "canonical_field_path": canonical_path,
                    "recommended_field": raw_path,
                    "section_id": canonical_section_id(fallback_section_id),
                    "group_index": group_index,
                    "group_name": str(group.get("group_name") or ""),
                }
            )
    return {
        "policy": (
            "A schema field is plan-supported only when its canonical path exactly "
            "matches a recommended field from the current task-derived field plan."
        ),
        "planned_field_count": len(entries),
        "planned_field_paths": [item["canonical_field_path"] for item in entries],
        "entries": entries,
    }


def normalize_reference_field_audit(reference_contract, raw_audit=None, field_groups=None):
    reference_contract = reference_contract if isinstance(reference_contract, dict) else {}
    catalog = [
        normalize_reference_field_path(item.get("path"))
        for item in reference_contract.get("leaf_fields", []) or []
        if isinstance(item, dict) and item.get("path")
    ]
    catalog = list(dict.fromkeys(path for path in catalog if path))
    if not catalog:
        return {
            "available": False,
            "policy": reference_contract.get("policy", "task_adaptive"),
            "catalog_leaf_count": 0,
            "included_leaf_paths": [],
            "adapted_leaf_mappings": [],
            "excluded_path_prefixes": [],
            "excluded_leaf_paths": [],
            "unresolved_leaf_paths": [],
            "invalid_decisions": [],
        }

    audit = raw_audit if isinstance(raw_audit, dict) else {}
    catalog_set = set(catalog)
    included = {
        normalize_reference_field_path(path)
        for path in audit.get("included_leaf_paths", []) or []
        if normalize_reference_field_path(path) in catalog_set
    }
    for group in field_groups or []:
        if not isinstance(group, dict):
            continue
        included.update(
            normalize_reference_field_path(path)
            for path in group.get("recommended_fields", []) or []
            if normalize_reference_field_path(path) in catalog_set
        )

    mappings = []
    mapped_sources = set()
    invalid_decisions = []
    for item in audit.get("adapted_leaf_mappings", []) or []:
        if not isinstance(item, dict):
            invalid_decisions.append("adapted mapping must be an object")
            continue
        source_path = normalize_reference_field_path(item.get("source_path"))
        target_path = normalize_reference_field_path(item.get("target_path"))
        reason = str(item.get("reason") or "").strip()
        if source_path not in catalog_set or not target_path or not reason:
            invalid_decisions.append(f"invalid adapted mapping: {source_path or '<missing>'}")
            continue
        mappings.append(
            {"source_path": source_path, "target_path": target_path, "reason": reason}
        )
        mapped_sources.add(source_path)

    excluded_prefixes = []
    excluded_paths = set()
    for item in audit.get("excluded_path_prefixes", []) or []:
        if isinstance(item, str):
            prefix = normalize_reference_field_path(item)
            reason = ""
        elif isinstance(item, dict):
            prefix = normalize_reference_field_path(item.get("path_prefix"))
            reason = str(item.get("reason") or "").strip()
        else:
            invalid_decisions.append("excluded prefix must be a string or object")
            continue
        matched = {
            path for path in catalog if path == prefix or path.startswith(prefix + ".")
        }
        if not prefix or not reason or not matched:
            invalid_decisions.append(f"invalid excluded prefix: {prefix or '<missing>'}")
            continue
        excluded_prefixes.append({"path_prefix": prefix, "reason": reason})
        excluded_paths.update(matched)

    resolved = included | mapped_sources | excluded_paths
    unresolved = [path for path in catalog if path not in resolved]
    return {
        "available": True,
        "policy": str(audit.get("policy") or reference_contract.get("policy") or "task_adaptive"),
        "catalog_leaf_count": len(catalog),
        "included_leaf_paths": [path for path in catalog if path in included],
        "adapted_leaf_mappings": mappings,
        "excluded_path_prefixes": excluded_prefixes,
        "excluded_leaf_paths": [path for path in catalog if path in excluded_paths],
        "unresolved_leaf_paths": unresolved,
        "invalid_decisions": invalid_decisions,
    }


def build_coverage_report(result):
    has_contract = isinstance(result, dict) and "requirement_contract" in result
    contract = result.get("requirement_contract") if has_contract else {}
    contract = contract if isinstance(contract, dict) else {}
    concepts = [item for item in contract.get("concepts", []) if isinstance(item, dict)]
    required_concepts = {
        str(item.get("concept_id")) for item in concepts if item.get("required") and item.get("concept_id")
    }
    fields = (((result or {}).get("schema_definition") or {}).get("field_registry")) or []
    valid_concept_ids = {
        str(item.get("concept_id")) for item in concepts if item.get("concept_id")
    }
    mapped_concepts = {
        str(concept_id)
        for field in fields
        if isinstance(field, dict)
        for concept_id in field.get("concept_ids", []) or []
        if concept_id and str(concept_id) in valid_concept_ids
    }
    missing_concepts = sorted(required_concepts - mapped_concepts)
    concept_ratio = (
        1.0
        if not required_concepts
        else (len(required_concepts) - len(missing_concepts)) / len(required_concepts)
    )

    top_level_keys = {
        str(item.get("key"))
        for item in (((result or {}).get("schema_definition") or {}).get("top_level_keys") or [])
        if isinstance(item, dict) and item.get("key")
    }
    field_paths = [str(field.get("field_path") or "") for field in fields if isinstance(field, dict)]
    normalized_field_paths = {
        normalize_reference_field_path(path) for path in field_paths if path
    }
    reference_contract = (
        result.get("reference_field_contract")
        if isinstance(result, dict) and isinstance(result.get("reference_field_contract"), dict)
        else {}
    )
    reference_audit = normalize_reference_field_audit(
        reference_contract,
        result.get("reference_field_audit") if isinstance(result, dict) else None,
    )
    reference_paths = {
        normalize_reference_field_path(item.get("path"))
        for item in reference_contract.get("leaf_fields", []) or []
        if isinstance(item, dict) and item.get("path")
    }
    target_by_source = {
        item["source_path"]: item["target_path"]
        for item in reference_audit.get("adapted_leaf_mappings", []) or []
        if isinstance(item, dict) and item.get("source_path") and item.get("target_path")
    }
    applicable_reference_paths = list(reference_audit.get("included_leaf_paths", []) or []) + list(
        target_by_source
    )
    represented_reference_paths = {
        source_path
        for source_path in applicable_reference_paths
        if target_by_source.get(source_path, source_path) in normalized_field_paths
    }
    missing_reference_paths = sorted(
        set(applicable_reference_paths) - represented_reference_paths
    )
    catalog_leaf_count = int(reference_audit.get("catalog_leaf_count", 0) or 0)
    unresolved_reference_paths = list(reference_audit.get("unresolved_leaf_paths", []) or [])
    decision_ratio = (
        1.0
        if not catalog_leaf_count
        else (catalog_leaf_count - len(unresolved_reference_paths)) / catalog_leaf_count
    )
    applicable_ratio = (
        1.0
        if not applicable_reference_paths
        else len(represented_reference_paths) / len(set(applicable_reference_paths))
    )
    full_reference_ratio = (
        1.0
        if not catalog_leaf_count
        else len(represented_reference_paths) / catalog_leaf_count
    )
    entities = (
        [item for item in (result.get("entity_registry") or []) if isinstance(item, dict)]
        if isinstance(result, dict)
        else []
    )
    required_entities = [item for item in entities if item.get("required") and item.get("independent_owner")]
    missing_entities = sorted(
        str(item.get("entity_id"))
        for item in required_entities
        if str(item.get("owner_key") or "") not in top_level_keys
        or not any(path.startswith(f"{item.get('owner_key')}.") for path in field_paths)
    )

    incomplete_fields = []
    object_missing_slots = {}
    for field in fields:
        if not isinstance(field, dict):
            continue
        missing_slots = object_contract_missing_slots(field)
        if missing_slots:
            field_path = str(field.get("field_path") or "<unknown>")
            incomplete_fields.append(field_path)
            object_missing_slots[field_path] = sorted(missing_slots)

    required_evidence_missing = []
    for concept_id in required_concepts:
        concept_fields = [
            field
            for field in fields
            if isinstance(field, dict) and concept_id in (field.get("concept_ids") or [])
        ]
        if concept_fields and not any(field.get("source_basis") for field in concept_fields):
            required_evidence_missing.append(concept_id)

    registered_entity_identity_paths = {
        f"{str(entity.get('owner_key')).strip()}.identity"
        for entity in entities
        if str(entity.get("owner_key") or "").strip()
    }
    accepted_reference_targets = {
        normalize_reference_field_path(path)
        for path in reference_audit.get("included_leaf_paths", []) or []
    } | {
        normalize_reference_field_path(item.get("target_path"))
        for item in reference_audit.get("adapted_leaf_mappings", []) or []
        if isinstance(item, dict) and item.get("target_path")
    }
    field_plan_traceability = (
        result.get("field_plan_traceability")
        if isinstance(result, dict)
        and isinstance(result.get("field_plan_traceability"), dict)
        else {}
    )
    planned_field_paths = {
        normalize_reference_field_path(path)
        for path in field_plan_traceability.get("planned_field_paths", []) or []
        if normalize_reference_field_path(path)
    }
    invalid_concept_references = {}
    missing_utility_reasons = []
    for field in fields:
        if not isinstance(field, dict):
            continue
        field_path = str(field.get("field_path") or "")
        invalid_ids = sorted(
            {
                str(concept_id)
                for concept_id in field.get("concept_ids", []) or []
                if str(concept_id) not in valid_concept_ids
            }
        )
        if invalid_ids:
            invalid_concept_references[field_path or "<unknown>"] = invalid_ids
        if not str(field.get("reason") or "").strip():
            missing_utility_reasons.append(field_path or "<unknown>")
    unrelated_fields = [
        str(field.get("field_path") or "")
        for field in fields
        if isinstance(field, dict)
        and not (
            set(map(str, field.get("concept_ids", []) or [])) & valid_concept_ids
        )
        and normalize_reference_field_path(field.get("field_path")) not in accepted_reference_targets
        and normalize_reference_field_path(field.get("field_path")) not in planned_field_paths
        and str(field.get("field_path") or "") not in registered_entity_identity_paths
    ]
    semantic_groups = {}
    for field in fields:
        if not isinstance(field, dict) or not field.get("field_path") or not field.get("concept_ids"):
            continue
        field_path = str(field.get("field_path") or "")
        scientific_owner_path, _, leaf_role = field_path.rpartition(".")
        root_level_owners = {
            "paper_info",
            "primary_signature",
            "primary_signature_normalized",
            "section5",
            "normalization_aliases",
            "material_name_aliases",
            "formula_aliases",
            "sample_id_aliases",
        } | {f"material_info.section{index}" for index in range(5)}
        nested_field_role = "" if scientific_owner_path in root_level_owners else leaf_role
        semantic_contract = {
            # Reused record slots are distinct when they belong to different
            # scientific quantities, even if their slot contracts are identical.
            "scientific_owner_path": scientific_owner_path,
            "nested_field_role": nested_field_role,
            "section_id": str(field.get("section_id") or ""),
            "data_type": str(field.get("data_type") or "").strip().lower(),
            "source_basis": sorted(str(item) for item in field.get("source_basis") or []),
            "concept_ids": sorted(str(item) for item in field.get("concept_ids") or []),
            "description": re.sub(r"\s+", " ", str(field.get("description") or "").strip().lower()),
            "extraction_notes": re.sub(
                r"\s+", " ", str(field.get("extraction_notes") or "").strip().lower()
            ),
            "inclusion_rule": field.get("inclusion_rule"),
            "absence_rule": field.get("absence_rule"),
            "evidence_requirements": field.get("evidence_requirements"),
            "relation_constraints": field.get("relation_constraints"),
            "object_contract": field.get("object_contract"),
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                semantic_contract,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        semantic_groups.setdefault(fingerprint, []).append(field_path)
    duplicate_groups = [
        sorted(paths) for paths in semantic_groups.values() if len(paths) > 1
    ]
    redundant_fields = sorted(
        path for paths in duplicate_groups for path in paths[1:]
    )
    source_requirements = contract.get("source_requirements", [])
    traced_requirement_ids = {
        str(req_id)
        for concept in concepts
        for req_id in concept.get("source_requirement_ids", []) or []
        if req_id
    }
    missing_requirement_ids = sorted(
        str(item.get("requirement_id"))
        for item in source_requirements
        if isinstance(item, dict) and item.get("requirement_id") not in traced_requirement_ids
    )
    care_requirement_ids = sorted(
        str(item.get("requirement_id"))
        for item in source_requirements
        if isinstance(item, dict)
        and item.get("requirement_id")
        and item.get("source") == "care_counterfactual_query"
    )
    missing_care_requirement_ids = sorted(
        set(care_requirement_ids) - traced_requirement_ids
    )
    return {
        "legacy_mode": not has_contract,
        "required_concept_coverage": {
            "required": len(required_concepts),
            "covered": len(required_concepts) - len(missing_concepts),
            "ratio": concept_ratio,
            "missing_concept_ids": missing_concepts,
        },
        "entity_owner_coverage": {
            "required": len(required_entities),
            "covered": len(required_entities) - len(missing_entities),
            "missing_entity_ids": missing_entities,
        },
        "structured_object_completeness": {
            "incomplete_fields": sorted(incomplete_fields),
            "missing_slots": object_missing_slots,
        },
        "evidence_contract_coverage": {"missing_concept_ids": sorted(required_evidence_missing)},
        "query_requirement_traceability": {"missing_requirement_ids": missing_requirement_ids},
        "care_counterfactual_traceability": {
            "enabled": bool(care_requirement_ids),
            "required": len(care_requirement_ids),
            "covered": len(care_requirement_ids) - len(missing_care_requirement_ids),
            "required_requirement_ids": care_requirement_ids,
            "missing_requirement_ids": missing_care_requirement_ids,
        },
        "reference_field_coverage": {
            "available": bool(reference_audit.get("available")),
            "catalog_leaf_count": catalog_leaf_count,
            "decision_coverage_ratio": decision_ratio,
            "applicable_leaf_count": len(set(applicable_reference_paths)),
            "represented_applicable_leaf_count": len(represented_reference_paths),
            "applicable_coverage_ratio": applicable_ratio,
            "full_reference_coverage_ratio": full_reference_ratio,
            "excluded_leaf_count": len(reference_audit.get("excluded_leaf_paths", []) or []),
            "missing_applicable_paths": missing_reference_paths,
            "unresolved_leaf_paths": unresolved_reference_paths,
            "invalid_decisions": list(reference_audit.get("invalid_decisions", []) or []),
        },
        "unrelated_domain_field_count": len(unrelated_fields),
        "unrelated_domain_fields": sorted(unrelated_fields),
        "field_utility_traceability": {
            "unjustified_field_count": len(unrelated_fields),
            "unjustified_fields": sorted(unrelated_fields),
            "invalid_concept_references": invalid_concept_references,
            "missing_utility_reason_fields": sorted(missing_utility_reasons),
            "policy": (
                "Every field must trace to a valid task concept, an applicable reference decision, "
                "an exact current field-plan path, a registered entity identity, or a shared "
                "materials-record contract. Field count is not a target."
            ),
        },
        "field_plan_traceability": {
            "planned_field_count": len(planned_field_paths),
            "planned_field_paths": sorted(planned_field_paths),
            "represented_planned_field_count": len(
                planned_field_paths & normalized_field_paths
            ),
            "represented_planned_field_paths": sorted(
                planned_field_paths & normalized_field_paths
            ),
        },
        "semantic_redundancy": {
            "duplicate_groups": sorted(duplicate_groups),
            "redundant_field_count": len(redundant_fields),
            "redundant_fields": redundant_fields,
            "policy": "Exact semantic-contract duplicates are redundant regardless of total field count.",
        },
    }


def validate_coverage_report(report):
    if report.get("legacy_mode"):
        return []
    errors = []
    checks = (
        ("coverage:missing_concepts:", report.get("required_concept_coverage", {}).get("missing_concept_ids", [])),
        ("coverage:missing_entity_owners:", report.get("entity_owner_coverage", {}).get("missing_entity_ids", [])),
        (
            "coverage:incomplete_object_contracts:",
            report.get("structured_object_completeness", {}).get("incomplete_fields", []),
        ),
        (
            "coverage:missing_evidence_contracts:",
            report.get("evidence_contract_coverage", {}).get("missing_concept_ids", []),
        ),
        (
            "coverage:untraced_query_requirements:",
            report.get("query_requirement_traceability", {}).get("missing_requirement_ids", []),
        ),
        (
            "coverage:untraced_care_counterfactuals:",
            report.get("care_counterfactual_traceability", {}).get(
                "missing_requirement_ids", []
            ),
        ),
        (
            "coverage:missing_reference_fields:",
            report.get("reference_field_coverage", {}).get("missing_applicable_paths", []),
        ),
        (
            "coverage:unresolved_reference_fields:",
            report.get("reference_field_coverage", {}).get("unresolved_leaf_paths", []),
        ),
        (
            "coverage:invalid_reference_decisions:",
            report.get("reference_field_coverage", {}).get("invalid_decisions", []),
        ),
        ("coverage:unmapped_domain_fields:", report.get("unrelated_domain_fields", [])),
        (
            "coverage:invalid_concept_references:",
            [
                f"{path}=>{'|'.join(concept_ids)}"
                for path, concept_ids in (
                    report.get("field_utility_traceability", {}).get(
                        "invalid_concept_references", {}
                    )
                    or {}
                ).items()
            ],
        ),
        (
            "coverage:missing_field_utility_reasons:",
            report.get("field_utility_traceability", {}).get(
                "missing_utility_reason_fields", []
            ),
        ),
        (
            "coverage:redundant_fields:",
            report.get("semantic_redundancy", {}).get("redundant_fields", []),
        ),
    )
    for prefix, values in checks:
        if values:
            errors.append(prefix + ",".join(values))
    return errors


def coverage_blocker_sets(report):
    return {
        "concepts": set(report.get("required_concept_coverage", {}).get("missing_concept_ids", [])),
        "entities": set(report.get("entity_owner_coverage", {}).get("missing_entity_ids", [])),
        "objects": set(report.get("structured_object_completeness", {}).get("incomplete_fields", [])),
        "evidence": set(report.get("evidence_contract_coverage", {}).get("missing_concept_ids", [])),
        "requirements": set(report.get("query_requirement_traceability", {}).get("missing_requirement_ids", [])),
        "reference_fields": set(report.get("reference_field_coverage", {}).get("missing_applicable_paths", [])),
        "reference_unresolved": set(report.get("reference_field_coverage", {}).get("unresolved_leaf_paths", [])),
        "reference_invalid": set(report.get("reference_field_coverage", {}).get("invalid_decisions", [])),
        "unrelated": set(report.get("unrelated_domain_fields", [])),
        "redundant": set(report.get("semantic_redundancy", {}).get("redundant_fields", [])),
    }


def is_coverage_improvement(previous_report, candidate_report):
    previous_sets = coverage_blocker_sets(previous_report)
    candidate_sets = coverage_blocker_sets(candidate_report)
    if any(not candidate_sets[key].issubset(previous_sets[key]) for key in previous_sets):
        return False
    previous_total = sum(len(values) for values in previous_sets.values())
    candidate_total = sum(len(values) for values in candidate_sets.values())
    previous_ratio = float(previous_report.get("required_concept_coverage", {}).get("ratio", 0) or 0)
    candidate_ratio = float(candidate_report.get("required_concept_coverage", {}).get("ratio", 0) or 0)
    previous_reference_ratio = float(
        previous_report.get("reference_field_coverage", {}).get("applicable_coverage_ratio", 0) or 0
    )
    candidate_reference_ratio = float(
        candidate_report.get("reference_field_coverage", {}).get("applicable_coverage_ratio", 0) or 0
    )
    return (
        candidate_total < previous_total
        or candidate_ratio > previous_ratio
        or candidate_reference_ratio > previous_reference_ratio
    )


def validate_specialization(shared_context, modules, result):
    report = build_coverage_report(result)
    if isinstance(result, dict):
        result["coverage_report"] = report
    return validate_coverage_report(report)


def infer_figure_constraint(field):
    field_path = str(field.get("field_path", ""))
    section_id = str(field.get("section_id", ""))
    path_lower = field_path.lower()
    section_lower = section_id.lower()

    allowed_sections = []
    allowed_categories = []
    why_needed = ""

    if section_id:
        allowed_sections = [section_id]
    elif "." in field_path:
        allowed_sections = [field_path.rsplit(".", 1)[0]]

    if not allowed_sections:
        return None

    if path_lower.endswith(".figure"):
        allowed_categories = ["figure"]
    elif "table" in path_lower:
        allowed_categories = ["table"]
    elif "curve" in path_lower or "plot" in path_lower:
        allowed_categories = ["curve_or_plot"]
    elif any(token in path_lower for token in ["image", "microscopy", "xrd", "diffraction", "spectrum"]):
        allowed_categories = ["characterization_figure"]
    else:
        allowed_categories = ["domain_relevant_figure"]

    why_needed = (
        "This figure-linked field must preserve ownership by the section that "
        "contains the measured or observed evidence, instead of letting a "
        "neighboring interpretation section consume the same figure."
    )

    if not allowed_sections:
        return None

    return {
        "uses_figure_classification": True,
        "allowed_sections": allowed_sections,
        "allowed_figure_categories": allowed_categories,
        "why_needed": why_needed,
    }


def clone_figure_constraint(figure_constraint, why_needed=None, allowed_sections=None, allowed_categories=None):
    if not isinstance(figure_constraint, dict):
        return None
    cloned = {
        "uses_figure_classification": bool(figure_constraint.get("uses_figure_classification", True)),
        "allowed_sections": list(allowed_sections or figure_constraint.get("allowed_sections") or []),
        "allowed_figure_categories": list(allowed_categories or figure_constraint.get("allowed_figure_categories") or []),
        "why_needed": why_needed or figure_constraint.get("why_needed", ""),
    }
    if not cloned["allowed_sections"] or not cloned["allowed_figure_categories"] or not cloned["why_needed"]:
        return None
    return cloned


def normalize_source_basis_values(field_registry):
    basis_aliases = {
        "text": "text",
        "paper text": "text",
        "caption": "figure",
        "figure caption": "figure",
        "figure_caption": "figure",
        "image": "figure",
        "plot": "figure",
        "chart": "figure",
        "table": "table",
    }
    valid_basis = {"text", "table", "figure"}
    for field in field_registry:
        if not isinstance(field, dict):
            continue
        source_basis = field.get("source_basis")
        if not isinstance(source_basis, list):
            continue
        normalized = []
        for item in source_basis:
            normalized_item = basis_aliases.get(str(item).strip().lower(), str(item).strip().lower())
            if normalized_item in valid_basis and normalized_item not in normalized:
                normalized.append(normalized_item)
        field["source_basis"] = normalized or ["text"]


def detect_theory_section_id(result):
    section_design = result.get("section_design", {}) if isinstance(result, dict) else {}
    candidate_sections = []
    for group_name in ("core_sections", "non_core_sections"):
        for section in section_design.get(group_name, []) or []:
            if isinstance(section, dict) and section.get("section_id"):
                section_id = str(section["section_id"])
                section_name = str(section.get("section_name", "")).lower()
                if (
                    section_id.lower() == "section5"
                    or "theory" in section_name
                    or "mechanism" in section_name
                    or "simulation" in section_name
                ):
                    candidate_sections.append(section_id)
    if candidate_sections:
        return candidate_sections[0]
    return "section5"


def detect_general_section_id(result):
    section_design = result.get("section_design", {}) if isinstance(result, dict) else {}
    candidate_sections = []
    for group_name in ("core_sections", "non_core_sections"):
        for section in section_design.get(group_name, []) or []:
            if isinstance(section, dict) and section.get("section_id"):
                section_id = str(section["section_id"])
                section_name = str(section.get("section_name", "")).lower()
                section_id_lower = section_id.lower()
                if (
                    section_id_lower == "section0"
                    or "general" in section_id_lower
                    or "material_general" in section_id_lower
                    or "material system" in section_name
                    or "general information" in section_name
                    or "synthesis" in section_name
                ):
                    candidate_sections.append(section_id)
    if candidate_sections:
        return candidate_sections[0]
    return "section0"


def get_top_level_key_names(result):
    schema_definition = result.get("schema_definition", {}) if isinstance(result, dict) else {}
    top_level_keys = schema_definition.get("top_level_keys") or []
    names = set()
    for item in top_level_keys:
        if isinstance(item, dict) and item.get("key"):
            names.add(str(item["key"]))
    return names


def build_owner_field_path(result, owner_id, suffix):
    owner_id = str(owner_id)
    suffix = str(suffix).lstrip(".")
    top_level_keys = get_top_level_key_names(result)
    if owner_id in top_level_keys:
        return f"{owner_id}.{suffix}"
    if owner_id.startswith("material_info.") or owner_id.startswith("paper_info."):
        return f"{owner_id}.{suffix}"
    return f"material_info.{owner_id}.{suffix}"


PROCESS_STANDARD_SECTION_ALIASES = {
    "section0": "material_info.section0",
    "section1": "material_info.section1",
    "section2": "material_info.section2",
    "section3": "material_info.section3",
    "section4": "material_info.section4",
    "section5": "section5",
    "material_info.section0": "material_info.section0",
    "material_info.section1": "material_info.section1",
    "material_info.section2": "material_info.section2",
    "material_info.section3": "material_info.section3",
    "material_info.section4": "material_info.section4",
}
PROCESS_STANDARD_SECTIONS = set(PROCESS_STANDARD_SECTION_ALIASES.values())


def canonical_process_section_id(value):
    text = str(value or "").strip()
    return PROCESS_STANDARD_SECTION_ALIASES.get(text, text)


def collect_process_standard_sections(result, field_registry):
    sections = []
    seen = set()

    def add_section(raw_section_id):
        section_id = canonical_process_section_id(raw_section_id)
        if section_id in PROCESS_STANDARD_SECTIONS and section_id not in seen:
            seen.add(section_id)
            sections.append(section_id)

    for field in field_registry:
        if not isinstance(field, dict):
            continue
        add_section(field.get("section_id"))
        field_path = str(field.get("field_path", ""))
        for section_id in PROCESS_STANDARD_SECTIONS:
            if field_path.startswith(f"{section_id}."):
                add_section(section_id)

    section_design = result.get("section_design", {}) if isinstance(result, dict) else {}
    for group_name in ("core_sections", "non_core_sections"):
        for section in section_design.get(group_name, []) or []:
            if isinstance(section, dict):
                add_section(section.get("section_id"))

    return sections


def ensure_process_standard_fields(result, field_registry):
    """Attach reusable record contracts without creating synthetic field rows."""
    active_sections = set(collect_process_standard_sections(result, field_registry))
    schema_definition = result.setdefault("schema_definition", {})
    shared_contracts = schema_definition.setdefault("shared_record_contracts", {})
    shared_contracts.setdefault(
        "literature_provenance_v1",
        {
            "purpose": "Trace an extracted value to the scientific literature without duplicating provenance leaves under every quantity.",
            "required_slots": ["source_type", "locator", "confidence"],
            "optional_slots": ["source_text", "source_table", "source_figure", "value_origin"],
            "materialize_as_field_only_when": "The provenance value is independently queried or has task-specific semantics.",
        },
    )
    shared_contracts.setdefault(
        "measurement_context_v1",
        {
            "purpose": "Keep values bound to the conditions that change their scientific interpretation.",
            "optional_slots": [
                "temperature",
                "pressure",
                "external_field_or_stimulus",
                "direction_or_geometry",
                "protocol",
                "method",
                "criterion",
            ],
            "materialize_as_field_only_when": "A condition is independently queried, required by an applicable reference leaf, or has quantity-specific semantics.",
        },
    )
    shared_contracts.setdefault(
        "entity_binding_v1",
        {
            "purpose": "Bind every scientific claim to the correct material, sample, phase, interface, device, or other registered owner.",
            "required_slots": ["entity_ref"],
        },
    )

    for field in field_registry:
        if not isinstance(field, dict):
            continue
        field_path = str(field.get("field_path") or "")
        section_id = canonical_process_section_id(field.get("section_id"))
        if section_id not in active_sections or field_path.startswith("paper_info."):
            continue
        refs = list(field.get("contract_refs") or [])
        refs.extend(["literature_provenance_v1", "entity_binding_v1"])
        object_kind = str((field.get("object_contract") or {}).get("object_kind") or "")
        if section_id in {"material_info.section1", "material_info.section4"} or object_kind == "measurement":
            refs.append("measurement_context_v1")
        field["contract_refs"] = list(dict.fromkeys(refs))


def canonical_section_id(section_id):
    section_id = str(section_id or "").strip()
    if re.fullmatch(r"section[0-4]", section_id):
        return f"material_info.{section_id}"
    if section_id == "material_info.section5":
        return "section5"
    return section_id


def normalize_result_section_ids(result):
    section_design = result.get("section_design") or {}
    for group_name in ("core_sections", "non_core_sections"):
        for section in section_design.get(group_name, []) or []:
            if isinstance(section, dict):
                section["section_id"] = canonical_section_id(section.get("section_id"))
    schema_definition = result.get("schema_definition") or {}
    for field in schema_definition.get("field_registry", []) or []:
        if not isinstance(field, dict):
            continue
        field["section_id"] = canonical_section_id(field.get("section_id"))
        field_path = str(field.get("field_path") or "")
        if field_path.startswith("material_info.section5."):
            field["field_path"] = "section5." + field_path[len("material_info.section5.") :]


STANDARD_SECTION_DESCRIPTORS = {
    "material_info.section0": ("Material Identity And Structure", "Identity, composition, structure, sample, and tuning descriptors."),
    "material_info.section1": ("Task Relevant Properties", "Task-specific phases, properties, transitions, and derived quantities."),
    "material_info.section2": ("Preparation And Processing", "Synthesis, growth, fabrication, and post-processing records."),
    "material_info.section3": ("Microscopic Characterization", "Microscopic, spectroscopic, structural, and imaging evidence."),
    "material_info.section4": ("Macroscopic Measurements", "Macroscopic measurements, curves, maps, and response data."),
    "section5": ("Theory And Mechanisms", "Models, calculations, mechanisms, fits, and theoretical evidence."),
}


def section_descriptor(section_id):
    name, purpose = STANDARD_SECTION_DESCRIPTORS.get(
        section_id,
        (section_id.replace("material_info.", "").replace("_", " ").title(), "Task-adaptive material data section."),
    )
    return {"section_id": section_id, "section_name": name, "purpose": purpose}


def ensure_section_architecture_covers_fields(result, field_registry):
    section_design = result.setdefault("section_design", {})
    top_level_keys = get_top_level_key_names(result)
    field_counts = {}
    for field in field_registry:
        if not isinstance(field, dict):
            continue
        section_id = canonical_section_id(field.get("section_id"))
        field["section_id"] = section_id
        if section_id and section_id not in top_level_keys:
            field_counts[section_id] = field_counts.get(section_id, 0) + 1

    core_sections = []
    non_core_sections = []
    seen = set()
    for group_name, target in (
        ("core_sections", core_sections),
        ("non_core_sections", non_core_sections),
    ):
        for raw in section_design.get(group_name, []) or []:
            if not isinstance(raw, dict):
                continue
            section = dict(raw)
            section_id = canonical_section_id(section.get("section_id"))
            if not section_id or section_id in seen:
                continue
            if section_id.lower() in {"sectionx", "section_x"} and not field_counts.get(section_id):
                continue
            section["section_id"] = section_id
            target.append(section)
            seen.add(section_id)

    for section_id in sorted(field_counts):
        if section_id not in seen:
            non_core_sections.append(section_descriptor(section_id))
            seen.add(section_id)

    core_ids = {item["section_id"] for item in core_sections}
    promotion_order = [
        section_id
        for section_id in ("material_info.section0", "material_info.section1")
        if section_id in field_counts
    ]
    promotion_order.extend(
        section_id
        for section_id, _count in sorted(
            field_counts.items(), key=lambda item: (-item[1], item[0])
        )
        if section_id not in promotion_order
    )
    for section_id in promotion_order:
        if len(core_ids) >= 2:
            break
        if section_id in core_ids:
            continue
        existing = next(
            (item for item in non_core_sections if item.get("section_id") == section_id),
            section_descriptor(section_id),
        )
        non_core_sections = [
            item for item in non_core_sections if item.get("section_id") != section_id
        ]
        core_sections.append(existing)
        core_ids.add(section_id)

    section_design["core_sections"] = core_sections
    section_design["non_core_sections"] = non_core_sections


INSTRUCTION_ARTIFACT_LEAVES = {
    "keep_experimental",
    "fitted",
    "calculated_values_separate",
    "inferred_quantities_separate",
}


def prune_instruction_artifact_fields(field_registry):
    retained = []
    for field in field_registry:
        if not isinstance(field, dict):
            retained.append(field)
            continue
        field_path = str(field.get("field_path") or "")
        leaf = stable_concept_id(field_path.rsplit(".", 1)[-1])
        lowered = field_path.lower()
        if leaf in INSTRUCTION_ARTIFACT_LEAVES:
            continue
        if any(
            token in lowered
            for token in (
                "the_current_test_documents",
                "converted_from_the_pdf",
                "without_figure_images",
                "vlm_judge",
                "must_remain_unavailable",
            )
        ):
            continue
        retained.append(field)
    field_registry[:] = retained


def ensure_field_contract_metadata(result, field_registry):
    """Complete extraction metadata without inventing field justifications."""
    for field in field_registry:
        if not isinstance(field, dict):
            continue
        field_path = str(field.get("field_path") or "")
        section_id = canonical_section_id(field.get("section_id"))
        field["section_id"] = section_id
        if not str(field.get("extraction_notes") or "").strip():
            field["extraction_notes"] = (
                f"Extract only explicit {field_path.rsplit('.', 1)[-1] or 'field'} evidence for "
                "the target entity in the same evidence span. Preserve sample, method, units, "
                "conditions, and source location; do not infer or merge across records."
            )

        field["concept_ids"] = list(
            dict.fromkeys(
                [
                    stable_concept_id(item)
                    for item in field.get("concept_ids", []) or []
                    if stable_concept_id(item)
                ]
            )
        )


def ensure_literature_field_rule_contract(result, field_registry):
    """Attach stable, executable field rules without changing schema paths."""
    task_contract = dict(MATERIAL_LITERATURE_TASK_CONTRACT)
    result["task_contract"] = task_contract
    result["lineage_contract"] = {
        "contract_version": "materials-literature-lineage-v1",
        "dependency_chain": [
            "literature_evidence_unit",
            "schema_field_rule",
            "extraction_prompt",
            "candidate_field_value",
            "validator_result",
            "final_material_record",
        ],
        "required_version_keys": [
            "field_rule_version",
            "prompt_version",
            "model_version",
            "validator_version",
        ],
        "reprocess_policy": "rerun_only_affected_literature_documents_stages_and_field_paths",
    }

    core_section_ids = {
        canonical_section_id(section.get("section_id"))
        for section in (result.get("section_design", {}).get("core_sections") or [])
        if isinstance(section, dict) and section.get("section_id")
    }
    registry_versions = []
    for field in field_registry:
        if not isinstance(field, dict):
            continue
        field_path = str(field.get("field_path") or "").strip()
        section_id = canonical_section_id(field.get("section_id"))
        source_basis = list(field.get("source_basis") or ["text"])
        is_paper_metadata = field_path.startswith("paper_info.")
        object_contract = field.get("object_contract") or {}
        object_kind = str(object_contract.get("object_kind") or "").lower()
        contract_refs = set(field.get("contract_refs") or [])

        field["field_rule_id"] = f"field.{stable_concept_id(field_path) or 'unnamed'}"
        field.setdefault(
            "core_field",
            bool(field.get("required")) and section_id in core_section_ids,
        )
        field.setdefault("core_field_source", "step8_system_design")
        field.setdefault(
            "inclusion_rule",
            (
                "Populate from authoritative download-time bibliographic metadata; use literature text only "
                "when the metadata contract explicitly permits it."
                if is_paper_metadata
                else "Populate only when the scientific literature directly supports this field's exact semantics "
                "for the bound material or sample under the reported conditions."
            ),
        )
        field.setdefault(
            "absence_rule",
            "Use missing when the literature contains no supporting report; use unresolved when relevant evidence "
            "exists but ownership, conditions, interpretation, or support remains ambiguous. Never infer a value "
            "only to fill the schema.",
        )

        evidence_requirements = field.get("evidence_requirements")
        if not isinstance(evidence_requirements, dict):
            evidence_requirements = {}
        evidence_requirements.setdefault("direct_support_required", not is_paper_metadata)
        evidence_requirements.setdefault("locator_required", True)
        evidence_requirements["allowed_source_types"] = source_basis
        evidence_requirements.setdefault(
            "metadata_handoff_allowed",
            is_paper_metadata,
        )
        field["evidence_requirements"] = evidence_requirements

        relation_constraints = field.get("relation_constraints")
        if not isinstance(relation_constraints, dict):
            relation_constraints = {}
        relation_constraints.setdefault("entity_binding_required", not is_paper_metadata)
        relation_constraints.setdefault(
            "condition_binding_required",
            not is_paper_metadata
            and (
                object_kind in {"measurement", "process"}
                or "measurement_context_v1" in contract_refs
            ),
        )
        relation_constraints.setdefault(
            "separate_instances",
            not is_paper_metadata
            and (
                object_kind in {"measurement", "process", "classification"}
                or "array" in str(field.get("data_type") or "").lower()
            ),
        )
        field["relation_constraints"] = relation_constraints

        version_payload = {
            key: field.get(key)
            for key in (
                "field_path",
                "section_id",
                "description",
                "extraction_notes",
                "data_type",
                "required",
                "core_field",
                "source_basis",
                "object_contract",
                "contract_refs",
                "figure_constraint",
                "inclusion_rule",
                "absence_rule",
                "evidence_requirements",
                "relation_constraints",
            )
        }
        digest = hashlib.sha256(
            json.dumps(version_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest().upper()
        field["field_rule_version"] = f"sha256:{digest}"
        registry_versions.append((field["field_rule_id"], field["field_rule_version"]))

    registry_payload = json.dumps(sorted(registry_versions), ensure_ascii=False).encode("utf-8")
    result.setdefault("schema_definition", {})["field_rule_registry_version"] = (
        "sha256:" + hashlib.sha256(registry_payload).hexdigest().upper()
    )
    shared_contracts = result.setdefault("schema_definition", {}).get(
        "shared_record_contracts", {}
    )
    result["schema_definition"]["shared_record_contract_registry_version"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                shared_contracts,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest().upper()
    )


def postprocess_result(shared_context, result):
    if not isinstance(result, dict):
        return result

    normalize_result_section_ids(result)
    schema_definition = result.setdefault("schema_definition", {})
    field_registry = schema_definition.setdefault("field_registry", [])
    if not isinstance(field_registry, list):
        return result

    general_section_id = detect_general_section_id(result)
    top_level_keys = get_top_level_key_names(result)
    normalize_source_basis_values(field_registry)

    ensure_process_standard_fields(result, field_registry)
    prune_instruction_artifact_fields(field_registry)

    for field in field_registry:
        if not isinstance(field, dict):
            continue
        field_path = str(field.get("field_path", ""))
        section_id = str(field.get("section_id", ""))
        if field_path.startswith("material_info.material_info."):
            field["field_path"] = field_path.replace("material_info.material_info.", "material_info.", 1)
            field_path = str(field.get("field_path", ""))
        if field_path.startswith("material_info.section0.") and general_section_id != "section0":
            field["field_path"] = field_path.replace(
                "material_info.section0.",
                build_owner_field_path(result, general_section_id, ""),
                1,
            ).replace("..", ".").rstrip(".")
            field_path = str(field.get("field_path", ""))
        if section_id == "section0" and general_section_id != "section0":
            field["section_id"] = general_section_id
            section_id = general_section_id
        if section_id in top_level_keys and field_path.startswith(f"material_info.{section_id}."):
            field["field_path"] = field_path.replace(f"material_info.{section_id}.", f"{section_id}.", 1)

    for field in field_registry:
        if not isinstance(field, dict):
            continue
        source_basis = field.get("source_basis") or []
        figure_constraint = field.get("figure_constraint")
        needs_constraint_repair = not isinstance(figure_constraint, dict)
        if isinstance(figure_constraint, dict):
            needs_constraint_repair = (
                not isinstance(figure_constraint.get("allowed_sections"), list)
                or not figure_constraint.get("allowed_sections")
                or not isinstance(figure_constraint.get("allowed_figure_categories"), list)
                or not figure_constraint.get("allowed_figure_categories")
                or not isinstance(figure_constraint.get("why_needed"), str)
                or not figure_constraint.get("why_needed", "").strip()
            )
        if "figure" in source_basis and needs_constraint_repair:
            inferred = infer_figure_constraint(field)
            if not inferred:
                field_path = str(field.get("field_path", ""))
                if field_path.endswith(".figure"):
                    parent_path = field_path.rsplit(".", 1)[0]
                    inferred = infer_figure_constraint(
                        {
                            "field_path": parent_path,
                            "section_id": field.get("section_id", ""),
                        }
                    )
            if inferred:
                field["figure_constraint"] = inferred

    ensure_field_contract_metadata(result, field_registry)
    ensure_section_architecture_covers_fields(result, field_registry)
    ensure_literature_field_rule_contract(result, field_registry)

    # Default Step 8 postprocessing is intentionally domain-neutral. Any
    # domain-specific field injection should be implemented as an explicit,
    # opt-in profile outside this generic workflow.
    reference_field_contract = shared_context.get("reference_field_contract") or {}
    if reference_field_contract.get("available"):
        result["reference_field_contract"] = reference_field_contract
        result["reference_field_audit"] = normalize_reference_field_audit(
            reference_field_contract,
            result.get("reference_field_audit"),
        )
    return result

def finalize_result(shared_context, result):
    if not isinstance(result, dict):
        return result

    result = postprocess_result(shared_context, result)
    schema_definition = result.setdefault("schema_definition", {})
    field_registry = schema_definition.get("field_registry")
    if isinstance(field_registry, list):
        normalize_source_basis_values(field_registry)
    if "requirement_contract" in result:
        result["coverage_report"] = build_coverage_report(result)
    return result


def call_module(
    client,
    model,
    module_name,
    prompt,
    temperature=0,
    max_retries=1,
    context_validator=None,
):
    attempts = []
    current_prompt = prompt
    result = None
    errors = []
    for round_idx in range(max_retries + 1):
        raw_response = None
        try:
            raw_response = chat(
                client,
                model,
                current_prompt,
                temperature=temperature,
                max_tokens=MODULE_MAX_TOKENS.get(
                    module_name,
                    DEFAULT_MODULE_MAX_TOKENS,
                ),
            )
        except Exception as exc:
            errors = [f"{type(exc).__name__}: {exc}"]
            attempts.append(
                {
                    "round": round_idx + 1,
                    "prompt": current_prompt,
                    "raw_response": None,
                    "exception": errors[0],
                }
            )
            if round_idx < max_retries:
                current_prompt = build_module_redo_prompt(
                    module_name,
                    errors[0],
                    errors,
                    MODULE_SCHEMAS[module_name]["schema_text"],
                    original_prompt=prompt,
                )
                continue
            return result, errors, attempts

        attempts.append({"round": round_idx + 1, "prompt": current_prompt, "raw_response": raw_response})
        result, parse_errors = parse_json_response(raw_response)
        if parse_errors:
            errors = parse_errors
        else:
            result = normalize_module_result(module_name, result)
            errors = validate_module_result(module_name, result)
            if not errors and context_validator is not None:
                errors.extend(context_validator(result) or [])
        if not errors:
            break
        if round_idx < max_retries:
            current_prompt = build_module_redo_prompt(
                module_name,
                raw_response,
                errors,
                MODULE_SCHEMAS[module_name]["schema_text"],
                original_prompt=prompt,
            )
    return result, errors, attempts


def compact_shared_context_for_modules(
    shared_context,
    key_description_chars=500000,
    reference_chars=2000000,
):
    compact = dict(shared_context)
    if compact.get("key_description_text"):
        compact["key_description_text"] = _balanced_reference_excerpt(
            compact["key_description_text"],
            key_description_chars,
        )
    if compact.get("reference_paper_context"):
        compact["reference_paper_context"] = _truncate_text(
            compact["reference_paper_context"],
            reference_chars,
        )
    return compact


def get_checkpoint_path(args):
    checkpoint_output = getattr(args, "checkpoint_output", "")
    if checkpoint_output:
        return Path(checkpoint_output)
    output = Path(args.output)
    return output.with_suffix(output.suffix + ".state.json")


def save_pipeline_checkpoint(args, stage, module_outputs, module_attempts, module_errors, result=None, errors=None):
    checkpoint_path = get_checkpoint_path(args)
    payload = {
        "step": "step8_section_design_agent_checkpoint",
        "stage": stage,
        "model": getattr(args, "model", ""),
        "base_url": getattr(args, "base_url", ""),
        "inputs": {
            "database_goal": getattr(args, "database_goal", ""),
            "discipline": getattr(args, "discipline", ""),
            "query_requirements": getattr(args, "query_requirements", ""),
            "key_description_path": getattr(args, "key_description_path", ""),
            "reference_papers": getattr(args, "reference_papers", []),
        },
        "module_outputs": module_outputs,
        "module_errors": module_errors,
        "result": result,
        "validation_errors": errors or [],
        "attempts": {"modules": module_attempts},
        "run_identity": getattr(args, "_run_identity", None),
    }
    run_artifact_guard.atomic_write_json(
        checkpoint_path,
        payload,
        run_identity=getattr(args, "_run_identity", None),
    )


def should_stop_after_module(args, module_name):
    return getattr(args, "stop_after_module", "") == module_name


def stopped_after_module_error(module_name):
    return f"stopped_after_module: {module_name}"


def is_partial_stop(errors):
    return bool(errors) and len(errors) == 1 and str(errors[0]).startswith("stopped_after_module:")


def assemble_final_result_from_modules(
    shared_context,
    locating_result,
    query_result,
    subjective_result,
    topic_result,
    section_result,
    field_plan_result,
    supervisor_result,
    figure_result,
    schema_result,
    critic_result,
):
    shared_context = shared_context or {}
    locating_result = locating_result or {}
    query_result = query_result or {}
    subjective_result = subjective_result or {}
    topic_result = topic_result or {}
    section_result = section_result or {}
    field_plan_result = field_plan_result or {}
    supervisor_result = supervisor_result or {}
    figure_result = figure_result or {}
    schema_result = schema_result or {}
    critic_result = critic_result or {}
    topic_adjustments = list(topic_result.get("topic_specific_adjustments") or [])
    must_have_concepts = subjective_result.get("must_have_concepts") or []
    red_flags = subjective_result.get("red_flags") or []
    if must_have_concepts:
        topic_adjustments.append("Must-have concepts: " + ", ".join(map(str, must_have_concepts[:8])))
    if red_flags:
        topic_adjustments.append("Red flags checked: " + ", ".join(map(str, red_flags[:5])))
    topic_adjustments.append(
        "Figure classification enabled: "
        + str(bool(figure_result.get("enable_figure_classification")))
        + "; routing: "
        + str(supervisor_result.get("routing_decision", ""))
    )
    topic_adjustments.append(
        "Specialization critic: "
        + str(critic_result.get("specialization_status", "unknown"))
        + "; generic-template risk: "
        + str(bool(critic_result.get("is_generic")))
    )

    coverage_check = []
    for item in query_result.get("query_objects") or []:
        if not isinstance(item, dict):
            continue
        requirement = item.get("query_requirement", "")
        groups = item.get("recommended_field_groups") or []
        coverage_check.append(
            f"{requirement}: covered by {', '.join(map(str, groups[:5]))}"
            if groups
            else f"{requirement}: covered by section and field registry design"
        )
    if not coverage_check:
        coverage_check = [
            "Query requirements are covered by the section architecture, schema field registry, and evidence model."
        ]

    redo_needed = bool(critic_result.get("redo_needed")) or bool(critic_result.get("is_generic"))
    redo_reason = ""
    if redo_needed:
        redo_reason = "; ".join(map(str, critic_result.get("redo_directives") or []))

    requirement_contract = normalize_requirement_contract(shared_context, subjective_result)
    entity_registry = normalize_entity_registry(subjective_result, requirement_contract)
    reference_field_contract = shared_context.get("reference_field_contract") or {}
    reference_field_audit = normalize_reference_field_audit(
        reference_field_contract,
        field_plan_result.get("reference_field_audit"),
        field_plan_result.get("field_groups"),
    )
    field_plan_traceability = build_field_plan_traceability(field_plan_result)
    return {
        "database_positioning": {
            "database_goal": locating_result.get("database_goal", shared_context.get("database_goal", "")),
            "discipline": locating_result.get("discipline", shared_context.get("discipline", "")),
            "query_requirements": locating_result.get(
                "query_requirements",
                shared_context.get("query_requirements", []),
            ),
            "retrieval_unit": locating_result.get("retrieval_unit", ""),
            "design_rationale": locating_result.get("design_rationale", ""),
        },
        "section_design": {
            "core_sections": section_result.get("core_sections", []),
            "non_core_sections": section_result.get("non_core_sections", []),
        },
        "schema_definition": {
            "top_level_keys": schema_result.get("top_level_keys", []),
            "field_registry": schema_result.get("field_registry", []),
        },
        "reference_field_contract": reference_field_contract,
        "reference_field_audit": reference_field_audit,
        "field_plan_traceability": field_plan_traceability,
        "requirement_contract": requirement_contract,
        "entity_registry": entity_registry,
        "quality_check": {
            "topic_specific_adjustments": topic_adjustments,
            "coverage_check": coverage_check,
            "redo_needed": redo_needed,
            "redo_reason": redo_reason,
            "field_utility_audit": deepcopy(
                critic_result.get("field_utility_audit") or {}
            ),
        },
    }


def run_pipeline(client, args, shared_context):
    module_outputs = {}
    module_attempts = {}
    module_errors = {}

    locating_result, locating_errors, locating_attempts = call_module(
        client,
        args.model,
        "locating_module",
        build_locating_prompt(
            args.database_goal,
            args.discipline,
            shared_context["query_requirements"],
            shared_context["key_description_text"],
            shared_context.get("reference_paper_context", ""),
        ),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["locating_module"] = locating_result
    module_attempts["locating_module"] = locating_attempts
    module_errors["locating_module"] = locating_errors
    save_pipeline_checkpoint(args, "locating_module", module_outputs, module_attempts, module_errors, errors=locating_errors)
    if should_stop_after_module(args, "locating_module"):
        return None, [stopped_after_module_error("locating_module")], module_outputs, module_attempts, module_errors
    if locating_errors:
        return None, locating_errors, module_outputs, module_attempts, module_errors

    module_context = compact_shared_context_for_modules(shared_context)

    mechanism_result, mechanism_errors, mechanism_attempts = call_module(
        client,
        args.model,
        "mechanism_requirement_module",
        build_mechanism_requirement_prompt(module_context, locating_result),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["mechanism_requirement_module"] = mechanism_result
    module_attempts["mechanism_requirement_module"] = mechanism_attempts
    module_errors["mechanism_requirement_module"] = mechanism_errors
    save_pipeline_checkpoint(args, "mechanism_requirement_module", module_outputs, module_attempts, module_errors, errors=mechanism_errors)
    if should_stop_after_module(args, "mechanism_requirement_module"):
        return None, [stopped_after_module_error("mechanism_requirement_module")], module_outputs, module_attempts, module_errors
    if mechanism_errors:
        return None, mechanism_errors, module_outputs, module_attempts, module_errors

    query_result, query_errors, query_attempts = call_module(
        client,
        args.model,
        "query_semantics_module",
        build_query_semantics_prompt(module_context, locating_result),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["query_semantics_module"] = query_result
    module_attempts["query_semantics_module"] = query_attempts
    module_errors["query_semantics_module"] = query_errors
    save_pipeline_checkpoint(args, "query_semantics_module", module_outputs, module_attempts, module_errors, errors=query_errors)
    if should_stop_after_module(args, "query_semantics_module"):
        return None, [stopped_after_module_error("query_semantics_module")], module_outputs, module_attempts, module_errors
    if query_errors:
        return None, query_errors, module_outputs, module_attempts, module_errors

    evidence_result, evidence_errors, evidence_attempts = call_module(
        client,
        args.model,
        "evidence_model_module",
        build_evidence_model_prompt(module_context, locating_result),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["evidence_model_module"] = evidence_result
    module_attempts["evidence_model_module"] = evidence_attempts
    module_errors["evidence_model_module"] = evidence_errors
    save_pipeline_checkpoint(args, "evidence_model_module", module_outputs, module_attempts, module_errors, errors=evidence_errors)
    if should_stop_after_module(args, "evidence_model_module"):
        return None, [stopped_after_module_error("evidence_model_module")], module_outputs, module_attempts, module_errors
    if evidence_errors:
        return None, evidence_errors, module_outputs, module_attempts, module_errors

    subjective_result, subjective_errors, subjective_attempts = call_module(
        client,
        args.model,
        "subjective_supervisor_module",
        build_subjective_supervisor_prompt(
            module_context,
            locating_result,
            mechanism_result,
            query_result,
            evidence_result,
        ),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["subjective_supervisor_module"] = subjective_result
    module_attempts["subjective_supervisor_module"] = subjective_attempts
    module_errors["subjective_supervisor_module"] = subjective_errors
    save_pipeline_checkpoint(args, "subjective_supervisor_module", module_outputs, module_attempts, module_errors, errors=subjective_errors)
    if should_stop_after_module(args, "subjective_supervisor_module"):
        return None, [stopped_after_module_error("subjective_supervisor_module")], module_outputs, module_attempts, module_errors
    if subjective_errors:
        return None, subjective_errors, module_outputs, module_attempts, module_errors

    topic_result, topic_errors, topic_attempts = call_module(
        client,
        args.model,
        "topic_adaptation_module",
        build_topic_adaptation_prompt(module_context, locating_result),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["topic_adaptation_module"] = topic_result
    module_attempts["topic_adaptation_module"] = topic_attempts
    module_errors["topic_adaptation_module"] = topic_errors
    save_pipeline_checkpoint(args, "topic_adaptation_module", module_outputs, module_attempts, module_errors, errors=topic_errors)
    if should_stop_after_module(args, "topic_adaptation_module"):
        return None, [stopped_after_module_error("topic_adaptation_module")], module_outputs, module_attempts, module_errors
    if topic_errors:
        return None, topic_errors, module_outputs, module_attempts, module_errors

    section_result, section_errors, section_attempts = call_module(
        client,
        args.model,
        "section_partition_module",
        build_section_partition_prompt(
            module_context,
            locating_result,
            mechanism_result,
            query_result,
            evidence_result,
            subjective_result,
            topic_result,
        ),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["section_partition_module"] = section_result
    module_attempts["section_partition_module"] = section_attempts
    module_errors["section_partition_module"] = section_errors
    save_pipeline_checkpoint(args, "section_partition_module", module_outputs, module_attempts, module_errors, errors=section_errors)
    if should_stop_after_module(args, "section_partition_module"):
        return None, [stopped_after_module_error("section_partition_module")], module_outputs, module_attempts, module_errors
    if section_errors:
        return None, section_errors, module_outputs, module_attempts, module_errors

    field_plan_result, field_plan_errors, field_plan_attempts = call_module(
        client,
        args.model,
        "field_planning_module",
        build_field_planning_prompt(
            module_context,
            locating_result,
            mechanism_result,
            query_result,
            evidence_result,
            subjective_result,
            topic_result,
            section_result,
        ),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["field_planning_module"] = field_plan_result
    module_attempts["field_planning_module"] = field_plan_attempts
    module_errors["field_planning_module"] = field_plan_errors
    save_pipeline_checkpoint(args, "field_planning_module", module_outputs, module_attempts, module_errors, errors=field_plan_errors)
    if should_stop_after_module(args, "field_planning_module"):
        return None, [stopped_after_module_error("field_planning_module")], module_outputs, module_attempts, module_errors
    if field_plan_errors:
        return None, field_plan_errors, module_outputs, module_attempts, module_errors

    human_advice = shared_context.get("human_advice", "")
    if getattr(args, "require_human_advice_before_supervisor", False) and not human_advice:
        module_outputs["human_advice_gate"] = {
            "status": "waiting_for_human_advice",
            "position": "before_supervisor_module",
            "reason": (
                "Human advice is required before the supervisor can decide the "
                "figure-classification routing path."
            ),
            "expected_advice": (
                "Provide expert suggestions about figure ownership risks, sections "
                "that must be reviewed by the supervisor, and any fields that should "
                "or should not use figure evidence."
            ),
        }
        module_errors["human_advice_gate"] = ["needs_human_advice_before_supervisor"]
        module_attempts["human_advice_gate"] = []
        return (
            None,
            ["needs_human_advice_before_supervisor"],
            module_outputs,
            module_attempts,
            module_errors,
        )

    supervisor_result, supervisor_errors, supervisor_attempts = call_module(
        client,
        args.model,
        "supervisor_module",
        build_supervisor_prompt(
            module_context,
            locating_result,
            topic_result,
            section_result,
            field_plan_result,
            human_advice=human_advice,
        ),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["supervisor_module"] = supervisor_result
    module_attempts["supervisor_module"] = supervisor_attempts
    module_errors["supervisor_module"] = supervisor_errors
    save_pipeline_checkpoint(args, "supervisor_module", module_outputs, module_attempts, module_errors, errors=supervisor_errors)
    if should_stop_after_module(args, "supervisor_module"):
        return None, [stopped_after_module_error("supervisor_module")], module_outputs, module_attempts, module_errors
    if supervisor_errors:
        return None, supervisor_errors, module_outputs, module_attempts, module_errors

    figure_result, figure_errors, figure_attempts = call_module(
        client,
        args.model,
        "figure_classification_module",
        build_figure_classification_prompt(module_context, section_result, field_plan_result, supervisor_result),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["figure_classification_module"] = figure_result
    module_attempts["figure_classification_module"] = figure_attempts
    module_errors["figure_classification_module"] = figure_errors
    save_pipeline_checkpoint(args, "figure_classification_module", module_outputs, module_attempts, module_errors, errors=figure_errors)
    if should_stop_after_module(args, "figure_classification_module"):
        return None, [stopped_after_module_error("figure_classification_module")], module_outputs, module_attempts, module_errors
    if figure_errors:
        return None, figure_errors, module_outputs, module_attempts, module_errors
    if supervisor_result.get("enable_figure_classification") != figure_result.get("enable_figure_classification"):
        return (
            None,
            ["Supervisor decision and figure classification branch are inconsistent"],
            module_outputs,
            module_attempts,
            module_errors,
        )

    schema_result, schema_errors, schema_attempts = call_module(
        client,
        args.model,
        "schema_design_module",
        build_schema_design_prompt(
            module_context,
            locating_result,
            mechanism_result,
            query_result,
            evidence_result,
            subjective_result,
            topic_result,
            section_result,
            field_plan_result,
            supervisor_result,
            figure_result,
        ),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["schema_design_module"] = schema_result
    module_attempts["schema_design_module"] = schema_attempts
    module_errors["schema_design_module"] = schema_errors
    save_pipeline_checkpoint(args, "schema_design_module", module_outputs, module_attempts, module_errors, errors=schema_errors)
    if should_stop_after_module(args, "schema_design_module"):
        return None, [stopped_after_module_error("schema_design_module")], module_outputs, module_attempts, module_errors
    if schema_errors:
        return None, schema_errors, module_outputs, module_attempts, module_errors

    critic_result, critic_errors, critic_attempts = call_module(
        client,
        args.model,
        "specialization_critic_module",
        build_specialization_critic_prompt(
            module_context,
            mechanism_result,
            query_result,
            evidence_result,
            subjective_result,
            section_result,
            field_plan_result,
            schema_result,
        ),
        temperature=args.temperature,
        max_retries=args.max_retries,
        context_validator=lambda critic: validate_critic_field_utility(
            critic,
            schema_result,
        ),
    )
    module_outputs["specialization_critic_module"] = critic_result
    module_attempts["specialization_critic_module"] = critic_attempts
    module_errors["specialization_critic_module"] = critic_errors
    save_pipeline_checkpoint(args, "specialization_critic_module", module_outputs, module_attempts, module_errors, errors=critic_errors)
    if should_stop_after_module(args, "specialization_critic_module"):
        return None, [stopped_after_module_error("specialization_critic_module")], module_outputs, module_attempts, module_errors
    if critic_errors:
        return None, critic_errors, module_outputs, module_attempts, module_errors

    if getattr(args, "use_llm_aggregation", False):
        aggregation_prompt = build_aggregation_prompt(
            module_context,
            locating_result,
            mechanism_result,
            query_result,
            evidence_result,
            subjective_result,
            topic_result,
            section_result,
            field_plan_result,
            supervisor_result,
            figure_result,
            schema_result,
            critic_result,
        )
        final_result, final_errors, aggregation_attempts = call_module(
            client,
            args.model,
            "aggregation",
            aggregation_prompt,
            temperature=args.temperature,
            max_retries=0,
        )
    else:
        final_result = assemble_final_result_from_modules(
            shared_context,
            locating_result,
            query_result,
            subjective_result,
            topic_result,
            section_result,
            field_plan_result,
            supervisor_result,
            figure_result,
            schema_result,
            critic_result,
        )
        final_errors = []
        aggregation_attempts = [
            {
                "round": 1,
                "prompt": "deterministic_aggregation",
                "raw_response": json.dumps(final_result, ensure_ascii=False),
            }
        ]
    module_outputs["aggregation"] = final_result
    module_attempts["aggregation"] = aggregation_attempts
    module_errors["aggregation"] = final_errors
    save_pipeline_checkpoint(args, "aggregation", module_outputs, module_attempts, module_errors, result=final_result, errors=final_errors)
    if should_stop_after_module(args, "aggregation"):
        return None, [stopped_after_module_error("aggregation")], module_outputs, module_attempts, module_errors
    if final_errors:
        return None, final_errors, module_outputs, module_attempts, module_errors

    final_result = finalize_result(shared_context, final_result)
    validation_errors = validate_result(final_result)
    validation_errors.extend(validate_specialization(shared_context, module_outputs, final_result))
    save_pipeline_checkpoint(args, "final_validation", module_outputs, module_attempts, module_errors, result=final_result, errors=validation_errors)
    return final_result, validation_errors, module_outputs, module_attempts, module_errors


def build_step8_input_identity(args):
    input_paths = [
        getattr(args, "key_description_path", ""),
        *list(getattr(args, "reference_papers", []) or []),
    ]
    query_source = str(getattr(args, "query_requirements", "") or "")
    if query_source:
        try:
            if Path(query_source).exists():
                input_paths.append(query_source)
        except OSError:
            pass
    human_advice_path = str(getattr(args, "human_advice_path", "") or "")
    if human_advice_path:
        input_paths.append(human_advice_path)
    return run_artifact_guard.build_input_identity(
        "step8_section_design",
        {
            "database_goal": getattr(args, "database_goal", ""),
            "discipline": getattr(args, "discipline", ""),
            "query_requirements": query_source,
            "reference_papers": list(getattr(args, "reference_papers", []) or []),
            "model": getattr(args, "model", ""),
            "llm_backend": getattr(args, "llm_backend", ""),
            "temperature": getattr(args, "temperature", 0),
            "structured_protocol": bool(getattr(args, "structured_protocol", False)),
        },
        input_paths,
    )


def _run_section_design_reserved(args):
    client = get_client(
        base_url=args.base_url,
        api_key=args.api_key,
        backend=args.llm_backend,
        timeout=args.request_timeout,
    )
    query_requirements = load_query_requirements(args.query_requirements)
    key_description_text = load_key_description_text(args.key_description_path)
    reference_field_contract = build_reference_field_contract(
        key_description_text,
        args.key_description_path,
    )
    reference_paper_context = load_reference_paper_context(args.reference_papers)
    human_advice = (
        args.human_advice.strip()
        if args.human_advice
        else load_optional_text(args.human_advice_path)
    )
    shared_context = {
        "task_contract": dict(MATERIAL_LITERATURE_TASK_CONTRACT),
        "database_goal": args.database_goal,
        "discipline": args.discipline,
        "query_requirements": query_requirements,
        "key_description_text": key_description_text,
        "reference_field_contract": reference_field_contract,
        "reference_paper_context": reference_paper_context,
        "reference_paper_count": len(args.reference_papers or []),
        "human_advice": human_advice,
        "shared_context_block": build_shared_context_block(
            args.database_goal,
            args.discipline,
            query_requirements,
            key_description_text,
            reference_paper_context,
        ),
    }

    attach_to_context(shared_context, getattr(args, "domain_knowledge_pack", ""))
    result, errors, module_outputs, module_attempts, module_errors = run_pipeline(client, args, shared_context)
    final_attempts = []

    if result is not None and errors and args.max_retries > 0:
        redo_prompt = build_redo_prompt(json.dumps(result, ensure_ascii=False, indent=2), errors)
        raw_response = chat(client, args.model, redo_prompt, temperature=args.temperature)
        final_attempts.append({"round": 1, "prompt": redo_prompt, "raw_response": raw_response})
        redo_result, parse_errors = parse_json_response(raw_response)
        if not parse_errors:
            redo_result = finalize_result(shared_context, redo_result)
            redo_validation_errors = validate_result(redo_result)
            redo_validation_errors.extend(validate_specialization(shared_context, module_outputs, redo_result))
            previous_report = (result or {}).get("coverage_report") or build_coverage_report(result or {})
            candidate_report = redo_result.get("coverage_report") or build_coverage_report(redo_result)
            previous_structural = [error for error in errors if not str(error).startswith("coverage:")]
            candidate_structural = [
                error for error in redo_validation_errors if not str(error).startswith("coverage:")
            ]
            candidate_is_better = not redo_validation_errors or (
                len(candidate_structural) <= len(previous_structural)
                and is_coverage_improvement(previous_report, candidate_report)
            )
            if candidate_is_better:
                result = redo_result
                errors = redo_validation_errors

    output_inputs = {
        "task_contract": dict(MATERIAL_LITERATURE_TASK_CONTRACT),
        "database_goal": args.database_goal,
        "discipline": args.discipline,
        "query_requirements": query_requirements,
        "key_description_path": str(Path(args.key_description_path)),
        "reference_field_contract": {
            key: reference_field_contract.get(key)
            for key in ("source_sha256", "all_path_count", "leaf_count", "policy")
        },
        "reference_papers": args.reference_papers,
        "human_advice_required_before_supervisor": args.require_human_advice_before_supervisor,
        "human_advice_provided": bool(human_advice),
    }
    status = "success" if not errors else "needs_review"
    if is_partial_stop(errors):
        status = "partial_success"
    completed_modules = [
        name
        for name in REQUIRED_PIPELINE_MODULES
        if isinstance(module_outputs.get(name), dict) and not module_errors.get(name)
    ]
    all_required_modules_completed = len(completed_modules) == len(REQUIRED_PIPELINE_MODULES)

    output = {
        "step": "step8_section_design_agent",
        "framework": STEP8_FRAMEWORK,
        "run_identity": getattr(args, "_run_identity", None),
        "inputs": output_inputs,
        "agent_flow": build_agent_flow_trace(output_inputs, module_outputs, result, errors),
        "protocol_messages": build_step8_protocol_messages(
            output_inputs,
            module_outputs,
            result,
            errors,
            module_errors,
        ),
        "module_outputs": module_outputs,
        "module_errors": module_errors,
        "result": result,
        "validation_errors": errors,
        "status": status,
        "execution_provenance": {
            "mode": "online_full_pipeline" if all_required_modules_completed else "online_partial_pipeline",
            "fallback_used": False,
            "all_required_modules_completed": all_required_modules_completed,
            "completed_module_count": len(completed_modules),
            "required_module_count": len(REQUIRED_PIPELINE_MODULES),
            "request_timeout_seconds": args.request_timeout,
            "module_max_tokens": dict(MODULE_MAX_TOKENS),
        },
        "attempts": {
            "modules": module_attempts,
            "final_redo": final_attempts,
        },
    }
    output["protocol_validation"] = agent_protocol.validate_message_list(output["protocol_messages"])

    output_path = Path(args.output)
    run_artifact_guard.atomic_write_json(
        output_path,
        output,
        run_identity=getattr(args, "_run_identity", None),
    )
    print(f"Saved result to {output_path}")
    print(f"Status: {output['status']}")
    if errors:
        print("Validation errors:")
        for item in errors:
            print(f"- {item}")
    return output


def run_section_design(args):
    input_identity = build_step8_input_identity(args)
    checkpoint_path = get_checkpoint_path(args)
    run_identity = run_artifact_guard.reserve_fresh_run(
        pipeline="step8_section_design",
        output_path=args.output,
        input_identity=input_identity,
        artifact_paths=[args.output, checkpoint_path],
    )
    args._run_identity = run_identity
    try:
        output = _run_section_design_reserved(args)
    except BaseException as exc:
        run_artifact_guard.update_run_status(
            run_identity,
            "interrupted",
            error_type=type(exc).__name__,
        )
        raise
    run_artifact_guard.update_run_status(
        run_identity,
        "completed",
        result_status=output.get("status"),
    )
    return output


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-knowledge-pack", default="")
    parser.add_argument("--database-goal", required=True)
    parser.add_argument("--discipline", required=True)
    parser.add_argument("--query-requirements", required=True)
    parser.add_argument("--key-description-path", required=True)
    parser.add_argument("--reference-papers", nargs="*", default=[])
    parser.add_argument("--output", default="./section_design_output.json")
    parser.add_argument("--model", default=os.getenv("SECTION_AGENT_MODEL", DEFAULT_DEEPSEEK_MODEL))
    parser.add_argument("--base-url", default=os.getenv("SECTION_AGENT_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL))
    parser.add_argument("--api-key", default=os.getenv("SECTION_AGENT_API_KEY"))
    parser.add_argument(
        "--llm-backend",
        choices=["openai", "langchain", "qiniu"],
        default=os.getenv("SECTION_AGENT_LLM_BACKEND", DEFAULT_LLM_BACKEND),
        help=(
            "LLM client backend. Use `langchain` for ChatOpenAI or `qiniu` "
            "for the gateway's explicit SSE streaming adapter."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=float(os.getenv("SECTION_AGENT_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT_SECONDS)),
        help="Per-request timeout in seconds for OpenAI-compatible chat calls.",
    )
    parser.add_argument(
        "--checkpoint-output",
        default=os.getenv("SECTION_AGENT_CHECKPOINT_OUTPUT", ""),
        help="Optional checkpoint JSON path. Defaults to <output>.state.json.",
    )
    parser.add_argument(
        "--stop-after-module",
        choices=[
            "",
            "locating_module",
            "mechanism_requirement_module",
            "query_semantics_module",
            "evidence_model_module",
            "subjective_supervisor_module",
            "topic_adaptation_module",
            "section_partition_module",
            "field_planning_module",
            "supervisor_module",
            "figure_classification_module",
            "schema_design_module",
            "specialization_critic_module",
            "aggregation",
        ],
        default="",
        help="Stop after a module and save partial output. Useful for debugging slow providers.",
    )
    parser.add_argument(
        "--use-llm-aggregation",
        action="store_true",
        help="Use the legacy LLM aggregation module instead of deterministic assembly.",
    )
    parser.add_argument(
        "--require-human-advice-before-supervisor",
        action="store_true",
        help="Stop before supervisor_module unless human advice is provided.",
    )
    parser.add_argument(
        "--human-advice",
        default="",
        help="Human expert advice injected before supervisor_module.",
    )
    parser.add_argument(
        "--human-advice-path",
        default="",
        help="Path to a UTF-8 text file containing human expert advice.",
    )
    return parser


if __name__ == "__main__":
    run_section_design(build_parser().parse_args())
