"""Carry per-field semantics from planning through deterministic schema assembly."""

from copy import deepcopy
import os


VERSION = "materials-field-definitions/v1"
TEXT_KEYS = ("description", "extraction_notes", "data_type", "inclusion_rule", "absence_rule")
COPY_KEYS = (*TEXT_KEYS, "required", "source_basis", "concept_ids", "relation_constraints", "evidence_requirements", "object_contract")


def path_key(path):
    return str(path or "").strip().replace("[]", "")


def definitions_required(plan):
    return bool(plan.get("field_definition_contract_version")) or os.getenv("FIELD_DEFINITIONS_REQUIRED") == "1"


def validate_definitions(plan):
    if not isinstance(plan, dict):
        return ["field_definitions: plan must be an object"]
    if not definitions_required(plan):
        return []
    errors = []
    if plan.get("field_definition_contract_version") != VERSION:
        errors.append(f"field_definitions: field_definition_contract_version must be {VERSION}")
    seen = set()
    for group in plan.get("field_groups") or []:
        if not isinstance(group, dict):
            errors.append("field_definitions: field group must be an object")
            continue
        paths = group.get("recommended_fields") or []
        definitions = group.get("field_definitions")
        if not isinstance(definitions, list):
            errors.append("field_definitions: each group needs a field_definitions list")
            continue
        indexed = {}
        for definition in definitions:
            if not isinstance(definition, dict):
                errors.append("field_definitions: definition must be an object")
                continue
            path = str(definition.get("field_path") or "").strip()
            key = path_key(path)
            if not key or key in seen:
                errors.append(f"field_definitions: empty or duplicate field_path {path}")
            seen.add(key)
            indexed[path] = definition
            for name in TEXT_KEYS:
                if not isinstance(definition.get(name), str) or not definition[name].strip():
                    errors.append(f"field_definitions: {path} needs nonempty {name}")
            for name, limit in (("description", 1200), ("extraction_notes", 2400)):
                if isinstance(definition.get(name), str) and len(definition[name]) > limit:
                    errors.append(f"field_definitions: {path} {name} exceeds {limit} characters; retain semantics concisely")
            if definition.get("description") in (path, path.rsplit(".", 1)[-1], group.get("purpose")):
                errors.append(f"field_definitions: {path} needs its own scientific definition, not its name or group purpose")
            if not isinstance(definition.get("required"), bool):
                errors.append(f"field_definitions: {path} required must be boolean")
            sources = definition.get("source_basis")
            if not isinstance(sources, list) or not sources or any(not isinstance(s, str) or not s.strip() for s in sources):
                errors.append(f"field_definitions: {path} needs source_basis")
            relations = definition.get("relation_constraints")
            if not isinstance(relations, dict) or any(not isinstance(relations.get(k), bool) for k in
                    ("entity_binding_required", "condition_binding_required", "separate_instances")):
                errors.append(f"field_definitions: {path} needs boolean binding and instance rules")
            elif "[]" in path and not relations["separate_instances"]:
                errors.append(f"field_definitions: {path} repeated ancestor requires separate_instances=true")
            evidence = definition.get("evidence_requirements")
            if not isinstance(evidence, dict) or any(not isinstance(evidence.get(k), bool) for k in
                    ("direct_support_required", "locator_required")) or not isinstance(evidence.get("allowed_source_types"), list) or not evidence["allowed_source_types"]:
                errors.append(f"field_definitions: {path} needs an evidence contract")
            if "object" in str(definition.get("data_type", "")).lower():
                contract = definition.get("object_contract")
                if not isinstance(contract, dict) or not isinstance(contract.get("fields"), dict) or not contract["fields"]:
                    errors.append(f"field_definitions: {path} object needs typed payload children")
        if set(indexed) != set(paths):
            errors.append(f"field_definitions: definitions must exactly match recommended_fields (missing={sorted(set(paths)-set(indexed))}, extra={sorted(set(indexed)-set(paths))})")
    return errors


def apply_definitions(registry, plan):
    errors = validate_definitions(plan)
    if errors:
        raise ValueError("; ".join(errors))
    definitions = {
        path_key(item.get("field_path")): item
        for group in plan.get("field_groups") or [] if isinstance(group, dict)
        for item in group.get("field_definitions") or [] if isinstance(item, dict)
    }
    missing = set(definitions) - {path_key(field.get("field_path")) for field in registry}
    if missing:
        raise ValueError(f"field_definitions: compiled schema dropped planned fields {sorted(missing)}")
    for field in registry:
        definition = definitions.get(path_key(field.get("field_path")))
        if not definition:
            continue
        for name in COPY_KEYS:
            if name in definition:
                field[name] = deepcopy(definition[name])
        if "object" not in field["data_type"].lower():
            field.pop("object_contract", None)
        declared_path = definition["field_path"]
        repeated = []
        parts = declared_path.split(".")
        for index, part in enumerate(parts):
            if part.endswith("[]"):
                repeated.append(".".join(parts[:index + 1]))
        field.setdefault("field_plan_trace", {}).update({
            "definition_contract_version": VERSION,
            "recommended_field": declared_path,
        })
        # Canonical registry paths historically omit []; retain cardinality in the
        # executable relation contract passed to extraction instead of losing it.
        if repeated:
            field["relation_constraints"]["repeatable_ancestors"] = repeated
    return registry
