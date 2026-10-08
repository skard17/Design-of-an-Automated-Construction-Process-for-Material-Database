import hashlib
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import prompt_quality_code_agent as prompt_agent
import materials_agent_protocol as agent_protocol
import run_artifact_guard


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def read_text(path: str, max_chars: int) -> str:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.65)]
    tail = text[-int(max_chars * 0.35) :]
    return f"{head}\n\n[... middle truncated for bench extraction ...]\n\n{tail}"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:120] or "item"


LATEX_SYMBOLS = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "theta": "θ",
    "lambda": "λ",
    "mu": "μ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "phi": "φ",
    "omega": "ω",
    "Delta": "Δ",
    "Theta": "Θ",
    "Phi": "Φ",
    "Omega": "Ω",
    "times": "×",
    "cdot": "·",
    "pm": "±",
    "approx": "≈",
    "sim": "∼",
    "leq": "≤",
    "geq": "≥",
    "to": "→",
    "rightarrow": "→",
    "leftarrow": "←",
    "partial": "∂",
    "nabla": "∇",
    "infty": "∞",
    "degree": "°",
    "circ": "°",
    "hbar": "ℏ",
    "angstrom": "Å",
}
SUPERSCRIPTS = str.maketrans("0123456789+-=()nix", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱˣ")
SUBSCRIPTS = str.maketrans("0123456789+-=()aeijmnoprstux", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢⱼₘₙₒₚᵣₛₜᵤₓ")


def decode_unicode_escapes(text: str) -> str:
    if "\\u" not in text and "\\U" not in text:
        return text
    try:
        return text.encode("utf-8").decode("unicode_escape", errors="replace")
    except Exception:
        return text


def replace_latex_commands(text: str) -> str:
    return re.sub(r"\\([A-Za-z]+)", lambda m: LATEX_SYMBOLS.get(m.group(1), m.group(0)), text)


def replace_supersubs(text: str) -> str:
    placeholder = "__PROTECTED_UNDERSCORE__"
    text = re.sub(r"([A-Za-z]{2,})_([A-Za-z]{2,})", lambda m: m.group(1) + placeholder + m.group(2), text)
    text = re.sub(r"\^\{([^{}]+)\}", lambda m: m.group(1).translate(SUPERSCRIPTS), text)
    text = re.sub(r"_\{([^{}]+)\}", lambda m: m.group(1).translate(SUBSCRIPTS), text)
    text = re.sub(r"\^([+\-]?[0-9]+)", lambda m: m.group(1).translate(SUPERSCRIPTS), text)
    text = re.sub(r"_([+\-]?[0-9]+)", lambda m: m.group(1).translate(SUBSCRIPTS), text)
    return text.replace(placeholder, "_")


def normalize_formula_digits(text: str) -> str:
    element = r"(?:[A-Z][a-z]?)"
    token_pattern = re.compile(rf"\b(?:{element}(?:[0-9]+(?:\.[0-9]+)?|[A-Z][a-z]?))+[A-Za-z0-9.()\-]*\b")

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        if not re.search(rf"{element}[0-9]", token):
            return token
        return re.sub(r"(?<=[A-Za-z)])([0-9]+(?:\.[0-9]+)?)", lambda n: n.group(1).translate(SUBSCRIPTS), token)

    return token_pattern.sub(repl, text)


def normalize_scientific_unicode(text: str) -> str:
    normalized = unicodedata.normalize("NFC", decode_unicode_escapes(str(text)))
    replacements = {
        "渭0": "μ0",
        "米0": "μ0",
        "渭m": "μm",
        "米m": "μm",
        "渭": "μ",
        "卤": "±",
        "掳": "°",
        "N谷el": "Néel",
        "N谷el": "Néel",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = replace_supersubs(replace_latex_commands(normalized))
    normalized = normalize_formula_digits(normalized)
    return normalized


def normalize_strings_deep(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_scientific_unicode(value)
    if isinstance(value, list):
        return [normalize_strings_deep(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_strings_deep(item) for key, item in value.items()}
    return value


def stage_field_specs(prompt_output: dict[str, Any], stage_id: str) -> list[dict[str, Any]]:
    prompts = (
        (prompt_output.get("module_outputs") or {})
        .get("section_extraction_prompt_module", {})
        .get("section_extraction_prompts", {})
    )
    contract = (prompts.get(stage_id) or {}).get("output_contract") or {}
    specs = [
        dict(item)
        for item in contract.get("field_specs", []) or []
        if isinstance(item, dict) and item.get("field_path")
    ]
    if specs:
        return specs
    return [
        {"field_path": str(item), "data_type": "unspecified", "description": ""}
        for item in contract.get("fields", []) or []
        if item
    ]


def stage_fields(prompt_output: dict[str, Any], stage_id: str) -> list[str]:
    """Return the complete stage field inventory; batching happens separately."""
    return [item["field_path"] for item in stage_field_specs(prompt_output, stage_id)]


def chunk_fields(fields: list[str], max_fields: int) -> list[list[str]]:
    if not max_fields or max_fields <= 0 or len(fields) <= max_fields:
        return [fields]
    return [fields[index : index + max_fields] for index in range(0, len(fields), max_fields)]


def classify_extraction_error(error: Any) -> str:
    text = str(error or "").casefold()
    if "http error 429" in text or "rate limit" in text or "tpm" in text:
        return "rate_limit"
    if any(token in text for token in ("http error 500", "http error 502", "http error 503", "http error 504")):
        return "transient_http"
    if any(token in text for token in ("transport error", "timed out", "timeout")):
        return "transport_timeout"
    if any(token in text for token in ("jsondecodeerror", "expecting value", "unterminated string", "invalid json")):
        return "response_json_parse"
    return "extraction_error"


def merge_stage_payloads(
    payloads: list[dict[str, Any]],
    stage: dict[str, Any],
    document_name: str,
    field_batches: list[list[str]],
) -> dict[str, Any]:
    extracted = []
    missing = []
    unresolved = []
    quality_notes = []
    seen_extracted = set()
    seen_missing = set()
    seen_unresolved = set()

    for batch_index, payload in enumerate(payloads):
        if payload.get("error") or payload.get("raw_text"):
            return {
                "stage_id": stage.get("stage_id"),
                "section_id": stage.get("section_id"),
                "document": document_name,
                "error": payload.get("error", "batch extraction failed"),
                "error_type": payload.get("error_type") or classify_extraction_error(payload.get("error")),
                "raw_text": payload.get("raw_text", ""),
                "batch_index": batch_index,
            }
        for item in payload.get("extracted_fields") or []:
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("field_path") or ""),
                json.dumps(item.get("value"), ensure_ascii=False, sort_keys=True, default=str),
                str(item.get("evidence_text") or ""),
            )
            if key in seen_extracted:
                continue
            seen_extracted.add(key)
            extracted.append(item)
        for item in payload.get("missing_fields") or []:
            if not isinstance(item, dict):
                continue
            key = (str(item.get("field_path") or ""), str(item.get("missing_reason") or ""))
            if key in seen_missing:
                continue
            seen_missing.add(key)
            missing.append(item)
        for item in payload.get("unresolved_fields") or []:
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("field_path") or ""),
                str(item.get("unresolved_reason") or ""),
                json.dumps(item.get("candidate_value"), ensure_ascii=False, sort_keys=True, default=str),
            )
            if key in seen_unresolved:
                continue
            seen_unresolved.add(key)
            unresolved.append(item)
        quality_notes.extend(payload.get("quality_notes") or [])

    return {
        "stage_id": stage.get("stage_id"),
        "section_id": stage.get("section_id"),
        "document": document_name,
        "extracted_fields": extracted,
        "missing_fields": missing,
        "unresolved_fields": unresolved,
        "quality_notes": quality_notes,
        "batching": {
            "enabled": len(field_batches) > 1,
            "batch_count": len(field_batches),
            "fields_per_batch": [len(batch) for batch in field_batches],
        },
    }


def stage_specific_rules(stage: dict[str, Any]) -> list[str]:
    stage_id = str(stage.get("stage_id", ""))
    section_id = str(stage.get("section_id", ""))
    rules = []
    if section_id == "material_info.section2":
        rules.extend(
            [
                "For section2, extract only fabrication, growth, synthesis, processing, treatment, sample preparation, device fabrication, or gating/intercalation steps.",
                "Do not put characterization or property-measurement procedures into section2 unless the field explicitly asks for measurement-preparation conditions.",
                "If a sentence says measurements were performed, spectra were measured, MR/Hall/XRD/XAS/SEMPA/SQUID was carried out, route it to section3/section4 context instead of section2.",
                "Section2 figures/tables are allowed only when they describe fabrication, growth, synthesis, treatment, device fabrication, processing conditions, or sample preparation; reject property curves, characterization figures, and measurement-result figures.",
            ]
        )
    if section_id == "material_info.section3":
        rules.extend(
            [
                "For section3, extract microscopic, structural, compositional, spectroscopic, diffraction, microscopy, or electronic-structure evidence.",
                "Do not create extracted_fields for techniques that are not mentioned; put them in missing_fields.",
            ]
        )
    if section_id == "material_info.section4":
        rules.extend(
            [
                "For section4, extract only macroscopic property curves/maps/plots and their figure/table references.",
                "Do not create extracted_fields for absent curve types; put them in missing_fields.",
            ]
        )
    if section_id == "material_info.section1":
        rules.extend(
            [
                "For section1, extract scalar/structured property values only when the paper reports a concrete value, trend, transition, or qualitative property statement.",
                "Do not emit placeholder records for unrelated property families.",
            ]
        )
    if stage_id == "figure_classification":
        rules.extend(
            [
                "Use only canonical allowed section IDs: paper_info, material_info.section0, material_info.section1, material_info.section2, material_info.section3, material_info.section4, section5.",
                "Classify figures by what evidence they support, not by the paper's numbered section headings.",
                "Always emit field_path exactly as figure_classification.figures.",
                "Each extracted value must be an object with figure_id, panel_id, caption_or_evidence, evidence_type, assigned_section, allowed_sections, prohibited_sections, assignment_reason, and confidence.",
                "assigned_section must be one canonical section ID, not a paper section number.",
                "material_info.section2 is only for process, fabrication, synthesis, treatment, growth, device fabrication, or processing-condition evidence.",
                "Never assign magnetic/property curves, transport curves, Hall/MR plots, susceptibility, magnetization, thermodynamic, optical, electrochemical, mechanical, characterization, microscopy, diffraction, spectroscopy, theory, simulation, or mechanism figures to material_info.section2.",
                "Route property curves/maps/plots to material_info.section4; route microscopy/diffraction/spectroscopy/structural/electronic characterization to material_info.section3; route theory/simulation/mechanism schematics to section5.",
            ]
        )
    return rules


def build_stage_prompt(
    document_text: str,
    document_name: str,
    stage: dict[str, Any],
    fields: list[str],
    dependency_context: dict[str, Any],
    field_specs: list[dict[str, Any]] | None = None,
) -> str:
    specific_rules = "\n".join(f"- {rule}" for rule in stage_specific_rules(stage))
    specs = field_specs or [
        {"field_path": field_path, "data_type": "unspecified", "description": ""}
        for field_path in fields
    ]
    return f"""
You are running a section-wise materials literature extraction bench.

Document: {document_name}
Stage: {stage.get("stage_id")}
Section: {stage.get("section_id")}
Purpose: {stage.get("purpose")}
Dependencies: {stage.get("depends_on", [])}
Dependency outputs already available:
{json_dumps(dependency_context)[:6000]}

Allowed field paths for this stage:
{json_dumps(fields)}

Authoritative field definitions for this batch:
{json_dumps(specs)}

Rules:
- The fixed task is automated construction of a materials database from parsed scientific literature. Do not extract textbook knowledge, laboratory-note content, generic report summaries, or unrelated document facts.
- Return JSON only.
- Extract only information supported by the document text.
- Do not invent values.
- Match the exact semantics of each field path. Do not substitute a related descriptor, mechanism, oxidation state, electronic configuration, preparation statement, or hypothetical structure for the requested quantity.
- Treat each field definition's description, extraction_notes, data_type, source_basis, object_contract, and figure_constraint as binding. A similar-sounding term is not enough when it violates that contract.
- Keep independently queryable values, units, criteria, directions, conditions, methods, sample bindings, and evidence locations in their designated fields. Do not collapse them into a generic summary object.
- For array or object fields, emit one record per material/sample/condition instance as required by the object contract; do not merge measurements made under materially different conditions.
- A characterization technique counts as extracted only when the document reports an actual observation, result, spectrum, image, curve, map, or measured value from that technique. Merely naming a technique or describing sample preparation is not a result.
- Do not turn "implied", "likely", "consistent with", "assumed", "presumed", or model-dependent interpretations into explicit facts. Extract an inferred classification only when the allowed field explicitly models inference and the output records its assignment basis, source type, confidence, and direct evidence.
- Never put null, "not mentioned", "not reported", "none", "N/A", or empty values in extracted_fields.
- If a requested field is absent, put it under missing_fields with field_path and a short missing_reason.
- If relevant literature evidence exists but its entity ownership, conditions, interpretation, or support is genuinely ambiguous, put it under unresolved_fields with the candidate value when available, evidence_text, source_hint, and unresolved_reason. Do not force it into extracted_fields or missing_fields.
- Account for every allowed field path exactly once per material/sample: with direct evidence in extracted_fields, an explicit absence in missing_fields, or a reviewable ambiguity in unresolved_fields.
- Every extracted value must include field_path, non-null value, evidence_text, confidence, and source_hint.
- evidence_text must directly support the value and field meaning; nearby topical text is insufficient.
- If the document has multiple material systems, keep material_system or sample_id on each extracted item when possible.
- Preserve DOI/arXiv/title metadata when the stage is paper_info.
- For figure_classification, classify figures/tables/panels by evidence type and allowed section.
- Do not use the paper's Introduction/Results section numbering as the output section ID.
{specific_rules}

Required JSON shape:
{{
  "stage_id": "{stage.get("stage_id")}",
  "section_id": "{stage.get("section_id")}",
  "document": "{document_name}",
  "extracted_fields": [
    {{
      "field_path": "...",
      "value": "...",
      "unit": null,
      "material_system": null,
      "evidence_text": "...",
      "source_hint": "page/section/figure/table if available",
      "confidence": 0.0
    }}
  ],
  "missing_fields": [
    {{"field_path": "...", "missing_reason": "..."}}
  ],
  "unresolved_fields": [
    {{
      "field_path": "...",
      "candidate_value": null,
      "evidence_text": "...",
      "source_hint": "page/section/figure/table if available",
      "unresolved_reason": "ambiguous entity, conditions, interpretation, or evidence support"
    }}
  ],
  "quality_notes": []
}}

Document text:
<<<
{document_text}
>>>
""".strip()


NULL_LIKE_VALUES = {
    "",
    "null",
    "none",
    "not mentioned",
    "not reported",
    "not provided",
    "not available",
    "not observed",
    "not specified",
    "n/a",
    "na",
    "no",
    "not found",
}
NULL_LIKE_PREFIX_PATTERN = re.compile(
    r"^(?:"
    r"not\s+(?:explicitly\s+)?(?:mentioned|reported|provided|available|observed|found|specified|extracted|given|stated|measured|determined)"
    r"|no\s+(?:(?:explicit|reported|available|measurable)\s+)?(?:value|data|result|information|measurement|evidence)\b"
    r"|no\s+.{1,80}\s+(?:was|were|is|are)\s+(?:reported|provided|given|stated|measured|determined)\b"
    r")\b",
    flags=re.IGNORECASE,
)
MEASUREMENT_LEAK_PATTERNS = (
    "measurements were performed",
    "measurement was performed",
    "measurements were carried out",
    "measurement was carried out",
    "spectra",
    "spectrum",
    "xrd measurement",
    "xas measurement",
    "hall signal",
    "magnetoresistance",
    "diffraction pattern",
    "microscopy image",
    "magnetization",
    "susceptibility",
    "hysteresis",
    "transport curve",
    "resistivity",
    "property curve",
)
PROCESS_KEEP_PATTERNS = (
    "grown",
    "deposited",
    "sputter",
    "anneal",
    "synthes",
    "fabricat",
    "prepared",
    "milled",
    "doped",
    "gating",
    "intercalat",
    "growth condition",
    "te-rich",
    "te-poor",
)
CANONICAL_SECTION_IDS = {
    "paper_info",
    "material_info.section0",
    "material_info.section1",
    "material_info.section2",
    "material_info.section3",
    "material_info.section4",
    "section5",
}
FIGURE_PROPERTY_CURVE_PATTERNS = (
    "property curve",
    "m-h",
    "m_h",
    "m vs h",
    "m-h loop",
    "m-t",
    "m_t",
    "m vs t",
    "hysteresis",
    "magnetization",
    "susceptibility",
    "chi",
    "hall",
    "magnetoresistance",
    "resistivity",
    "transport",
    "ac susceptibility",
    "hall resistivity",
    "topological hall",
    "mr curve",
    "r-h",
    "r-t",
    "heat capacity",
    "specific heat",
    "iv curve",
    "i-v",
    "j-v",
    "stress-strain",
    "eis",
    "cv curve",
)
FIGURE_CHARACTERIZATION_PATTERNS = (
    "xrd",
    "xas",
    "xps",
    "tem",
    "stem",
    "sem",
    "stm",
    "afm",
    "ltem",
    "diffraction",
    "microscopy",
    "spectrum",
    "spectra",
    "spectroscopy",
    "raman",
    "eds",
    "edx",
    "arpes",
    "neutron",
    "composition map",
    "elemental map",
)
FIGURE_THEORY_PATTERNS = (
    "theory",
    "theoretical",
    "simulation",
    "calculation",
    "calculated",
    "dft",
    "model",
    "mechanism",
    "schematic",
    "dmi",
    "bloch",
    "neel",
    "free energy",
    "phase diagram",
)
FIGURE_PROCESS_PATTERNS = (
    "synthesis",
    "growth",
    "grown",
    "deposition",
    "deposited",
    "fabrication",
    "fabricated",
    "anneal",
    "annealing",
    "treatment",
    "processing",
    "prepared",
    "sample preparation",
    "device fabrication",
    "intercalation",
    "gating",
    "sputter",
    "milled",
    "reaction route",
)


def is_null_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        text = value.strip()
        return text.lower() in NULL_LIKE_VALUES or bool(NULL_LIKE_PREFIX_PATTERN.match(text))
    if isinstance(value, list):
        return not value or all(is_null_like(item) for item in value)
    if isinstance(value, dict):
        semantic_values = [item for key, item in value.items() if key not in {"unit", "confidence", "source_hint"}]
        return not semantic_values or all(is_null_like(item) for item in semantic_values)
    return False


def record_owner(item: dict[str, Any]) -> str:
    return str(item.get("material_system") or item.get("sample_id") or "").strip().casefold()


def extracted_identity(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(item.get("field_path") or "").strip(),
        json.dumps(item.get("value"), ensure_ascii=False, sort_keys=True, default=str),
        str(item.get("unit") or "").strip().casefold(),
        record_owner(item),
    )


def evidence_item(item: dict[str, Any]) -> dict[str, Any] | None:
    evidence = {
        key: item.get(key)
        for key in ("evidence_text", "source_hint", "confidence")
        if item.get(key) not in {None, ""}
    }
    return evidence or None


def deduplicate_extracted_fields(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for item in items:
        key = extracted_identity(item)
        if key not in merged:
            merged[key] = dict(item)
            existing_evidence = merged[key].get("evidence_items")
            if not isinstance(existing_evidence, list):
                existing_evidence = []
            current_evidence = evidence_item(item)
            if current_evidence and current_evidence not in existing_evidence:
                existing_evidence.append(current_evidence)
            if existing_evidence:
                merged[key]["evidence_items"] = existing_evidence
            continue

        target = merged[key]
        evidence_items = target.setdefault("evidence_items", [])
        for candidate in item.get("evidence_items") or []:
            if isinstance(candidate, dict) and candidate not in evidence_items:
                evidence_items.append(candidate)
        current_evidence = evidence_item(item)
        if current_evidence and current_evidence not in evidence_items:
            evidence_items.append(current_evidence)
        confidences = [value for value in (target.get("confidence"), item.get("confidence")) if isinstance(value, (int, float))]
        if confidences:
            target["confidence"] = max(confidences)
    return list(merged.values())


def reconcile_missing_fields(
    extracted: list[dict[str, Any]], missing: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    extracted_owners: dict[str, set[str]] = {}
    for item in extracted:
        extracted_owners.setdefault(str(item.get("field_path") or "").strip(), set()).add(record_owner(item))

    reconciled = []
    seen = set()
    for item in missing:
        if not isinstance(item, dict):
            continue
        field_path = str(item.get("field_path") or "").strip()
        owner = record_owner(item)
        owners_with_values = extracted_owners.get(field_path, set())
        if owners_with_values and (not owner or owner in owners_with_values or "" in owners_with_values):
            continue
        key = (field_path, owner, str(item.get("missing_reason") or "").strip())
        if key not in seen:
            seen.add(key)
            reconciled.append(item)
    return reconciled


def assess_stage_yield(
    document_checks: list[dict[str, Any]], expected_fields: list[str], *, stage_id: str
) -> dict[str, Any]:
    expected = {str(field).strip() for field in expected_fields if str(field).strip()}
    extracted_total = sum(int(check.get("extracted_count", 0) or 0) for check in document_checks)
    missing_total = sum(int(check.get("missing_count", 0) or 0) for check in document_checks)
    unresolved_total = sum(int(check.get("unresolved_count", 0) or 0) for check in document_checks)
    if expected and document_checks:
        addressed_total = sum(
            len(expected.intersection({str(path).strip() for path in check.get("addressed_field_paths", [])}))
            for check in document_checks
        )
        coverage_ratio = addressed_total / (len(expected) * len(document_checks))
    else:
        coverage_ratio = 1.0 if document_checks else 0.0

    if extracted_total:
        status = "passed"
        outcome = "values_extracted"
    elif unresolved_total and coverage_ratio >= 1.0:
        status = "passed"
        outcome = "unresolved_requires_review"
    elif stage_id == "figure_classification" or coverage_ratio >= 1.0:
        status = "passed"
        outcome = "no_values_found"
    else:
        status = "low_quality"
        outcome = "unaccounted_fields"
    return {
        "status": status,
        "extraction_outcome": outcome,
        "field_coverage_ratio": round(coverage_ratio, 4),
        "extracted_total": extracted_total,
        "missing_total": missing_total,
        "unresolved_total": unresolved_total,
    }


def is_section2_measurement_leak(item: dict[str, Any], payload: dict[str, Any]) -> bool:
    if payload.get("section_id") != "material_info.section2":
        return False
    text = " ".join(str(item.get(key) or "") for key in ("value", "evidence_text", "source_hint")).lower()
    if any(pattern in text for pattern in PROCESS_KEEP_PATTERNS):
        return False
    return any(pattern in text for pattern in MEASUREMENT_LEAK_PATTERNS)


def is_unsupported_inference(item: dict[str, Any]) -> bool:
    value = item.get("value")
    if not isinstance(value, str) or not re.search(
        r"\b(?:implied|likely|presumed|assumed|inferred)\b", value, flags=re.IGNORECASE
    ):
        return False
    has_basis = bool(item.get("assignment_basis") or item.get("inference_basis"))
    has_source_type = bool(item.get("source_type") or item.get("provenance_type"))
    has_confidence = item.get("confidence") not in {None, ""}
    return not (has_basis and has_source_type and has_confidence)


def normalize_doi(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    match = re.match(r"^(10\.\d{4,9})[_/](.+)$", text, flags=re.I)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return text


def canonical_section_id(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    normalized = re.sub(r"[\s_-]+", "", text.lower())
    if normalized in {"section0", "section1", "section2", "section3", "section4"}:
        return f"material_info.{normalized}"
    if normalized == "materialinfosection0":
        return "material_info.section0"
    if normalized == "materialinfosection1":
        return "material_info.section1"
    if normalized == "materialinfosection2":
        return "material_info.section2"
    if normalized == "materialinfosection3":
        return "material_info.section3"
    if normalized == "materialinfosection4":
        return "material_info.section4"
    if text in {"section5", "paper_info"} or text.startswith("material_info."):
        return text
    return text


PAPER_METADATA_KEYS = {
    "title": ("title",),
    "authors": ("authors", "author_details"),
    "doi": ("doi", "external_ids.doi"),
    "arxiv_id": ("arxiv_id", "external_ids.arxiv", "source_id"),
    "abstract": ("abstract",),
    "publication_date": ("published_date", "publication_date", "year"),
    "published_date": ("published_date", "publication_date", "year"),
    "journal": ("journal", "venue"),
    "venue": ("venue", "journal"),
    "url": ("paper_url", "url", "raw_links.landing_url"),
    "paper_url": ("paper_url", "url", "raw_links.landing_url"),
    "pdf_url": ("pdf_url", "raw_links.pdf_url"),
    "keywords": ("keywords", "subjects"),
    "subjects": ("subjects", "keywords"),
    "source": ("source",),
    "retrieved_at": ("retrieved_at",),
    "publisher": ("publisher",),
    "paper_type": ("paper_type",),
    "license": ("license",),
}


def nested_metadata_value(record: dict[str, Any], dotted_key: str) -> Any:
    value: Any = record
    for key in dotted_key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def normalize_metadata_value(field_name: str, value: Any) -> Any:
    if field_name == "authors" and isinstance(value, list):
        names = []
        for item in value:
            if isinstance(item, dict):
                name = str(item.get("name") or "").strip()
            else:
                name = str(item or "").strip()
            if name:
                names.append(name)
        return names
    if field_name == "doi":
        return normalize_doi(value)
    return value


def metadata_value_for_field(record: dict[str, Any], field_path: str) -> Any:
    field_name = str(field_path or "").rsplit(".", 1)[-1].casefold()
    candidates = PAPER_METADATA_KEYS.get(field_name, (field_name,))
    for candidate in candidates:
        value = normalize_metadata_value(field_name, nested_metadata_value(record, candidate))
        if not is_null_like(value):
            return value
    return None


def build_paper_metadata_payload(
    record: dict[str, Any], fields: list[str], document_name: str
) -> dict[str, Any]:
    source = str(record.get("source") or "upstream_metadata").strip() or "upstream_metadata"
    source_path = str(record.get("_metadata_source_path") or "").strip()
    source_hint = source_path or str(record.get("paper_url") or record.get("url") or source)
    extracted = []
    missing = []
    for field_path in fields:
        value = metadata_value_for_field(record, field_path)
        if is_null_like(value):
            missing.append(
                {
                    "field_path": field_path,
                    "missing_reason": f"Not provided by upstream {source} metadata; PDF text was not used to invent bibliographic metadata.",
                    "source_hint": source_hint,
                }
            )
            continue
        extracted.append(
            {
                "field_path": field_path,
                "value": value,
                "unit": None,
                "material_system": None,
                "evidence_text": f"Value supplied by the upstream {source} metadata record.",
                "source_hint": source_hint,
                "source_type": source,
                "confidence": 1.0,
            }
        )
    return {
        "stage_id": "paper_info",
        "section_id": "paper_info",
        "document": document_name,
        "extracted_fields": extracted,
        "missing_fields": missing,
        "unresolved_fields": [],
        "quality_notes": ["paper_info was prefilled deterministically from retrieval/download metadata; no LLM call was used."],
        "metadata_handoff": {
            "source": source,
            "source_path": source_path,
            "paper_id": record.get("paper_id"),
        },
    }


def metadata_for_document(
    paper_metadata_by_document: dict[str, dict[str, Any]], document_path: str
) -> dict[str, Any] | None:
    path = Path(document_path)
    candidates = [str(path), str(path.resolve()), path.name, path.stem]
    for candidate in candidates:
        value = paper_metadata_by_document.get(candidate)
        if isinstance(value, dict):
            return value
    return None


def normalize_figure_sections(value: Any) -> Any:
    if isinstance(value, list):
        return [normalize_figure_sections(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {}
    for key, item in value.items():
        if key == "allowed_section":
            normalized["allowed_sections"] = [canonical_section_id(item)]
        elif key == "allowed_sections" and isinstance(item, list):
            normalized[key] = [canonical_section_id(entry) for entry in item]
        else:
            normalized[key] = normalize_figure_sections(item)
    return normalized


def flatten_for_search(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{flatten_for_search(key)} {flatten_for_search(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(flatten_for_search(item) for item in value)
    return str(value or "")


def text_has_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(rf"(?<![a-z0-9]){re.escape(pattern)}(?![a-z0-9])", text) for pattern in patterns)


def infer_figure_section_from_text(text: str) -> str | None:
    lowered = text.lower()
    if text_has_any(lowered, FIGURE_CHARACTERIZATION_PATTERNS):
        return "material_info.section3"
    if text_has_any(lowered, FIGURE_PROPERTY_CURVE_PATTERNS):
        return "material_info.section4"
    if text_has_any(lowered, FIGURE_THEORY_PATTERNS):
        return "section5"
    if text_has_any(lowered, FIGURE_PROCESS_PATTERNS):
        return "material_info.section2"
    return None


def infer_figure_evidence_type(section_id: str | None) -> str:
    if section_id == "material_info.section2":
        return "process_or_fabrication"
    if section_id == "material_info.section3":
        return "microscopic_or_structural_characterization"
    if section_id == "material_info.section4":
        return "macroscopic_property_curve_or_map"
    if section_id == "section5":
        return "theory_or_mechanism"
    return "section_evidence"


def canonicalize_figure_value(raw_value: Any, source_item: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    value = normalize_figure_sections(raw_value)
    if not isinstance(value, dict):
        possible_section = canonical_section_id(value)
        value = {"raw_value": raw_value}
        if isinstance(possible_section, str) and possible_section in CANONICAL_SECTION_IDS:
            value["assigned_section"] = possible_section

    field_tail = str(source_item.get("field_path") or "").rsplit(".", 1)[-1]
    if value.get("figure_label") and not value.get("figure_id"):
        value["figure_id"] = value.get("figure_label")
    if field_tail and field_tail not in {"figure_classification", "figures"}:
        value.setdefault("figure_id", field_tail)
    if value.get("reason") and not value.get("assignment_reason"):
        value["assignment_reason"] = value.get("reason")
    if value.get("caption") and not value.get("caption_or_evidence"):
        value["caption_or_evidence"] = value.get("caption")

    assigned_section = (
        value.get("assigned_section")
        or value.get("section_assignment")
        or value.get("assigned_to")
        or value.get("section")
        or value.get("section_id")
        or value.get("target_section")
        or (value.get("allowed_sections") or [None])[0]
    )
    assigned_section = canonical_section_id(assigned_section)

    search_text = " ".join(
        [
            flatten_for_search(value),
            str(source_item.get("evidence_text") or ""),
            str(source_item.get("source_hint") or ""),
        ]
    ).lower()
    inferred_section = infer_figure_section_from_text(search_text)
    note = None
    if assigned_section == "material_info.section2" and inferred_section in {
        "material_info.section3",
        "material_info.section4",
        "section5",
    }:
        value.setdefault("original_assigned_section", assigned_section)
        assigned_section = inferred_section
        note = (
            f"Rerouted {value.get('figure_id', 'figure')} from material_info.section2 "
            f"to {assigned_section} because its evidence is not process/fabrication evidence."
        )
    elif inferred_section == "material_info.section4" and assigned_section in {
        "material_info.section1",
        "material_info.section2",
        "material_info.section3",
    }:
        value.setdefault("original_assigned_section", assigned_section)
        assigned_section = inferred_section
        note = (
            f"Rerouted {value.get('figure_id', 'figure')} from {value.get('original_assigned_section')} "
            "to material_info.section4 because its evidence is a property curve, map, or measurement result."
        )
    elif inferred_section == "material_info.section3" and assigned_section in {
        "material_info.section1",
        "material_info.section2",
    }:
        value.setdefault("original_assigned_section", assigned_section)
        assigned_section = inferred_section
        note = (
            f"Rerouted {value.get('figure_id', 'figure')} from {value.get('original_assigned_section')} "
            "to material_info.section3 because its evidence is characterization evidence."
        )
    elif inferred_section == "section5" and assigned_section in {
        "material_info.section1",
        "material_info.section2",
    }:
        value.setdefault("original_assigned_section", assigned_section)
        assigned_section = inferred_section
        note = (
            f"Rerouted {value.get('figure_id', 'figure')} from {value.get('original_assigned_section')} "
            "to section5 because its evidence is theory, simulation, or mechanism evidence."
        )
    elif (not isinstance(assigned_section, str) or assigned_section not in CANONICAL_SECTION_IDS) and inferred_section:
        assigned_section = inferred_section

    if isinstance(assigned_section, str) and assigned_section in CANONICAL_SECTION_IDS:
        value["assigned_section"] = assigned_section
        value.setdefault("allowed_sections", [assigned_section])

    if isinstance(value.get("allowed_sections"), list):
        value["allowed_sections"] = [canonical_section_id(item) for item in value["allowed_sections"]]
    if isinstance(value.get("prohibited_sections"), list):
        value["prohibited_sections"] = [canonical_section_id(item) for item in value["prohibited_sections"]]
    elif isinstance(value.get("blocked_sections"), list):
        value["prohibited_sections"] = [canonical_section_id(item) for item in value["blocked_sections"]]
    else:
        value.setdefault("prohibited_sections", [])

    value.setdefault("panel_id", "")
    value.setdefault("caption_or_evidence", source_item.get("evidence_text") or value.get("caption") or "")
    value.setdefault("evidence_type", infer_figure_evidence_type(value.get("assigned_section")))
    value.setdefault("assignment_reason", "Assigned by figure classification evidence.")
    value.setdefault("confidence", source_item.get("confidence") if source_item.get("confidence") is not None else "unspecified")
    return value, note


def canonicalize_figure_classification_item(item: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    raw_value = item.get("value")
    raw_values = raw_value if isinstance(raw_value, list) else [raw_value]
    canonical_items = []
    notes = []
    for value_item in raw_values:
        value, note = canonicalize_figure_value(value_item, item)
        canonical_item = dict(item)
        canonical_item["field_path"] = "figure_classification.figures"
        canonical_item["value"] = value
        canonical_items.append(canonical_item)
        if note:
            notes.append(note)
    return canonical_items, notes


def field_matches_any(field_path: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    path = str(field_path or "")
    return any(pattern in path for pattern in patterns)


def normalize_payload_for_unicode_stage(payload: dict[str, Any], target_field_patterns: list[str]) -> tuple[dict[str, Any], int]:
    if payload.get("error") or payload.get("raw_text"):
        return payload, 0
    changed = 0
    for item in payload.get("extracted_fields", []) or []:
        if not isinstance(item, dict):
            continue
        before = json_dumps(item)
        item["value"] = normalize_strings_deep(item.get("value"))
        item["unit"] = normalize_strings_deep(item.get("unit"))
        item["material_system"] = normalize_strings_deep(item.get("material_system"))
        item["evidence_text"] = normalize_strings_deep(item.get("evidence_text"))
        if str(item.get("field_path") or "").lower().endswith(".doi"):
            item["value"] = normalize_doi(item.get("value"))
        if json_dumps(item) != before:
            changed += 1
    payload.setdefault("quality_notes", []).append(
        f"unicode_normalization_stage normalized {changed} extracted field records."
    )
    return payload, changed


def run_unicode_normalization_stage(run_dir: Path, workflow_plan: dict[str, Any], selected_docs: list[str]) -> dict[str, Any]:
    policy = workflow_plan.get("unicode_normalization_policy") or {}
    target_patterns = [str(item) for item in policy.get("target_field_patterns", []) or []]
    normalized_files = 0
    normalized_records = 0
    invalid_files = []
    for stage_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        if stage_dir.name == "unicode_normalization":
            continue
        for output_file in sorted(stage_dir.glob("*.json")):
            try:
                payload = json.loads(output_file.read_text(encoding="utf-8"))
                payload, changed = normalize_payload_for_unicode_stage(payload, target_patterns)
                output_file.write_text(json_dumps(payload), encoding="utf-8")
                normalized_files += 1
                normalized_records += changed
            except Exception as exc:  # noqa: BLE001 - diagnostics for the supervisor
                invalid_files.append({"file": str(output_file), "error": str(exc)})
    status = "passed" if not invalid_files else "invalid_json"
    return {
        "stage_id": "unicode_normalization",
        "section_id": "postprocess.unicode_normalization",
        "depends_on": workflow_plan.get("execution_order", []),
        "status": status,
        "test_documents": selected_docs,
        "field_count_tested": len(target_patterns),
        "document_results": [],
        "checks": ["unicode_normalization", "json_validity", "schema_aware_target_selection"],
        "summary": {
            "documents": len(selected_docs),
            "invalid_json": len(invalid_files),
            "extracted_total": normalized_records,
            "null_like_total": 0,
            "unicode_candidate_total": 0,
            "normalized_files": normalized_files,
            "normalized_records": normalized_records,
            "failed_docs": invalid_files,
            "target_field_patterns": target_patterns,
        },
    }


def sanitize_payload(payload: dict[str, Any], normalize_unicode: bool = False) -> dict[str, Any]:
    if payload.get("error") or payload.get("raw_text"):
        return payload
    if normalize_unicode:
        payload = normalize_strings_deep(payload)
    extracted = payload.get("extracted_fields")
    missing = payload.get("missing_fields")
    unresolved = payload.get("unresolved_fields")
    if not isinstance(extracted, list):
        extracted = []
    if not isinstance(missing, list):
        missing = []
    if not isinstance(unresolved, list):
        unresolved = []
    clean_extracted = []
    moved_to_missing = []
    figure_quality_notes = []
    is_figure_stage = str(payload.get("stage_id") or "") == "figure_classification"
    for item in extracted:
        if not isinstance(item, dict):
            continue
        if is_null_like(item.get("value")):
            moved_to_missing.append(
                {
                    "field_path": item.get("field_path"),
                    "missing_reason": item.get("evidence_text") or "Value was absent or explicitly unavailable in the source text.",
                    "source_hint": item.get("source_hint"),
                }
            )
        elif is_section2_measurement_leak(item, payload):
            moved_to_missing.append(
                {
                    "field_path": item.get("field_path"),
                    "missing_reason": "Removed from section2 because the evidence describes characterization/property measurement rather than fabrication or processing.",
                    "source_hint": item.get("source_hint"),
                }
            )
        elif is_unsupported_inference(item):
            moved_to_missing.append(
                {
                    "field_path": item.get("field_path"),
                    "missing_reason": "Removed because the value is an inferred label without a structured inference basis, source type, and confidence.",
                    "source_hint": item.get("source_hint"),
                }
            )
        else:
            field_path = str(item.get("field_path") or "").lower()
            if field_path.endswith(".doi") or field_path == "paper_info.metadata.doi":
                item["value"] = normalize_doi(item.get("value"))
            if is_figure_stage:
                canonical_items, notes = canonicalize_figure_classification_item(item)
                clean_extracted.extend(canonical_items)
                figure_quality_notes.extend(notes)
                continue
            clean_extracted.append(item)
    if moved_to_missing:
        payload.setdefault("quality_notes", []).append(
            f"Moved {len(moved_to_missing)} null-like or section-mismatched extracted_fields into missing_fields."
        )
    if figure_quality_notes:
        payload.setdefault("quality_notes", []).extend(figure_quality_notes)
    clean_extracted = deduplicate_extracted_fields(clean_extracted)
    extracted_paths = {
        str(item.get("field_path") or "").strip()
        for item in clean_extracted
        if isinstance(item, dict) and str(item.get("field_path") or "").strip()
    }
    clean_unresolved = []
    unresolved_paths = set()
    for item in unresolved:
        if not isinstance(item, dict):
            continue
        field_path = str(item.get("field_path") or "").strip()
        reason = str(item.get("unresolved_reason") or "").strip()
        if not field_path or not reason or field_path in extracted_paths or field_path in unresolved_paths:
            continue
        item["field_path"] = field_path
        item["decision_status"] = "unresolved"
        clean_unresolved.append(item)
        unresolved_paths.add(field_path)
    clean_missing = [
        item
        for item in reconcile_missing_fields(clean_extracted, [*missing, *moved_to_missing])
        if str(item.get("field_path") or "").strip() not in unresolved_paths
    ]
    payload["extracted_fields"] = clean_extracted
    payload["missing_fields"] = clean_missing
    payload["unresolved_fields"] = clean_unresolved
    return payload


def needs_unicode_normalization(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    return bool(
        re.search(r"\\[A-Za-z]+|\^\{[^{}]+\}|_\{[^{}]+\}|\^[+\-]?[0-9]+|_[+\-]?[0-9]+", text)
        or re.search(r"\b(?:[A-Z][a-z]?)(?:[0-9]+(?:\.[0-9]+)?)(?:[A-Z][a-z]?[0-9]*(?:\.[0-9]+)?)*\b", text)
    )


def count_unicode_candidates(value: Any) -> int:
    if isinstance(value, str):
        return 1 if needs_unicode_normalization(value) else 0
    if isinstance(value, list):
        return sum(count_unicode_candidates(item) for item in value)
    if isinstance(value, dict):
        return sum(count_unicode_candidates(item) for item in value.values())
    return 0


def stable_contract_version(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest().upper()


def annotate_payload_lineage(
    payload: dict[str, Any],
    field_specs: list[dict[str, Any]],
    *,
    document_path: str,
    stage_id: str,
    model_version: str,
) -> dict[str, Any]:
    """Bind each decision to its literature source and executable field rule."""
    specs_by_path = {
        str(item.get("field_path") or ""): item
        for item in field_specs
        if isinstance(item, dict) and item.get("field_path")
    }
    prompt_version = stable_contract_version(
        {
            "task": "automated_materials_database_construction_from_scientific_literature",
            "stage_id": stage_id,
            "field_specs": field_specs,
            "output_states": ["extracted", "missing", "unresolved"],
        }
    )
    validator_version = "downstream-extraction-validator-v2"
    decision_groups = (
        ("extracted_fields", "extracted", "value"),
        ("missing_fields", "missing", "missing_reason"),
        ("unresolved_fields", "unresolved", "candidate_value"),
    )
    for group_name, decision_status, value_key in decision_groups:
        for item in payload.get(group_name, []) or []:
            if not isinstance(item, dict):
                continue
            field_path = str(item.get("field_path") or "")
            spec = specs_by_path.get(field_path) or {}
            field_rule_version = str(spec.get("field_rule_version") or "")
            if not field_rule_version.startswith("sha256:"):
                field_rule_version = stable_contract_version(spec or {"field_path": field_path})
            item["decision_status"] = decision_status
            identity_payload = {
                "document_path": str(Path(document_path)),
                "stage_id": stage_id,
                "field_path": field_path,
                "material_system": item.get("material_system"),
                "sample_id": item.get("sample_id"),
                "entity_ref": item.get("entity_ref"),
                "source_hint": item.get("source_hint"),
                "decision_status": decision_status,
                "decision_value": item.get(value_key),
            }
            item["record_id"] = "record:" + hashlib.sha256(
                json.dumps(identity_payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
            ).hexdigest()[:24].upper()
            item["lineage"] = {
                "source_document": str(Path(document_path)),
                "source_artifact_type": "parsed_literature_markdown",
                "stage_id": stage_id,
                "field_rule_id": spec.get("field_rule_id") or f"field.{safe_name(field_path).lower()}",
                "field_rule_version": field_rule_version,
                "prompt_version": prompt_version,
                "model_version": model_version,
                "validator_version": validator_version,
            }

    payload["task_contract"] = agent_protocol.materials_literature_task_contract()
    payload["lineage_manifest"] = {
        "contract_version": "materials-literature-lineage-v1",
        "document_path": str(Path(document_path)),
        "stage_id": stage_id,
        "prompt_version": prompt_version,
        "model_version": model_version,
        "validator_version": validator_version,
        "field_rule_versions": {
            path: (spec.get("field_rule_version") or stable_contract_version(spec))
            for path, spec in sorted(specs_by_path.items())
        },
    }
    payload["protocol_envelope"] = agent_protocol.make_message(
        sender="section_extraction_runner",
        receiver="step9_supervisor",
        phase=stage_id,
        status="completed",
        task_contract=payload["task_contract"],
        payload_refs={
            "document": str(Path(document_path)),
            "stage_id": stage_id,
            "lineage": "lineage_manifest",
        },
        decision={
            "output_states": ["extracted", "missing", "unresolved"],
            "field_decision_count": sum(
                len(payload.get(group_name, []) or [])
                for group_name in ("extracted_fields", "missing_fields", "unresolved_fields")
            ),
        },
        produced_artifacts=[{"ref": "stage_payload"}],
        next_route="extraction_eval",
    )
    return payload


def validate_stage_payload(payload: dict[str, Any], normalize_unicode: bool = False) -> dict[str, Any]:
    payload = sanitize_payload(payload, normalize_unicode=normalize_unicode)
    if payload.get("error") or payload.get("raw_text"):
        return {
            "extracted_count": 0,
            "missing_count": 0,
            "null_like_count": 0,
            "evidence_count": 0,
            "has_json": False,
            "has_evidence": False,
            "protocol_valid": False,
            "protocol_errors": ["payload was not successfully annotated"],
            "error": payload.get("error", "raw_text payload indicates an earlier parse failure"),
            "error_type": payload.get("error_type") or classify_extraction_error(payload.get("error")),
            "retryable_transport_error": (
                payload.get("error_type") or classify_extraction_error(payload.get("error"))
            ) in {"rate_limit", "transient_http", "transport_timeout"},
        }
    extracted = payload.get("extracted_fields")
    missing = payload.get("missing_fields")
    unresolved = payload.get("unresolved_fields")
    if not isinstance(extracted, list):
        extracted = []
    if not isinstance(missing, list):
        missing = []
    if not isinstance(unresolved, list):
        unresolved = []
    evidence_count = 0
    null_like_count = 0
    unicode_candidate_count = 0
    for item in extracted:
        if not isinstance(item, dict):
            continue
        unicode_candidate_count += count_unicode_candidates(
            {
                "value": item.get("value"),
                "unit": item.get("unit"),
                "material_system": item.get("material_system"),
                "evidence_text": item.get("evidence_text"),
            }
        )
        value = item.get("value")
        if is_null_like(value):
            null_like_count += 1
        if item.get("evidence_text"):
            evidence_count += 1
    unresolved_evidence_count = sum(
        1
        for item in unresolved
        if isinstance(item, dict) and str(item.get("evidence_text") or "").strip()
    )
    evidence_count += unresolved_evidence_count
    addressed_field_paths = sorted(
        {
            str(item.get("field_path") or "").strip()
            for item in [*extracted, *missing, *unresolved]
            if isinstance(item, dict) and str(item.get("field_path") or "").strip()
        }
    )
    protocol_errors = agent_protocol.validate_message(payload.get("protocol_envelope"))
    return {
        "extracted_count": len(extracted),
        "missing_count": len(missing),
        "unresolved_count": len(unresolved),
        "null_like_count": null_like_count,
        "unicode_candidate_count": unicode_candidate_count,
        "evidence_count": evidence_count,
        "addressed_field_paths": addressed_field_paths,
        "has_json": True,
        "has_evidence": evidence_count > 0 or (len(extracted) == 0 and len(unresolved) == 0),
        "protocol_valid": not protocol_errors,
        "protocol_errors": protocol_errors,
    }


def _selector_tokens(path: str) -> set[str]:
    value = str(path or "").strip()
    if not value:
        return set()
    path_obj = Path(value)
    return {
        value.casefold(),
        str(path_obj).casefold(),
        path_obj.name.casefold(),
        path_obj.stem.casefold(),
    }


def build_selective_reprocess_scope(
    workflow_plan: dict[str, Any],
    prompt_output: dict[str, Any],
    documents: list[str],
    impact_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    all_documents = [str(Path(path)) for path in documents]
    all_stages = [
        stage
        for stage in workflow_plan.get("section_test_plan", [])
        if isinstance(stage, dict)
        and stage.get("stage_id")
        and stage.get("stage_id") != "unicode_normalization"
        and not str(stage.get("section_id", "")).startswith("postprocess.")
    ]
    all_stage_ids = [str(stage["stage_id"]) for stage in all_stages]
    if not impact_manifest:
        return {
            "enabled": False,
            "approved": True,
            "errors": [],
            "selected_documents": all_documents,
            "selected_stage_ids": all_stage_ids,
            "target_stage_ids": all_stage_ids,
            "dependency_stage_ids": [],
            "selected_field_paths_by_stage": {},
            "requested_documents": [],
            "requested_stage_ids": [],
            "requested_field_paths": [],
            "skipped_documents": [],
            "skipped_stage_ids": [],
        }

    errors = []
    approval = impact_manifest.get("supervisor_approval") or {}
    approved = approval.get("status") == "approved"
    if not approved:
        errors.append("Impact manifest is not approved by impact_supervisor.")
    if impact_manifest.get("task_type") != agent_protocol.materials_literature_task_contract()["task_type"]:
        errors.append("Impact manifest task_type is outside automated materials-database construction.")
    if impact_manifest.get("source_scope") != agent_protocol.materials_literature_task_contract()["source_scope"]:
        errors.append("Impact manifest source_scope must be scientific_literature_only.")

    requested_documents = sorted({str(value) for value in impact_manifest.get("affected_documents", []) if str(value).strip()})
    requested_stage_ids = sorted({str(value) for value in impact_manifest.get("affected_stage_ids", []) if str(value).strip()})
    requested_field_paths = sorted({str(value) for value in impact_manifest.get("affected_field_paths", []) if str(value).strip()})
    requested_doc_tokens = set().union(*(_selector_tokens(value) for value in requested_documents)) if requested_documents else set()
    selected_documents = [
        document
        for document in all_documents
        if not requested_doc_tokens or _selector_tokens(document).intersection(requested_doc_tokens)
    ]
    if requested_documents and not selected_documents:
        errors.append("No input literature document matches affected_documents.")

    known_stage_ids = set(all_stage_ids)
    target_stage_ids = {stage_id for stage_id in requested_stage_ids if stage_id in known_stage_ids}
    unknown_stage_ids = sorted(set(requested_stage_ids) - known_stage_ids)
    if unknown_stage_ids:
        errors.append("Unknown affected_stage_ids: " + ", ".join(unknown_stage_ids))

    selected_fields_by_stage: dict[str, list[str]] = {}
    matched_fields = set()
    if requested_field_paths:
        requested_field_set = set(requested_field_paths)
        for stage_id in all_stage_ids:
            matched = sorted(
                requested_field_set.intersection(
                    spec.get("field_path")
                    for spec in stage_field_specs(prompt_output, stage_id)
                    if isinstance(spec, dict) and spec.get("field_path")
                )
            )
            if matched:
                selected_fields_by_stage[stage_id] = matched
                target_stage_ids.add(stage_id)
                matched_fields.update(matched)
        unmatched_fields = sorted(set(requested_field_paths) - matched_fields)
        if unmatched_fields:
            errors.append("Unknown affected_field_paths: " + ", ".join(unmatched_fields))

    if not requested_stage_ids and not requested_field_paths:
        target_stage_ids.update(all_stage_ids)
    if not target_stage_ids and impact_manifest.get("status") != "no_reprocessing_required":
        errors.append("Impact manifest does not select any executable extraction stage.")

    stage_by_id = {str(stage["stage_id"]): stage for stage in all_stages}
    selected_stage_ids = set(target_stage_ids)
    pending = list(target_stage_ids)
    while pending:
        stage_id = pending.pop()
        for dependency in stage_by_id.get(stage_id, {}).get("depends_on", []) or []:
            dependency = str(dependency)
            if dependency in known_stage_ids and dependency not in selected_stage_ids:
                selected_stage_ids.add(dependency)
                pending.append(dependency)

    ordered_selected_stages = [stage_id for stage_id in all_stage_ids if stage_id in selected_stage_ids]
    dependency_stage_ids = [stage_id for stage_id in ordered_selected_stages if stage_id not in target_stage_ids]
    return {
        "enabled": True,
        "approved": approved and not errors,
        "errors": errors,
        "selected_documents": selected_documents,
        "selected_stage_ids": ordered_selected_stages,
        "target_stage_ids": [stage_id for stage_id in all_stage_ids if stage_id in target_stage_ids],
        "dependency_stage_ids": dependency_stage_ids,
        "selected_field_paths_by_stage": selected_fields_by_stage,
        "requested_documents": requested_documents,
        "requested_stage_ids": requested_stage_ids,
        "requested_field_paths": requested_field_paths,
        "skipped_documents": [document for document in all_documents if document not in selected_documents],
        "skipped_stage_ids": [stage_id for stage_id in all_stage_ids if stage_id not in selected_stage_ids],
    }


def run_extraction_bench(
    workflow_plan: dict[str, Any],
    prompt_output: dict[str, Any],
    documents: list[str],
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float = 0,
    max_tokens: int = 384000,
    output_dir: str = "step9_extraction_runs",
    max_documents: int = 4,
    max_document_chars: int = 200000,
    max_fields_per_stage: int = 80,
    max_workers: int = 4,
    paper_metadata_by_document: dict[str, dict[str, Any]] | None = None,
    impact_manifest: dict[str, Any] | None = None,
    run_identity: dict[str, Any] | None = None,
    reuse_existing_outputs: bool = False,
) -> dict[str, Any]:
    run_dir = Path(output_dir)
    if run_identity:
        run_artifact_guard.assert_active_run(run_identity)
        if run_dir.exists():
            return {
                "status": "clean_run_rejected",
                "reason": (
                    "The extraction run directory already exists. Start a new Step9 run with a new "
                    "output namespace; existing document or batch outputs are never reused by an official run."
                ),
                "run_dir": str(run_dir),
                "documents": [],
                "selective_reprocess": {},
                "section_results": [],
                "run_identity": run_identity,
            }
    run_dir.mkdir(parents=True, exist_ok=True)
    candidate_docs = [str(Path(path)) for path in documents[:max_documents]]
    selective_scope = build_selective_reprocess_scope(
        workflow_plan,
        prompt_output,
        candidate_docs,
        impact_manifest,
    )
    if selective_scope["errors"]:
        return {
            "status": "selection_rejected",
            "reason": "Selective reprocessing was rejected before extraction.",
            "run_dir": str(run_dir),
            "documents": [],
            "selective_reprocess": selective_scope,
            "section_results": [],
        }
    if selective_scope["enabled"] and (impact_manifest or {}).get("status") == "no_reprocessing_required":
        return {
            "status": "completed_noop",
            "reason": "The approved impact manifest requires no historical reprocessing.",
            "run_dir": str(run_dir),
            "documents": [],
            "selective_reprocess": selective_scope,
            "section_results": [],
        }
    selected_docs = selective_scope["selected_documents"]
    selected_stage_ids = set(selective_scope["selected_stage_ids"])
    target_stage_ids = set(selective_scope["target_stage_ids"])
    selected_fields_by_stage = selective_scope["selected_field_paths_by_stage"]
    worker_limit = max(1, int(max_workers or 1))
    section_results = []
    dependency_outputs: dict[str, Any] = {}
    paper_metadata_by_document = paper_metadata_by_document or {}
    metadata_prefilled_documents = set()

    def bind_run_identity(payload: dict[str, Any]) -> dict[str, Any]:
        if run_identity:
            payload["run_identity"] = dict(run_identity)
        return payload

    for stage in workflow_plan.get("section_test_plan", []):
        stage_id = stage["stage_id"]
        if stage_id == "unicode_normalization" or str(stage.get("section_id", "")).startswith("postprocess."):
            continue
        if stage_id not in selected_stage_ids:
            continue
        stage_dir = run_dir / safe_name(stage_id)
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_policy = (workflow_plan.get("stage_batching_policy") or {}).get(stage_id) or {}
        stage_max_fields = int(stage_policy.get("max_fields_per_call") or max_fields_per_stage)
        all_field_specs = stage_field_specs(prompt_output, stage_id)
        if stage_id in selected_fields_by_stage:
            selected_field_paths = set(selected_fields_by_stage[stage_id])
            all_field_specs = [
                spec for spec in all_field_specs if spec.get("field_path") in selected_field_paths
            ]
        all_fields = [item["field_path"] for item in all_field_specs]
        specs_by_path = {item["field_path"]: item for item in all_field_specs}
        if stage_max_fields and len(all_fields) > stage_max_fields:
            field_batches = chunk_fields(all_fields, stage_max_fields)
            fields = all_fields
        else:
            fields = all_fields
            field_batches = [fields]
        effective_workers = (
            1
            if stage_id == "paper_info" or len(selected_docs) <= 1
            else min(worker_limit, len(selected_docs))
        )

        def process_document(doc_path: str) -> dict[str, Any]:
            doc_name = Path(doc_path).name
            output_file = stage_dir / f"{safe_name(Path(doc_path).stem)}.json"
            metadata_record = metadata_for_document(paper_metadata_by_document, doc_path)
            if stage_id == "paper_info":
                payload = build_paper_metadata_payload(metadata_record or {}, fields, doc_name)
                if not metadata_record:
                    payload.setdefault("quality_notes", []).append(
                        "No upstream metadata record matched this document; paper_info was not extracted from PDF text."
                    )
                payload = sanitize_payload(payload, normalize_unicode=False)
                payload = annotate_payload_lineage(
                    payload,
                    all_field_specs,
                    document_path=doc_path,
                    stage_id=stage_id,
                    model_version="metadata_handoff",
                )
                payload = bind_run_identity(payload)
                checks = validate_stage_payload(payload, normalize_unicode=False)
                run_artifact_guard.atomic_write_json(
                    output_file,
                    payload,
                    run_identity=run_identity,
                )
                return {
                    "item": {
                        "document": doc_name,
                        "status": (
                            "passed"
                            if metadata_record and checks["has_json"]
                            else "metadata_missing"
                            if checks["has_json"]
                            else "invalid_json"
                        ),
                        "checks": checks,
                        "output_file": str(output_file),
                        "metadata_prefilled": bool(metadata_record),
                    },
                    "metadata_prefilled": bool(metadata_record),
                    "metadata_missing": not bool(metadata_record),
                    "failed": not checks["has_json"],
                }
            force_targeted_output = selective_scope["enabled"] and stage_id in target_stage_ids
            if reuse_existing_outputs and output_file.exists() and not force_targeted_output:
                try:
                    payload = json.loads(output_file.read_text(encoding="utf-8"))
                    payload = sanitize_payload(payload, normalize_unicode=False)
                    payload = annotate_payload_lineage(
                        payload,
                        all_field_specs,
                        document_path=doc_path,
                        stage_id=stage_id,
                        model_version=model,
                    )
                    checks = validate_stage_payload(payload, normalize_unicode=False)
                    doc_status = "passed" if checks["has_json"] else "invalid_json"
                    if doc_status == "passed":
                        run_artifact_guard.atomic_write_json(
                            output_file,
                            payload,
                            run_identity=run_identity,
                        )
                        return {
                            "item": {
                                "document": doc_name,
                                "status": doc_status,
                                "checks": checks,
                                "output_file": str(output_file),
                                "resumed": True,
                            },
                            "metadata_prefilled": False,
                            "metadata_missing": False,
                            "failed": False,
                        }
                except json.JSONDecodeError:
                    pass
            document_text = read_text(doc_path, max_document_chars)
            dependency_context = {
                dep: dependency_outputs.get(dep, {"status": "not_available"})
                for dep in stage.get("depends_on", [])
            }
            batch_payloads = []
            batch_dir = stage_dir / "_batches" / safe_name(Path(doc_path).stem)
            if len(field_batches) > 1:
                batch_dir.mkdir(parents=True, exist_ok=True)
            for batch_index, batch_fields in enumerate(field_batches):
                batch_field_specs = [specs_by_path[field_path] for field_path in batch_fields]
                batch_output_file = batch_dir / f"batch_{batch_index + 1:02d}.json"
                if (
                    reuse_existing_outputs
                    and len(field_batches) > 1
                    and batch_output_file.exists()
                    and not force_targeted_output
                ):
                    try:
                        batch_payload = json.loads(batch_output_file.read_text(encoding="utf-8"))
                        batch_payload = sanitize_payload(batch_payload, normalize_unicode=False)
                        if validate_stage_payload(batch_payload, normalize_unicode=False)["has_json"]:
                            batch_payloads.append(batch_payload)
                            continue
                    except json.JSONDecodeError:
                        pass

                prompt = build_stage_prompt(
                    document_text,
                    doc_name,
                    stage,
                    batch_fields,
                    dependency_context,
                    batch_field_specs,
                )
                raw_text = ""
                try:
                    raw_text = prompt_agent.litellm_chat(
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    batch_payload = prompt_agent.parse_llm_json(raw_text)
                    batch_payload = sanitize_payload(batch_payload, normalize_unicode=False)
                except Exception as exc:  # noqa: BLE001 - keep bench diagnostics explicit
                    batch_payload = {
                        "stage_id": stage.get("stage_id"),
                        "section_id": stage.get("section_id"),
                        "document": doc_name,
                        "raw_text": raw_text,
                        "error": str(exc),
                        "error_type": classify_extraction_error(exc),
                    }
                batch_payload = bind_run_identity(batch_payload)
                if len(field_batches) > 1:
                    run_artifact_guard.atomic_write_json(
                        batch_output_file,
                        batch_payload,
                        run_identity=run_identity,
                    )
                batch_payloads.append(batch_payload)

            payload = merge_stage_payloads(batch_payloads, stage, doc_name, field_batches)
            payload = sanitize_payload(payload, normalize_unicode=False)
            payload = annotate_payload_lineage(
                payload,
                all_field_specs,
                document_path=doc_path,
                stage_id=stage_id,
                model_version=model,
            )
            payload = bind_run_identity(payload)
            checks = validate_stage_payload(payload, normalize_unicode=False)
            doc_status = (
                "passed"
                if checks["has_json"]
                else "transport_error"
                if checks.get("retryable_transport_error")
                else "invalid_json"
            )
            item = {
                "document": doc_name,
                "status": doc_status,
                "checks": checks,
                "output_file": str(output_file),
            }
            if len(field_batches) > 1:
                item["batch_count"] = len(field_batches)
            run_artifact_guard.atomic_write_json(
                item["output_file"],
                payload,
                run_identity=run_identity,
            )
            return {
                "item": item,
                "metadata_prefilled": False,
                "metadata_missing": False,
                "failed": doc_status in {"invalid_json", "transport_error"},
            }

        if effective_workers == 1:
            document_outcomes = [process_document(doc_path) for doc_path in selected_docs]
        else:
            with ThreadPoolExecutor(
                max_workers=effective_workers,
                thread_name_prefix=f"step9-{safe_name(stage_id)[:24]}",
            ) as executor:
                # executor.map preserves source-document order while API calls run concurrently.
                document_outcomes = list(executor.map(process_document, selected_docs))

        doc_results = [outcome["item"] for outcome in document_outcomes]
        failed_docs = [
            outcome["item"]["document"] for outcome in document_outcomes if outcome["failed"]
        ]
        metadata_missing_docs = [
            outcome["item"]["document"]
            for outcome in document_outcomes
            if outcome["metadata_missing"]
        ]
        metadata_prefilled_documents.update(
            outcome["item"]["document"]
            for outcome in document_outcomes
            if outcome["metadata_prefilled"]
        )

        document_checks = [item.get("checks") or {} for item in doc_results]
        yield_assessment = assess_stage_yield(document_checks, fields, stage_id=stage_id)
        extracted_total = yield_assessment["extracted_total"]
        missing_total = yield_assessment["missing_total"]
        unresolved_total = yield_assessment["unresolved_total"]
        null_like_total = sum((item.get("checks") or {}).get("null_like_count", 0) for item in doc_results)
        unicode_candidate_total = sum((item.get("checks") or {}).get("unicode_candidate_count", 0) for item in doc_results)
        invalid_count = sum(1 for item in doc_results if item.get("status") == "invalid_json")
        transport_error_count = sum(1 for item in doc_results if item.get("status") == "transport_error")
        stage_status = "passed"
        if transport_error_count:
            stage_status = "transport_error"
        elif invalid_count:
            stage_status = "invalid_json"
        elif stage_id == "paper_info" and metadata_missing_docs:
            stage_status = "metadata_missing"
        else:
            stage_status = yield_assessment["status"]

        section_result = {
            "stage_id": stage_id,
            "section_id": stage.get("section_id"),
            "depends_on": stage.get("depends_on", []),
            "status": stage_status,
            "test_documents": selected_docs,
            "field_count_tested": len(fields),
            "batching": {
                "enabled": len(field_batches) > 1,
                "batch_count": len(field_batches),
                "max_fields_per_call": stage_max_fields,
            },
            "parallelism": {
                "strategy": "bounded_document_thread_pool",
                "configured_max_workers": worker_limit,
                "effective_workers": effective_workers,
                "deterministic_result_order": True,
            },
            "document_results": doc_results,
            "checks": [
                "dependency_inputs_available",
                "json_validity",
                "field_coverage",
                "evidence_provenance",
            ],
            "summary": {
                "documents": len(doc_results),
                "invalid_json": invalid_count,
                "transport_errors": transport_error_count,
                "extracted_total": extracted_total,
                "missing_total": missing_total,
                "unresolved_total": unresolved_total,
                "field_coverage_ratio": yield_assessment["field_coverage_ratio"],
                "extraction_outcome": yield_assessment["extraction_outcome"],
                "null_like_total": null_like_total,
                "unicode_candidate_total": unicode_candidate_total,
                "failed_docs": failed_docs,
                "metadata_missing_docs": metadata_missing_docs,
            },
        }
        section_results.append(section_result)
        dependency_outputs[stage_id] = {
            "status": stage_status,
            "summary": section_result["summary"],
            "sample_outputs": [item["output_file"] for item in doc_results[:2]],
        }

    unicode_policy = workflow_plan.get("unicode_normalization_policy") or {}
    if unicode_policy.get("enabled"):
        section_results.append(run_unicode_normalization_stage(run_dir, workflow_plan, selected_docs))

    return {
        "status": "completed",
        "reason": "Live section-wise extraction bench completed.",
        "run_dir": str(run_dir),
        "documents": selected_docs,
        "selective_reprocess": selective_scope,
        "paper_metadata_handoff": {
            "available_records": len(paper_metadata_by_document),
            "prefilled_documents": sorted(metadata_prefilled_documents),
        },
        "parallelism": {
            "strategy": "stages_in_dependency_order_documents_concurrent",
            "configured_max_workers": worker_limit,
        },
        "section_results": section_results,
        "run_identity": run_identity or {},
    }
