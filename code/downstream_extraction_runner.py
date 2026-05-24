import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import prompt_quality_code_agent as prompt_agent


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


def stage_fields(prompt_output: dict[str, Any], stage_id: str, max_fields: int | None = None) -> list[str]:
    prompts = (
        (prompt_output.get("module_outputs") or {})
        .get("section_extraction_prompt_module", {})
        .get("section_extraction_prompts", {})
    )
    contract = (prompts.get(stage_id) or {}).get("output_contract") or {}
    fields = [str(item) for item in contract.get("fields", []) if item]
    if max_fields and max_fields > 0:
        return fields[:max_fields]
    return fields


def chunk_fields(fields: list[str], max_fields: int) -> list[list[str]]:
    if not max_fields or max_fields <= 0 or len(fields) <= max_fields:
        return [fields]
    return [fields[index : index + max_fields] for index in range(0, len(fields), max_fields)]


def merge_stage_payloads(
    payloads: list[dict[str, Any]],
    stage: dict[str, Any],
    document_name: str,
    field_batches: list[list[str]],
) -> dict[str, Any]:
    extracted = []
    missing = []
    quality_notes = []
    seen_extracted = set()
    seen_missing = set()

    for batch_index, payload in enumerate(payloads):
        if payload.get("error") or payload.get("raw_text"):
            return {
                "stage_id": stage.get("stage_id"),
                "section_id": stage.get("section_id"),
                "document": document_name,
                "error": payload.get("error", "batch extraction failed"),
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
        quality_notes.extend(payload.get("quality_notes") or [])

    return {
        "stage_id": stage.get("stage_id"),
        "section_id": stage.get("section_id"),
        "document": document_name,
        "extracted_fields": extracted,
        "missing_fields": missing,
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
            ]
        )
    return rules


def build_stage_prompt(
    document_text: str,
    document_name: str,
    stage: dict[str, Any],
    fields: list[str],
    dependency_context: dict[str, Any],
) -> str:
    specific_rules = "\n".join(f"- {rule}" for rule in stage_specific_rules(stage))
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

Rules:
- Return JSON only.
- Extract only information supported by the document text.
- Do not invent values.
- Never put null, "not mentioned", "not reported", "none", "N/A", or empty values in extracted_fields.
- If a requested field is absent, put it under missing_fields with field_path and a short missing_reason.
- Every extracted value must include field_path, non-null value, evidence_text, confidence, and source_hint.
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
  "quality_notes": []
}}

Document text:
<<<
{document_text}
>>>
""".strip()


NULL_LIKE_VALUES = {"", "null", "none", "not mentioned", "not reported", "n/a", "na", "no", "not found"}
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


def is_null_like(value: Any) -> bool:
    return value is None or str(value).strip().lower() in NULL_LIKE_VALUES


def is_section2_measurement_leak(item: dict[str, Any], payload: dict[str, Any]) -> bool:
    if payload.get("section_id") != "material_info.section2":
        return False
    text = " ".join(str(item.get(key) or "") for key in ("value", "evidence_text", "source_hint")).lower()
    if any(pattern in text for pattern in PROCESS_KEEP_PATTERNS):
        return False
    return any(pattern in text for pattern in MEASUREMENT_LEAK_PATTERNS)


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
    if text in {"section0", "section1", "section2", "section3", "section4"}:
        return f"material_info.{text}"
    if text in {"section5", "paper_info"} or text.startswith("material_info."):
        return text
    return text


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
    if not isinstance(extracted, list):
        extracted = []
    if not isinstance(missing, list):
        missing = []
    clean_extracted = []
    moved_to_missing = []
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
        else:
            field_path = str(item.get("field_path") or "").lower()
            if field_path.endswith(".doi") or field_path == "paper_info.metadata.doi":
                item["value"] = normalize_doi(item.get("value"))
            if str(payload.get("stage_id") or "") == "figure_classification":
                item["value"] = normalize_figure_sections(item.get("value"))
            clean_extracted.append(item)
    if moved_to_missing:
        payload.setdefault("quality_notes", []).append(
            f"Moved {len(moved_to_missing)} null-like extracted_fields into missing_fields."
        )
    payload["extracted_fields"] = clean_extracted
    payload["missing_fields"] = [*missing, *moved_to_missing]
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
            "error": payload.get("error", "raw_text payload indicates an earlier parse failure"),
        }
    extracted = payload.get("extracted_fields")
    missing = payload.get("missing_fields")
    if not isinstance(extracted, list):
        extracted = []
    if not isinstance(missing, list):
        missing = []
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
    return {
        "extracted_count": len(extracted),
        "missing_count": len(missing),
        "null_like_count": null_like_count,
        "unicode_candidate_count": unicode_candidate_count,
        "evidence_count": evidence_count,
        "has_json": True,
        "has_evidence": evidence_count > 0 or len(extracted) == 0,
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
    max_tokens: int = 4096,
    output_dir: str = "step9_extraction_runs",
    max_documents: int = 4,
    max_document_chars: int = 26000,
    max_fields_per_stage: int = 80,
) -> dict[str, Any]:
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    selected_docs = [str(Path(path)) for path in documents[:max_documents]]
    section_results = []
    dependency_outputs: dict[str, Any] = {}

    for stage in workflow_plan.get("section_test_plan", []):
        stage_id = stage["stage_id"]
        if stage_id == "unicode_normalization" or str(stage.get("section_id", "")).startswith("postprocess."):
            continue
        stage_dir = run_dir / safe_name(stage_id)
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_policy = (workflow_plan.get("stage_batching_policy") or {}).get(stage_id) or {}
        stage_max_fields = int(stage_policy.get("max_fields_per_call") or max_fields_per_stage)
        all_fields = stage_fields(prompt_output, stage_id)
        if stage_policy.get("max_fields_per_call"):
            field_batches = chunk_fields(all_fields, stage_max_fields)
            fields = all_fields
        else:
            fields = all_fields[:stage_max_fields]
            field_batches = [fields]
        doc_results = []
        failed_docs = []

        for doc_path in selected_docs:
            doc_name = Path(doc_path).name
            output_file = stage_dir / f"{safe_name(Path(doc_path).stem)}.json"
            if output_file.exists():
                try:
                    payload = json.loads(output_file.read_text(encoding="utf-8"))
                    payload = sanitize_payload(payload, normalize_unicode=False)
                    checks = validate_stage_payload(payload, normalize_unicode=False)
                    doc_status = "passed" if checks["has_json"] else "invalid_json"
                    if doc_status == "passed":
                        output_file.write_text(json_dumps(payload), encoding="utf-8")
                        doc_results.append(
                            {
                                "document": doc_name,
                                "status": doc_status,
                                "checks": checks,
                                "output_file": str(output_file),
                                "resumed": True,
                            }
                        )
                        continue
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
                batch_output_file = batch_dir / f"batch_{batch_index + 1:02d}.json"
                if len(field_batches) > 1 and batch_output_file.exists():
                    try:
                        batch_payload = json.loads(batch_output_file.read_text(encoding="utf-8"))
                        batch_payload = sanitize_payload(batch_payload, normalize_unicode=False)
                        if validate_stage_payload(batch_payload, normalize_unicode=False)["has_json"]:
                            batch_payloads.append(batch_payload)
                            continue
                    except json.JSONDecodeError:
                        pass

                prompt = build_stage_prompt(document_text, doc_name, stage, batch_fields, dependency_context)
                raw_text = prompt_agent.litellm_chat(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                try:
                    batch_payload = prompt_agent.parse_llm_json(raw_text)
                    batch_payload = sanitize_payload(batch_payload, normalize_unicode=False)
                except Exception as exc:  # noqa: BLE001 - keep bench diagnostics explicit
                    batch_payload = {"raw_text": raw_text, "error": str(exc)}
                if len(field_batches) > 1:
                    batch_output_file.write_text(json_dumps(batch_payload), encoding="utf-8")
                batch_payloads.append(batch_payload)

            payload = merge_stage_payloads(batch_payloads, stage, doc_name, field_batches)
            payload = sanitize_payload(payload, normalize_unicode=False)
            checks = validate_stage_payload(payload, normalize_unicode=False)
            doc_status = "passed" if checks["has_json"] else "invalid_json"
            if doc_status == "invalid_json":
                failed_docs.append(doc_name)

            item = {
                "document": doc_name,
                "status": doc_status,
                "checks": checks,
                "output_file": str(output_file),
            }
            if len(field_batches) > 1:
                item["batch_count"] = len(field_batches)
            Path(item["output_file"]).write_text(json_dumps(payload), encoding="utf-8")
            doc_results.append(item)

        extracted_total = sum((item.get("checks") or {}).get("extracted_count", 0) for item in doc_results)
        null_like_total = sum((item.get("checks") or {}).get("null_like_count", 0) for item in doc_results)
        unicode_candidate_total = sum((item.get("checks") or {}).get("unicode_candidate_count", 0) for item in doc_results)
        invalid_count = sum(1 for item in doc_results if item.get("status") == "invalid_json")
        stage_status = "passed"
        if invalid_count:
            stage_status = "invalid_json"
        elif extracted_total == 0 and stage_id not in {"figure_classification"}:
            stage_status = "low_quality"

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
                "extracted_total": extracted_total,
                "null_like_total": null_like_total,
                "unicode_candidate_total": unicode_candidate_total,
                "failed_docs": failed_docs,
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
        "section_results": section_results,
    }
