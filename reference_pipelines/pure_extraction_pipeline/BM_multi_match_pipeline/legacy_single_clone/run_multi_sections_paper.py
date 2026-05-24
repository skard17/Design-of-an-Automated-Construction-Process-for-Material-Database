from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_prompt_bank import load_multi_prompt
from utils.ds_client import chat
from utils.io_utils import ensure_dir, load_config, load_json, load_supplementary_information, read_text, select_paper_file, slugify_filename, write_text
from utils.key_pool import KeyPool
from utils.log_utils import setup_logger
from utils.prompt_utils import clean_llm_output, inject_blocks


SECTION_ORDER = ["section0", "section1", "section2_1", "section2_2", "section3", "section4", "section5"]
SCOPE_DIR = Path(os.getenv("BM_MULTI_SCOPE_DIR", str(ROOT / "outputs" / "multi_scope")))
FIG_DIR = Path(os.getenv("BM_MULTI_FIG_DIR", str(ROOT / "outputs" / "multi_fig_classify")))
RAW_DIR = Path(os.getenv("BM_MULTI_RAW_DIR", str(ROOT / "outputs" / "multi_section_raw")))
FINAL_DIR = Path(os.getenv("BM_MULTI_FINAL_DIR", str(ROOT / "outputs" / "multi_paper_final")))
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
FABRICATION_DIR = PROMPTS_DIR / "s2_2_fabrication"
VERIFIED_MULTI_DIR = Path(os.getenv("BM_MULTI_META_DIR", str(ROOT.parent / "verified_multi_inputs")))
VERIFIED_MANIFEST_CSV = VERIFIED_MULTI_DIR / "manifest.csv"
VERIFIED_SUMMARY_JSON = VERIFIED_MULTI_DIR / "summary.json"

PROMPT_AGENT2_METHOD_FILES = {
    "CVD": "CVD.md",
    "CVT": "CVT.md",
    "MBE": "MBE.md",
    "PVD": "PVD.md",
    "Solution": "Solution-Based.md",
    "SSR": "SSR.md",
    "Flux": "Flux_Growth.md",
    "Melt": "Melt_Growth.md",
    "Nano": "Nanofabrication.md",
}

METHOD_ALLOWED_KEYS = {
    "CVD": {"reactor_type", "process_gases", "substrate_temperature", "chamber_pressure", "deposition_time", "deposition_rate", "gas_ratios", "pulse_sequence", "rf_power", "substrate", "patterning_method", "post_treatment"},
    "CVT": {"source_material", "transport_agent", "transport_agent_loading", "ampoule_material", "ampoule_dimensions", "charge_conditions", "source_temperature", "growth_temperature", "temperature_gradient", "growth_duration", "atmosphere", "cooling_procedure", "crystal_morphology", "post_treatment"},
    "MBE": {"chamber_base_pressure", "effusion_cells", "gas_sources", "substrate_temperature", "growth_rate", "growth_monitoring", "shutter_sequence", "growth_sequence", "laser_parameters", "plasma_parameters", "cooling_rate", "patterning_method", "post_treatment"},
    "PVD": {"base_pressure", "deposition_gas", "deposition_pressure", "substrate_temperature", "target_substrate_distance", "deposition_rate", "deposition_duration", "energy_source_parameters", "laser_energy", "laser_frequency", "patterning_method", "post_treatment"},
    "Solution": {"solvents", "solution_concentration", "additives", "stirring_conditions", "ph_adjustment", "drying_conditions", "pyrolysis_conditions", "annealing_conditions", "autoclave_conditions", "spin_coating_parameters", "electrodeposition_parameters", "electrolyte_composition", "epd_parameters", "spray_parameters", "post_treatment"},
    "SSR": {"precursor_preparation", "starting_composition", "calcination_conditions", "sintering_conditions", "heating_rate", "cooling_rate", "pressure_technique", "applied_pressure", "encapsulation", "post_treatment"},
    "Flux": {"flux_material", "flux_composition_ratio", "crucible_material", "encapsulation", "max_temperature", "soak_time", "cooling_rate", "decanting_temperature", "removal_method", "post_treatment"},
    "Melt": {"method_variant", "heat_source", "atmosphere", "crucible_material", "growth_rate", "rotation_speed", "feed_rod_preparation", "homogenization_steps", "post_treatment"},
    "Nano": {"starting_material", "exfoliation_method", "substrate", "transfer_method", "encapsulation", "patterning_method", "contact_fabrication", "post_treatment"},
}

METHOD_KEY_ALIASES = {
    "Melt": {
        "fabrication_method": "method_variant",
        "melting_cycles": "homogenization_steps",
    },
    "Flux": {
        "cooling_conditions": "cooling_rate",
    },
}


def call_llm(config: dict, prompt: str) -> str:
    provider = config["provider"]
    api_key = KeyPool.get_next_key(config_keys=provider.get("api_keys"))
    response = chat(
        prompt=prompt,
        model=provider["model"],
        base_url=provider["base_url"],
        api_key=api_key,
        temperature=0.0,
        timeout_sec=int(os.getenv("BM_MULTI_TIMEOUT_SEC", str(provider.get("timeout_sec", 60)))),
        system_prompt="You are an expert in superconducting materials information extraction.",
    )
    return clean_llm_output(response)


def paper_output_dir(base: Path, paper_id: str) -> Path:
    path = base / paper_id
    ensure_dir(str(path))
    return path


def save_json(path: Path, data: dict) -> None:
    write_text(str(path), json.dumps(data, ensure_ascii=False, indent=2))


def parse_llm_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)
        return json.loads(repaired)


def build_single_system_result(multi_system_result: dict, signature: str, signature_type: str) -> dict:
    counted_system = None
    for item in multi_system_result.get("counted_systems", []):
        if str(item.get("system_signature") or "").strip() == signature:
            counted_system = item
            break

    evidence = counted_system.get("evidence", {}) if isinstance(counted_system, dict) else {}
    secondary = multi_system_result.get("secondary_or_control_materials", [])
    return {
        "single_system": 1,
        "primary_signature": signature,
        "primary_signature_type": signature_type,
        "counted_signatures": [signature],
        "secondary_or_control_materials": secondary if isinstance(secondary, list) else [],
        "decision_reason": "legacy-style multi runner: treat one counted system as a pseudo-single target for section extraction",
        "evidence": {
            "primary_system_quotes": evidence.get("system_quotes", []) if isinstance(evidence, dict) else [],
            "other_system_quotes": [],
        },
    }


def load_verified_meta(paper_id: str) -> dict:
    meta: dict = {"paper_id": paper_id}

    if VERIFIED_SUMMARY_JSON.exists():
        try:
            summary = load_json(str(VERIFIED_SUMMARY_JSON))
            if isinstance(summary, dict):
                meta["verified_multi_summary"] = summary
        except Exception:
            pass

    if VERIFIED_MANIFEST_CSV.exists():
        try:
            with VERIFIED_MANIFEST_CSV.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if str(row.get("paper_id") or "").strip() != paper_id:
                        continue
                    meta.update(
                        {
                            "doi": str(row.get("doi") or "").strip(),
                            "task": str(row.get("task") or "").strip(),
                            "source_md": str(row.get("source_md") or "").strip(),
                            "curated_md": str(row.get("curated_md") or "").strip(),
                        }
                    )
                    break
        except Exception:
            pass

    return meta


def build_legacy_prompt(template_name: str, blocks: dict[str, str]) -> str:
    template = read_text(str(PROMPTS_DIR / template_name))
    return inject_blocks(template, blocks)


def extract_methods_from_section2_1(section2_1_payload: dict) -> list[str]:
    methods: list[str] = []
    seen: set[str] = set()
    for entry in section2_1_payload.get("section2_1", []):
        if not isinstance(entry, dict):
            continue
        method = str(entry.get("method") or "").strip()
        if method and method not in seen:
            seen.add(method)
            methods.append(method)
    return methods


def build_legacy_s2_2_prompt(section2_1_payload: dict, paper_text: str, supplementary_information: str) -> str:
    prefix = read_text(str(PROMPTS_DIR / "section2_2_prefix_single.md"))
    suffix = read_text(str(PROMPTS_DIR / "section2_2_suffix_single.md"))
    parts = [prefix]
    for method in extract_methods_from_section2_1(section2_1_payload):
        filename = PROMPT_AGENT2_METHOD_FILES.get(method)
        if not filename:
            continue
        path = FABRICATION_DIR / filename
        if path.exists():
            parts.append(read_text(str(path)))
    parts.append(suffix)
    return inject_blocks(
        "\n\n".join(parts),
        {
            "fabrication_extracton1": json.dumps(section2_1_payload.get("section2_1", []), ensure_ascii=False, indent=2),
            "paper_text": paper_text,
            "supplementary_information": supplementary_information,
        },
    )


def get_system_records(multi_system_result: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    counted_signatures = multi_system_result.get("counted_signatures", [])
    counted_systems = multi_system_result.get("counted_systems", [])
    type_map = {}
    if isinstance(counted_systems, list):
        for item in counted_systems:
            if isinstance(item, dict):
                sig = str(item.get("system_signature") or "").strip()
                sig_type = str(item.get("system_signature_type") or "").strip()
                if sig:
                    type_map[sig] = sig_type or "formula_based"
    for signature in counted_signatures if isinstance(counted_signatures, list) else []:
        sig = str(signature or "").strip()
        if sig:
            out.append((sig, type_map.get(sig, "formula_based")))
    return out


def load_multi_inputs(paper_id: str, multi_system_json: str | None, figure_json: str | None) -> tuple[dict, dict | None]:
    multi_path = Path(multi_system_json) if multi_system_json else SCOPE_DIR / paper_id / "multi_system_result.json"
    multi_system_result = load_json(str(multi_path))
    figure_result = None
    figure_path = Path(figure_json) if figure_json else FIG_DIR / paper_id / "figure_classification.json"
    if figure_path.exists():
        figure_result = load_json(str(figure_path))
    return multi_system_result, figure_result


def build_prompt(section_name: str, multi_system_result: dict, paper_text: str, supplementary_information: str, figure_result: dict | None, section2_1_result: dict | None) -> str:
    blocks = {
        "multi_system_result": json.dumps(multi_system_result, ensure_ascii=False, indent=2),
        "paper_text": paper_text,
        "supplementary_information": supplementary_information,
        "figure_classification": json.dumps(figure_result or {}, ensure_ascii=False, indent=2),
        "fabrication_extracton1": json.dumps(section2_1_result or {}, ensure_ascii=False, indent=2),
    }
    return inject_blocks(load_multi_prompt(section_name), blocks)


def split_by_system(section_name: str, payload: dict) -> dict[str, dict]:
    if section_name in {"section2_1", "section2_2"}:
        value = payload.get(section_name)
        if isinstance(value, list):
            out: dict[str, dict] = {}
            child_key = "section2_1" if section_name == "section2_1" else "section2_2"
            for item in value:
                if not isinstance(item, dict):
                    continue
                signature = str(item.get("system_signature") or "").strip()
                if not signature:
                    continue
                out[signature] = {child_key: {child_key: item.get("fabrication_entries", [])}}
            return out
        return {}
    value = payload.get(section_name)
    if isinstance(value, list):
        out: dict[str, dict] = {}
        for item in value:
            signature = str(item.get("system_signature") or "").strip()
            if not signature:
                continue
            out[signature] = {section_name: item}
        return out
    return {}


def merge_final_record(existing: dict, section_name: str, system_payload: dict) -> dict:
    existing.setdefault("material_info", {"section0": {}, "section1": {}, "section2": {}, "section3": {}, "section4": {}})
    existing.setdefault("section5", {})
    existing.setdefault("paper_info", {})
    if section_name == "section0":
        existing["material_info"]["section0"] = system_payload["section0"]
    elif section_name == "section1":
        existing["material_info"]["section1"] = system_payload["section1"]
    elif section_name in {"section2_1", "section2_2"}:
        existing["material_info"]["section2"][section_name] = system_payload[section_name]
    elif section_name == "section3":
        existing["material_info"]["section3"] = system_payload["section3"]
    elif section_name == "section4":
        existing["material_info"]["section4"] = system_payload["section4"]
    elif section_name == "section5":
        existing["section5"] = system_payload["section5"]
    return existing


def run_legacy_s2_1(
    config: dict,
    multi_system_result: dict,
    paper_text: str,
    supplementary_information: str,
) -> dict:
    system_records = []
    for signature, signature_type in get_system_records(multi_system_result):
        pseudo_single = build_single_system_result(multi_system_result, signature, signature_type)
        prompt = build_legacy_prompt(
            "section2_1_single.md",
            {
                "single_system_result": json.dumps(pseudo_single, ensure_ascii=False, indent=2),
                "paper_text": paper_text,
                "supplementary_information": supplementary_information,
            },
        )
        payload = parse_llm_json(call_llm(config, prompt))
        system_records.append(
            {
                "system_signature": signature,
                "system_signature_type": signature_type,
                "fabrication_entries": payload.get("section2_1", []) if isinstance(payload, dict) else [],
            }
        )
    return {"section2_1": system_records}


def run_legacy_s2_2(
    config: dict,
    section2_1_result: dict,
    paper_text: str,
    supplementary_information: str,
) -> dict:
    out_records = []
    for system_record in section2_1_result.get("section2_1", []):
        if not isinstance(system_record, dict):
            continue
        signature = str(system_record.get("system_signature") or "").strip()
        signature_type = str(system_record.get("system_signature_type") or "formula_based").strip()
        fabrication_entries = system_record.get("fabrication_entries", [])
        single_payload = {"section2_1": fabrication_entries if isinstance(fabrication_entries, list) else []}
        prompt = build_legacy_s2_2_prompt(single_payload, paper_text, supplementary_information)
        payload = parse_llm_json(call_llm(config, prompt))
        out_records.append(
            {
                "system_signature": signature,
                "system_signature_type": signature_type,
                "fabrication_entries": normalize_section2_2_entries(
                    method_entries=payload.get("section2_2", []) if isinstance(payload, dict) else []
                ),
            }
        )
    return {"section2_2": out_records}


def run_legacy_s5(config: dict, paper_text: str, supplementary_information: str) -> dict:
    prompt = build_legacy_prompt(
        "section5_single.md",
        {
            "paper_text": paper_text,
            "supplementary_information": supplementary_information,
        },
    )
    payload = parse_llm_json(call_llm(config, prompt))
    if isinstance(payload, dict) and isinstance(payload.get("section5"), dict):
        payload["section5"].pop("theoretical_keywords", None)
    return payload


def normalize_condition_key(method: str, key: str) -> str | None:
    clean_key = str(key or "").strip()
    if not clean_key:
        return None
    alias_map = METHOD_KEY_ALIASES.get(method, {})
    normalized = alias_map.get(clean_key, clean_key)
    if normalized in METHOD_ALLOWED_KEYS.get(method, set()):
        return normalized
    return None


def normalize_section2_2_entries(method_entries: list) -> list[dict]:
    out: list[dict] = []
    for entry in method_entries if isinstance(method_entries, list) else []:
        if not isinstance(entry, dict):
            continue
        method = str(entry.get("method") or "").strip()
        conditions = entry.get("conditions", {})
        normalized_conditions: dict = {}
        if isinstance(conditions, dict):
            for raw_key, raw_value in conditions.items():
                normalized_key = normalize_condition_key(method, str(raw_key))
                if not normalized_key:
                    continue
                normalized_conditions[normalized_key] = raw_value
        out.append(
            {
                "entry_id": entry.get("entry_id"),
                "method": method,
                "conditions": normalized_conditions,
            }
        )
    return out


def sync_section2_1_conditions_with_section2_2(final_record: dict) -> dict:
    section2 = final_record.get("material_info", {}).get("section2", {})
    section2_1 = section2.get("section2_1", {}).get("section2_1", [])
    section2_2 = section2.get("section2_2", {}).get("section2_2", [])
    if not isinstance(section2_1, list) or not isinstance(section2_2, list):
        return final_record
    keyed = {}
    for entry in section2_2:
        if isinstance(entry, dict):
            keyed[entry.get("entry_id")] = entry.get("conditions", {})
    for entry in section2_1:
        if not isinstance(entry, dict):
            continue
        conditions = keyed.get(entry.get("entry_id"))
        if isinstance(conditions, dict):
            entry["conditions"] = list(conditions.keys())
    return final_record


def run_paper_sections(
    paper_id: str,
    multi_system_json: str | None = None,
    figure_json: str | None = None,
    only_sections: list[str] | None = None,
) -> str:
    logger = setup_logger("multi_sections_paper")
    config = load_config(str(ROOT / "config.yaml"))
    papers_dir = str((ROOT / config["paths"]["papers_dir"]).resolve())
    selected_paper_id, paper_path = select_paper_file({"papers_dir": papers_dir}, paper_id)
    paper_text = read_text(paper_path)
    supplementary_information = load_supplementary_information(paper_path, selected_paper_id)
    multi_system_result, figure_result = load_multi_inputs(selected_paper_id, multi_system_json, figure_json)
    paper_meta = load_verified_meta(selected_paper_id)

    raw_paper_dir = paper_output_dir(RAW_DIR, selected_paper_id)
    final_paper_dir = paper_output_dir(FINAL_DIR, selected_paper_id)

    raw_section2_1_path = raw_paper_dir / "section2_1.json"
    section2_1_result: dict | None = load_json(str(raw_section2_1_path)) if raw_section2_1_path.exists() else None
    final_records: dict[str, dict] = {}
    for existing_file in final_paper_dir.glob("*.json"):
        if existing_file.name == "run_summary.json":
            continue
        existing_payload = load_json(str(existing_file))
        signature = str(existing_payload.get("system_signature") or "").strip()
        if signature:
            existing_payload.setdefault("paper_info", {}).update(paper_meta)
            final_records[signature] = existing_payload

    section_order = only_sections or SECTION_ORDER
    for section_name in section_order:
        if section_name in {"section3", "section4"} and figure_result is None:
            logger.info("skip %s because figure classification is missing", section_name)
            continue
        if section_name == "section2_2" and section2_1_result is None:
            logger.info("skip section2_2 because section2_1 is missing")
            continue

        if section_name == "section2_1":
            payload = run_legacy_s2_1(config, multi_system_result, paper_text, supplementary_information)
        elif section_name == "section2_2":
            payload = run_legacy_s2_2(config, section2_1_result or {"section2_1": []}, paper_text, supplementary_information)
        elif section_name == "section5":
            payload = run_legacy_s5(config, paper_text, supplementary_information)
        else:
            prompt = build_prompt(section_name, multi_system_result, paper_text, supplementary_information, figure_result, section2_1_result)
            out_text = call_llm(config, prompt)
            payload = parse_llm_json(out_text)
        save_json(raw_paper_dir / f"{section_name}.json", payload)
        logger.info("done paper_id=%s section=%s", selected_paper_id, section_name)

        if section_name == "section2_1":
            section2_1_result = payload

        if section_name == "section5":
            section5_payload = payload.get("section5", {}) if isinstance(payload, dict) else {}
            for signature, current in list(final_records.items()):
                current = merge_final_record(current, "section5", {"section5": section5_payload})
                current.setdefault("paper_info", {}).update(paper_meta)
                final_records[signature] = current
                save_json(final_paper_dir / f"{slugify_filename(signature)}.json", current)
            for signature, _signature_type in get_system_records(multi_system_result):
                if signature in final_records:
                    continue
                current = merge_final_record({"system_signature": signature}, "section5", {"section5": section5_payload})
                current.setdefault("paper_info", {}).update(paper_meta)
                final_records[signature] = current
                save_json(final_paper_dir / f"{slugify_filename(signature)}.json", current)
            continue

        split_payloads = split_by_system(section_name, payload)
        for signature, system_payload in split_payloads.items():
            slug = slugify_filename(signature)
            current = final_records.get(signature, {"system_signature": signature})
            current = merge_final_record(current, section_name, system_payload)
            if section_name == "section2_2":
                current = sync_section2_1_conditions_with_section2_2(current)
            current.setdefault("paper_info", {}).update(paper_meta)
            final_records[signature] = current
            save_json(final_paper_dir / f"{slug}.json", current)

    summary = {
        "paper_id": selected_paper_id,
        "paper_info": paper_meta,
        "systems": sorted(final_records.keys()),
        "raw_dir": str(raw_paper_dir),
        "final_dir": str(final_paper_dir),
    }
    save_json(final_paper_dir / "run_summary.json", summary)
    return str(final_paper_dir / "run_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-example sections paper-by-paper and split outputs by system.")
    parser.add_argument("--paper_id", required=True)
    parser.add_argument("--multi_system_json", default=None)
    parser.add_argument("--figure_classification_json", default=None)
    parser.add_argument("--only_sections", nargs="*", default=None)
    args = parser.parse_args()
    print(run_paper_sections(args.paper_id, args.multi_system_json, args.figure_classification_json, args.only_sections))


if __name__ == "__main__":
    main()
