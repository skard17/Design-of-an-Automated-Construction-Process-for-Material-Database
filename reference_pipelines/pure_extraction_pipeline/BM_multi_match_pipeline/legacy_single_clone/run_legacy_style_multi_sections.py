from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.ds_client import chat
from utils.io_utils import (
    ensure_dir,
    load_config,
    load_json,
    load_supplementary_information,
    read_text,
    select_paper_file,
    slugify_filename,
    write_text,
)
from utils.key_pool import KeyPool
from utils.log_utils import setup_logger
from utils.prompt_utils import clean_llm_output, inject_blocks


CLONE_DIR = ROOT / "legacy_single_clone"
PROMPTS_DIR = CLONE_DIR / "prompts"
FABRICATION_DIR = PROMPTS_DIR / "s2_2_fabrication"

SECTION_PROMPTS = {
    "section0": PROMPTS_DIR / "section0_single.md",
    "section1": PROMPTS_DIR / "section1_single.md",
    "section2_1": PROMPTS_DIR / "section2_1_single.md",
    "section3": PROMPTS_DIR / "section3_single.md",
    "section4": PROMPTS_DIR / "section4_single.md",
    "section5": PROMPTS_DIR / "section5_single.md",
}

METHOD_TO_FABRICATION_FILE = {
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


def build_pseudo_single_result(multi_system_result: dict, system_entry: dict) -> dict:
    signature = str(system_entry.get("system_signature") or "")
    signature_type = str(system_entry.get("system_signature_type") or "formula_based")
    counted_systems = list(multi_system_result.get("counted_systems", []))
    secondary = list(multi_system_result.get("secondary_or_control_materials", []))
    primary_signature = multi_system_result.get("primary_signature")
    if primary_signature is None:
        primary_signature = signature

    current_quotes: list[str] = []
    for item in counted_systems:
        if str(item.get("system_signature") or "") == signature:
            current_quotes = list(item.get("evidence_quotes", []))
            break

    other_quotes: list[str] = []
    for item in counted_systems:
        if str(item.get("system_signature") or "") == signature:
            continue
        other_quotes.extend(item.get("evidence_quotes", []))

    return {
        "single_system": 1,
        "primary_signature": primary_signature,
        "primary_signature_type": "formula_based" if signature_type == "formula_based" else "stack_based",
        "counted_signatures": [signature],
        "counted_systems": [
            {
                "system_signature": signature,
                "system_signature_type": signature_type,
                "evidence_quotes": current_quotes,
            }
        ],
        "secondary_or_control_materials": secondary,
        "decision_reason": "single_counted_signature",
        "evidence": {
            "primary_system_quotes": current_quotes,
            "other_system_quotes": other_quotes,
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
        timeout_sec=int(provider.get("timeout_sec", 60)),
        system_prompt="You are an expert in superconducting materials information extraction.",
    )
    return clean_llm_output(response)


def load_multi_system_result(paper_id: str, explicit_path: str | None) -> dict:
    if explicit_path:
        return load_json(explicit_path)
    default_path = ROOT / "outputs" / "multi_scope" / paper_id / "multi_system_result.json"
    return load_json(str(default_path))


def output_dir_for_system(paper_id: str, system_signature: str) -> Path:
    return ROOT / "outputs" / "legacy_style_sections" / paper_id / slugify_filename(system_signature)


def write_section_output(paper_id: str, system_signature: str, section_name: str, content: str) -> Path:
    out_dir = output_dir_for_system(paper_id, system_signature)
    ensure_dir(str(out_dir))
    out_path = out_dir / f"{section_name}.json"
    write_text(str(out_path), content)
    return out_path


def build_standard_prompt(prompt_path: Path, single_result: dict, paper_text: str, supplementary_information: str) -> str:
    template = prompt_path.read_text(encoding="utf-8")
    return inject_blocks(
        template,
        {
            "single_system_result": json.dumps(single_result, ensure_ascii=False, indent=2),
            "paper_text": paper_text,
            "supplementary_information": supplementary_information,
        },
    )


def extract_methods_from_section2_1(section2_1_json: dict) -> list[str]:
    entries = section2_1_json.get("section2_1", [])
    seen: list[str] = []
    for entry in entries:
        method = str(entry.get("method") or "").strip()
        if method and method not in seen:
            seen.append(method)
    return seen


def build_section2_2_prompt(section2_1_result: dict, paper_text: str, supplementary_information: str) -> str:
    prefix = (PROMPTS_DIR / "section2_2_prefix_single.md").read_text(encoding="utf-8")
    suffix = (PROMPTS_DIR / "section2_2_suffix_single.md").read_text(encoding="utf-8")
    methods = extract_methods_from_section2_1(section2_1_result)
    fabrication_parts: list[str] = []
    for method in methods:
        filename = METHOD_TO_FABRICATION_FILE.get(method)
        if not filename:
            continue
        path = FABRICATION_DIR / filename
        if path.exists():
            fabrication_parts.append(path.read_text(encoding="utf-8"))

    template = prefix
    if fabrication_parts:
        template += "\n\n" + "\n\n".join(fabrication_parts)
    template += "\n\n" + suffix

    return inject_blocks(
        template,
        {
            "fabrication_extracton1": json.dumps(section2_1_result.get("section2_1", []), ensure_ascii=False, indent=2),
            "paper_text": paper_text,
            "supplementary_information": supplementary_information,
        },
    )


def run_sections_for_system(
    config: dict,
    paper_id: str,
    paper_text: str,
    supplementary_information: str,
    multi_system_result: dict,
    system_entry: dict,
    figure_classification: dict | None,
) -> dict[str, str]:
    logger = setup_logger("legacy_multi_sections")
    signature = str(system_entry.get("system_signature") or "")
    single_result = build_pseudo_single_result(multi_system_result, system_entry)
    outputs: dict[str, str] = {}

    for section_name in ("section0", "section1", "section2_1", "section5"):
        prompt = build_standard_prompt(SECTION_PROMPTS[section_name], single_result, paper_text, supplementary_information)
        out_text = call_llm(config, prompt)
        out_path = write_section_output(paper_id, signature, section_name, out_text)
        outputs[section_name] = str(out_path)

    section2_1_result = json.loads(read_text(outputs["section2_1"]))
    prompt_2_2 = build_section2_2_prompt(section2_1_result, paper_text, supplementary_information)
    out_text_2_2 = call_llm(config, prompt_2_2)
    out_path_2_2 = write_section_output(paper_id, signature, "section2_2", out_text_2_2)
    outputs["section2_2"] = str(out_path_2_2)

    if figure_classification is not None:
        for section_name in ("section3", "section4"):
            template = SECTION_PROMPTS[section_name].read_text(encoding="utf-8")
            prompt = inject_blocks(
                template,
                {
                    "figure_classification": json.dumps(figure_classification, ensure_ascii=False, indent=2),
                    "paper_text": paper_text,
                    "supplementary_information": supplementary_information,
                },
            )
            out_text = call_llm(config, prompt)
            out_path = write_section_output(paper_id, signature, section_name, out_text)
            outputs[section_name] = str(out_path)
    else:
        logger.info(
            "paper_id=%s system=%s skip section3/4 because no figure classification JSON was provided",
            paper_id,
            signature,
        )

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run copied single-system section prompts separately for each counted system in a multi-system paper."
    )
    parser.add_argument("--paper_id", required=True)
    parser.add_argument("--multi_system_json", default=None)
    parser.add_argument("--figure_classification_json", default=None)
    args = parser.parse_args()

    logger = setup_logger("legacy_multi_sections")
    config = load_config(str(ROOT / "config.yaml"))
    papers_dir = str((ROOT / config["paths"]["papers_dir"]).resolve())
    paper_id, paper_path = select_paper_file({"papers_dir": papers_dir}, args.paper_id)
    paper_text = read_text(paper_path)
    supplementary_information = load_supplementary_information(paper_path, paper_id)
    multi_system_result = load_multi_system_result(paper_id, args.multi_system_json)
    figure_classification = load_json(args.figure_classification_json) if args.figure_classification_json else None

    summary: dict[str, dict[str, str]] = {}
    for system_entry in multi_system_result.get("counted_systems", []):
        signature = str(system_entry.get("system_signature") or "").strip()
        if not signature:
            continue
        logger.info("run paper_id=%s system=%s", paper_id, signature)
        summary[signature] = run_sections_for_system(
            config=config,
            paper_id=paper_id,
            paper_text=paper_text,
            supplementary_information=supplementary_information,
            multi_system_result=multi_system_result,
            system_entry=system_entry,
            figure_classification=figure_classification,
        )

    summary_path = ROOT / "outputs" / "legacy_style_sections" / paper_id / "run_summary.json"
    ensure_dir(str(summary_path.parent))
    write_text(str(summary_path), json.dumps(summary, ensure_ascii=False, indent=2))
    print(summary_path)


if __name__ == "__main__":
    main()
