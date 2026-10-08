from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from utils.ds_client import chat
from utils.io_utils import load_config, load_supplementary_information, read_text, select_paper_file, validate_papers, write_named_output
from utils.key_pool import KeyPool
from utils.log_utils import setup_logger
from utils.prompt_utils import clean_llm_output, inject_blocks, load_prompt_template
from utils.io_utils import slugify_filename


PROMPT_FILE = "manifest.md"


REFERENCE_MARKERS = (
    "reference compound",
    "reference material",
    "reference phase",
    "parent compound",
    "parent material",
    "3d analogue",
    "analogue",
    "solid solution",
    "comparison-only",
    "comparison only",
)


def normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def title_mentions_target(title: str, target: dict) -> bool:
    normalized_title = normalize_text(title)
    if not normalized_title:
        return False
    names = [str(target.get("canonical_name") or ""), *[str(x) for x in target.get("aliases", [])]]
    for name in names:
        normalized_name = normalize_text(name)
        if normalized_name and normalized_name in normalized_title:
            return True
    return False


def looks_like_reference_only_target(target: dict, scope_notes: list[str]) -> bool:
    haystacks = [
        str(target.get("family_context") or ""),
        " ".join(str(x) for x in target.get("evidence_quotes", [])),
        " ".join(str(x) for x in scope_notes),
    ]
    normalized = normalize_text(" ".join(haystacks))
    return any(marker in normalized for marker in REFERENCE_MARKERS)


def postprocess_manifest(data: dict) -> dict:
    title = str(data.get("paper_title") or "")
    scope_notes = [str(x) for x in data.get("paper_scope_notes", [])]
    filtered_targets: list[dict] = []
    removed_targets: list[str] = []

    for target in data.get("material_targets", []):
        if title and not title_mentions_target(title, target) and looks_like_reference_only_target(target, scope_notes):
            removed_targets.append(str(target.get("canonical_name") or target.get("target_id") or ""))
            continue
        filtered_targets.append(target)

    if removed_targets:
        notes = list(data.get("paper_scope_notes", []))
        notes.append(
            "Post-filter removed reference-only targets not named in the title: " + ", ".join(removed_targets)
        )
        data["paper_scope_notes"] = notes
    data["material_targets"] = filtered_targets
    return data


def cleanup_stale_outputs(outputs_dir: str, manifest_data: dict) -> list[str]:
    paper_id = str(manifest_data.get("paper_id") or "")
    if not paper_id:
        return []

    keep_names = {
        f"{paper_id}__{slugify_filename(str(target['target_id']))}.json"
        for target in manifest_data.get("material_targets", [])
    }
    removed: list[str] = []

    for subdir_name in ("matched", "final_targets"):
        subdir = Path(outputs_dir) / subdir_name
        if not subdir.exists():
            continue
        for path in sorted(subdir.glob(f"{paper_id}__*.json")):
            if path.name in keep_names:
                continue
            path.unlink(missing_ok=True)
            removed.append(f"{subdir_name}/{path.name}")

    return removed


def run_manifest(paper_id: str | None = None) -> None:
    logger = setup_logger("manifest")
    root = Path(__file__).resolve().parent
    config = load_config(str(root / "config.yaml"))
    provider = config["provider"]
    paths = config["paths"]
    stage_cfg = config["stages"]["manifest"]

    resolved_paths = {
        **paths,
        "papers_dir": str((root / paths["papers_dir"]).resolve()),
        "prompts_dir": str((root / paths["prompts_dir"]).resolve()),
        "outputs_dir": str((root / paths["outputs_dir"]).resolve()),
    }

    paper_id, paper_path = select_paper_file(resolved_paths, paper_id)
    paper_text = read_text(paper_path)
    supplementary_information = load_supplementary_information(paper_path, paper_id)
    template = load_prompt_template(resolved_paths["prompts_dir"], PROMPT_FILE)
    prompt = inject_blocks(
        template,
        {
            "paper_id": paper_id,
            "paper_text": paper_text,
            "supplementary_information": supplementary_information,
        },
    )

    api_key = KeyPool.get_next_key(config_keys=provider.get("api_keys"))
    response = chat(
        prompt=prompt,
        model=provider["model"],
        base_url=provider["base_url"],
        api_key=api_key,
        temperature=float(stage_cfg.get("temperature", 0.0)),
        timeout_sec=int(provider.get("timeout_sec", 60)),
        system_prompt="You are an expert in materials science information extraction and database construction.",
    )
    out_text = clean_llm_output(response)
    manifest_data = postprocess_manifest(json.loads(out_text))
    manifest_data["paper_id"] = paper_id
    out_path = write_named_output(
        resolved_paths["outputs_dir"],
        "manifests",
        paper_id,
        json.dumps(manifest_data, ensure_ascii=False, indent=2),
    )
    removed_outputs = cleanup_stale_outputs(resolved_paths["outputs_dir"], manifest_data)
    if removed_outputs:
        logger.info("removed stale outputs: %s", ", ".join(removed_outputs))
    logger.info("done output=%s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_id", default=None)
    args = parser.parse_args()
    if args.paper_id:
        run_manifest(args.paper_id)
        return

    root = Path(__file__).resolve().parent
    config = load_config(str(root / "config.yaml"))
    paper_ids = validate_papers({"papers_dir": str((root / config["paths"]["papers_dir"]).resolve())})
    for paper_id in paper_ids:
        run_manifest(paper_id)


if __name__ == "__main__":
    main()
