"""
Extract section3 information to outputs/s3/{paper_id}.json.
"""
from __future__ import annotations

import argparse
import json

from utils.ds_client import chat
from utils.io_utils import (
    load_config,
    load_supplementary_information,
    read_input_json,
    read_text,
    select_paper_file,
    validate_paper_input_pairs,
    write_stage_output,
)
from utils.key_pool import KeyPool
from utils.log_utils import setup_logger
from utils.prompt_utils import (
    clean_llm_output,
    repair_json_text,
    inject_figure_classification,
    inject_paper_text,
    inject_supplementary_information,
    load_prompt_template,
)


PROMPT_FILE = "ex_s3.md"


def run_s3(paper_id: str | None = None) -> None:
    logger = setup_logger("s3")
    config = load_config("config.yaml")
    provider = config["provider"]
    paths = config["paths"]
    stage_cfg = config["stages"]["s3"]

    prompts_dir = paths["prompts_dir"]
    outputs_dir = paths["outputs_dir"]

    paper_id, paper_path = select_paper_file(paths, paper_id)
    logger.info(f"start paper_id={paper_id}")
    paper_text = read_text(paper_path)
    fig_classify_json = read_input_json(paths, paper_id)
    supplementary_information = load_supplementary_information(paper_path, paper_id)

    template = load_prompt_template(prompts_dir, PROMPT_FILE)
    prompt = inject_figure_classification(template, fig_classify_json)
    prompt = inject_paper_text(prompt, paper_text, tag="paper_text")
    prompt = inject_supplementary_information(prompt, supplementary_information)

    with KeyPool.leased_key(config_keys=provider.get("api_keys")) as api_key:
        resp = chat(
            prompt=prompt,
            model=provider["model"],
            base_url=provider["base_url"],
            api_key=api_key,
            temperature=float(stage_cfg.get("temperature", 0.0)),
            timeout_sec=int(provider.get("timeout_sec", 60)),
            system_prompt="You are an expert in superconducting materials information extraction.",
        )

    out_text = clean_llm_output(resp)
    out_path = write_stage_output(outputs_dir, "s3", paper_id, out_text, ext=".json")

    try:
        json.loads(out_text)
        logger.info("s3 JSON validated")
    except json.JSONDecodeError as exc:
        logger.warning(f"s3 produced invalid JSON (error: {exc}); attempting repair")
        fixed_text = repair_json_text(out_text)
        try:
            json.loads(fixed_text)
            logger.info("s3 JSON repaired successfully")
            write_stage_output(outputs_dir, "s3", paper_id, fixed_text, ext=".json")
        except json.JSONDecodeError:
            logger.error("s3 JSON repair failed")

    logger.info(f"done output={out_path}")


def run_s3_all() -> None:
    logger = setup_logger("s3")
    config = load_config("config.yaml")
    paths = config["paths"]
    paper_ids = validate_paper_input_pairs(paths)
    logger.info(f"batch_count={len(paper_ids)}")
    for pid in paper_ids:
        try:
            run_s3(paper_id=pid)
        except Exception as exc:
            logger.error(f"paper_id={pid} error={exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run s3 extraction to outputs/s3/{paper_id}.json")
    parser.add_argument("--paper_id", type=str, default=None, help="Process a single paper id")
    args = parser.parse_args()
    if args.paper_id:
        run_s3(paper_id=args.paper_id)
        return
    run_s3_all()


if __name__ == "__main__":
    main()
