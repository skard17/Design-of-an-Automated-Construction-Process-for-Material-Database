"""
运行 section5 抽取：
- prompt：prompts/ex_s5.md
- 输出：outputs/s5/{paper_id}.json

输入论文文件夹在哪里改：
- config.yaml -> paths.papers_md_dir / paths.papers_tex_dir / paths.input_format
"""

from __future__ import annotations

import argparse
import json

from utils.ds_client import chat
from utils.io_utils import (
    load_config,
    load_supplementary_information,
    read_text,
    select_paper_file,
    validate_papers,
    write_stage_output,
)
from utils.key_pool import KeyPool
from utils.log_utils import setup_logger
from utils.prompt_utils import (
    clean_llm_output,
    inject_paper_text,
    inject_supplementary_information,
    load_prompt_template,
)


PROMPT_FILE = "ex_s5.md"


def run_s5(paper_id: str | None = None) -> None:
    logger = setup_logger("s5")
    config = load_config("config.yaml")
    provider = config["provider"]
    paths = config["paths"]
    stage_cfg = config["stages"]["s5"]

    prompts_dir = paths["prompts_dir"]
    outputs_dir = paths["outputs_dir"]

    paper_id, paper_path = select_paper_file(paths, paper_id)
    logger.info(f"start paper_id={paper_id}")
    paper_text = read_text(paper_path)
    supplementary_information = load_supplementary_information(paper_path, paper_id)

    template = load_prompt_template(prompts_dir, PROMPT_FILE)
    prompt = inject_paper_text(template, paper_text, tag="paper_text")
    prompt = inject_supplementary_information(prompt, supplementary_information)

    # ✅ 修正后的缩进
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
    out_path = write_stage_output(outputs_dir, "s5", paper_id, out_text, ext=".json")
    logger.info(f"done output={out_path}")

    try:
        json.loads(out_text)
    except Exception:
        logger.warning("s5 输出不是合法 JSON（已按原样保存）。")


def run_s5_all() -> None:
    logger = setup_logger("s5")
    config = load_config("config.yaml")
    paths = config["paths"]
    paper_ids = validate_papers(paths)
    logger.info(f"batch_count={len(paper_ids)}")
    for pid in paper_ids:
        try:
            run_s5(paper_id=pid)
        except Exception as e:
            logger.error(f"paper_id={pid} error={e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 section5 抽取 -> outputs/s5/{paper_id}.json")
    parser.add_argument("--paper_id", type=str, default=None, help="论文文件名（不含后缀）")
    args = parser.parse_args()
    if args.paper_id:
        run_s5(paper_id=args.paper_id)
        return
    run_s5_all()


if __name__ == "__main__":
    main()

