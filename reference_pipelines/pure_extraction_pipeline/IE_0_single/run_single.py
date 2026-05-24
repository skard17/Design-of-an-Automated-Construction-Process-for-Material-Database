"""
运行 single 任务：
- prompt：prompts/1.md
- 输出：outputs/{paper_id}.json

输入论文文件夹在哪里改：
- config.yaml -> paths.papers_dir
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from utils.ds_client import chat
from utils.io_utils import (
    load_config,
    load_supplementary_information,
    read_text,
    select_paper_file,
    validate_papers,
    write_text,
)
from utils.key_pool import KeyPool
from utils.log_utils import setup_logger
from utils.prompt_utils import (
    clean_llm_output,
    inject_paper_text,
    inject_supplementary_information,
    load_prompt_template,
)


PROMPT_FILE = "1.md"


def apply_env_overrides(paths: dict, provider: dict) -> tuple[dict, dict]:
    paths = dict(paths)
    provider = dict(provider)
    if os.getenv("IE_SINGLE_PAPERS_DIR"):
        paths["papers_dir"] = os.getenv("IE_SINGLE_PAPERS_DIR")
    if os.getenv("IE_SINGLE_OUTPUTS_DIR"):
        paths["outputs_dir"] = os.getenv("IE_SINGLE_OUTPUTS_DIR")
    if os.getenv("SILICONFLOW_BASE_URL") or os.getenv("LOCAL_DEEPSEEK_BASE_URL"):
        provider["base_url"] = os.getenv("SILICONFLOW_BASE_URL") or os.getenv("LOCAL_DEEPSEEK_BASE_URL")
    if os.getenv("SILICONFLOW_MODEL") or os.getenv("LOCAL_DEEPSEEK_MODEL"):
        provider["model"] = os.getenv("SILICONFLOW_MODEL") or os.getenv("LOCAL_DEEPSEEK_MODEL")
    if os.getenv("SILICONFLOW_TIMEOUT_SEC"):
        provider["timeout_sec"] = int(os.getenv("SILICONFLOW_TIMEOUT_SEC", "300"))
    return paths, provider


def run_single(paper_id: str | None = None) -> None:
    logger = setup_logger("single")
    root = Path(__file__).resolve().parent
    config = load_config(str(root / "config.yaml"))
    provider = config["provider"]
    paths = config["paths"]
    paths, provider = apply_env_overrides(paths, provider)
    stage_cfg = config["stages"]["single"]

    prompts_dir = str((root / paths["prompts_dir"]).resolve())
    papers_dir = str((root / paths["papers_dir"]).resolve())
    outputs_dir = str((root / paths["outputs_dir"]).resolve())
    paths = {**paths, "prompts_dir": prompts_dir, "papers_dir": papers_dir, "outputs_dir": outputs_dir}

    paper_id, paper_path = select_paper_file(paths, paper_id)
    logger.info(f"start paper_id={paper_id}")
    paper_text = read_text(paper_path)
    supplementary_information = load_supplementary_information(paper_path, paper_id)

    template = load_prompt_template(prompts_dir, PROMPT_FILE)
    prompt = inject_paper_text(template, paper_text, tag="paper_text")
    prompt = inject_supplementary_information(prompt, supplementary_information)

    api_key = KeyPool.get_next_key(config_keys=provider.get("api_keys"))
    # 准备chat参数
    chat_kwargs = {
        "prompt": prompt,
        "model": provider["model"],
        "base_url": provider["base_url"],
        "api_key": api_key,
        "temperature": float(stage_cfg.get("temperature", 0.0)),
        "timeout_sec": int(provider.get("timeout_sec", 60)),
        "system_prompt": "You are an expert in superconducting materials information extraction.",
    }

    # 如果设置了max_tokens，添加到参数中
    if "max_tokens" in provider:
        chat_kwargs["max_tokens"] = int(provider["max_tokens"])

    resp = chat(**chat_kwargs)

    out_text = clean_llm_output(resp)
    out_path = os.path.join(outputs_dir, f"{paper_id}.json")
    write_text(out_path, out_text)
    logger.info(f"done output={out_path}")

    try:
        json.loads(out_text)
    except Exception:
        logger.warning("single 输出不是合法 JSON（已按原样保存）。")


def run_single_all() -> None:
    logger = setup_logger("single")
    root = Path(__file__).resolve().parent
    config = load_config(str(root / "config.yaml"))
    paths = config["paths"]
    paths = {
        **paths,
        "prompts_dir": str((root / paths["prompts_dir"]).resolve()),
        "papers_dir": str((root / paths["papers_dir"]).resolve()),
        "outputs_dir": str((root / paths["outputs_dir"]).resolve()),
    }
    paper_ids = validate_papers(paths)
    logger.info(f"batch_count={len(paper_ids)}")
    for pid in paper_ids:
        try:
            run_single(paper_id=pid)
        except Exception as e:
            logger.error(f"paper_id={pid} error={e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 single 任务 -> outputs/{paper_id}.json")
    parser.add_argument("--paper_id", type=str, default=None, help="论文文件名（不含后缀）")
    args = parser.parse_args()
    if args.paper_id:
        run_single(paper_id=args.paper_id)
        return
    run_single_all()


if __name__ == "__main__":
    main()
