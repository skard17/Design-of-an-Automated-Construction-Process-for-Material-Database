"""
运行 section0 抽取：
- prompt：prompts/ex_s0.md
- 输出：outputs/s0/{paper_id}.json

输入论文文件夹在哪里改：
- config.yaml -> paths.papers_md_dir / paths.papers_tex_dir / paths.input_format
"""

from __future__ import annotations

import argparse
import json

from utils.ds_client import chat
from utils.io_utils import (
    load_config,
    read_input_json,
    read_text,
    load_supplementary_information,
    select_paper_file,
    validate_paper_input_pairs,
    write_stage_output,
)
from utils.key_pool import KeyPool
from utils.log_utils import setup_logger
from utils.prompt_utils import (
    clean_llm_output,
    inject_single_system_result,
    inject_paper_text,
    inject_supplementary_information,
    load_prompt_template,
)


PROMPT_FILE = "ex_s0.md"


def run_s0(paper_id: str | None = None) -> None:
    logger = setup_logger("s0")
    config = load_config("config.yaml")
    provider = config["provider"]
    paths = config["paths"]
    stage_cfg = config["stages"]["s0"]

    prompts_dir = paths["prompts_dir"]
    outputs_dir = paths["outputs_dir"]

    paper_id, paper_path = select_paper_file(paths, paper_id)
    logger.info(f"start paper_id={paper_id}")
    paper_text = read_text(paper_path)
    single_system_json = read_input_json(paths, paper_id)
    supplementary_information = load_supplementary_information(paper_path, paper_id)

    template = load_prompt_template(prompts_dir, PROMPT_FILE)
    prompt = inject_single_system_result(template, single_system_json)
    prompt = inject_paper_text(prompt, paper_text, tag="paper_text")
    prompt = inject_supplementary_information(prompt, supplementary_information)

    api_key = KeyPool.get_next_key(config_keys=provider.get("api_keys"))
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
    out_path = write_stage_output(outputs_dir, "s0", paper_id, out_text, ext=".json")
    logger.info(f"done output={out_path}")

    # 增强 JSON 合法性检查和自动修复
    try:
        parsed = json.loads(out_text)
        logger.info("s0 JSON 格式验证通过")
    except json.JSONDecodeError as e:
        logger.warning(f"s0 输出不是合法 JSON (错误: {e})，尝试自动修复...")
        # 尝试简单的修复：移除可能的 markdown 残留
        import re
        fixed_text = re.sub(r'[^\x20-\x7E\n\r\t]', '', out_text)  # 移除非 ASCII 字符
        try:
            parsed = json.loads(fixed_text)
            logger.info("自动修复成功，保存修复后的 JSON")
            write_stage_output(outputs_dir, "s0", paper_id, fixed_text, ext=".json")
        except json.JSONDecodeError:
            logger.error(f"自动修复失败，保存原始输出用于手动检查")
            # 可以在这里添加更复杂的修复逻辑，比如使用 json-repair 库


def run_s0_all() -> None:
    logger = setup_logger("s0")
    config = load_config("config.yaml")
    paths = config["paths"]
    paper_ids = validate_paper_input_pairs(paths)
    logger.info(f"batch_count={len(paper_ids)}")
    for pid in paper_ids:
        try:
            run_s0(paper_id=pid)
        except Exception as e:
            logger.error(f"paper_id={pid} error={e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 section0 抽取 -> outputs/s0/{paper_id}.json")
    parser.add_argument("--paper_id", type=str, default=None, help="论文文件名（不含后缀）")
    args = parser.parse_args()
    if args.paper_id:
        run_s0(paper_id=args.paper_id)
        return
    run_s0_all()


if __name__ == "__main__":
    main()


