"""
运行 section2_2 精抽：
- prompt：prompts/agent2_1.md + fabrication/{methods}.md + prompts/agent2_2.md
- 输出：outputs/s2_2/{paper_id}.json
"""

from __future__ import annotations

import argparse
import json
import os
import re  # 补充导入 re

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
    inject_fabrication_extraction1,
    inject_paper_text,
    inject_supplementary_information,
    load_prompt_template,
)


PROMPT_AGENT1 = "agent2_1.md"
PROMPT_AGENT2 = "agent2_2.md"

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

METHOD_TO_DISPLAY_NAME = {
    "CVD": "CVD",
    "CVT": "CVT",
    "MBE": "MBE",
    "PVD": "PVD",
    "Solution": "Solution-Based",
    "SSR": "SSR",
    "Flux": "Flux Growth",
    "Melt": "Melt Growth",
    "Nano": "Nanofabrication",
}


def extract_methods_from_input(input_json_str: str) -> list[str]:
    """从第一层粗抽结果（JSON数组）中提取所有不同的 method 字段。"""
    try:
        data = json.loads(input_json_str)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    seen = set()
    methods = []
    for entry in data:
        if isinstance(entry, dict):
            method = entry.get("method")
            if method and method not in seen:
                seen.add(method)
                methods.append(method)
    return methods


def build_fabrication_prompt(fabrication_dir: str, methods: list[str]) -> str:
    """根据 methods 列表，读取对应的 fabrication prompt 文件并拼接。"""
    parts = []
    for method in methods:
        filename = METHOD_TO_FABRICATION_FILE.get(method)
        display_name = METHOD_TO_DISPLAY_NAME.get(method, method)
        if filename:
            path = os.path.join(fabrication_dir, filename)
            if os.path.exists(path):
                content = read_text(path)
                parts.append(f"### {display_name} Method\n\n{content}")
    return "\n\n".join(parts)


def build_full_prompt(
    prompts_dir: str,
    fabrication_dir: str,
    methods: list[str],
    input_json_str: str,
    paper_text: str,
    supplementary_information: str,
) -> str:
    """拼接完整的 prompt：agent2_1.md + fabrication prompts + agent2_2.md"""
    agent1_content = load_prompt_template(prompts_dir, PROMPT_AGENT1)
    fabrication_content = build_fabrication_prompt(fabrication_dir, methods)
    agent2_content = load_prompt_template(prompts_dir, PROMPT_AGENT2)

    prompt = agent1_content
    if fabrication_content:
        prompt += "\n\n" + fabrication_content
    prompt += "\n\n" + agent2_content

    prompt = inject_fabrication_extraction1(prompt, input_json_str)
    prompt = inject_paper_text(prompt, paper_text, tag="paper_text")
    prompt = inject_supplementary_information(prompt, supplementary_information)

    return prompt


def run_s2_2(paper_id: str | None = None) -> None:
    logger = setup_logger("s2_2")
    config = load_config("config.yaml")
    provider = config["provider"]
    paths = config["paths"]
    stage_cfg = config["stages"]["s2_2"]

    prompts_dir = paths["prompts_dir"]
    outputs_dir = paths["outputs_dir"]
    fabrication_dir = paths.get("fabrication_dir", "prompts/fabrication")

    paper_id, paper_path = select_paper_file(paths, paper_id)
    logger.info(f"start paper_id={paper_id}")

    paper_text = read_text(paper_path)
    input_json_str = read_input_json(paths, paper_id)
    supplementary_information = load_supplementary_information(paper_path, paper_id)

    methods = extract_methods_from_input(input_json_str)
    logger.info(f"detected methods={methods}")

    prompt = build_full_prompt(
        prompts_dir=prompts_dir,
        fabrication_dir=fabrication_dir,
        methods=methods,
        input_json_str=input_json_str,
        paper_text=paper_text,
        supplementary_information=supplementary_information,
    )

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
    out_path = write_stage_output(outputs_dir, "s2_2", paper_id, out_text, ext=".json")

    try:
        parsed = json.loads(out_text)
        logger.info("s2_2 JSON 格式验证通过")
    except json.JSONDecodeError as e:
        logger.warning(f"s2_2 输出不是合法 JSON (错误: {e})，尝试自动修复...")
        fixed_text = re.sub(r'[^\x20-\x7E\n\r\t]', '', out_text)
        try:
            parsed = json.loads(fixed_text)
            logger.info("自动修复成功，保存修复后的 JSON")
            write_stage_output(outputs_dir, "s2_2", paper_id, fixed_text, ext=".json")
        except json.JSONDecodeError:
            logger.error(f"自动修复失败，保存原始输出用于手动检查")

    logger.info(f"done output={out_path}")


def run_s2_2_all() -> None:
    logger = setup_logger("s2_2")
    config = load_config("config.yaml")
    paths = config["paths"]
    paper_ids = validate_paper_input_pairs(paths)
    logger.info(f"batch_count={len(paper_ids)}")
    for pid in paper_ids:
        try:
            run_s2_2(paper_id=pid)
        except Exception as e:
            logger.error(f"paper_id={pid} error={e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 section2_2 精抽 -> outputs/s2_2/{paper_id}.json")
    parser.add_argument("--paper_id", type=str, default=None, help="论文文件名（不含后缀）")
    args = parser.parse_args()
    if args.paper_id:
        run_s2_2(paper_id=args.paper_id)
        return
    run_s2_2_all()


if __name__ == "__main__":
    main()