"""
批量并行运行 single 任务（论文之间并行）。

支持的 stages：single

默认不传参数时：运行全部论文的 single。

输入论文文件夹在哪里改：
- config.yaml -> paths.papers_dir
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from utils.io_utils import load_config, validate_papers
from utils.key_pool import KeyPool
from utils.log_utils import setup_logger

from run_single import run_single


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


def filter_paper_ids(paper_ids: list[str], outputs_dir: str, logger) -> list[str]:
    sample_file = os.getenv("IE_SINGLE_SAMPLE_FILE", "").strip()
    if sample_file:
        with open(sample_file, "r", encoding="utf-8-sig") as f:
            sample_ids = {line.strip() for line in f if line.strip()}
        before = len(paper_ids)
        paper_ids = [pid for pid in paper_ids if pid in sample_ids]
        logger.info(f"sample_file={sample_file} sample_ids={len(sample_ids)} selected={len(paper_ids)}/{before}")

    if os.getenv("IE_SINGLE_SKIP_EXISTING", "1").lower() in {"1", "true", "yes"}:
        before = len(paper_ids)
        paper_ids = [
            pid for pid in paper_ids
            if not os.path.exists(os.path.join(outputs_dir, f"{pid}.json"))
        ]
        logger.info(f"skip_existing selected={len(paper_ids)}/{before}")

    return paper_ids


def generate_single_txt(outputs_dir: str, logger) -> None:
    """
    扫描 outputs_dir 中的所有 JSON 文件，
    找出 single_system=1 的文章，生成 single.txt 文件
    """
    single_txt_path = os.path.join(os.path.dirname(outputs_dir), "single.txt")
    single_paper_ids = []

    if not os.path.exists(outputs_dir):
        logger.warning(f"输出目录不存在: {outputs_dir}")
        return

    def _read_single_system(json_path: str) -> int | None:
        """
        读取输出 JSON 中的 single_system 值。
        - 优先严格 JSON 解析
        - 解析失败（如包含 LaTeX 反斜杠）时，用正则兜底只提取 single_system
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                value = data.get("single_system")
                return value if isinstance(value, int) else None
            return None
        except (json.JSONDecodeError, IOError) as e:
            try:  # ✅ 修正：在except块内缩进8个空格
                with open(json_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                match = re.search(r"\"single_system\"\s*:\s*(\d+)", content)
                if match:
                    return int(match.group(1))
            except Exception:
                pass
            logger.warning(f"读取文件失败 {json_path}: {e}")
            return None

    total_files = 0
    matched = 0
    skipped = 0

    # 扫描所有 JSON 文件
    for filename in sorted(os.listdir(outputs_dir)):
        if not filename.endswith(".json"):
            continue

        total_files += 1
        json_path = os.path.join(outputs_dir, filename)
        single_system = _read_single_system(json_path)
        if single_system == 1:
            # 从文件名提取 paper_id（去掉 .json 扩展名）
            paper_id = os.path.splitext(filename)[0]
            single_paper_ids.append(paper_id)
            matched += 1
        else:
            skipped += 1

    # 排序并写入 single.txt
    single_paper_ids.sort()
    try:
        with open(single_txt_path, 'w', encoding='utf-8') as f:
            for paper_id in single_paper_ids:
                f.write(f"{paper_id}\n")

        logger.info(
            f"生成 single.txt 文件: {single_txt_path} ({len(single_paper_ids)} 篇文章, "
            f"total_json={total_files}, skipped={skipped})"
        )
    except IOError as e:
        logger.error(f"写入 single.txt 失败: {e}")


def dispatch(stage: str, paper_id: str) -> None:
    stage = stage.strip().lower()
    if stage == "single":
        return run_single(paper_id=paper_id)
    raise ValueError(f"未知 stage: {stage}（仅支持 single）")


def main() -> None:
    logger = setup_logger("batch")
    parser = argparse.ArgumentParser(description="批量并行运行 Stage 抽取（single）")
    parser.add_argument(
        "--stages",
        type=str,
        default="single",
        help="要运行的 Stage 名称列表，逗号分隔（默认：single）",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="最大并行线程数（默认：API key 数量）",
    )
    args = parser.parse_args()

    stages_list = [s.strip() for s in args.stages.split(",") if s.strip()]
    if not stages_list:
        logger.error("未指定有效的 Stage 列表")
        raise SystemExit(1)

    root = Path(__file__).resolve().parent
    config = load_config(str(root / "config.yaml"))
    provider = config.get("provider", {})
    paths = config.get("paths", {})
    paths, provider = apply_env_overrides(paths, provider)
    paths = {
        **paths,
        "prompts_dir": str((root / paths.get("prompts_dir", "prompts")).resolve()),
        "papers_dir": str((root / paths.get("papers_dir", "papers")).resolve()),
        "outputs_dir": str((root / paths.get("outputs_dir", "outputs")).resolve()),
    }

    paper_ids = validate_papers(paths)
    paper_ids = filter_paper_ids(paper_ids, paths["outputs_dir"], logger)
    if not paper_ids:
        logger.info("no pending papers")
        generate_single_txt(paths["outputs_dir"], logger)
        return

    KeyPool.initialize(config_keys=provider.get("api_keys"))
    key_count = KeyPool.key_count(config_keys=provider.get("api_keys"))
    if key_count <= 0:
        logger.error("未配置任何 API key。")
        raise SystemExit(1)

    total_tasks = len(paper_ids) * len(stages_list)
    max_workers = args.max_workers or min(key_count, total_tasks)

    logger.info(f"papers={len(paper_ids)} stages={stages_list} total_tasks={total_tasks}")
    logger.info(f"max_workers={max_workers} key_count={key_count}")

    start_total = time.time()
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(dispatch, stage, pid): (pid, stage)
            for pid in paper_ids
            for stage in stages_list
        }
        for future in as_completed(future_to_task):
            pid, stage = future_to_task[future]
            try:
                future.result()
                success_count += 1
                logger.info(f"done paper_id={pid} stage={stage}")
            except Exception as exc:
                fail_count += 1
                logger.error(f"fail paper_id={pid} stage={stage} error={exc}")

    total_duration = time.time() - start_total
    logger.info("=== 处理完成 ===")
    logger.info(f"total_duration_sec={total_duration:.2f}")
    logger.info(f"success={success_count} fail={fail_count}")

    # 生成 single.txt 文件：收集所有 single_system=1 的文章文件名
    generate_single_txt(paths["outputs_dir"], logger)


if __name__ == "__main__":
    main()
