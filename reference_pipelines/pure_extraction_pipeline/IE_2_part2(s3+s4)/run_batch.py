"""
批量并行运行（论文之间并行，section 之间并行）。

支持的 stages：s3, s4

默认不传参数时：运行全部论文的 s3,s4。

输入论文文件夹在哪里改：
- config.yaml -> paths.papers_md_dir / paths.papers_tex_dir / paths.input_format
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.io_utils import load_config, validate_paper_input_pairs
from utils.key_pool import KeyPool
from utils.log_utils import setup_logger

from run_s3 import run_s3
from run_s4 import run_s4


def dispatch(stage: str, paper_id: str) -> None:
    stage = stage.strip().lower()
    if stage == "s3":
        return run_s3(paper_id=paper_id)
    if stage == "s4":
        return run_s4(paper_id=paper_id)
    raise ValueError(f"未知 stage: {stage}（仅支持 s3/s4）")


def main() -> None:
    logger = setup_logger("batch")
    parser = argparse.ArgumentParser(description="批量并行运行 Stage 抽取（s3/s4）")
    parser.add_argument(
        "--stages",
        type=str,
        default="s3,s4",
        help="要运行的 Stage 名称列表，逗号分隔（默认：s3,s4）",
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

    config = load_config("config.yaml")
    provider = config.get("provider", {})
    paths = config.get("paths", {})

    paper_ids = validate_paper_input_pairs(paths)
    if not paper_ids:
        logger.error("未找到任何论文 ID。")
        raise SystemExit(1)

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
    logger.info(f"total_duration_sec={total_duration:.2f}")
    logger.info(f"success={success_count} fail={fail_count}")


if __name__ == "__main__":
    main()
