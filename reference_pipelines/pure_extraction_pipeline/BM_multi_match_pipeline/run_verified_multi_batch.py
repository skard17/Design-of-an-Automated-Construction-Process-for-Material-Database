from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(r"C:\Users\Administrator\Desktop\fsdownload\sc-ie")
PIPELINE_ROOT = ROOT / "BM_multi_match_pipeline"
STATUS_FILE = ROOT / "verified_status_lists" / "multi.txt"
STATUS_MANIFEST_CSV = ROOT / "verified_status_lists" / "status_manifest.csv"
PAPERS_DIR = PIPELINE_ROOT / "papers"
LOG_DIR = PIPELINE_ROOT / "logs"


@dataclass(frozen=True)
class PaperTask:
    doi: str
    paper_id: str
    task: str
    source_md: Path
    source_si: Path


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def load_verified_tasks(log_path: Path) -> tuple[list[PaperTask], int]:
    doi_list = [
        line.strip().lower()
        for line in STATUS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    manifest: dict[str, dict[str, str]] = {}
    with STATUS_MANIFEST_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            doi = (row.get("doi") or "").strip().lower()
            paper_id = (row.get("paper_id") or "").strip()
            task = (row.get("task") or "").strip()
            latest_status = (row.get("latest_status") or "").strip()
            if doi and paper_id and task:
                manifest[doi] = {
                    "paper_id": paper_id,
                    "task": task,
                    "latest_status": latest_status,
                }

    tasks: list[PaperTask] = []
    missing_count = 0
    seen: set[str] = set()
    for doi in doi_list:
        row = manifest.get(doi)
        if not row or row.get("latest_status") != "multi":
            append_log(log_path, f"[MISSING] {doi} :: not found as latest multi in status_manifest.csv")
            missing_count += 1
            continue
        paper_id = row["paper_id"]
        if paper_id in seen:
            continue
        seen.add(paper_id)
        task_name = row["task"]
        source_md = ROOT / task_name / "papers_md" / f"{paper_id}.md"
        source_si = ROOT / task_name / "papers_md" / f"{paper_id}_si.md"
        if not source_md.exists():
            append_log(log_path, f"[MISSING] {doi} -> {paper_id} :: source md not found: {source_md}")
            missing_count += 1
            continue
        tasks.append(
            PaperTask(
                doi=doi,
                paper_id=paper_id,
                task=task_name,
                source_md=source_md,
                source_si=source_si,
            )
        )
    return tasks, missing_count


def sync_papers(tasks: list[PaperTask]) -> None:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    for item in tasks:
        shutil.copyfile(item.source_md, PAPERS_DIR / f"{item.paper_id}.md")
        if item.source_si.exists():
            shutil.copyfile(item.source_si, PAPERS_DIR / f"{item.paper_id}_si.md")


def stage_missing_ie0(tasks: list[PaperTask], scope_dir: Path) -> tuple[list[PaperTask], int]:
    queue: list[PaperTask] = []
    skip = 0
    for item in tasks:
        out_path = scope_dir / item.paper_id / "multi_system_result.json"
        if out_path.exists():
            skip += 1
        else:
            queue.append(item)
    return queue, skip


def stage_missing_fig(tasks: list[PaperTask], scope_dir: Path, fig_dir: Path) -> tuple[list[PaperTask], int]:
    queue: list[PaperTask] = []
    skip = 0
    for item in tasks:
        scope_path = scope_dir / item.paper_id / "multi_system_result.json"
        fig_path = fig_dir / item.paper_id / "figure_classification.json"
        if fig_path.exists():
            skip += 1
        elif scope_path.exists():
            queue.append(item)
    return queue, skip


def stage_missing_sections(tasks: list[PaperTask], fig_dir: Path, final_dir: Path) -> tuple[list[PaperTask], int]:
    queue: list[PaperTask] = []
    skip = 0
    for item in tasks:
        fig_path = fig_dir / item.paper_id / "figure_classification.json"
        final_path = final_dir / item.paper_id / "run_summary.json"
        if final_path.exists():
            skip += 1
        elif fig_path.exists():
            queue.append(item)
    return queue, skip


def run_stage(
    stage_name: str,
    queue: list[PaperTask],
    max_workers: int,
    log_path: Path,
    run_func,
) -> tuple[int, int]:
    if not queue:
        append_log(log_path, f"[INFO] stage={stage_name} queued=0")
        return 0, 0

    done = 0
    fail = 0
    append_log(log_path, f"[INFO] stage={stage_name} queued={len(queue)} workers={max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(run_func, item): item for item in queue}
        for future in as_completed(future_map):
            item = future_map[future]
            try:
                output_path = future.result()
                done += 1
                append_log(log_path, f"[DONE] stage={stage_name} {item.paper_id} -> {output_path}")
            except Exception as exc:  # noqa: BLE001
                fail += 1
                append_log(log_path, f"[FAIL] stage={stage_name} {item.paper_id} :: {exc}")
                append_log(log_path, traceback.format_exc())
    return done, fail


def main() -> int:
    run_namespace = os.getenv("RUN_NAMESPACE", "verified_multi")
    parallel_requested = int(os.getenv("PARALLEL", "200"))
    max_workers = parallel_requested

    scope_dir = PIPELINE_ROOT / "outputs" / f"{run_namespace}_scope"
    fig_dir = PIPELINE_ROOT / "outputs" / f"{run_namespace}_fig_classify"
    raw_dir = PIPELINE_ROOT / "outputs" / f"{run_namespace}_section_raw"
    final_dir = PIPELINE_ROOT / "outputs" / f"{run_namespace}_paper_final"
    for path in (scope_dir, fig_dir, raw_dir, final_dir, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["BM_MULTI_SCOPE_DIR"] = str(scope_dir)
    os.environ["BM_MULTI_FIG_DIR"] = str(fig_dir)
    os.environ["BM_MULTI_RAW_DIR"] = str(raw_dir)
    os.environ["BM_MULTI_FINAL_DIR"] = str(final_dir)

    legacy_dir = PIPELINE_ROOT / "legacy_single_clone"
    if str(legacy_dir) not in sys.path:
        sys.path.insert(0, str(legacy_dir))
    if str(PIPELINE_ROOT) not in sys.path:
        sys.path.insert(0, str(PIPELINE_ROOT))

    from run_ie0_multi import run_ie0_multi  # type: ignore
    from run_fig_classify_multi import run_fig_classify_multi  # type: ignore
    from run_multi_sections_paper import run_paper_sections  # type: ignore

    log_path = LOG_DIR / f"run_verified_multi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    append_log(log_path, f"[INFO] started_at={now()}")
    append_log(log_path, f"[INFO] status_file={STATUS_FILE}")
    append_log(log_path, f"[INFO] status_manifest={STATUS_MANIFEST_CSV}")
    append_log(log_path, f"[INFO] namespace={run_namespace}")
    append_log(log_path, f"[INFO] parallel_requested={parallel_requested}")
    append_log(log_path, f"[INFO] parallel_used={max_workers}")
    append_log(log_path, f"[INFO] scope_dir={scope_dir}")
    append_log(log_path, f"[INFO] fig_dir={fig_dir}")
    append_log(log_path, f"[INFO] raw_dir={raw_dir}")
    append_log(log_path, f"[INFO] final_dir={final_dir}")

    tasks, missing_count = load_verified_tasks(log_path)
    sync_papers(tasks)

    ie0_queue, ie0_skip = stage_missing_ie0(tasks, scope_dir)
    fig_queue, fig_skip = stage_missing_fig(tasks, scope_dir, fig_dir)
    sec_queue, sec_skip = stage_missing_sections(tasks, fig_dir, final_dir)

    ie0_done, ie0_fail = run_stage(
        "ie0",
        ie0_queue,
        max_workers,
        log_path,
        lambda item: run_ie0_multi(item.paper_id),
    )
    fig_queue, fig_skip = stage_missing_fig(tasks, scope_dir, fig_dir)
    fig_done, fig_fail = run_stage(
        "fig_classify",
        fig_queue,
        max_workers,
        log_path,
        lambda item: run_fig_classify_multi(item.paper_id),
    )
    sec_queue, sec_skip = stage_missing_sections(tasks, fig_dir, final_dir)
    sec_done, sec_fail = run_stage(
        "sections",
        sec_queue,
        max_workers,
        log_path,
        lambda item: run_paper_sections(item.paper_id),
    )

    completed = sum(1 for item in tasks if (final_dir / item.paper_id / "run_summary.json").exists())
    summary = {
        "started_at": now(),
        "run_namespace": run_namespace,
        "parallel_requested": parallel_requested,
        "parallel_used": max_workers,
        "expected": len(tasks),
        "completed": completed,
        "remaining": len(tasks) - completed,
        "missing": missing_count,
        "stage_counts": {
            "ie0": {"queued": len(ie0_queue), "done": ie0_done, "fail": ie0_fail, "skip": ie0_skip},
            "fig_classify": {"queued": len(fig_queue), "done": fig_done, "fail": fig_fail, "skip": fig_skip},
            "sections": {"queued": len(sec_queue), "done": sec_done, "fail": sec_fail, "skip": sec_skip},
        },
        "scope_dir": str(scope_dir),
        "fig_dir": str(fig_dir),
        "raw_dir": str(raw_dir),
        "final_dir": str(final_dir),
        "log_path": str(log_path),
    }
    append_log(log_path, "[INFO] summary=" + json.dumps(summary, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if completed == len(tasks) and missing_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
