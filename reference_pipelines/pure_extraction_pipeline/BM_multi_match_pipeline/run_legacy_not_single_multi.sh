#!/usr/bin/env bash
set -uo pipefail

ROOT="/c/Users/Administrator/Desktop/fsdownload/sc-ie"
PIPELINE_ROOT="$ROOT/BM_multi_match_pipeline"
PAPERS_DIR="$PIPELINE_ROOT/papers"
LOG_DIR="$PIPELINE_ROOT/logs"
KEY_FILE="$ROOT/.siliconflow_keys.local.sh"

PARALLEL="${PARALLEL:-200}"
TIMEOUT_SEC="${BM_MULTI_TIMEOUT_SEC:-900}"
OUTPUT_DIR="$PIPELINE_ROOT/outputs/multi_paper_final"

mkdir -p "$LOG_DIR" "$PAPERS_DIR"

if [[ ! -f "$KEY_FILE" ]]; then
  echo "[ERROR] missing key file: $KEY_FILE" >&2
  exit 1
fi

API_KEY="$(grep -o 'sk-[A-Za-z0-9]*' "$KEY_FILE" | head -n 1)"
if [[ -z "${API_KEY:-}" ]]; then
  echo "[ERROR] failed to read SILICONFLOW_API_KEY from $KEY_FILE" >&2
  exit 1
fi

export SILICONFLOW_API_KEY="$API_KEY"
export SILICONFLOW_MODEL="${SILICONFLOW_MODEL:-Pro/deepseek-ai/DeepSeek-V3.2}"
export SILICONFLOW_BASE_URL="${SILICONFLOW_BASE_URL:-https://api.siliconflow.cn/v1/chat/completions}"
export BM_MULTI_TIMEOUT_SEC="$TIMEOUT_SEC"

RUN_LOG="$LOG_DIR/run_legacy_not_single_multi_$(date +%Y%m%d_%H%M%S).log"
QUEUE_FILE="$(mktemp)"

echo "[INFO] log file: $RUN_LOG"
echo "[INFO] parallel: $PARALLEL"
echo "[INFO] timeout: $BM_MULTI_TIMEOUT_SEC"

export ROOT PIPELINE_ROOT PAPERS_DIR OUTPUT_DIR RUN_LOG

python - <<'PY' > "$QUEUE_FILE"
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
papers_dir = Path(os.environ["PAPERS_DIR"])
output_dir = Path(os.environ["OUTPUT_DIR"])
run_log = Path(os.environ["RUN_LOG"])

papers_dir.mkdir(parents=True, exist_ok=True)
run_log.parent.mkdir(parents=True, exist_ok=True)

task_dirs = []
for path in root.iterdir():
    if not path.is_dir():
        continue
    if not path.name.startswith("317-"):
        continue
    if (path / "classification_out" / "experimental.txt").exists() and (path / "single_out").exists() and (path / "papers_md").exists():
        task_dirs.append(path)

task_dirs.sort(key=lambda p: p.name)

selected = {}
for task_dir in task_dirs:
    experimental_file = task_dir / "classification_out" / "experimental.txt"
    single_out_dir = task_dir / "single_out"
    source_papers_dir = task_dir / "papers_md"

    experimental_ids = [line.strip() for line in experimental_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    for paper_id in experimental_ids:
        single_json = single_out_dir / f"{paper_id}.json"
        if not single_json.exists():
            continue
        try:
            data = json.loads(single_json.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("single_system") != 0:
            continue
        md_path = source_papers_dir / f"{paper_id}.md"
        if not md_path.exists():
            continue
        selected[paper_id] = {
            "task": task_dir.name,
            "source_md": str(md_path),
            "source_si": str(source_papers_dir / f"{paper_id}_si.md"),
        }

manifest_out = root / "BM_multi_match_pipeline" / "legacy_not_single_manifest.csv"
with manifest_out.open("w", encoding="utf-8", newline="") as f:
    f.write("paper_id,task,source_md,source_si\n")
    for paper_id in sorted(selected):
        row = selected[paper_id]
        f.write(f"{paper_id},{row['task']},{row['source_md']},{row['source_si']}\n")

for paper_id in sorted(selected):
    row = selected[paper_id]
    src_md = Path(row["source_md"])
    target_md = papers_dir / f"{paper_id}.md"
    target_md.write_text(src_md.read_text(encoding="utf-8"), encoding="utf-8")

    src_si = Path(row["source_si"])
    if src_si.exists():
        target_si = papers_dir / f"{paper_id}_si.md"
        target_si.write_text(src_si.read_text(encoding="utf-8"), encoding="utf-8")

    run_summary = output_dir / paper_id / "run_summary.json"
    if run_summary.exists():
        with run_log.open("a", encoding="utf-8") as logf:
            logf.write(f"[SKIP] {paper_id}\n")
        continue
    print(paper_id)
PY

if [[ ! -s "$QUEUE_FILE" ]]; then
  echo "[INFO] nothing to run; all legacy not-single papers already have run_summary.json" | tee -a "$RUN_LOG"
  rm -f "$QUEUE_FILE"
  exit 0
fi

run_one() {
  local paper_id="$1"
  local ie0="$PIPELINE_ROOT/legacy_single_clone/run_ie0_multi.py"
  local fig="$PIPELINE_ROOT/legacy_single_clone/run_fig_classify_multi.py"
  local sec="$PIPELINE_ROOT/legacy_single_clone/run_multi_sections_paper.py"

  {
    printf '[START] %s %s\n' "$(date '+%F %T')" "$paper_id"
    python "$ie0" --paper_id "$paper_id"
    python "$fig" --paper_id "$paper_id"
    python "$sec" --paper_id "$paper_id"
    printf '[DONE] %s %s\n' "$(date '+%F %T')" "$paper_id"
  } >> "$RUN_LOG" 2>&1
}

export -f run_one

xargs -a "$QUEUE_FILE" -I{} -P "$PARALLEL" bash -lc 'run_one "$@"' _ {}
status=$?

rm -f "$QUEUE_FILE"

if [[ $status -ne 0 ]]; then
  echo "[WARN] batch finished with some failures; check $RUN_LOG" >&2
  exit $status
fi

echo "[INFO] batch finished successfully; log: $RUN_LOG"
