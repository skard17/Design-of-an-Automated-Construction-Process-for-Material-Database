#!/usr/bin/env bash
set -uo pipefail

ROOT="/c/Users/Administrator/Desktop/fsdownload/sc-ie"
PIPELINE_ROOT="$ROOT/BM_multi_match_pipeline"
SOURCE_DIR="$ROOT/multi"
PAPERS_DIR="$PIPELINE_ROOT/papers"
LOG_DIR="$PIPELINE_ROOT/logs"
KEY_FILE="$ROOT/.siliconflow_keys.local.sh"
PARALLEL="${PARALLEL:-20}"
TIMEOUT_SEC="${BM_MULTI_TIMEOUT_SEC:-900}"

mkdir -p "$LOG_DIR" "$PAPERS_DIR"

if [[ ! -f "$KEY_FILE" ]]; then
  echo "missing key file: $KEY_FILE" >&2
  exit 1
fi

API_KEY="$(grep -o 'sk-[A-Za-z0-9]*' "$KEY_FILE" | head -n 1)"
if [[ -z "${API_KEY:-}" ]]; then
  echo "failed to read SILICONFLOW_API_KEY from $KEY_FILE" >&2
  exit 1
fi

export SILICONFLOW_API_KEY="$API_KEY"
export SILICONFLOW_MODEL="${SILICONFLOW_MODEL:-Pro/deepseek-ai/DeepSeek-V3.2}"
export SILICONFLOW_BASE_URL="${SILICONFLOW_BASE_URL:-https://api.siliconflow.cn/v1/chat/completions}"
export BM_MULTI_TIMEOUT_SEC="$TIMEOUT_SEC"

shopt -s nullglob
md_files=("$SOURCE_DIR"/*.md)
shopt -u nullglob

if [[ ${#md_files[@]} -eq 0 ]]; then
  echo "no markdown files found in $SOURCE_DIR" >&2
  exit 1
fi

cp -f "$SOURCE_DIR"/*.md "$PAPERS_DIR"/

RUN_LOG="$LOG_DIR/run_multi_remaining_$(date +%Y%m%d_%H%M%S).log"
QUEUE_FILE="$(mktemp)"

echo "[INFO] log file: $RUN_LOG"
echo "[INFO] parallel: $PARALLEL"
echo "[INFO] timeout: $BM_MULTI_TIMEOUT_SEC"

for file in "${md_files[@]}"; do
  paper_id="$(basename "$file" .md)"
  summary="$PIPELINE_ROOT/outputs/multi_paper_final/$paper_id/run_summary.json"
  if [[ -f "$summary" ]]; then
    printf '[SKIP] %s\n' "$paper_id" | tee -a "$RUN_LOG"
    continue
  fi
  printf '%s\n' "$paper_id" >> "$QUEUE_FILE"
done

if [[ ! -s "$QUEUE_FILE" ]]; then
  echo "[INFO] nothing to run; all papers already have run_summary.json" | tee -a "$RUN_LOG"
  rm -f "$QUEUE_FILE"
  exit 0
fi

export ROOT PIPELINE_ROOT PAPERS_DIR RUN_LOG

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
