#!/usr/bin/env bash
set -euo pipefail

ROOT="/c/Users/Administrator/Desktop/fsdownload/sc-ie"
PIPELINE_ROOT="$ROOT/BM_multi_match_pipeline"
KEY_FILE="$ROOT/.siliconflow_keys.local.sh"
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_NAMESPACE="${RUN_NAMESPACE:-verified_multi}"
PARALLEL="${PARALLEL:-200}"
TIMEOUT_SEC="${BM_MULTI_TIMEOUT_SEC:-900}"

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
export RUN_NAMESPACE
export PARALLEL

echo "[INFO] namespace: $RUN_NAMESPACE"
echo "[INFO] parallel requested: $PARALLEL"
echo "[INFO] timeout: $BM_MULTI_TIMEOUT_SEC"
echo "[INFO] status file: $ROOT/verified_status_lists/multi.txt"
echo "[INFO] status manifest: $ROOT/verified_status_lists/status_manifest.csv"

"$PYTHON_BIN" "$PIPELINE_ROOT/run_verified_multi_batch.py"
