#!/usr/bin/env bash
set -euo pipefail

# Clean single-material extraction pipeline.
# This copy contains code and prompts only. It reads API keys from the environment.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
JOB_NAME="${1:?Usage: ./run_pure_extraction.sh <job_name> [papers_md_dir]}"
TASK_DIR="${TASK_DIR:-$ROOT_DIR/runs/$JOB_NAME}"
PAPERS_DIR="${2:-${PAPERS_DIR:-$TASK_DIR/papers_md}}"

BASE_URL="${LLM_BASE_URL:-${SILICONFLOW_BASE_URL:-https://api.siliconflow.cn/v1/chat/completions}}"
MODEL="${LLM_MODEL:-${SILICONFLOW_MODEL:-Pro/deepseek-ai/DeepSeek-V3.2}}"
TIMEOUT_SEC="${LLM_TIMEOUT_SEC:-300}"
MAX_WORKERS="${MAX_WORKERS:-8}"
UNICODE_MAX_WORKERS="${UNICODE_MAX_WORKERS:-16}"

if [[ -z "${LLM_API_KEYS:-${SILICONFLOW_API_KEY:-}}" ]]; then
  echo "ERROR: set LLM_API_KEYS='key1,key2' or SILICONFLOW_API_KEY before running." >&2
  exit 2
fi

to_py_path() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -m "$1"
  else
    echo "$1"
  fi
}

api_keys_yaml() {
  local raw="${LLM_API_KEYS:-${SILICONFLOW_API_KEY:-}}"
  raw="${raw//;/,}"
  IFS=',' read -ra keys <<< "$raw"
  echo "  api_keys:"
  for key in "${keys[@]}"; do
    key="$(echo "$key" | xargs)"
    [[ -n "$key" ]] && echo "    - \"$key\""
  done
}

log_step() {
  echo
  echo "=================================================="
  echo ">>> $1"
  echo "=================================================="
}

copy_jsons() {
  local src="$1"
  local dst="$2"
  mkdir -p "$dst"
  if compgen -G "$src/*.json" >/dev/null; then
    cp "$src"/*.json "$dst"/
  fi
}

reset_dir() {
  mkdir -p "$1"
  find "$1" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
}

mkdir -p "$TASK_DIR"/{classification_out,single_out,logs,results_part,post_process_history,post_processed/final_merged_json}
for part in s0 s1 s2_1 s2_2 s3 s4 s5 resources fig_classify_out metadata; do
  mkdir -p "$TASK_DIR/results_part/$part"
done

if [[ ! -d "$PAPERS_DIR" ]]; then
  echo "ERROR: papers dir not found: $PAPERS_DIR" >&2
  exit 2
fi

log_step "0. experimental-paper classification"
cd "$ROOT_DIR/IE_0_classification"
cat > config.yaml <<EOF
api:
  base_url: "$BASE_URL"
$(api_keys_yaml)
  timeout_seconds: $TIMEOUT_SEC
  model: "$MODEL"
  temperature: 0.1
run:
  prompt_dir: "prompts"
  input_dir: "$(to_py_path "$PAPERS_DIR")"
  output_dir: "$(to_py_path "$TASK_DIR/classification_out")"
  experimental_file: "experimental.txt"
EOF
"$PYTHON_BIN" classify.py

EXP_LINK_DIR="$TASK_DIR/md_links_experimental"
reset_dir "$EXP_LINK_DIR"
if [[ -f "$TASK_DIR/classification_out/experimental.txt" ]]; then
  while IFS= read -r pid; do
    pid="$(echo "$pid" | xargs)"
    [[ -n "$pid" && -f "$PAPERS_DIR/$pid.md" ]] && cp "$PAPERS_DIR/$pid.md" "$EXP_LINK_DIR/$pid.md"
  done < "$TASK_DIR/classification_out/experimental.txt"
fi

log_step "1. single-system filtering"
cd "$ROOT_DIR/IE_0_single"
cat > config.yaml <<EOF
provider:
  base_url: "$BASE_URL"
  model: "$MODEL"
  timeout_sec: $TIMEOUT_SEC
$(api_keys_yaml)
paths:
  prompts_dir: "prompts"
  papers_dir: "$(to_py_path "$EXP_LINK_DIR")"
  outputs_dir: "$(to_py_path "$TASK_DIR/single_out")"
stages:
  single:
    temperature: 0.0
EOF
"$PYTHON_BIN" run_batch.py

FINAL_LINK_DIR="$TASK_DIR/md_links_final"
reset_dir "$FINAL_LINK_DIR"
if [[ -f "$TASK_DIR/single_out/single.txt" ]]; then
  while IFS= read -r pid; do
    pid="$(echo "$pid" | xargs)"
    [[ -n "$pid" && -f "$PAPERS_DIR/$pid.md" ]] && cp "$PAPERS_DIR/$pid.md" "$FINAL_LINK_DIR/$pid.md"
  done < "$TASK_DIR/single_out/single.txt"
fi

log_step "2. s5/resources extraction"
cd "$ROOT_DIR/IE_1_part0(s5+resource)"
cat > config.yaml <<EOF
provider:
  base_url: "$BASE_URL"
  model: "$MODEL"
  timeout_sec: $TIMEOUT_SEC
$(api_keys_yaml)
paths:
  prompts_dir: "prompts"
  papers_dir: "$(to_py_path "$FINAL_LINK_DIR")"
  outputs_dir: "$(to_py_path "$TASK_DIR/results_part")"
stages:
  s5: {temperature: 0.1}
  resources: {temperature: 0.1}
EOF
"$PYTHON_BIN" run_batch.py

log_step "3. figure classification"
cd "$ROOT_DIR/IE_1_part2(fig_classify)"
cat > config.yaml <<EOF
provider:
  base_url: "$BASE_URL"
  model: "$MODEL"
  timeout_sec: $TIMEOUT_SEC
$(api_keys_yaml)
paths:
  prompts_dir: "prompts"
  papers_dir: "$(to_py_path "$FINAL_LINK_DIR")"
  outputs_dir: "$(to_py_path "$TASK_DIR/results_part/fig_classify_out")"
stages:
  fig_classify: {temperature: 0.1}
EOF
"$PYTHON_BIN" run_batch.py

log_step "4. s0/s1/s2_1 extraction"
cd "$ROOT_DIR/IE_1_part1(s0+s1+s2_1)"
cat > config.yaml <<EOF
provider:
  base_url: "$BASE_URL"
  model: "$MODEL"
  timeout_sec: $TIMEOUT_SEC
$(api_keys_yaml)
paths:
  prompts_dir: "prompts"
  papers_dir: "$(to_py_path "$FINAL_LINK_DIR")"
  outputs_dir: "$(to_py_path "$TASK_DIR/results_part")"
stages:
  s0: {temperature: 0.1}
  s1: {temperature: 0.1}
  s2_1: {temperature: 0.1}
EOF
"$PYTHON_BIN" run_batch.py

log_step "5. s2_2 extraction"
cd "$ROOT_DIR/IE_2_part1(s2_2)"
cat > config.yaml <<EOF
provider:
  base_url: "$BASE_URL"
  model: "$MODEL"
  timeout_sec: $TIMEOUT_SEC
$(api_keys_yaml)
paths:
  prompts_dir: "prompts"
  papers_dir: "$(to_py_path "$FINAL_LINK_DIR")"
  fig_classify_dir: "$(to_py_path "$TASK_DIR/results_part/fig_classify_out")"
  outputs_dir: "$(to_py_path "$TASK_DIR/results_part/s2_2")"
stages:
  s2_2: {temperature: 0.1}
EOF
"$PYTHON_BIN" run_batch.py

log_step "6. s3/s4 extraction"
cd "$ROOT_DIR/IE_2_part2(s3+s4)"
cat > config.yaml <<EOF
provider:
  base_url: "$BASE_URL"
  model: "$MODEL"
  timeout_sec: $TIMEOUT_SEC
$(api_keys_yaml)
paths:
  prompts_dir: "prompts"
  papers_dir: "$(to_py_path "$FINAL_LINK_DIR")"
  outputs_dir: "$(to_py_path "$TASK_DIR/results_part")"
stages:
  s3: {temperature: 0.1}
  s4: {temperature: 0.1}
EOF
"$PYTHON_BIN" run_batch.py

log_step "7. JSON check"
cd "$ROOT_DIR/IE_3_check"
reset_dir inputs
reset_dir outputs
for part in single s0 s1 s2_1 s2_2 s3 s4 s5 metadata resources; do
  mkdir -p "inputs/$part"
done
copy_jsons "$TASK_DIR/single_out" "inputs/single"
copy_jsons "$TASK_DIR/results_part/s0" "inputs/s0"
copy_jsons "$TASK_DIR/results_part/s1" "inputs/s1"
copy_jsons "$TASK_DIR/results_part/s2_1" "inputs/s2_1"
copy_jsons "$TASK_DIR/results_part/s2_2" "inputs/s2_2"
copy_jsons "$TASK_DIR/results_part/s3" "inputs/s3"
copy_jsons "$TASK_DIR/results_part/s4" "inputs/s4"
copy_jsons "$TASK_DIR/results_part/s5" "inputs/s5"
copy_jsons "$TASK_DIR/results_part/metadata" "inputs/metadata"
copy_jsons "$TASK_DIR/results_part/resources" "inputs/resources"
"$PYTHON_BIN" check.py
mkdir -p "$TASK_DIR/post_process_history/1_check"
cp -r outputs/* "$TASK_DIR/post_process_history/1_check/" 2>/dev/null || true

log_step "8. schema cleanup"
cd "$ROOT_DIR/IE_4_preprocess(others)"
reset_dir inputs
reset_dir outputs
for part in single s0 s1 s3 s4 s5 metadata resources; do
  copy_jsons "$TASK_DIR/post_process_history/1_check/$part" "inputs/$part"
done
"$PYTHON_BIN" preprocess.py
mkdir -p "$TASK_DIR/post_process_history/2_pre_others"
cp -r outputs/* "$TASK_DIR/post_process_history/2_pre_others/" 2>/dev/null || true

cd "$ROOT_DIR/IE_4_preprocess1(s2)"
reset_dir inputs
reset_dir outputs
copy_jsons "$TASK_DIR/post_process_history/1_check/s2_1" "inputs/s2_1"
copy_jsons "$TASK_DIR/post_process_history/1_check/s2_2" "inputs/s2_2"
"$PYTHON_BIN" merge_s2.py

cd "$ROOT_DIR/IE_4_preprocess2(s2)"
reset_dir inputs
reset_dir outputs
copy_jsons "$ROOT_DIR/IE_4_preprocess1(s2)/outputs" "inputs"
"$PYTHON_BIN" method_mapping.py
mkdir -p "$TASK_DIR/post_process_history/4_pre_s2_map/s2"
cp outputs/*.json "$TASK_DIR/post_process_history/4_pre_s2_map/s2/" 2>/dev/null || true

log_step "9. unicode/semantic normalization"
cd "$ROOT_DIR/IE_5_unicode"
reset_dir inputs
reset_dir outputs
for part in single s0 s1 s3 s4 s5 metadata resources; do
  copy_jsons "$TASK_DIR/post_process_history/2_pre_others/$part" "inputs/$part"
done
copy_jsons "$TASK_DIR/post_process_history/4_pre_s2_map/s2" "inputs/s2"
cat > config.yaml <<EOF
llm:
  model: "$MODEL"
  base_url: "$BASE_URL"
$(api_keys_yaml)
  prompt_file: prompts/mode3_transform.md
  system_prompt: |
    You are a strict text normalizer.
    Return only transformed text content, with no explanation.
  temperature: 0.0
  timeout_sec: $TIMEOUT_SEC
  max_workers: $UNICODE_MAX_WORKERS
  max_llm_inflight: $MAX_WORKERS
paths:
  prompts_dir: prompts
  input_root: inputs
  output_root: outputs
  schema_root: schema_unicode_transfer
  log_root: logs
  list_path: preprocess_list.txt
EOF
"$PYTHON_BIN" preprocess.py
mkdir -p "$TASK_DIR/post_process_history/5_unicode"
cp -r outputs/* "$TASK_DIR/post_process_history/5_unicode/" 2>/dev/null || true

log_step "10. final merge"
cd "$ROOT_DIR/IE_6_merge"
reset_dir inputs
reset_dir outputs
for part in single s0 s1 s2 s3 s4 s5 metadata resources; do
  copy_jsons "$TASK_DIR/post_process_history/5_unicode/$part" "inputs/$part"
done
"$PYTHON_BIN" merge.py
cp outputs/*.json "$TASK_DIR/post_processed/final_merged_json/" 2>/dev/null || true

echo
echo "Done. Final JSON dir: $TASK_DIR/post_processed/final_merged_json"
