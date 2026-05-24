#!/usr/bin/env bash
set -euo pipefail

# Unified entrypoint:
# 1. Run the clean single-material pipeline.
# 2. Import papers rejected or undecided by the single-system gate into the
#    multi-material pipeline.
# 3. Run the multi-material pipeline for those imported papers.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
JOB_NAME="${1:?Usage: ./run_unified_extraction.sh <job_name> [papers_md_dir]}"
PAPERS_DIR="${2:-${PAPERS_DIR:-$ROOT_DIR/runs/$JOB_NAME/papers_md}}"
TASK_DIR="${TASK_DIR:-$ROOT_DIR/runs/$JOB_NAME}"

"$ROOT_DIR/run_pure_extraction.sh" "$JOB_NAME" "$PAPERS_DIR"

cd "$ROOT_DIR/BM_multi_match_pipeline"

"$PYTHON_BIN" run_from_legacy_gate.py \
  --source_papers_dir "$PAPERS_DIR" \
  --experimental_file "$TASK_DIR/classification_out/experimental.txt" \
  --single_outputs_dir "$TASK_DIR/single_out" \
  --include_uncertain \
  --run_pipeline

echo
echo "Unified extraction done."
echo "Single final JSON dir: $TASK_DIR/post_processed/final_merged_json"
echo "Multi final JSON dir: $ROOT_DIR/BM_multi_match_pipeline/outputs/final_targets"
