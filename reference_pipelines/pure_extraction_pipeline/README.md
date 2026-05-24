# Unified Pure Extraction Pipeline

This directory is a clean copy of the extraction flows. It was copied from the working `sc-ie` tree without historical run outputs, logs, task data, or generated configs with embedded API keys.

## What Is Included

### Single-material flow

- `IE_0_classification/` experimental paper classification
- `IE_0_single/` single-system paper filtering
- `IE_1_part0(s5+resource)/` section 5 and resource extraction
- `IE_1_part2(fig_classify)/` figure/table classification
- `IE_1_part1(s0+s1+s2_1)/` section 0, section 1, and coarse section 2 extraction
- `IE_2_part1(s2_2)/` fine section 2 extraction
- `IE_2_part2(s3+s4)/` section 3 and section 4 extraction
- `IE_3_check/` JSON repair/check
- `IE_4_preprocess(others)/`, `IE_4_preprocess1(s2)/`, `IE_4_preprocess2(s2)/` schema cleanup and section 2 merge/mapping
- `IE_5_unicode/` Unicode and semantic normalization
- `IE_6_merge/` final merged JSON assembly
- `IE_schema/` shared schema files
- `run_pure_extraction.sh` clean end-to-end launcher

### Multi-material flow

- `BM_multi_match_pipeline/` multi-material extraction and matching
- `BM_multi_match_pipeline/run_manifest.py` paper-level target manifest
- `BM_multi_match_pipeline/run_fact_candidates.py` paper-level candidate fact extraction
- `BM_multi_match_pipeline/run_matcher.py` target-level attribution
- `BM_multi_match_pipeline/run_aggregate.py` final target JSON aggregation
- `BM_multi_match_pipeline/run_pipeline.py` end-to-end multi-material launcher
- `BM_multi_match_pipeline/run_from_legacy_gate.py` bridge from the single-system gate into multi-material extraction
- `BM_multi_match_pipeline/legacy_single_clone/` legacy-compatible multi-material batch path used by prior batch runs

### Unified launcher

- `run_unified_extraction.sh` runs the single flow first, then sends non-single or uncertain papers into the multi-material flow.

## What Was Deliberately Excluded

- task folders such as `317-*`, `semantic-8370`, `scihub-*`
- stage `logs/`, `outputs/`, `inputs/`, `papers/`, `__pycache__/`
- generated stage `config.yaml` files containing old absolute paths or API keys
- packaging, retry, PPT, report, historical benchmark outputs, and batch-control artifacts that are not part of the reusable extraction code

## Usage

Set keys through environment variables.

Run the unified flow when you do not want to care upfront whether a paper is single-material or multi-material:

```bash
export LLM_API_KEYS="key1,key2"
export LLM_BASE_URL="https://api.siliconflow.cn/v1/chat/completions"
export LLM_MODEL="Pro/deepseek-ai/DeepSeek-V3.2"
./run_unified_extraction.sh my_job /path/to/papers_md
```

Run only the single-material flow:

```bash
export LLM_API_KEYS="key1,key2"
export LLM_BASE_URL="https://api.siliconflow.cn/v1/chat/completions"
export LLM_MODEL="Pro/deepseek-ai/DeepSeek-V3.2"
./run_pure_extraction.sh my_job /path/to/papers_md
```

Single-material output goes under:

```text
pure_extraction_pipeline/runs/my_job/post_processed/final_merged_json/
```

Multi-material output goes under:

```text
pure_extraction_pipeline/BM_multi_match_pipeline/outputs/final_targets/
```

Each multi-material target is emitted with the same top-level shape as the single-material final JSON: `primary_signature`, `material_info`, `section5`, and `paper_info`. Multi-material-only bookkeeping such as `target_id`, sibling exclusions, attribution provenance, and quality-control flags is stored under `paper_info.metadata`.

The launchers copy input markdown into each stage's working area as needed. They do not move or delete source data outside this directory.
