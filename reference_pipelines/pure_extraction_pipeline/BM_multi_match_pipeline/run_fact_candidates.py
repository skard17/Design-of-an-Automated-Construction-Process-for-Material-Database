from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from utils.ds_client import chat
from utils.io_utils import load_config, load_json, load_supplementary_information, read_text, select_paper_file, write_named_output
from utils.key_pool import KeyPool
from utils.log_utils import setup_logger
from utils.prompt_utils import clean_llm_output, inject_blocks, load_prompt_template


PROMPT_FILE = "fact_candidates.md"
MAX_CHARS_PER_CHUNK = 12000


def has_numeric_signal(text: str) -> bool:
    return bool(re.search(r"\d", text))


def parse_llm_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Repair lone backslashes such as "\m" or "\T" that occasionally leak from
        # model-generated pseudo-LaTeX inside otherwise valid JSON strings.
        repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)
        return json.loads(repaired)


def is_bad_candidate(candidate: dict) -> tuple[bool, str | None]:
    fact_type = str(candidate.get("fact_type") or "")
    evidence = str(candidate.get("verbatim_evidence") or "").lower()
    value = str(candidate.get("value") or "").lower()
    characteristics = str((candidate.get("conditions") or {}).get("characteristics") or "").lower()
    combined = " ".join([evidence, value, characteristics])

    if fact_type == "stack_descriptor":
        bad_patterns = [
            "atomic coordinates",
            "table i presents",
            "table ii presents",
            "x-ray analysis",
            "crystal structure",
            "bcc structure",
            "all alloys crystallized",
            "alpha-w",
            "quasi-one-dimensional",
            "quasi-1d",
            "electronic structure",
            "reduced dimensional properties",
        ]
        if any(pattern in combined for pattern in bad_patterns):
            return True, "Dropped non-stack crystallographic or qualitative dimensionality statement."

    if fact_type in {"tuning", "electronic_state_tuning_mechanism"}:
        bad_patterns = [
            "midpoint between zero and full resistance",
            "measurement definition",
            "strong normal state electronic anisotropy",
            "normal state electronic anisotropy",
            "field angle",
            "field orientation",
            "ionic size",
            "chain spacing",
            "spacing between quasi-one-dimensional chains",
            "spacing between",
            "normalized gap",
            "uemura",
            "outside unconventional region",
            "unconventional range",
            "t_c/t_f",
        ]
        if any(pattern in combined for pattern in bad_patterns):
            return True, "Dropped non-tuning statement misclassified as electronic_state_tuning_mechanism."

    if fact_type == "carrier_concentration":
        if not has_numeric_signal(value) and not has_numeric_signal(evidence):
            return True, "Dropped non-numeric carrier_concentration statement."

    return False, None


def dedupe_candidates(candidates: list[dict]) -> list[dict]:
    seen: set[tuple[str, str, str, str, str]] = set()
    deduped: list[dict] = []
    for candidate in candidates:
        target_key = ",".join(candidate.get("attribution_hint", {}).get("candidate_targets", []))
        anchor = candidate.get("source_anchor", {}) or {}
        anchor_key = "|".join(
            [
                str(anchor.get("section_label") or ""),
                str(anchor.get("figure") or ""),
                str(anchor.get("table") or ""),
            ]
        )
        key = (
            str(candidate.get("fact_type") or ""),
            json.dumps(candidate.get("value"), ensure_ascii=False, sort_keys=True),
            str(candidate.get("unit") or ""),
            target_key,
            anchor_key,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def cleanup_chunk_data(data: dict) -> dict:
    filtered_candidates = []
    notes = list(data.get("paper_level_unassigned_notes", []))
    for candidate in data.get("paper_level_candidates", []):
        bad, note = is_bad_candidate(candidate)
        if bad:
            if note:
                notes.append(note)
            continue
        filtered_candidates.append(candidate)
    data["paper_level_candidates"] = filtered_candidates
    data["paper_level_unassigned_notes"] = notes
    return data


def persist_merged_output(outputs_dir: str, paper_id: str, merged: dict) -> None:
    merged["paper_level_candidates"] = dedupe_candidates(merged.get("paper_level_candidates", []))
    write_named_output(
        outputs_dir,
        "fact_candidates",
        paper_id,
        json.dumps(merged, ensure_ascii=False, indent=2),
    )


def split_paper_text(paper_text: str) -> list[str]:
    if len(paper_text) <= MAX_CHARS_PER_CHUNK:
        return [paper_text]

    blocks = paper_text.split("\n## Page ")
    chunks: list[str] = []
    current = ""

    for index, block in enumerate(blocks):
        piece = block if index == 0 else f"## Page {block}"
        if not current:
            current = piece
            continue
        if len(current) + len(piece) + 2 <= MAX_CHARS_PER_CHUNK:
            current = current + "\n\n" + piece
        else:
            chunks.append(current)
            current = piece

    if current:
        chunks.append(current)
    return chunks


def renumber_candidates(data: dict, start_index: int) -> tuple[dict, int]:
    candidates = data.get("paper_level_candidates", [])
    next_index = start_index
    for item in candidates:
        item["candidate_id"] = f"cand_{next_index:03d}"
        next_index += 1
    return data, next_index


def run_fact_candidates(paper_id: str, max_new_chunks: int | None = None) -> None:
    logger = setup_logger("fact_candidates")
    root = Path(__file__).resolve().parent
    config = load_config(str(root / "config.yaml"))
    provider = config["provider"]
    paths = config["paths"]
    stage_cfg = config["stages"]["fact_candidates"]

    resolved_paths = {
        **paths,
        "papers_dir": str((root / paths["papers_dir"]).resolve()),
        "prompts_dir": str((root / paths["prompts_dir"]).resolve()),
        "outputs_dir": str((root / paths["outputs_dir"]).resolve()),
    }

    _, paper_path = select_paper_file(resolved_paths, paper_id)
    paper_text = read_text(paper_path)
    supplementary_information = load_supplementary_information(paper_path, paper_id)
    manifest = load_json(str(Path(resolved_paths["outputs_dir"]) / "manifests" / f"{paper_id}.json"))

    api_key = KeyPool.get_next_key(config_keys=provider.get("api_keys"))
    template = load_prompt_template(resolved_paths["prompts_dir"], PROMPT_FILE)
    chunks = split_paper_text(paper_text)
    merged = {
        "paper_id": paper_id,
        "candidate_version": "v1",
        "paper_level_candidates": [],
        "paper_level_unassigned_notes": [],
    }
    next_index = 1
    new_chunk_count = 0
    chunks_dir = Path(resolved_paths["outputs_dir"]) / "fact_candidates_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    for chunk_no, chunk_text in enumerate(chunks, start=1):
        logger.info("chunk %s/%s chars=%s", chunk_no, len(chunks), len(chunk_text))
        chunk_path = chunks_dir / f"{paper_id}__chunk{chunk_no:02d}.json"
        if chunk_path.exists():
            logger.info("reuse existing chunk file %s", chunk_path.name)
            chunk_data = load_json(str(chunk_path))
            chunk_data = cleanup_chunk_data(chunk_data)
            chunk_path.parent.mkdir(parents=True, exist_ok=True)
            chunk_path.write_text(json.dumps(chunk_data, ensure_ascii=False, indent=2), encoding="utf-8")
            if chunk_data.get("paper_level_candidates"):
                max_id = max(
                    int(str(item.get("candidate_id", "cand_000")).split("_")[-1])
                    for item in chunk_data["paper_level_candidates"]
                    if str(item.get("candidate_id", "")).startswith("cand_")
                )
                next_index = max(next_index, max_id + 1)
            merged["paper_level_candidates"].extend(chunk_data.get("paper_level_candidates", []))
            merged["paper_level_unassigned_notes"].extend(chunk_data.get("paper_level_unassigned_notes", []))
            persist_merged_output(resolved_paths["outputs_dir"], paper_id, merged)
            continue

        if max_new_chunks is not None and new_chunk_count >= max_new_chunks:
            logger.info("skip new chunk %s due to max_new_chunks=%s", chunk_no, max_new_chunks)
            continue

        prompt = inject_blocks(
            template,
            {
                "manifest_json": json.dumps(manifest, ensure_ascii=False, indent=2),
                "paper_text": chunk_text,
                "supplementary_information": supplementary_information if chunk_no == 1 else "",
            },
        )
        response = chat(
            prompt=prompt,
            model=provider["model"],
            base_url=provider["base_url"],
            api_key=api_key,
            temperature=float(stage_cfg.get("temperature", 0.0)),
            timeout_sec=int(provider.get("timeout_sec", 60)),
            system_prompt="You are an expert in materials science information extraction and database construction.",
        )
        out_text = clean_llm_output(response)
        chunk_data = parse_llm_json(out_text)
        chunk_data = cleanup_chunk_data(chunk_data)
        chunk_data, next_index = renumber_candidates(chunk_data, next_index)
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_path.write_text(json.dumps(chunk_data, ensure_ascii=False, indent=2), encoding="utf-8")
        new_chunk_count += 1
        merged["paper_level_candidates"].extend(chunk_data.get("paper_level_candidates", []))
        merged["paper_level_unassigned_notes"].extend(chunk_data.get("paper_level_unassigned_notes", []))
        persist_merged_output(resolved_paths["outputs_dir"], paper_id, merged)

    persist_merged_output(resolved_paths["outputs_dir"], paper_id, merged)
    out_path = Path(resolved_paths["outputs_dir"]) / "fact_candidates" / f"{paper_id}.json"
    logger.info("done output=%s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_id", required=True)
    parser.add_argument("--max_new_chunks", type=int, default=None)
    args = parser.parse_args()
    run_fact_candidates(args.paper_id, max_new_chunks=args.max_new_chunks)


if __name__ == "__main__":
    main()
