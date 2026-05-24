#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import csv
from pathlib import Path
from urllib.parse import quote

# ==============================================================================
# Configuration
# ==============================================================================
DEFAULT_JOB_NAME = "316-8k"
META_SOURCE_DIR = Path(r"C:\Users\Administrator\Desktop\fsdownload\sc\scihub_output_metax\scihub_output_metadata")
ROOT_DIR = Path(r"C:\Users\Administrator\Desktop\fsdownload\sc-ie")
ALL_DOIS_CSV = Path(r"C:\Users\Administrator\Desktop\fsdownload\sc\all_dois.csv")

EXPECTED_METADATA_KEYS = (
    "title",
    "year",
    "paper_type",
    "doi",
    "arxiv_id",
    "url",
    "venue",
)

DOI_PATTERN = re.compile(r"(10\.\d{4,9}/[^\s<>\")]+)", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")


def normalize_metadata_payload(content: dict) -> dict:
    """Normalize source metadata into the flat shape expected by preprocess."""
    if not isinstance(content, dict):
        return {key: None for key in EXPECTED_METADATA_KEYS}

    payload = content.get("metadata", content)
    if not isinstance(payload, dict):
        payload = {}

    basic_info = payload.get("Basic Information of the Paper")
    if isinstance(basic_info, dict):
        payload = basic_info

    return {
        "title": payload.get("title"),
        "year": payload.get("year"),
        "paper_type": payload.get("paper_type"),
        "doi": payload.get("doi"),
        "arxiv_id": payload.get("arxiv_id"),
        "url": payload.get("url", payload.get("paper_url")),
        "venue": payload.get("venue"),
    }


def doi_to_meta_filename(doi: str) -> str:
    normalized = doi.strip().rstrip(".,;")
    lowered = normalized.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        if lowered.startswith(prefix):
            normalized = normalized[len(prefix):]
            lowered = normalized.lower()
            break
    return quote(normalized, safe="").replace("%", "~") + ".json"


def metadata_filename_to_doi_suffix(meta_path: Path) -> str:
    stem = meta_path.stem
    variants = [stem]
    if "~2F" in stem:
        variants.append(stem.replace("~2F", "/"))
    if "_" in stem:
        variants.append(stem.replace("_", "/"))
    for decoded in variants:
        if "/" in decoded:
            return decoded.split("/", 1)[1]
    return stem


def normalize_title(text: str | None) -> str:
    if not text:
        return ""
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = text.lower()
    text = NON_ALNUM_PATTERN.sub(" ", text)
    return " ".join(text.split())


def extract_doi_from_md(md_path: Path) -> str | None:
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    match = DOI_PATTERN.search(text)
    if not match:
        return None
    return match.group(1).rstrip(".,;")


def extract_title_from_md(md_path: Path) -> str | None:
    try:
        lines = md_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None

    for line in lines[:80]:
        line = line.strip()
        if not line:
            continue
        if line.startswith("# Paper: "):
            title = line[len("# Paper: "):].strip()
            if title.lower().endswith(".pdf"):
                title = title[:-4].strip()
            if title:
                return title
        if line.startswith("# "):
            title = line[2:].strip()
            if title and not title.lower().startswith("paper:"):
                return title
    return None


def build_suffix_index() -> dict[str, Path]:
    suffix_index: dict[str, Path] = {}
    for meta_path in META_SOURCE_DIR.glob("*.json"):
        suffix = metadata_filename_to_doi_suffix(meta_path).strip().lower()
        if suffix and suffix not in suffix_index:
            suffix_index[suffix] = meta_path
    return suffix_index


def doi_to_md_variants(doi: str) -> tuple[str, ...]:
    normalized = doi.strip().lower()
    variants = {
        normalized,
        normalized.replace("/", "_"),
        normalized.replace("/", "-"),
        normalized.replace("/", "%2f"),
        normalized.replace("/", "."),
    }
    if "/" in normalized:
        variants.add(normalized.split("/", 1)[1])
    return tuple(v for v in variants if v)


def build_paper_id_to_doi_map(md_files: list[Path]) -> dict[str, str]:
    if not ALL_DOIS_CSV.exists():
        return {}

    md_stems = {md_file.stem.lower(): md_file.stem for md_file in md_files}
    mapping: dict[str, str] = {}

    with ALL_DOIS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doi = (row.get("DOI") or "").strip()
            if not doi:
                continue
            for variant in doi_to_md_variants(doi):
                paper_id = md_stems.get(variant)
                if paper_id and paper_id not in mapping:
                    mapping[paper_id] = doi
                    break

    return mapping


def build_title_index() -> dict[str, Path]:
    title_index: dict[str, Path] = {}
    for meta_path in META_SOURCE_DIR.glob("*.json"):
        try:
            content = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        normalized = normalize_metadata_payload(content)
        title_key = normalize_title(normalized.get("title"))
        if title_key and title_key not in title_index:
            title_index[title_key] = meta_path
    return title_index


def resolve_metadata_source(
    md_path: Path,
    paper_id: str,
    suffix_index: dict[str, Path],
    paper_id_to_doi: dict[str, str],
    title_index: dict[str, Path],
) -> tuple[Path | None, str]:
    direct = META_SOURCE_DIR / f"{paper_id}.json"
    if direct.exists():
        return direct, "direct_id"
    direct_lower = META_SOURCE_DIR / f"{paper_id.lower()}.json"
    if direct_lower.exists():
        return direct_lower, "direct_id_lower"

    doi = extract_doi_from_md(md_path)
    if doi:
        by_doi = META_SOURCE_DIR / doi_to_meta_filename(doi)
        if by_doi.exists():
            return by_doi, "doi_lookup"

    mapped_doi = paper_id_to_doi.get(paper_id)
    if mapped_doi:
        by_mapped_doi = META_SOURCE_DIR / doi_to_meta_filename(mapped_doi)
        if by_mapped_doi.exists():
            return by_mapped_doi, "csv_doi_lookup"

    by_suffix = suffix_index.get(paper_id.lower())
    if by_suffix is not None:
        return by_suffix, "suffix_lookup"

    md_title = extract_title_from_md(md_path) or paper_id
    by_title = title_index.get(normalize_title(md_title))
    if by_title is not None:
        return by_title, "title_lookup"

    return None, "missing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy metadata into a task directory.")
    parser.add_argument("job_name", nargs="?", default=DEFAULT_JOB_NAME, help="Task name, e.g. 317-1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job_name = args.job_name

    print(f"--- Copy metadata for task: {job_name} ---")

    job_dir = ROOT_DIR / job_name
    papers_md_dir = job_dir / "papers_md"
    target_meta_dir = job_dir / "results_part" / "metadata"

    if not papers_md_dir.exists():
        print(f"ERROR: markdown directory not found: {papers_md_dir}")
        return

    if not META_SOURCE_DIR.exists():
        print(f"ERROR: metadata source directory not found: {META_SOURCE_DIR}")
        return

    target_meta_dir.mkdir(parents=True, exist_ok=True)

    md_files = list(papers_md_dir.glob("*.md"))
    if not md_files:
        print(f"WARNING: no markdown files found in {papers_md_dir}")
        return

    print(f"Found {len(md_files)} markdown files; syncing metadata...")

    suffix_index = build_suffix_index()
    paper_id_to_doi = build_paper_id_to_doi_map(md_files)
    title_index = build_title_index()
    success_count = 0
    missing_count = 0
    doi_match_count = 0
    csv_doi_match_count = 0
    suffix_match_count = 0
    title_match_count = 0
    error_count = 0

    for md_file in md_files:
        paper_id = md_file.stem
        dst_json = target_meta_dir / f"{paper_id}.json"
        src_json, mode = resolve_metadata_source(md_file, paper_id, suffix_index, paper_id_to_doi, title_index)

        if src_json is None:
            missing_count += 1
            continue

        try:
            with src_json.open("r", encoding="utf-8") as f:
                content = json.load(f)

            output_data = {"metadata": normalize_metadata_payload(content)}

            with dst_json.open("w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            success_count += 1
            if mode == "doi_lookup":
                doi_match_count += 1
            elif mode == "csv_doi_lookup":
                csv_doi_match_count += 1
            elif mode == "suffix_lookup":
                suffix_match_count += 1
            elif mode == "title_lookup":
                title_match_count += 1
        except Exception as exc:
            error_count += 1
            print(f"ERROR: failed to process {paper_id} from {src_json.name}: {exc}")

    print("--------------------------------------------------")
    print("Metadata copy finished.")
    print(f"Copied and normalized: {success_count}")
    print(f"Matched via DOI lookup: {doi_match_count}")
    print(f"Matched via all_dois.csv filename normalization: {csv_doi_match_count}")
    print(f"Matched via DOI suffix fallback: {suffix_match_count}")
    print(f"Matched via title lookup: {title_match_count}")
    print(f"Missing source metadata: {missing_count}")
    print(f"Processing errors: {error_count}")
    print(f"Target path: {target_meta_dir}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
