from __future__ import annotations

import json
import os
import re
from typing import Iterable

import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_text(path: str, encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def write_text(path: str, content: str, encoding: str = "utf-8") -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding=encoding) as f:
        f.write(content)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_paper_id(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def list_files(dir_path: str, suffixes: Iterable[str]) -> list[str]:
    if not os.path.exists(dir_path):
        return []
    suffixes = tuple(suffixes)
    return [
        os.path.join(dir_path, name)
        for name in sorted(os.listdir(dir_path))
        if name.endswith(suffixes) and os.path.isfile(os.path.join(dir_path, name))
    ]


_SUPP_SUFFIX = "_si"


def select_paper_file(paths_cfg: dict, paper_id: str | None) -> tuple[str, str]:
    papers_dir = paths_cfg.get("papers_dir", "papers")
    if paper_id is None:
        files = list_files(papers_dir, (".md",))
        if not files:
            raise FileNotFoundError(f"no markdown papers found in {papers_dir}")
        path = files[0]
        return get_paper_id(path), path
    path = os.path.join(papers_dir, f"{paper_id}.md")
    if not os.path.exists(path):
        raise FileNotFoundError(f"paper not found: {path}")
    return paper_id, path


def load_supplementary_information(paper_path: str, paper_id: str) -> str:
    paper_dir = os.path.dirname(paper_path)
    supp_path = os.path.join(paper_dir, f"{paper_id}{_SUPP_SUFFIX}.md")
    if os.path.exists(supp_path) and os.path.getsize(supp_path) > 0:
        return read_text(supp_path)
    return ""


def validate_papers(paths_cfg: dict) -> list[str]:
    papers_dir = paths_cfg.get("papers_dir", "papers")
    paper_ids = [get_paper_id(p) for p in list_files(papers_dir, (".md",)) if not get_paper_id(p).endswith(_SUPP_SUFFIX)]
    if not paper_ids:
        raise FileNotFoundError(f"no markdown papers found in {papers_dir}")
    return sorted(paper_ids)


def write_named_output(outputs_dir: str, subdir: str, file_stem: str, content: str) -> str:
    target_dir = os.path.join(outputs_dir, subdir)
    ensure_dir(target_dir)
    out_path = os.path.join(target_dir, f"{file_stem}.json")
    write_text(out_path, content)
    return out_path


def write_paper_scoped_output(outputs_dir: str, subdir: str, paper_id: str, file_stem: str, content: str) -> str:
    target_dir = os.path.join(outputs_dir, subdir, paper_id)
    ensure_dir(target_dir)
    out_path = os.path.join(target_dir, f"{file_stem}.json")
    write_text(out_path, content)
    return out_path


def resolve_paper_scoped_json_path(outputs_dir: str, subdir: str, paper_id: str, file_stem: str) -> str:
    return os.path.join(outputs_dir, subdir, paper_id, f"{file_stem}.json")


def resolve_target_output_path(outputs_dir: str, subdir: str, paper_id: str, file_stem: str) -> str:
    paper_scoped = resolve_paper_scoped_json_path(outputs_dir, subdir, paper_id, file_stem)
    if os.path.exists(paper_scoped):
        return paper_scoped
    legacy_flat = os.path.join(outputs_dir, subdir, f"{paper_id}__{file_stem}.json")
    return legacy_flat


def slugify_filename(text: str) -> str:
    value = re.sub(r"[^\w\-\.]+", "_", text, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value).strip("._")
    return value or "target"
