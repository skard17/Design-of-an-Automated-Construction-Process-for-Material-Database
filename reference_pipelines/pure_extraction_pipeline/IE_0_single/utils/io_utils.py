from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import yaml


def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到配置文件：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_text(path: str, encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def write_text(path: str, content: str, encoding: str = "utf-8") -> None:
    d = os.path.dirname(path)
    if d:
        ensure_dir(d)
    with open(path, "w", encoding=encoding) as f:
        f.write(content)


def get_paper_id(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def list_files(dir_path: str, suffixes: Iterable[str]) -> list[str]:
    if not os.path.exists(dir_path):
        return []
    suf = tuple(suffixes)
    files: list[str] = []
    if os.getenv("IE_SINGLE_RECURSIVE_INPUT", "0").lower() in {"1", "true", "yes"}:
        for root, _dirs, names in os.walk(dir_path):
            for name in sorted(names):
                if name.endswith(suf):
                    p = os.path.join(root, name)
                    if os.path.isfile(p):
                        files.append(p)
        return sorted(files)
    for name in sorted(os.listdir(dir_path)):
        if name.endswith(suf):
            p = os.path.join(dir_path, name)
            if os.path.isfile(p):
                files.append(p)
    return files


_SUPP_SUFFIX = "_si"


def _is_supplementary_paper_id(paper_id: str) -> bool:
    return str(paper_id or "").lower().endswith(_SUPP_SUFFIX)


def list_paper_ids(paths_cfg: dict) -> list[str]:
    papers_dir = paths_cfg.get("papers_dir", "papers")
    return [
        get_paper_id(p)
        for p in list_files(papers_dir, (".md",))
        if not _is_supplementary_paper_id(get_paper_id(p))
    ]


def select_paper_file(paths_cfg: dict, paper_id: str | None) -> tuple[str, str]:
    papers_dir = paths_cfg.get("papers_dir", "papers")
    if paper_id is None:
        files = list_files(papers_dir, (".md",))
        if not files:
            raise FileNotFoundError(
                f"未找到任何论文文件。已检查：{papers_dir}（后缀 .md）。"
                "请把 markdown 论文放到 papers 文件夹。"
            )
        path = files[0]
        return get_paper_id(path), path

    p = os.path.join(papers_dir, f"{paper_id}.md")
    if os.path.exists(p):
        return paper_id, p

    if os.getenv("IE_SINGLE_RECURSIVE_INPUT", "0").lower() in {"1", "true", "yes"}:
        for root, _dirs, names in os.walk(papers_dir):
            if f"{paper_id}.md" in names:
                return paper_id, os.path.join(root, f"{paper_id}.md")

    raise FileNotFoundError(
        f"未找到 paper_id={paper_id} 对应的论文文件。已检查：{p}"
    )


def write_stage_output(outputs_dir: str, stage: str, paper_id: str, content: str, ext: str = ".json") -> str:
    stage_dir = os.path.join(outputs_dir, stage)
    ensure_dir(stage_dir)
    out_path = os.path.join(stage_dir, f"{paper_id}{ext}")
    write_text(out_path, content)
    return out_path


def load_supplementary_information(paper_path: str, paper_id: str) -> str:
    paper_dir = os.path.dirname(paper_path)
    supp_path = os.path.join(paper_dir, f"{paper_id}{_SUPP_SUFFIX}.md")
    if os.path.exists(supp_path) and os.path.getsize(supp_path) > 0:
        return read_text(supp_path)
    return ""


def validate_papers(paths_cfg: dict) -> list[str]:
    paper_ids = list_paper_ids(paths_cfg)
    if not paper_ids:
        raise FileNotFoundError("papers 目录下未找到可处理的 markdown 论文文件。")
    return sorted(paper_ids)


def resolve_outputs_dir(config: dict) -> str:
    outputs_dir = config.get("paths", {}).get("outputs_dir", "outputs")
    return str(Path(outputs_dir))
