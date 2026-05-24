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

    raise FileNotFoundError(
        f"未找到 paper_id={paper_id} 对应的论文文件。已检查：{p}"
    )


def write_stage_output(outputs_dir: str, stage: str, paper_id: str, content: str, ext: str = ".json") -> str:
    stage_dir = os.path.join(outputs_dir, stage)
    ensure_dir(stage_dir)
    out_path = os.path.join(stage_dir, f"{paper_id}{ext}")
    write_text(out_path, content)
    return out_path


def resolve_inputs_dir(paths_cfg: dict) -> str:
    return str(paths_cfg.get("inputs_dir", "inputs"))


def read_input_json(paths_cfg: dict, paper_id: str) -> str:
    inputs_dir = resolve_inputs_dir(paths_cfg)
    path = os.path.join(inputs_dir, f"{paper_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到输入 JSON：{path}")
    if os.path.getsize(path) <= 0:
        raise ValueError(f"输入 JSON 为空：{path}")
    return read_text(path)


def load_supplementary_information(paper_path: str, paper_id: str) -> str:
    paper_dir = os.path.dirname(paper_path)
    supp_path = os.path.join(paper_dir, f"{paper_id}{_SUPP_SUFFIX}.md")
    if os.path.exists(supp_path) and os.path.getsize(supp_path) > 0:
        return read_text(supp_path)
    return ""


def list_input_ids(paths_cfg: dict) -> list[str]:
    inputs_dir = resolve_inputs_dir(paths_cfg)
    return [get_paper_id(p) for p in list_files(inputs_dir, (".json",))]


def validate_paper_input_pairs(paths_cfg: dict) -> list[str]:
    paper_ids = list_paper_ids(paths_cfg)
    input_ids = list_input_ids(paths_cfg)

    if not paper_ids:
        raise FileNotFoundError("papers 目录下未找到可处理的 markdown 论文文件。")
    if not input_ids:
        raise FileNotFoundError("inputs 目录下未找到任何 JSON 输入文件。")

    paper_set = set(paper_ids)
    input_set = set(input_ids)
    missing_inputs = sorted(paper_set - input_set)
    extra_inputs = sorted(input_set - paper_set)
    if missing_inputs or extra_inputs:
        msg = []
        if missing_inputs:
            msg.append(f"缺少对应 inputs JSON 的论文: {', '.join(missing_inputs)}")
        if extra_inputs:
            msg.append(f"多余的 inputs JSON（无对应论文）: {', '.join(extra_inputs)}")
        raise FileNotFoundError("；".join(msg))

    return sorted(paper_set)


def resolve_outputs_dir(config: dict) -> str:
    outputs_dir = config.get("paths", {}).get("outputs_dir", "outputs")
    return str(Path(outputs_dir))
