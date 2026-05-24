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


def _pick_input_format(paths_cfg: dict) -> str:
    fmt = str(paths_cfg.get("input_format", "auto")).strip().lower()
    if fmt in {"md", "tex", "txt"}:
        return fmt
    return "auto"


def _resolve_paper_dir(paths_cfg: dict, input_format: str) -> tuple[str, tuple[str, ...]]:
    if input_format == "md":
        return paths_cfg["papers_md_dir"], (".md",)
    if input_format == "tex":
        return paths_cfg["papers_tex_dir"], (".tex",)
    if input_format == "txt":
        return paths_cfg.get("papers_txt_dir", "papers/txt"), (".txt",)

    md_dir = paths_cfg["papers_md_dir"]
    tex_dir = paths_cfg["papers_tex_dir"]
    txt_dir = paths_cfg.get("papers_txt_dir", "papers/txt")

    if list_files(md_dir, (".md",)):
        return md_dir, (".md",)
    if list_files(tex_dir, (".tex",)):
        return tex_dir, (".tex",)
    return txt_dir, (".txt",)


def list_paper_ids(paths_cfg: dict) -> list[str]:
    fmt = _pick_input_format(paths_cfg)
    paper_dir, suffixes = _resolve_paper_dir(paths_cfg, fmt)
    return [
        get_paper_id(p)
        for p in list_files(paper_dir, suffixes)
        if not _is_supplementary_paper_id(get_paper_id(p))
    ]


def select_paper_file(paths_cfg: dict, paper_id: str | None) -> tuple[str, str]:
    fmt = _pick_input_format(paths_cfg)
    paper_dir, suffixes = _resolve_paper_dir(paths_cfg, fmt)
    if paper_id is None:
        files = list_files(paper_dir, suffixes)
        if not files:
            raise FileNotFoundError(
                f"未找到任何论文文件。已检查：{paper_dir}（后缀 {suffixes}）。"
                "请把论文放到 papers/md 或 papers/tex（或 papers/txt），并在 config.yaml 里修改路径配置。"
            )
        path = files[0]
        return get_paper_id(path), path

    for suf in suffixes:
        p = os.path.join(paper_dir, f"{paper_id}{suf}")
        if os.path.exists(p):
            return paper_id, p

    raise FileNotFoundError(
        f"未找到 paper_id={paper_id} 对应的论文文件。已检查目录={paper_dir}，后缀={suffixes}"
    )


def write_stage_output(outputs_dir: str, stage: str, paper_id: str, content: str, ext: str = ".json") -> str:
    stage_dir = os.path.join(outputs_dir, stage)
    ensure_dir(stage_dir)
    out_path = os.path.join(stage_dir, f"{paper_id}{ext}")
    write_text(out_path, content)
    return out_path


_SUPP_SUFFIX = "_si"


def _is_supplementary_paper_id(paper_id: str) -> bool:
    return str(paper_id or "").lower().endswith(_SUPP_SUFFIX)


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
        raise FileNotFoundError("papers 目录下未找到可处理的论文文件。")
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

