from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from run_pipeline import run_one
from utils.io_utils import load_config


def read_id_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"id list not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_single_flag(single_json_path: Path) -> int | None:
    if not single_json_path.exists():
        return None
    try:
        data = json.loads(single_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = data.get("single_system")
    return value if isinstance(value, int) else None


def copy_if_needed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.read_bytes() == src.read_bytes():
        return
    shutil.copy2(src, dst)


def collect_not_single_ids(experimental_ids: list[str], single_outputs_dir: Path) -> tuple[list[str], list[str]]:
    not_single_ids: list[str] = []
    undecided_ids: list[str] = []
    for paper_id in experimental_ids:
        flag = read_single_flag(single_outputs_dir / f"{paper_id}.json")
        if flag == 1:
            continue
        if flag == 0:
            not_single_ids.append(paper_id)
        else:
            undecided_ids.append(paper_id)
    return not_single_ids, undecided_ids


def import_papers(paper_ids: list[str], source_papers_dir: Path, target_papers_dir: Path) -> list[str]:
    imported: list[str] = []
    for paper_id in paper_ids:
        src = source_papers_dir / f"{paper_id}.md"
        if not src.exists():
            continue
        copy_if_needed(src, target_papers_dir / src.name)
        supp_src = source_papers_dir / f"{paper_id}_si.md"
        if supp_src.exists():
            copy_if_needed(supp_src, target_papers_dir / supp_src.name)
        imported.append(paper_id)
    return imported


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import papers that failed the legacy single-system gate into the multi-material pipeline."
    )
    parser.add_argument("--source_papers_dir", required=True, help="Legacy markdown papers directory.")
    parser.add_argument("--experimental_file", required=True, help="Path to legacy experimental.txt.")
    parser.add_argument("--single_outputs_dir", required=True, help="Path to legacy single JSON outputs.")
    parser.add_argument(
        "--include_uncertain",
        action="store_true",
        help="Also import papers whose legacy single JSON is missing or unparsable.",
    )
    parser.add_argument(
        "--run_pipeline",
        action="store_true",
        help="Run the multi-material pipeline immediately after import.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = load_config(str(root / "config.yaml"))
    target_papers_dir = (root / config["paths"]["papers_dir"]).resolve()

    source_papers_dir = Path(args.source_papers_dir).resolve()
    experimental_file = Path(args.experimental_file).resolve()
    single_outputs_dir = Path(args.single_outputs_dir).resolve()

    experimental_ids = read_id_list(experimental_file)
    not_single_ids, undecided_ids = collect_not_single_ids(experimental_ids, single_outputs_dir)
    selected_ids = list(not_single_ids)
    if args.include_uncertain:
        selected_ids.extend(undecided_ids)

    imported_ids = import_papers(selected_ids, source_papers_dir, target_papers_dir)

    imported_list_path = root / "legacy_not_single_ids.txt"
    imported_list_path.write_text(
        "\n".join(imported_ids) + ("\n" if imported_ids else ""),
        encoding="utf-8",
    )

    undecided_list_path = root / "legacy_uncertain_ids.txt"
    undecided_list_path.write_text(
        "\n".join(undecided_ids) + ("\n" if undecided_ids else ""),
        encoding="utf-8",
    )

    print(f"experimental_total={len(experimental_ids)}")
    print(f"legacy_not_single={len(not_single_ids)}")
    print(f"legacy_uncertain={len(undecided_ids)}")
    print(f"imported={len(imported_ids)}")
    print(f"imported_list={imported_list_path}")
    print(f"uncertain_list={undecided_list_path}")

    if args.run_pipeline:
        for paper_id in imported_ids:
            print(f"run paper_id={paper_id}")
            run_one(paper_id)


if __name__ == "__main__":
    main()
