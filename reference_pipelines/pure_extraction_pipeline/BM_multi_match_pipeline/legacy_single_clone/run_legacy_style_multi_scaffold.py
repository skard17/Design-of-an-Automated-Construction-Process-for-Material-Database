from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io_utils import ensure_dir, load_json, read_text, write_text


def build_scaffold_manifest(paper_id: str, multi_system_result: dict) -> dict:
    counted_systems = list(multi_system_result.get("counted_systems", []))
    system_entries: list[dict] = []
    for item in counted_systems:
        system_signature = str(item.get("system_signature") or "").strip()
        if not system_signature:
            continue
        system_entries.append(
            {
                "system_signature": system_signature,
                "system_signature_type": item.get("system_signature_type"),
                "legacy_single_prompt_bundle": {
                    "section0": "section0_single.md",
                    "section1": "section1_single.md",
                    "section2_1": "section2_1_single.md",
                    "section2_2_prefix": "section2_2_prefix_single.md",
                    "section2_2_suffix": "section2_2_suffix_single.md",
                    "section2_2_fabrication_dir": "s2_2_fabrication",
                    "section3": "section3_single.md",
                    "section4": "section4_single.md",
                    "section5": "section5_single.md",
                },
            }
        )

    return {
        "paper_id": paper_id,
        "multi_system_result": multi_system_result,
        "systems": system_entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a paper-level scaffold showing how the copied single-system prompts will be reused for each multi-system target."
    )
    parser.add_argument("--paper_id", required=True)
    parser.add_argument("--multi_system_json", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = ROOT
    multi_system_path = Path(args.multi_system_json).resolve()
    multi_system_result = load_json(str(multi_system_path))
    scaffold = build_scaffold_manifest(args.paper_id, multi_system_result)

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = root / "outputs" / "legacy_style_scaffold" / args.paper_id / "scaffold.json"

    ensure_dir(str(output_path.parent))
    write_text(str(output_path), json.dumps(scaffold, ensure_ascii=False, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
