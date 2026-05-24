from __future__ import annotations

import argparse
import json
from pathlib import Path


SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉₊₋", "0123456789+-")

SLUG_MAP = {
    "CeCoIn5，CeRhIn5，CeIrIn5": "CeCoIn5_CeRhIn5_CeIrIn5",
    "CeIrIn5，CeRhIn5": "CeIrIn5_CeRhIn5",
    "K2Cr3As3，Rb2Cr3As3": "K2Cr3As3_Rb2Cr3As3",
    "La-, Pr-, Nd-nickelates": "La_Pr_Nd_nickelates",
    "LnFeAsO1-y": "LnFeAsO1_y",
    "Nb0.85X0.15 (X = Ti, Zr, Hf)": "Nb085X015_Ti_Zr_Hf",
    "Rb2Mo3As3 and Cs2Mo3As3": "Rb2Mo3As3_Cs2Mo3As3",
    "RbLn2Fe4As4O2": "RbLn2Fe4As4O2",
    "Sc metal and Li(Mg) alloy": "Sc_metal_LiMg_alloy",
    "Sr1-xLnxCuO2(Ln=La, Nd, Sm, Gd)": "Sr1_xLnxCuO2_Ln_La_Nd_Sm_Gd",
    "T-R2CuO4 (R=Pr, Nd, Sm, Eu, Gd)": "T_R2CuO4_R_Pr_Nd_Sm_Eu_Gd",
}


def normalize_source_name(name: str) -> str:
    normalized = name.translate(SUBSCRIPT_MAP)
    normalized = normalized.replace("−", "-")
    return normalized


def slugify_fallback(name: str) -> str:
    slug = name.replace(" ", "_")
    slug = slug.replace(",", "_")
    slug = slug.replace("，", "_")
    slug = slug.replace("(", "_").replace(")", "_")
    slug = slug.replace("=", "_")
    slug = slug.replace("/", "_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="bm-scmulti-md")
    parser.add_argument("--target_dir", default="BM_multi_match_pipeline/papers")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing markdown files in target_dir before importing.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    target_dir = Path(args.target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for stale_md in target_dir.glob("*.md"):
            stale_md.unlink()

    mapping: list[dict[str, str]] = []
    for paper_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
        source_md = paper_dir / f"{paper_dir.name}.md"
        if not source_md.exists():
            continue

        normalized_name = normalize_source_name(paper_dir.name)
        paper_id = SLUG_MAP.get(normalized_name, slugify_fallback(normalized_name))
        target_md = target_dir / f"{paper_id}.md"
        target_md.write_text(source_md.read_text(encoding="utf-8"), encoding="utf-8")
        mapping.append(
            {
                "paper_id": paper_id,
                "source_dir": paper_dir.name,
                "normalized_source_dir": normalized_name,
                "source_md": str(source_md),
                "target_md": str(target_md),
            }
        )

    mapping_path = target_dir.parent / "benchmark_mapping.json"
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"prepared {len(mapping)} papers")
    print(mapping_path)


if __name__ == "__main__":
    main()
