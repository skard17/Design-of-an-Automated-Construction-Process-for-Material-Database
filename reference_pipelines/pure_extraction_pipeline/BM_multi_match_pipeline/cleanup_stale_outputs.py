from __future__ import annotations

import argparse
from pathlib import Path

from utils.io_utils import load_config, load_json, slugify_filename


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_id", default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = load_config(str(root / "config.yaml"))
    outputs_dir = (root / config["paths"]["outputs_dir"]).resolve()
    manifests_dir = outputs_dir / "manifests"

    manifest_paths = [manifests_dir / f"{args.paper_id}.json"] if args.paper_id else sorted(manifests_dir.glob("*.json"))

    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            continue
        manifest = load_json(str(manifest_path))
        paper_id = str(manifest.get("paper_id") or manifest_path.stem)
        keep_suffixes = {
            f"{slugify_filename(str(target['target_id']))}.json"
            for target in manifest.get("material_targets", [])
        }

        for subdir_name in ("matched", "final_targets"):
            subdir = outputs_dir / subdir_name / paper_id
            if not subdir.exists():
                continue
            for path in sorted(subdir.glob("*.json")):
                if path.name in keep_suffixes:
                    continue
                action = "DELETE" if args.apply else "STALE"
                print(f"{action} {subdir_name}/{paper_id} {path.name}")
                if args.apply:
                    path.unlink()


if __name__ == "__main__":
    main()
