from __future__ import annotations

import argparse
from pathlib import Path

from run_aggregate import run_aggregate
from run_fact_candidates import run_fact_candidates
from run_manifest import run_manifest
from run_matcher import run_matcher
from utils.io_utils import load_config, load_json, validate_papers
from utils.log_utils import setup_logger


def run_one(paper_id: str) -> None:
    root = Path(__file__).resolve().parent
    config = load_config(str(root / "config.yaml"))
    outputs_dir = str((root / config["paths"]["outputs_dir"]).resolve())

    run_manifest(paper_id)
    run_fact_candidates(paper_id)
    manifest = load_json(str(Path(outputs_dir) / "manifests" / f"{paper_id}.json"))
    for target in manifest.get("material_targets", []):
        target_id = target["target_id"]
        run_matcher(paper_id, target_id)
        run_aggregate(paper_id, target_id)


def main() -> None:
    logger = setup_logger("pipeline")
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_id", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = load_config(str(root / "config.yaml"))
    papers_dir = str((root / config["paths"]["papers_dir"]).resolve())
    paper_ids = [args.paper_id] if args.paper_id else validate_papers({"papers_dir": papers_dir})
    for paper_id in paper_ids:
        logger.info("run paper_id=%s", paper_id)
        run_one(paper_id)


if __name__ == "__main__":
    main()
