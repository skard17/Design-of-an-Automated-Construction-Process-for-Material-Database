from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.io_utils import load_config, load_json, resolve_target_output_path, slugify_filename


def summarize_target(data: dict) -> dict:
    material_info = data.get("material_info", {}) or {}
    section0 = material_info.get("section0", {}) or {}
    section1 = material_info.get("section1", {}) or {}
    metadata = ((data.get("paper_info") or {}).get("metadata") or {})
    quality_control = data.get("quality_control") or (metadata.get("quality_control") or {})
    provenance = data.get("provenance") or (metadata.get("multi_material_provenance") or {})
    return {
        "target_id": data.get("target_id") or metadata.get("target_id"),
        "canonical_name": data.get("canonical_name") or metadata.get("canonical_name"),
        "section0_counts": {
            key: len(value)
            for key, value in section0.items()
            if isinstance(value, list) and value
        },
        "section1_counts": {
            key: len(value)
            for key, value in section1.items()
            if isinstance(value, list) and value
        },
        "accepted_candidate_count": len(provenance.get("accepted_candidate_ids", [])),
        "ambiguous_candidate_count": len(provenance.get("ambiguous_candidate_ids", [])),
        "ambiguity_flags": list(quality_control.get("ambiguity_flags", [])),
        "has_target_specific_property_evidence": bool(
            quality_control.get(
                "has_target_specific_property_evidence",
                quality_control.get("has_target_specific_superconducting_evidence", False),
            )
        ),
        "omission_reasons": list(quality_control.get("omission_reasons", [])),
    }


def build_snapshot(outputs_dir: Path) -> dict:
    manifests_dir = outputs_dir / "manifests"

    papers: dict[str, dict] = {}
    for manifest_path in sorted(manifests_dir.glob("*.json")):
        manifest = load_json(str(manifest_path))
        paper_id = str(manifest.get("paper_id") or manifest_path.stem)
        targets = []
        for target in manifest.get("material_targets", []):
            target_id = str(target["target_id"])
            final_path = Path(resolve_target_output_path(str(outputs_dir), "final_targets", paper_id, slugify_filename(target_id)))

            target_summary = {
                "target_id": target_id,
                "canonical_name": target.get("canonical_name"),
                "final_exists": final_path.exists(),
            }
            if final_path.exists():
                target_summary.update(summarize_target(load_json(str(final_path))))
            targets.append(target_summary)

        papers[paper_id] = {
            "paper_title": manifest.get("paper_title"),
            "target_count": len(targets),
            "targets": targets,
        }

    totals = {
        "paper_count": len(papers),
        "target_count": sum(item["target_count"] for item in papers.values()),
        "targets_with_properties": sum(
            1
            for paper in papers.values()
            for target in paper["targets"]
            if target.get("section1_counts")
        ),
        "targets_with_ambiguity": sum(
            1
            for paper in papers.values()
            for target in paper["targets"]
            if target.get("ambiguity_flags")
        ),
        "targets_without_final": sum(
            1
            for paper in papers.values()
            for target in paper["targets"]
            if not target.get("final_exists")
        ),
    }

    return {
        "baseline_version": "v1",
        "totals": totals,
        "papers": papers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="BM_multi_match_pipeline/regression_baseline.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = load_config(str(root / "config.yaml"))
    outputs_dir = (root / config["paths"]["outputs_dir"]).resolve()
    snapshot = build_snapshot(outputs_dir)

    output_path = Path(args.output).resolve()
    output_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
