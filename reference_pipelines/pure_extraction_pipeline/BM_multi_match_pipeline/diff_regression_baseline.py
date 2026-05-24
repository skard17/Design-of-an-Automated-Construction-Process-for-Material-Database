from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def flatten_targets(snapshot: dict) -> dict[tuple[str, str], dict]:
    flat: dict[tuple[str, str], dict] = {}
    for paper_id, paper in snapshot.get("papers", {}).items():
        for target in paper.get("targets", []):
            key = (paper_id, str(target.get("target_id") or ""))
            flat[key] = target
    return flat


def summarize_target_change(old: dict | None, new: dict | None) -> str:
    if old is None and new is not None:
        return "added"
    if old is not None and new is None:
        return "removed"
    if old is None or new is None:
        return "unknown"

    old_s1 = sum((old.get("section1_counts") or {}).values())
    new_s1 = sum((new.get("section1_counts") or {}).values())
    old_amb = len(old.get("ambiguity_flags") or [])
    new_amb = len(new.get("ambiguity_flags") or [])

    if new_s1 > old_s1 and new_amb <= old_amb:
        return "improved"
    if new_s1 < old_s1:
        return "regressed"
    if new_amb < old_amb:
        return "cleaner"
    if new_amb > old_amb:
        return "noisier"
    if old.get("accepted_candidate_count") != new.get("accepted_candidate_count"):
        return "changed"
    return "unchanged"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    args = parser.parse_args()

    old_snapshot = load(args.old)
    new_snapshot = load(args.new)
    old_targets = flatten_targets(old_snapshot)
    new_targets = flatten_targets(new_snapshot)

    all_keys = sorted(set(old_targets) | set(new_targets))
    changes: list[dict] = []
    for key in all_keys:
        old_target = old_targets.get(key)
        new_target = new_targets.get(key)
        status = summarize_target_change(old_target, new_target)
        if status == "unchanged":
            continue
        paper_id, target_id = key
        changes.append(
            {
                "paper_id": paper_id,
                "target_id": target_id,
                "status": status,
                "old_section1_counts": (old_target or {}).get("section1_counts", {}),
                "new_section1_counts": (new_target or {}).get("section1_counts", {}),
                "old_ambiguity_count": len((old_target or {}).get("ambiguity_flags", [])),
                "new_ambiguity_count": len((new_target or {}).get("ambiguity_flags", [])),
                "old_accepted_candidate_count": (old_target or {}).get("accepted_candidate_count"),
                "new_accepted_candidate_count": (new_target or {}).get("accepted_candidate_count"),
            }
        )

    output = {
        "old_totals": old_snapshot.get("totals", {}),
        "new_totals": new_snapshot.get("totals", {}),
        "change_count": len(changes),
        "changes": changes,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
