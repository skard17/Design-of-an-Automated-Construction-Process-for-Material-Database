from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from utils.ds_client import chat
from utils.io_utils import load_config, load_json, slugify_filename, write_paper_scoped_output
from utils.key_pool import KeyPool
from utils.log_utils import setup_logger
from utils.prompt_utils import clean_llm_output, inject_blocks, load_prompt_template


PROMPT_FILE = "matcher.md"
PROPERTY_FACT_TYPES = {
    "property",
    "property_value",
    "performance_metric",
    "transition_temperature",
    "critical_field",
    "critical_current_density",
}


def normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def choose_display_name(target: dict) -> str:
    aliases = [str(x) for x in target.get("aliases", []) if str(x).strip()]
    for alias in aliases:
        if all(ord(ch) < 128 for ch in alias):
            return alias
    canonical = str(target.get("canonical_name") or "").strip()
    if canonical:
        return canonical
    return str(target.get("target_id") or "")


def normalize_display_name(text: str) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"[^A-Za-z0-9.\-+_]+", " ", cleaned).strip()
    return cleaned or str(text or "")


def extract_element_token(target_id: str) -> str | None:
    match = re.search(r"(ti|zr|hf)(?:\d+)?$", target_id.lower())
    return match.group(1) if match else None


def infer_context_labels(target: dict) -> set[str]:
    target_id = str(target.get("target_id") or "")
    labels: set[str] = set()

    nickelate_match = re.match(r"([A-Za-z]+)_nickelate$", target_id)
    if nickelate_match:
        labels.add(normalize_text(nickelate_match.group(1)))

    slco_match = re.match(r"SLCO_([A-Za-z]+)$", target_id)
    if slco_match:
        labels.add(normalize_text(slco_match.group(1)))

    family_element_patterns = [
        r"^Rb([A-Z][a-z]?)\d+Fe",
        r"^([A-Z][a-z]?)2CuO4$",
        r"^([A-Z][a-z]?)FeAsO",
    ]
    canonical = str(target.get("canonical_name") or "")
    for pattern in family_element_patterns:
        match = re.match(pattern, canonical)
        if match:
            labels.add(normalize_text(match.group(1)))

    return {label for label in labels if label}


def build_target_profile(target: dict) -> dict:
    aliases = set()
    for item in [target.get("canonical_name"), *target.get("aliases", []), target.get("target_id")]:
        if item:
            aliases.add(str(item))

    element = extract_element_token(str(target.get("target_id") or ""))
    if element:
        aliases.update(
            {
                element,
                f"x {element}",
                f"x {element}  ",
                f"x {element}".replace(" ", "="),
                f"nb6{element}",
                f"nb0 85 {element} 0 15",
                f"nb0.85{element}0.15",
            }
        )

    normalized_aliases = {normalize_text(alias) for alias in aliases if normalize_text(alias)}
    return {
        "target_id": target["target_id"],
        "canonical_name": target["canonical_name"],
        "normalized_aliases": normalized_aliases,
        "context_labels": infer_context_labels(target),
    }


def any_alias_present(alias_set: set[str], text: str) -> bool:
    if not text:
        return False
    padded = f" {text} "
    return any(f" {alias} " in padded for alias in alias_set if alias)


def candidate_text_blob(candidate: dict) -> str:
    texts: list[str] = []
    texts.extend(str(x) for x in candidate.get("local_material_mentions", []))
    texts.extend(str(x) for x in candidate.get("local_series_mentions", []))
    texts.extend(str(x) for x in (candidate.get("attribution_hint", {}) or {}).get("candidate_targets", []))
    texts.append(str(candidate.get("verbatim_evidence") or ""))
    texts.append(str(candidate.get("value") or ""))
    texts.append(str((candidate.get("conditions") or {}).get("characteristics") or ""))
    return normalize_text(" ".join(texts))


def normalized_aliases_from_target_names(names: list[str]) -> set[str]:
    aliases: set[str] = set()
    for name in names:
        normalized = normalize_text(name)
        if normalized:
            aliases.add(normalized)
    return aliases


def evidence_mentions_alias(alias_set: set[str], evidence: str) -> bool:
    normalized_evidence = normalize_text(evidence)
    return any_alias_present(alias_set, normalized_evidence)


def count_targets_mentioned_in_evidence(profiles: list[dict], evidence: str) -> int:
    normalized_evidence = normalize_text(evidence)
    return sum(
        1
        for profile in profiles
        if any_alias_present(profile["normalized_aliases"], normalized_evidence)
        or any_alias_present(profile.get("context_labels", set()), normalized_evidence)
    )


def is_fragmentary_comparative_fact(candidate: dict, target_profile: dict) -> bool:
    evidence = str(candidate.get("verbatim_evidence") or "")
    normalized_evidence = normalize_text(evidence)
    if not normalized_evidence:
        return False
    if evidence_mentions_alias(target_profile["normalized_aliases"], evidence):
        return False
    comparative_markers = (
        "increase",
        "decrease",
        "highest",
        "lowest",
        "varies",
        "from",
        "to",
        "versus",
        "compared",
        "relative",
    )
    if not any(marker in normalized_evidence for marker in comparative_markers):
        return False
    return True


def is_multi_target_group_fact(candidate: dict, all_profiles: list[dict]) -> bool:
    reason = str((candidate.get("attribution_hint", {}) or {}).get("reason") or "")
    if reason not in {"same_sentence", "same_caption", "same_table_column", "family_level"}:
        return False
    target_names = [str(x) for x in (candidate.get("attribution_hint", {}) or {}).get("candidate_targets", []) if str(x).strip()]
    if len(target_names) > 1:
        return True
    evidence = str(candidate.get("verbatim_evidence") or "")
    if count_targets_mentioned_in_evidence(all_profiles, evidence) > 1:
        return True
    local_mentions = [str(x) for x in candidate.get("local_material_mentions", []) if str(x).strip()]
    return len(local_mentions) > 1


def is_shared_series_property_fact(candidate: dict, manifest: dict) -> bool:
    fact_type = str(candidate.get("fact_type") or "")
    if fact_type not in PROPERTY_FACT_TYPES and not candidate.get("property_name"):
        return False

    target_names = [str(x) for x in (candidate.get("attribution_hint", {}) or {}).get("candidate_targets", []) if str(x).strip()]
    manifest_names = {
        normalize_text(str(item.get("canonical_name") or ""))
        for item in manifest.get("material_targets", [])
        if str(item.get("canonical_name") or "").strip()
    }
    hinted_names = {normalize_text(name) for name in target_names if normalize_text(name)}
    if not hinted_names or hinted_names != manifest_names:
        return False

    evidence = normalize_text(str(candidate.get("verbatim_evidence") or ""))
    shared_markers = (
        "remains almost constant",
        "remains at",
        "same value",
        "nearly unchanged",
        "unchanged",
        "always",
        "independent of x",
        "constant at",
        "for all samples",
        "for all compounds",
        "across the series",
    )
    return any(marker in evidence for marker in shared_markers)


def candidate_id_number(candidate_id: str) -> int:
    match = re.search(r"(\d+)$", candidate_id)
    return int(match.group(1)) if match else 0


def ordered_group_key(candidate: dict) -> tuple[str, str, str, str, str, str]:
    anchor = candidate.get("source_anchor", {}) or {}
    conditions = candidate.get("conditions", {}) or {}
    return (
        str(candidate.get("fact_type") or ""),
        str(anchor.get("section_label") or ""),
        str(anchor.get("figure") or ""),
        str(anchor.get("table") or ""),
        str(conditions.get("characteristics") or ""),
        str((candidate.get("attribution_hint", {}) or {}).get("reason") or ""),
    )


def build_ordered_assignment_map(manifest: dict, fact_candidates: dict) -> dict[str, str]:
    target_order = [item["target_id"] for item in manifest.get("material_targets", [])]
    target_count = len(target_order)
    grouped: dict[tuple[str, str, str, str, str, str], list[dict]] = {}

    for candidate in fact_candidates.get("paper_level_candidates", []):
        hinted_targets = (candidate.get("attribution_hint", {}) or {}).get("candidate_targets", [])
        if len(hinted_targets) != target_count:
            continue
        grouped.setdefault(ordered_group_key(candidate), []).append(candidate)

    assignments: dict[str, str] = {}
    for candidates in grouped.values():
        if len(candidates) != target_count:
            continue
        ordered_candidates = sorted(candidates, key=lambda item: candidate_id_number(str(item.get("candidate_id") or "")))
        for candidate, target_id in zip(ordered_candidates, target_order, strict=False):
            assignments[str(candidate["candidate_id"])] = target_id
    return assignments


def split_candidates_for_target(manifest: dict, target: dict, fact_candidates: dict) -> tuple[list[dict], list[dict], list[dict], list[str]]:
    profiles = {item["target_id"]: build_target_profile(item) for item in manifest.get("material_targets", [])}
    target_profile = profiles[target["target_id"]]
    sibling_profiles = [profiles[item["target_id"]] for item in manifest.get("material_targets", []) if item["target_id"] != target["target_id"]]
    all_profiles = list(profiles.values())
    ordered_assignments = build_ordered_assignment_map(manifest, fact_candidates)

    accepted: list[dict] = []
    rejected: list[dict] = []
    unresolved: list[dict] = []
    notes: list[str] = []

    for candidate in fact_candidates.get("paper_level_candidates", []):
        blob = candidate_text_blob(candidate)
        reason = str((candidate.get("attribution_hint", {}) or {}).get("reason") or "")
        target_present = any_alias_present(target_profile["normalized_aliases"], blob)
        sibling_present = any(any_alias_present(profile["normalized_aliases"], blob) for profile in sibling_profiles)
        target_list = [str(x) for x in (candidate.get("attribution_hint", {}) or {}).get("candidate_targets", [])]
        normalized_target_list = {normalize_text(x) for x in target_list if normalize_text(x)}
        reason_is_family_level = reason == "family_level"
        multi_target_group_fact = is_multi_target_group_fact(candidate, all_profiles)
        fragmentary_comparative_fact = is_fragmentary_comparative_fact(candidate, target_profile)
        shared_series_property_fact = is_shared_series_property_fact(candidate, manifest)
        evidence_text = str(candidate.get("verbatim_evidence") or "")
        target_context_present = any_alias_present(target_profile.get("context_labels", set()), normalize_text(evidence_text))
        sibling_context_present = any(
            any_alias_present(profile.get("context_labels", set()), normalize_text(evidence_text))
            for profile in sibling_profiles
        )

        decision = None
        decision_reason = None

        assigned_target_id = ordered_assignments.get(str(candidate.get("candidate_id") or ""))
        if assigned_target_id == target["target_id"]:
            decision = "accept"
            decision_reason = "auto: ordered candidate group mapped to current target"
        elif assigned_target_id is not None:
            decision = "reject"
            decision_reason = "auto: ordered candidate group mapped to sibling"

        if decision is None and shared_series_property_fact:
            decision = "accept"
            decision_reason = "auto: shared series-level property fact applies to every material target in the manifest"

        if decision is None and reason_is_family_level and candidate.get("fact_type") != "stack_descriptor":
            decision = "reject"
            decision_reason = "auto: family-level candidate is kept out of single-target output unless explicitly ordered-mapped"

        if decision is None and multi_target_group_fact and candidate.get("fact_type") != "stack_descriptor":
            decision = "reject"
            decision_reason = "auto: grouped multi-target candidate without deterministic one-to-one mapping"

        if len(normalized_target_list) == 1:
            only_target = next(iter(normalized_target_list))
            if any(f" {alias} " in f" {only_target} " for alias in target_profile["normalized_aliases"]):
                if fragmentary_comparative_fact and target_context_present and not sibling_context_present:
                    decision = "accept"
                    decision_reason = "auto: comparative fragment explicitly names the current family member label"
                elif fragmentary_comparative_fact:
                    decision = "reject"
                    decision_reason = "auto: singleton target match is only a comparative fragment without explicit target mention in evidence"
                else:
                    decision = "accept"
                    decision_reason = "auto: singleton candidate target matches current target"
            elif any(
                any(f" {alias} " in f" {only_target} " for alias in profile["normalized_aliases"])
                for profile in sibling_profiles
            ):
                decision = "reject"
                decision_reason = "auto: singleton candidate target matches sibling"

        if decision is None and target_present and not sibling_present and reason in {"same_sentence", "same_caption", "same_table_column"}:
            decision = "accept"
            decision_reason = f"auto: explicit local alias for current target via {reason}"

        if decision is None and not target_present and sibling_present and reason in {"same_sentence", "same_caption", "same_table_column"}:
            decision = "reject"
            decision_reason = f"auto: explicit local alias for sibling via {reason}"

        if decision == "accept":
            accepted.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "fact_type": candidate["fact_type"],
                    "match_decision": "accept",
                    "match_confidence": 0.99,
                    "reason": decision_reason,
                }
            )
            continue

        if decision == "reject":
            rejected.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "fact_type": candidate["fact_type"],
                    "match_decision": "reject",
                    "match_confidence": 0.99,
                    "reason": decision_reason,
                }
            )
            continue

        unresolved.append(candidate)

    notes.append(
        f"Auto-matcher preclassified {len(accepted)} accepts and {len(rejected)} rejects; {len(unresolved)} candidates left for LLM review."
    )
    return accepted, rejected, unresolved, notes


def run_matcher(paper_id: str, target_id: str) -> None:
    logger = setup_logger("matcher")
    root = Path(__file__).resolve().parent
    config = load_config(str(root / "config.yaml"))
    provider = config["provider"]
    paths = config["paths"]
    stage_cfg = config["stages"]["matcher"]
    outputs_dir = str((root / paths["outputs_dir"]).resolve())
    prompts_dir = str((root / paths["prompts_dir"]).resolve())

    manifest = load_json(str(Path(outputs_dir) / "manifests" / f"{paper_id}.json"))
    fact_candidates = load_json(str(Path(outputs_dir) / "fact_candidates" / f"{paper_id}.json"))
    target = next(item for item in manifest["material_targets"] if item["target_id"] == target_id)

    auto_accepted, auto_rejected, unresolved_candidates, auto_notes = split_candidates_for_target(manifest, target, fact_candidates)

    llm_result = {
        "paper_id": paper_id,
        "target_id": target_id,
        "canonical_name": normalize_display_name(choose_display_name(target)),
        "matched_version": "v1",
        "accepted_candidates": [],
        "rejected_candidates": [],
        "ambiguous_candidates": [],
        "global_notes": [],
    }

    if unresolved_candidates:
        template = load_prompt_template(prompts_dir, PROMPT_FILE)
        reduced_candidates = {
            **fact_candidates,
            "paper_level_candidates": unresolved_candidates,
        }
        prompt = inject_blocks(
            template,
            {
                "manifest_json": json.dumps(manifest, ensure_ascii=False, indent=2),
                "target_json": json.dumps(target, ensure_ascii=False, indent=2),
                "fact_candidates_json": json.dumps(reduced_candidates, ensure_ascii=False, indent=2),
            },
        )

        api_key = KeyPool.get_next_key(config_keys=provider.get("api_keys"))
        response = chat(
            prompt=prompt,
            model=provider["model"],
            base_url=provider["base_url"],
            api_key=api_key,
            temperature=float(stage_cfg.get("temperature", 0.0)),
            timeout_sec=int(provider.get("timeout_sec", 60)),
            system_prompt="You are an expert in materials science information extraction and database construction.",
        )
        out_text = clean_llm_output(response)
        llm_result = json.loads(out_text)
    else:
        llm_result["global_notes"].append("LLM stage skipped because all candidates were preclassified by deterministic matcher.")

    merged = {
        "paper_id": paper_id,
        "target_id": target_id,
        "canonical_name": normalize_display_name(choose_display_name(target)),
        "matched_version": "v1",
        "accepted_candidates": auto_accepted + llm_result.get("accepted_candidates", []),
        "rejected_candidates": auto_rejected + llm_result.get("rejected_candidates", []),
        "ambiguous_candidates": llm_result.get("ambiguous_candidates", []),
        "global_notes": auto_notes + llm_result.get("global_notes", []),
    }

    out_path = write_paper_scoped_output(
        outputs_dir,
        "matched",
        paper_id,
        slugify_filename(target_id),
        json.dumps(merged, ensure_ascii=False, indent=2),
    )
    logger.info("done output=%s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_id", required=True)
    parser.add_argument("--target_id", required=True)
    args = parser.parse_args()
    run_matcher(args.paper_id, args.target_id)


if __name__ == "__main__":
    main()
