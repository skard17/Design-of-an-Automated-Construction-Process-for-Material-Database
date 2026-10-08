from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import run_three_version_parallel_step8 as runner  # noqa: E402


def make_papers(root: Path, count: int) -> list[Path]:
    papers = []
    for index in range(count):
        path = root / f"paper_{index:02d}" / f"paper_{index:02d}.md"
        path.parent.mkdir(parents=True)
        path.write_text("# Paper\n" + "evidence " * 200, encoding="utf-8")
        papers.append(path)
    return papers


def test_discover_corpus_requires_exact_count_and_unique_stems(tmp_path: Path) -> None:
    make_papers(tmp_path, 3)
    assert len(runner.discover_corpus(tmp_path, 3)) == 3
    try:
        runner.discover_corpus(tmp_path, 4)
    except ValueError as exc:
        assert "expected exactly 4" in str(exc)
    else:
        raise AssertionError("count mismatch must fail")


def test_blind_advice_never_contains_gold_and_protocol_validates() -> None:
    plain = runner.blind_advice_payload(False, care=False)
    care = runner.blind_advice_payload(False, care=True)
    structured = runner.blind_advice_payload(True, care=True)
    assert plain["forbidden_information_used"] is False
    assert structured["forbidden_information_used"] is False
    assert structured["protocol_validation"]["valid"] is True
    assert len(structured["schema_concepts"]) >= 1
    assert "counterfactual_queries" not in plain
    assert len(care["counterfactual_queries"]) >= 5
    assert structured["counterfactual_queries"] == care["counterfactual_queries"]
    assert len({item["query_id"] for item in care["counterfactual_queries"]}) == len(
        care["counterfactual_queries"]
    )
    serialized = json.dumps([plain, care, structured], ensure_ascii=False).casefold()
    assert "schema2_store" not in serialized
    assert "section5.theoretical_keywords" not in serialized


def test_select_versions_supports_isolated_single_version() -> None:
    selected = runner.select_versions(["v2_care"])
    assert [item["version_id"] for item in selected] == ["v2_care"]
    with pytest.raises(ValueError, match="duplicates"):
        runner.select_versions(["v2_care", "v2_care"])


def test_named_api_key_loader_returns_only_requested_entry(tmp_path: Path) -> None:
    key_file = tmp_path / "keys.txt"
    key_file.write_text("key1=alpha-secret\nkey2=beta-secret\n", encoding="utf-8")
    assert runner.load_named_api_key(key_file, "key2") == "beta-secret"
    with pytest.raises(ValueError, match="exactly one"):
        runner.load_named_api_key(key_file, "key3")


def test_step8_commands_are_version_local_and_never_resume(tmp_path: Path) -> None:
    records = []
    for version in runner.VERSIONS:
        version_root = tmp_path / version["version_id"]
        workspace = version_root / "workspace"
        (workspace / "code").mkdir(parents=True)
        (workspace / "tests" / "fixtures").mkdir(parents=True)
        corpus = version_root / "inputs" / "corpus"
        corpus.mkdir(parents=True)
        paper = corpus / "paper.md"
        paper.write_text("# paper", encoding="utf-8")
        advice_path = version_root / "inputs" / "expert_round1.json"
        advice_path.parent.mkdir(parents=True, exist_ok=True)
        advice_path.write_text("{}", encoding="utf-8")
        advice = str(advice_path)
        records.append(
            {
                "version_id": version["version_id"],
                "root": str(version_root),
                "workspace": str(workspace),
                "corpus_paths": [str(paper)],
                "advice_path": advice,
                "structured_protocol": version["structured_protocol"],
            }
        )

    commands = [
        runner.build_step8_command("python", record, "model", "https://example.test/v1")
        for record in records
    ]
    output_paths = []
    checkpoint_paths = []
    for command, record in zip(commands, records):
        assert "--resume-from-state" not in command
        assert "--api-key" not in command
        assert "--human-advice-path" in command
        assert "--skip-human-expert-review" not in command
        assert command[command.index("--schema-inspection-interval") + 1] == "10"
        assert (
            command[command.index("--repeated-blocker-inspection-threshold") + 1]
            == "3"
        )
        output_paths.append(command[command.index("--output") + 1])
        checkpoint_paths.append(command[command.index("--checkpoint-output") + 1])
        assert all(
            sibling["version_id"] not in " ".join(command)
            for sibling in records
            if sibling["version_id"] != record["version_id"]
        )
    assert len(set(output_paths)) == 3
    assert len(set(checkpoint_paths)) == 3

    retry_root = tmp_path / "v2_retry"
    retry_workspace = retry_root / "workspace"
    (retry_workspace / "code").mkdir(parents=True)
    (retry_workspace / "tests" / "fixtures").mkdir(parents=True)
    retry_inputs = retry_root / "inputs"
    retry_inputs.mkdir(parents=True)
    retry_advice = retry_inputs / "expert_round1.json"
    retry_advice.write_text("{}", encoding="utf-8")
    retry_record = {
        **records[1],
        "root": str(retry_root),
        "workspace": str(retry_workspace),
        "advice_path": str(retry_advice),
    }
    retry_command = runner.build_step8_command(
        "python",
        retry_record,
        "deepseek/deepseek-v4-pro",
        "https://example.test/v1",
        max_supervisor_retries=10,
        max_total_supervisor_repairs=60,
    )
    assert retry_command[retry_command.index("--max-supervisor-retries") + 1] == "10"
    assert retry_command[retry_command.index("--max-total-supervisor-repairs") + 1] == "60"
    assert retry_command[retry_command.index("--schema-inspection-interval") + 1] == "10"
    assert (
        retry_command[
            retry_command.index("--repeated-blocker-inspection-threshold") + 1
        ]
        == "3"
    )


def test_all_three_versions_copy_the_same_common_schema_guard(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "code").mkdir(parents=True)
    (repo_root / "reference_pipelines" / "pure_extraction_pipeline").mkdir(
        parents=True
    )
    (repo_root / "tests" / "fixtures").mkdir(parents=True)
    marker = "schema_inspection_interval"
    (repo_root / "code" / "section_design_langgraph_human_gate.py").write_text(
        marker, encoding="utf-8"
    )
    (repo_root / "tests" / "fixtures" / "no_reference_field_catalog.txt").write_text(
        "none", encoding="utf-8"
    )

    copied = []
    for version in runner.VERSIONS:
        workspace = tmp_path / version["version_id"] / "workspace"
        runner.copy_workspace(repo_root, workspace)
        copied_text = (
            workspace / "code" / "section_design_langgraph_human_gate.py"
        ).read_text(encoding="utf-8")
        copied.append(copied_text)

    assert copied == [marker, marker, marker]


def test_prepared_campaign_rejects_existing_launch_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    records = []
    hashes = set()
    for version in runner.VERSIONS:
        version_root = root / version["version_id"]
        workspace = version_root / "workspace"
        workspace.mkdir(parents=True)
        corpus = version_root / "inputs" / "corpus"
        corpus.mkdir(parents=True)
        paper = corpus / "paper.md"
        paper.write_text("# paper\n" + "evidence " * 200, encoding="utf-8")
        advice = version_root / "inputs" / "expert_round1.json"
        advice.write_text("{}", encoding="utf-8")
        inventory = runner.corpus_inventory([paper])
        digest = runner.stable_digest(
            [
                {
                    "paper_id": item["paper_id"],
                    "bytes": item["bytes"],
                    "sha256": item["sha256"],
                }
                for item in inventory
            ]
        )
        hashes.add(digest)
        records.append(
            {
                "version_id": version["version_id"],
                "root": str(version_root),
                "workspace": str(workspace),
                "corpus_paths": [str(paper)],
                "advice_path": str(advice),
            }
        )
    assert len(hashes) == 1
    runner.atomic_write_json(
        root / "RUN_STATUS.json",
        {
            "status": "prepared",
            "paper_count": 1,
            "corpus_aggregate_sha256": next(iter(hashes)),
            "versions": records,
        },
    )
    assert runner.load_prepared_campaign(root)["status"] == "prepared"
    (root / "v1_baseline" / "artifacts").mkdir()
    with pytest.raises(runner.run_artifact_guard.CleanRunRequiredError):
        runner.load_prepared_campaign(root)
