from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import run_knowledge_augmented_step8 as workflow
import run_three_version_parallel_step8 as runner
import task_domain_knowledge as knowledge


def test_automatic_workflow_stops_at_incomplete_research(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["workflow", "--repo-root", str(tmp_path), "--corpus-dir", str(tmp_path),
                                      "--metadata-dir", str(tmp_path), "--output-root", str(tmp_path / "campaign"),
                                      "--research-output", str(tmp_path / "research")])
    monkeypatch.setattr(runner, "load_runtime_credentials", lambda *a: {"CODE_AGENT_API_KEY": "unused", "CODE_AGENT_BASE_URL": "unused", "CODE_AGENT_MODEL": "unused"})
    monkeypatch.setattr(workflow, "QiniuModelClient", lambda **k: object())
    monkeypatch.setattr(knowledge, "research", lambda *a, **k: {"status": "research_incomplete"})
    monkeypatch.setattr(runner, "prepare_campaign", lambda *a: pytest.fail("Incomplete knowledge must not silently launch Step8"))
    assert workflow.main() == 2
    assert not (tmp_path / "campaign").exists()


def test_step8_command_rejects_changed_knowledge(tmp_path):
    root = tmp_path / "version"
    (root / "inputs").mkdir(parents=True)
    path = root / "inputs" / "knowledge.json"
    path.write_text("{}")
    record = {"root": str(root), "workspace": str(root / "workspace"), "corpus_paths": [],
              "version_id": "v3_care_protocol", "advice_path": "advice", "structured_protocol": True,
              "domain_knowledge": {"path": str(path), "sha256": "wrong"}}
    with pytest.raises(ValueError, match="Frozen domain knowledge"):
        runner.build_step8_command(sys.executable, record, "model", "endpoint")


def test_step8_command_passes_frozen_knowledge_flag(tmp_path):
    root = tmp_path / "version"
    (root / "inputs").mkdir(parents=True)
    path = root / "inputs" / "knowledge.json"
    path.write_text("{}")
    record = {"root": str(root), "workspace": str(root / "workspace"), "corpus_paths": [],
              "version_id": "v3_care_protocol", "advice_path": "advice", "structured_protocol": True,
              "domain_knowledge": {"path": str(path), "sha256": knowledge.sha(path.read_bytes())}}
    command = runner.build_step8_command(sys.executable, record, "model", "endpoint")
    assert command[command.index("--domain-knowledge-pack") + 1] == str(path.resolve())


def test_task_rejection_precedes_campaign_creation(tmp_path):
    from argparse import Namespace
    task = {"database_goal": "wrong", "discipline": "wrong", "query_requirements": ["wrong"]}
    path = tmp_path / "pack.json"
    path.write_text(json.dumps({"version": knowledge.VERSION, "task": task,
                                "task_sha256": knowledge.sha(json.dumps(task, sort_keys=True).encode())}))
    output = tmp_path / "campaign"
    with pytest.raises(ValueError, match="another task"):
        runner.prepare_campaign(Namespace(domain_knowledge_pack=str(path), output_root=str(output)))
    assert not output.exists()
