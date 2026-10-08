import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import run_legacy_targeted_extraction_parallel as runner


def write_legacy_runner(path):
    path.write_text(
        """
def build_prompt(paper_id, targets, evidence):
    return targets[0]["target_id"]

def validate_response(payload, targets):
    facts = payload.get("facts")
    if not isinstance(facts, list):
        raise ValueError("facts required")
    expected = {item["target_id"] for item in targets}
    actual = {item.get("target_id") for item in facts}
    if actual != expected:
        raise ValueError("target mismatch")
    return facts
""".strip(),
        encoding="utf-8",
    )


def args_for(tmp_path, legacy_path, suffix=""):
    return runner.build_parser().parse_args(
        [
            "--legacy-runner",
            str(legacy_path),
            "--targets",
            str(tmp_path / "targets.json"),
            "--pack-dir",
            str(tmp_path / "packs"),
            "--run-dir",
            str(tmp_path / f"runs{suffix}"),
            "--output",
            str(tmp_path / f"predictions{suffix}.json"),
            "--base-url",
            "https://example.invalid",
            "--model",
            "test-model",
            "--max-attempts",
            "1",
            "--workers",
            "1",
            "--max-targets-per-call",
            "1",
        ]
    )


def test_each_attempt_requires_a_fresh_namespace_and_reruns_all_batches(tmp_path, monkeypatch):
    targets = [
        {"paper_id": "p1", "target_id": "t1"},
        {"paper_id": "p1", "target_id": "t2"},
    ]
    (tmp_path / "targets.json").write_text(
        json.dumps({"targets": targets}), encoding="utf-8"
    )
    packs = tmp_path / "packs"
    packs.mkdir()
    (packs / "p1.md").write_text("evidence", encoding="utf-8")
    legacy_path = tmp_path / "legacy.py"
    write_legacy_runner(legacy_path)
    monkeypatch.setenv("CODE_AGENT_API_KEY", "test-key")

    first_calls = []

    def first_chat(**kwargs):
        target_id = kwargs["prompt"]
        first_calls.append(target_id)
        if target_id == "t2":
            raise RuntimeError("provider 504")
        return json.dumps({"facts": [{"target_id": target_id, "value": 1}]})

    monkeypatch.setattr(runner.prompt_agent, "litellm_chat", first_chat)
    first = runner.run(args_for(tmp_path, legacy_path))

    assert first["prediction_count"] == 0
    assert first["failure_count"] == 1
    assert first["failures"][0]["partial_prediction_count"] == 1
    assert first_calls == ["t1", "t2"]

    second_calls = []

    def second_chat(**kwargs):
        target_id = kwargs["prompt"]
        second_calls.append(target_id)
        return json.dumps({"facts": [{"target_id": target_id, "value": 2}]})

    monkeypatch.setattr(runner.prompt_agent, "litellm_chat", second_chat)
    try:
        runner.run(args_for(tmp_path, legacy_path))
    except runner.run_artifact_guard.CleanRunRequiredError:
        pass
    else:
        raise AssertionError("a second run must not reuse the first run's artifacts")

    second = runner.run(args_for(tmp_path, legacy_path, "-fresh"))

    assert second["prediction_count"] == 2
    assert second["failure_count"] == 0
    assert second_calls == ["t1", "t2"]
    assert {item["target_id"] for item in second["facts"]} == {"t1", "t2"}
    assert second["run_identity"]["run_id"] != first["run_identity"]["run_id"]


def test_new_batch_checkpoints_use_short_hashed_paper_directory(tmp_path):
    path = runner.batch_checkpoint_path(
        tmp_path,
        "doi__10.1016_j.aop.2026.170637" * 5,
        [{"target_id": "target-1"}],
    )

    assert path.parent.parent.name == "_b"
    assert len(path.parent.name) == 12
    assert path.parent.exists()
