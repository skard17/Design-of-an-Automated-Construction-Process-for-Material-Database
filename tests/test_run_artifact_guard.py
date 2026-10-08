import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import run_artifact_guard as guard  # noqa: E402


def input_identity(tmp_path, value="one"):
    source = tmp_path / "source.md"
    source.write_text(value, encoding="utf-8")
    return guard.build_input_identity(
        "test_pipeline",
        {"goal": "test"},
        [source],
    )


def test_fresh_run_reservation_is_exclusive(tmp_path):
    output = tmp_path / "result.json"
    identity = input_identity(tmp_path)

    run = guard.reserve_fresh_run(
        pipeline="test_pipeline",
        output_path=output,
        input_identity=identity,
        artifact_paths=[output, tmp_path / "result.state.json"],
    )

    manifest = json.loads(Path(run["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "running"
    assert manifest["input_fingerprint"] == identity["fingerprint"]
    with pytest.raises(guard.CleanRunRequiredError):
        guard.reserve_fresh_run(
            pipeline="test_pipeline",
            output_path=output,
            input_identity=identity,
            artifact_paths=[output],
        )


def test_fresh_run_refuses_and_preserves_existing_artifact(tmp_path):
    output = tmp_path / "result.json"
    output.write_text("old result", encoding="utf-8")

    with pytest.raises(guard.CleanRunRequiredError):
        guard.reserve_fresh_run(
            pipeline="test_pipeline",
            output_path=output,
            input_identity=input_identity(tmp_path),
            artifact_paths=[output],
        )

    assert output.read_text(encoding="utf-8") == "old result"


def test_resume_requires_matching_immutable_inputs(tmp_path):
    output = tmp_path / "result.json"
    identity = input_identity(tmp_path, "original")
    run = guard.reserve_fresh_run(
        pipeline="test_pipeline",
        output_path=output,
        input_identity=identity,
        artifact_paths=[output],
    )
    snapshot = {"run_identity": run}

    assert guard.validate_resume_identity(
        snapshot,
        pipeline="test_pipeline",
        input_identity=identity,
        output_path=output,
    )["run_id"] == run["run_id"]

    changed = input_identity(tmp_path, "changed")
    with pytest.raises(guard.RunIdentityError):
        guard.validate_resume_identity(
            snapshot,
            pipeline="test_pipeline",
            input_identity=changed,
            output_path=output,
        )


def test_completed_run_cannot_be_written_or_resumed(tmp_path):
    output = tmp_path / "result.json"
    identity = input_identity(tmp_path)
    run = guard.reserve_fresh_run(
        pipeline="test_pipeline",
        output_path=output,
        input_identity=identity,
        artifact_paths=[output],
    )
    guard.atomic_write_json(output, {"value": 1}, run_identity=run)
    guard.update_run_status(run, "completed")

    with pytest.raises(guard.RunIdentityError):
        guard.atomic_write_json(output, {"value": 2}, run_identity=run)
    with pytest.raises(guard.RunIdentityError):
        guard.validate_resume_identity(
            {"run_identity": run},
            pipeline="test_pipeline",
            input_identity=identity,
            output_path=output,
        )
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}


def test_atomic_text_write_uses_the_active_run(tmp_path):
    output = tmp_path / "result.txt"
    identity = input_identity(tmp_path)
    run = guard.reserve_fresh_run(
        pipeline="test_pipeline",
        output_path=output,
        input_identity=identity,
        artifact_paths=[output],
    )

    guard.atomic_write_text(output, "fresh", run_identity=run)

    assert output.read_text(encoding="utf-8") == "fresh"


def test_atomic_json_write_streams_without_json_dumps(tmp_path, monkeypatch):
    output = tmp_path / "large-state.json"
    payload = {"records": [{"index": index, "value": "x" * 1000} for index in range(200)]}

    monkeypatch.setattr(
        guard.json,
        "dumps",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("atomic JSON writes must not materialize the full document")
        ),
    )

    guard.atomic_write_json(output, payload)

    with output.open("r", encoding="utf-8") as handle:
        assert json.load(handle) == payload


def test_atomic_temp_name_does_not_repeat_long_target_name(tmp_path):
    target = tmp_path / ("very-long-output-name-" * 8 + ".json")

    temporary = guard._temporary_sibling_path(target)

    assert temporary.parent == target.parent
    assert temporary.name.startswith(".tmp-")
    assert target.name not in temporary.name
    assert len(temporary.name) < len(target.name)


def test_atomic_write_retries_transient_permission_error(tmp_path, monkeypatch):
    output = tmp_path / "state.json"
    real_replace = guard.os.replace
    attempts = []

    def flaky_replace(source, target):
        attempts.append((source, target))
        if len(attempts) < 3:
            raise PermissionError("temporarily locked")
        real_replace(source, target)

    monkeypatch.setattr(guard.os, "replace", flaky_replace)
    monkeypatch.setattr(guard.time, "sleep", lambda _seconds: None)

    guard._atomic_write_text(output, '{"revision": 12}', run_id="run")

    assert len(attempts) == 3
    assert output.read_text(encoding="utf-8") == '{"revision": 12}'
    assert list(tmp_path.glob(".tmp-*")) == []


def test_atomic_write_reports_persistent_permission_error_and_cleans_tmp(
    tmp_path, monkeypatch
):
    output = tmp_path / "state.json"
    monkeypatch.setattr(
        guard.os,
        "replace",
        lambda _source, _target: (_ for _ in ()).throw(
            PermissionError("persistently locked")
        ),
    )
    monkeypatch.setattr(guard.time, "sleep", lambda _seconds: None)

    with pytest.raises(PermissionError, match="persistently locked"):
        guard._atomic_write_text(output, "{}", run_id="run")

    assert not output.exists()
    assert list(tmp_path.glob(".tmp-*")) == []
