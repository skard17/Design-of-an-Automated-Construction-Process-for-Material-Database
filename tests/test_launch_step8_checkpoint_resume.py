import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from launch_step8_checkpoint_resume import (  # noqa: E402
    build_resume_command,
    finalize_process_record,
)


def test_resume_command_contains_no_api_key_and_keeps_supervisor_limits(tmp_path):
    command = build_resume_command(
        "python.exe",
        tmp_path / "workspace",
        tmp_path / "artifacts" / "step8.state.json",
        model="deepseek/deepseek-v4-pro",
        base_url="https://api.qnaigc.com/v1",
        thread_id="v1-resume",
    )

    assert "--api-key" not in command
    assert "10" in command
    assert "60" in command
    assert command[command.index("--max-retries") + 1] == "10"
    assert "384000" not in command


def test_terminal_record_requires_success_output_with_run_identity(tmp_path):
    state_path = tmp_path / "step8.state.json"
    output_path = tmp_path / "step8.json"
    state_path.write_text(
        '{"status":"success","next_node":"write_output"}', encoding="utf-8"
    )
    output_path.write_text(
        '{"status":"success","run_identity":{"run_id":"fresh-run"}}',
        encoding="utf-8",
    )

    record, passed = finalize_process_record(
        {"status": "running"},
        return_code=0,
        state_path=state_path,
        output_path=output_path,
    )

    assert passed is True
    assert record["status"] == "passed"
    assert record["checkpoint_status"] == "success"
    assert record["step8_run_id"] == "fresh-run"


def test_terminal_record_exposes_checkpoint_when_output_is_missing(tmp_path):
    state_path = tmp_path / "step8.state.json"
    state_path.write_text(
        '{"status":"awaiting_supervisor_decision","next_node":"supervisor_router"}',
        encoding="utf-8",
    )

    record, passed = finalize_process_record(
        {"status": "running"},
        return_code=1,
        state_path=state_path,
        output_path=tmp_path / "missing.json",
    )

    assert passed is False
    assert record["status"] == "needs_review"
    assert record["checkpoint_status"] == "awaiting_supervisor_decision"
    assert record["checkpoint_next_node"] == "supervisor_router"
