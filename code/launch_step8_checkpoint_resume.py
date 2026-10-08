#!/usr/bin/env python3
"""Launch one isolated Step8 checkpoint fork without exposing credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import qiniu_model_client
import run_artifact_guard
from run_three_version_parallel_step8 import load_named_api_key


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize_process_record(
    process_record: dict,
    *,
    return_code: int,
    state_path: Path,
    output_path: Path,
) -> tuple[dict, bool]:
    state_status = "missing_state"
    next_node = ""
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state_status = str(state.get("status") or "unknown")
            next_node = str(state.get("next_node") or "")
        except (OSError, json.JSONDecodeError):
            state_status = "invalid_state_json"

    output_status = "missing_output"
    output_run_id = ""
    if output_path.exists():
        try:
            output = json.loads(output_path.read_text(encoding="utf-8"))
            output_status = str(output.get("status") or "unknown")
            output_run_id = str((output.get("run_identity") or {}).get("run_id") or "")
        except (OSError, json.JSONDecodeError):
            output_status = "invalid_output_json"

    passed = return_code == 0 and output_status == "success" and bool(output_run_id)
    process_record.update(
        {
            "status": "passed" if passed else "needs_review",
            "exit_code": return_code,
            "finished_at": utc_now(),
            "checkpoint_status": state_status,
            "checkpoint_next_node": next_node,
            "step8_result_status": output_status,
            "step8_run_id": output_run_id,
        }
    )
    return process_record, passed


def build_resume_command(
    python_executable: str,
    workspace: Path,
    state_path: Path,
    *,
    model: str,
    base_url: str,
    thread_id: str,
) -> list[str]:
    return [
        python_executable,
        str(workspace / "code" / "section_design_langgraph_human_gate.py"),
        "--resume-from-state",
        str(state_path),
        "--model",
        model,
        "--base-url",
        base_url,
        "--llm-backend",
        "qiniu",
        "--request-timeout",
        "1200",
        "--max-retries",
        "10",
        "--max-supervisor-retries",
        "10",
        "--max-total-supervisor-repairs",
        "60",
        "--thread-id",
        thread_id,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--credential-key-file", required=True)
    parser.add_argument("--credential-key-label", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--thread-id", default="step8-checkpoint-resume")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for the child and atomically persist its terminal process state.",
    )
    args = parser.parse_args()

    state_path = Path(args.state).resolve(strict=True)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    fork = state.get("checkpoint_fork") or {}
    if fork.get("contract_version") != "step8-checkpoint-fork/v1":
        raise ValueError("state is not an isolated Step8 checkpoint fork")
    run_identity = state.get("run_identity") or {}
    manifest = run_artifact_guard.load_manifest(run_identity)
    if manifest.get("status") != "interrupted":
        raise ValueError(
            f"checkpoint fork is not launchable from status={manifest.get('status')}"
        )

    version_root = state_path.parent.parent
    workspace = version_root / "workspace"
    if not (workspace / "code" / "section_design_langgraph_human_gate.py").is_file():
        raise FileNotFoundError("isolated workspace is incomplete")
    output_path = Path(state["args"]["output"])
    if output_path.exists():
        raise run_artifact_guard.CleanRunRequiredError(
            f"checkpoint fork already has an output: {output_path}"
        )
    process_path = version_root / "PROCESS.json"
    if process_path.exists():
        raise run_artifact_guard.CleanRunRequiredError(
            f"checkpoint fork already has a process record: {process_path}"
        )

    api_key = load_named_api_key(
        Path(args.credential_key_file).resolve(strict=True),
        args.credential_key_label,
    )
    model = str(state["args"]["model"])
    base_url = str(state["args"]["base_url"])
    probe = qiniu_model_client.QiniuModelClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    selected_model = probe.select_available_model()
    if selected_model != model:
        raise ValueError(
            f"credential does not expose the checkpoint model: {model}"
        )

    log_dir = version_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = log_dir / "step8.stdout.log"
    stderr_path = log_dir / "step8.stderr.log"
    command = build_resume_command(
        args.python,
        workspace,
        state_path,
        model=model,
        base_url=base_url,
        thread_id=args.thread_id,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "CODE_AGENT_BASE_URL": base_url,
            "CODE_AGENT_MODEL": model,
            "CODE_AGENT_API_KEY": api_key,
            "SECTION_AGENT_BASE_URL": base_url,
            "SECTION_AGENT_MODEL": model,
            "SECTION_AGENT_API_KEY": api_key,
            "SECTION_AGENT_LLM_BACKEND": "qiniu",
        }
    )
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )
    with stdout_path.open("x", encoding="utf-8", buffering=1) as stdout_handle, stderr_path.open(
        "x", encoding="utf-8", buffering=1
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=str(workspace),
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=creationflags,
        )

    process_record = {
        "status": "running",
        "pid": process.pid,
        "started_at": utc_now(),
        "state": str(state_path),
        "state_sha256_at_launch": sha256_file(state_path),
        "run_id": run_identity.get("run_id"),
        "source_run_id": fork.get("source_run_id"),
        "restart_node": fork.get("restart_node"),
        "model": model,
        "backend": "qiniu_direct",
        "credential_label": args.credential_key_label,
        "credential_values_persisted": False,
        "max_supervisor_retries": 10,
        "max_total_supervisor_repairs": 60,
        "max_api_retries": 10,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    with process_path.open("x", encoding="utf-8") as handle:
        json.dump(process_record, handle, ensure_ascii=False, indent=2)

    time.sleep(1)
    return_code = process.poll()
    if return_code is not None:
        process_record, _ = finalize_process_record(
            process_record,
            return_code=return_code,
            state_path=state_path,
            output_path=output_path,
        )
        process_record["launch_failure"] = "immediate_exit"
        run_artifact_guard.atomic_write_json(process_path, process_record)
        raise RuntimeError(
            f"checkpoint resume exited immediately with code {return_code}; see {stderr_path}"
        )
    if args.wait:
        return_code = process.wait()
        process_record, passed = finalize_process_record(
            process_record,
            return_code=return_code,
            state_path=state_path,
            output_path=output_path,
        )
        run_artifact_guard.atomic_write_json(process_path, process_record)
        print(
            json.dumps(
                {
                    "status": process_record["status"],
                    "pid": process.pid,
                    "exit_code": return_code,
                    "checkpoint_status": process_record["checkpoint_status"],
                    "step8_result_status": process_record["step8_result_status"],
                },
                ensure_ascii=False,
            )
        )
        return 0 if passed else 2
    print(
        json.dumps(
            {
                "status": "running",
                "pid": process.pid,
                "run_id": run_identity.get("run_id"),
                "restart_node": fork.get("restart_node"),
                "credential_label": args.credential_key_label,
                "credential_values_persisted": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
