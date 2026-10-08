"""Strict run isolation and atomic artifact writes for Step 8/Step 9.

Fresh runs never overwrite an existing artifact. Interrupted runs may continue
only through an explicit resume whose immutable input fingerprint matches the
original run manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


RUN_MANIFEST_VERSION = "materials-literature-clean-run/v1"


class CleanRunRequiredError(RuntimeError):
    """Raised when a fresh run would reuse or overwrite existing artifacts."""


class RunIdentityError(RuntimeError):
    """Raised when a resume or write does not belong to the active run."""


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def path_digest(raw_path: str | Path) -> dict[str, Any]:
    """Return a deterministic identity for a file, directory, or missing path."""
    path = Path(raw_path).expanduser().resolve(strict=False)
    if not path.exists():
        return {"path": str(path), "kind": "missing"}
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return {
            "path": str(path),
            "kind": "file",
            "size": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    if path.is_dir():
        entries = []
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            child_digest = path_digest(child)
            child_digest["path"] = child.relative_to(path).as_posix()
            entries.append(child_digest)
        return {
            "path": str(path),
            "kind": "directory",
            "entry_count": len(entries),
            "sha256": _sha256_bytes(_json_text(entries).encode("utf-8")),
        }
    return {"path": str(path), "kind": "other"}


def build_input_identity(
    pipeline: str,
    immutable_inputs: dict[str, Any],
    input_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    path_inputs = [path_digest(path) for path in input_paths if str(path or "").strip()]
    contract = {
        "pipeline": str(pipeline),
        "immutable_inputs": immutable_inputs,
        "input_paths": path_inputs,
    }
    return {
        "fingerprint": _sha256_bytes(_json_text(contract).encode("utf-8")),
        "contract": contract,
    }


def manifest_path_for_output(output_path: str | Path) -> Path:
    output = Path(output_path)
    return output.with_suffix(output.suffix + ".run.json")


def _unique_paths(paths: Iterable[str | Path]) -> list[Path]:
    unique = []
    seen = set()
    for raw_path in paths:
        if not str(raw_path or "").strip():
            continue
        path = Path(raw_path).expanduser().resolve(strict=False)
        key = os.path.normcase(str(path))
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _atomic_write_text(path: Path, text: str, run_id: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _temporary_sibling_path(path)
    tmp_path.write_text(text, encoding="utf-8")
    try:
        for attempt in range(8):
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _temporary_sibling_path(target: Path) -> Path:
    """Keep atomic-write temp names short enough for legacy Windows paths."""
    return target.parent / f".tmp-{uuid.uuid4().hex}"


def reserve_fresh_run(
    *,
    pipeline: str,
    output_path: str | Path,
    input_identity: dict[str, Any],
    artifact_paths: Iterable[str | Path],
) -> dict[str, Any]:
    """Atomically reserve a fresh output namespace and reject every stale target."""
    output = Path(output_path).expanduser().resolve(strict=False)
    manifest_path = manifest_path_for_output(output).resolve(strict=False)
    targets = _unique_paths([*artifact_paths, output, manifest_path])
    stale = [str(path) for path in targets if path.exists()]
    if stale:
        formatted = "\n- ".join(stale)
        raise CleanRunRequiredError(
            "Fresh run refused because its output namespace is not clean. "
            "Use new output/checkpoint/run-directory paths, or explicitly resume the matching state."
            f"\n- {formatted}"
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    identity = {
        "manifest_version": RUN_MANIFEST_VERSION,
        "run_id": run_id,
        "pipeline": str(pipeline),
        "input_fingerprint": str(input_identity["fingerprint"]),
        "manifest_path": str(manifest_path),
        "output_path": str(output),
        "artifact_paths": [str(path) for path in targets if path != manifest_path],
        "created_at": now,
    }
    manifest = {
        **identity,
        "status": "running",
        "updated_at": now,
        "input_contract": input_identity.get("contract", {}),
    }
    try:
        with manifest_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, ensure_ascii=False, indent=2))
    except FileExistsError as exc:
        raise CleanRunRequiredError(
            f"Fresh run refused because another run already reserved {manifest_path}."
        ) from exc
    return identity


def load_manifest(run_identity: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(str(run_identity.get("manifest_path") or ""))
    if not manifest_path.exists():
        raise RunIdentityError(f"Run manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunIdentityError(f"Run manifest is unreadable: {manifest_path}") from exc
    for key in ("run_id", "pipeline", "input_fingerprint", "output_path"):
        if str(manifest.get(key) or "") != str(run_identity.get(key) or ""):
            raise RunIdentityError(f"Run identity mismatch for {key}: {manifest_path}")
    return manifest


def validate_resume_identity(
    snapshot: dict[str, Any],
    *,
    pipeline: str,
    input_identity: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    identity = snapshot.get("run_identity")
    if not isinstance(identity, dict):
        raise RunIdentityError(
            "Resume refused because the snapshot predates clean-run identity tracking. "
            "Start a fresh run with new output paths."
        )
    expected_output = str(Path(output_path).expanduser().resolve(strict=False))
    checks = {
        "pipeline": str(pipeline),
        "input_fingerprint": str(input_identity["fingerprint"]),
        "output_path": expected_output,
    }
    for key, expected in checks.items():
        if str(identity.get(key) or "") != expected:
            raise RunIdentityError(
                f"Resume refused: {key} differs from the run that produced the snapshot."
            )
    manifest = load_manifest(identity)
    if manifest.get("status") not in {"running", "interrupted", "needs_review"}:
        raise RunIdentityError(
            f"Resume refused because run {identity.get('run_id')} is {manifest.get('status')}."
        )
    return identity


def assert_active_run(run_identity: dict[str, Any] | None) -> None:
    if not run_identity:
        return
    manifest = load_manifest(run_identity)
    if manifest.get("status") not in {"running", "interrupted", "needs_review"}:
        raise RunIdentityError(
            f"Run {run_identity.get('run_id')} is not writable ({manifest.get('status')})."
        )


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    run_identity: dict[str, Any] | None = None,
) -> None:
    assert_active_run(run_identity)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _temporary_sibling_path(target)
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            # Stream large checkpoints instead of allocating a second full JSON
            # representation in memory before the atomic replacement.
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(8):
            try:
                os.replace(tmp_path, target)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_write_text(
    path: str | Path,
    payload: str,
    *,
    run_identity: dict[str, Any] | None = None,
) -> None:
    assert_active_run(run_identity)
    _atomic_write_text(
        Path(path),
        str(payload),
        str((run_identity or {}).get("run_id") or ""),
    )


def update_run_status(
    run_identity: dict[str, Any] | None,
    status: str,
    **details: Any,
) -> None:
    if not run_identity:
        return
    manifest = load_manifest(run_identity)
    manifest["status"] = str(status)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest.update(details)
    _atomic_write_text(
        Path(run_identity["manifest_path"]),
        json.dumps(manifest, ensure_ascii=False, indent=2),
        str(run_identity.get("run_id") or ""),
    )
