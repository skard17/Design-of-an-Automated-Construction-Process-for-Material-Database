from __future__ import annotations

import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


@dataclass
class Issue:
    level: str
    file: str
    detail: str


@dataclass
class FileProcessResult:
    output_written: bool
    mapped_methods: int
    skipped_due_to_empty_list: bool


@dataclass
class RunStats:
    status: str
    started_at: str
    ended_at: str
    duration_seconds: float
    inputs_dir: str
    outputs_dir: str
    mapping_path: str
    total_input_files: int
    output_files_written: int
    files_skipped_empty_top_list: int
    mapped_methods_total: int
    issues_total: int
    issues_fail: int
    issues_warn: int
    issues_skip: int
    check_list_generated: bool


def load_mapping(mapping_path: Path) -> dict[str, str]:
    # Load mapping via PyYAML when available; otherwise use a simple line parser.
    text = mapping_path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("mapping file is not a YAML object")
        mapping = {}
        for key, value in data.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("mapping keys/values must be strings")
            mapping[key] = value
        if not mapping:
            raise ValueError("mapping file is empty")
        return mapping
    except ModuleNotFoundError:
        mapping = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(f"invalid mapping line: {raw_line}")
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key or not value:
                raise ValueError(f"invalid key/value in mapping line: {raw_line}")
            mapping[key] = value
        if not mapping:
            raise ValueError("mapping file is empty")
        return mapping


def write_output_json(output_file: Path, payload: object) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process_json_file(
    input_file: Path,
    output_file: Path,
    mapping: dict[str, str],
    issues: list[Issue],
) -> FileProcessResult:
    rel_path = input_file.as_posix()
    mapped_methods = 0

    # Parse and validate top-level JSON.
    try:
        payload = json.loads(input_file.read_text(encoding="utf-8"))
    except Exception as error:
        issues.append(Issue("FAIL", rel_path, f"invalid JSON: {error}"))
        return FileProcessResult(output_written=False, mapped_methods=0, skipped_due_to_empty_list=False)

    if not isinstance(payload, dict):
        issues.append(Issue("FAIL", rel_path, "top-level JSON is not an object"))
        return FileProcessResult(output_written=False, mapped_methods=0, skipped_due_to_empty_list=False)

    has_processable_list = False
    has_non_empty_list = False
    skipped_due_to_empty_list = False

    # Process each top-level list and map `method` values when possible.
    for top_key, top_value in payload.items():
        if not isinstance(top_value, list):
            issues.append(Issue("FAIL", rel_path, f"top-level key '{top_key}' is not a list"))
            continue

        has_processable_list = True
        if not top_value:
            issues.append(Issue("SKIP", rel_path, f"top-level key '{top_key}' contains an empty list"))
            skipped_due_to_empty_list = True
            continue

        has_non_empty_list = True
        for idx, item in enumerate(top_value):
            location = f"{top_key}[{idx}]"
            if not isinstance(item, dict):
                issues.append(Issue("FAIL", rel_path, f"{location} is not an object"))
                continue

            if "method" not in item:
                issues.append(Issue("WARN", rel_path, f"{location} missing 'method' key (left unchanged)"))
                continue

            method_value = item["method"]
            if not isinstance(method_value, str):
                issues.append(Issue("FAIL", rel_path, f"{location}.method is not a string"))
                continue

            mapped_value = mapping.get(method_value)
            if mapped_value is None:
                issues.append(Issue("FAIL", rel_path, f"{location}.method='{method_value}' not found in method_mapping.yaml"))
                continue

            item["method"] = mapped_value
            mapped_methods += 1

    if not has_processable_list:
        issues.append(Issue("FAIL", rel_path, "no top-level list found for processing"))
        return FileProcessResult(
            output_written=False,
            mapped_methods=mapped_methods,
            skipped_due_to_empty_list=False,
        )

    # Keep files with empty lists in outputs so output coverage matches inputs.
    if not has_non_empty_list:
        write_output_json(output_file, payload)
        return FileProcessResult(
            output_written=True,
            mapped_methods=mapped_methods,
            skipped_due_to_empty_list=skipped_due_to_empty_list,
        )

    write_output_json(output_file, payload)
    return FileProcessResult(
        output_written=True,
        mapped_methods=mapped_methods,
        skipped_due_to_empty_list=False,
    )


def count_issues(issues: list[Issue]) -> tuple[int, int, int]:
    fail_count = sum(1 for issue in issues if issue.level == "FAIL")
    warn_count = sum(1 for issue in issues if issue.level == "WARN")
    skip_count = sum(1 for issue in issues if issue.level == "SKIP")
    return fail_count, warn_count, skip_count


def remove_file_if_exists(path: Path | None) -> None:
    if path is not None and path.exists():
        path.unlink()


def write_check_list(check_path: Path, issues: list[Issue], legacy_check_path: Path | None = None) -> bool:
    # Write check_list at the project root and clean stale legacy file if present.

    if not issues:
        remove_file_if_exists(check_path)
        remove_file_if_exists(legacy_check_path)
        return False

    lines = ["# Processing issues", ""]
    for issue in issues:
        lines.append(f"[{issue.level}] {issue.file} :: {issue.detail}")

    check_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if legacy_check_path != check_path:
        remove_file_if_exists(legacy_check_path)
    return True


def write_run_log(logs_dir: Path, stats: RunStats) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"method_mapping_{timestamp}.log"

    lines = [
        "# Method mapping run summary",
        f"status: {stats.status}",
        f"started_at: {stats.started_at}",
        f"ended_at: {stats.ended_at}",
        f"duration_seconds: {stats.duration_seconds:.3f}",
        f"inputs_dir: {stats.inputs_dir}",
        f"outputs_dir: {stats.outputs_dir}",
        f"mapping_path: {stats.mapping_path}",
        f"total_input_files: {stats.total_input_files}",
        f"output_files_written: {stats.output_files_written}",
        f"files_skipped_empty_top_list: {stats.files_skipped_empty_top_list}",
        f"mapped_methods_total: {stats.mapped_methods_total}",
        f"issues_total: {stats.issues_total}",
        f"issues_fail: {stats.issues_fail}",
        f"issues_warn: {stats.issues_warn}",
        f"issues_skip: {stats.issues_skip}",
        f"check_list_generated: {stats.check_list_generated}",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def write_failure_run_log(
    logs_dir: Path,
    started: datetime,
    ended: datetime,
    duration_seconds: float,
    inputs_dir: Path,
    outputs_dir: Path,
    mapping_path: Path,
    error: Exception,
) -> Path:
    failure_stats = RunStats(
        status="failed",
        started_at=started.isoformat(timespec="seconds"),
        ended_at=ended.isoformat(timespec="seconds"),
        duration_seconds=duration_seconds,
        inputs_dir=str(inputs_dir),
        outputs_dir=str(outputs_dir),
        mapping_path=str(mapping_path),
        total_input_files=0,
        output_files_written=0,
        files_skipped_empty_top_list=0,
        mapped_methods_total=0,
        issues_total=1,
        issues_fail=1,
        issues_warn=0,
        issues_skip=0,
        check_list_generated=False,
    )
    log_path = write_run_log(logs_dir, failure_stats)
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(f"error: {type(error).__name__}: {error}\n")
    return log_path


def run(inputs_dir: Path, outputs_dir: Path, mapping_path: Path, logs_dir: Path) -> Path:
    # Orchestrate one full batch run and collect summary statistics.
    started = datetime.now()
    t0 = perf_counter()

    if not inputs_dir.exists():
        raise FileNotFoundError(f"inputs directory not found: {inputs_dir}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"mapping file not found: {mapping_path}")

    mapping = load_mapping(mapping_path)
    issues: list[Issue] = []
    output_files_written = 0
    files_skipped_empty_top_list = 0
    mapped_methods_total = 0

    json_files = sorted(path for path in inputs_dir.rglob("*.json") if path.is_file())
    for input_file in json_files:
        relative = input_file.relative_to(inputs_dir)
        output_file = outputs_dir / relative
        result = process_json_file(input_file, output_file, mapping, issues)
        if result.output_written:
            output_files_written += 1
        if result.skipped_due_to_empty_list:
            files_skipped_empty_top_list += 1
        mapped_methods_total += result.mapped_methods

    check_path = outputs_dir.parent / "check_list.txt"
    legacy_check_path = outputs_dir / "check_list.txt"
    check_list_generated = write_check_list(check_path, issues, legacy_check_path=legacy_check_path)
    ended = datetime.now()

    fail_count, warn_count, skip_count = count_issues(issues)

    stats = RunStats(
        status="success",
        started_at=started.isoformat(timespec="seconds"),
        ended_at=ended.isoformat(timespec="seconds"),
        duration_seconds=perf_counter() - t0,
        inputs_dir=str(inputs_dir),
        outputs_dir=str(outputs_dir),
        mapping_path=str(mapping_path),
        total_input_files=len(json_files),
        output_files_written=output_files_written,
        files_skipped_empty_top_list=files_skipped_empty_top_list,
        mapped_methods_total=mapped_methods_total,
        issues_total=len(issues),
        issues_fail=fail_count,
        issues_warn=warn_count,
        issues_skip=skip_count,
        check_list_generated=check_list_generated,
    )
    return write_run_log(logs_dir, stats)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    inputs_dir = base_dir / "inputs"
    outputs_dir = base_dir / "outputs"
    mapping_path = base_dir / "method_mapping.yaml"
    logs_dir = base_dir / "logs"

    started = datetime.now()
    t0 = perf_counter()
    try:
        log_path = run(
            inputs_dir=inputs_dir,
            outputs_dir=outputs_dir,
            mapping_path=mapping_path,
            logs_dir=logs_dir,
        )
        print(f"Run summary log written to: {log_path}")
    except Exception as error:
        logs_dir.mkdir(parents=True, exist_ok=True)
        ended = datetime.now()
        duration = perf_counter() - t0
        write_failure_run_log(
            logs_dir=logs_dir,
            started=started,
            ended=ended,
            duration_seconds=duration,
            inputs_dir=inputs_dir,
            outputs_dir=outputs_dir,
            mapping_path=mapping_path,
            error=error,
        )
        raise


if __name__ == "__main__":
    main()
