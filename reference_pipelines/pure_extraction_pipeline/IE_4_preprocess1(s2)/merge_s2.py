from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

# --- Configuration ---

LOG_NAME = "merge_s2"
BASE_DIR = Path(__file__).resolve().parent
MERGE_FIELDS = ("method", "description", "geometry")

S2_1_DIR = BASE_DIR / "inputs" / "s2_1"
S2_2_DIR = BASE_DIR / "inputs" / "s2_2"
OUT_DIR = BASE_DIR / "outputs" / "s2"
LOG_DIR = BASE_DIR / "logs"

# Explicit check execution/report order.
CHECK_SEQUENCE = ("check-1", "check-json", "check-2", "check-3", "check-4")
PRECHECK_NAME = "check-0"
FULL_CHECK_SEQUENCE = (PRECHECK_NAME, *CHECK_SEQUENCE)

CHECK_CONFIG = {
    "check-0": {
        "path": BASE_DIR / "check_0.csv",
        "headers": ["filename", "dataset", "action", "detail"],
    },
    "check-1": {"path": BASE_DIR / "check_1.csv", "headers": ["filename", "description"]},
    "check-json": {"path": BASE_DIR / "check_json.csv", "headers": ["filename", "description"]},
    "check-2": {"path": BASE_DIR / "check_2.csv", "headers": ["filename", "entry_id", "description"]},
    "check-3": {"path": BASE_DIR / "check_3.csv", "headers": ["filename", "entry_id", "missing_field"]},
    "check-4": {
        "path": BASE_DIR / "check_4.csv",
        "headers": ["filename", "entry_id", "only_in_section2_1", "only_in_section2_2"],
    },
}

CheckRows = dict[str, list[list[str]]]


# --- Basic utilities ---

def validate_check_config() -> None:
    """Ensure check sequence and config stay consistent."""
    sequence_set = set(FULL_CHECK_SEQUENCE)
    config_set = set(CHECK_CONFIG.keys())
    if sequence_set != config_set:
        missing_in_config = sorted(sequence_set - config_set)
        missing_in_sequence = sorted(config_set - sequence_set)
        raise ValueError(
            f"CHECK_SEQUENCE/CHECK_CONFIG mismatch: "
            f"missing_in_config={missing_in_config}, missing_in_sequence={missing_in_sequence}"
        )


def create_logger(log_dir: Path) -> logging.Logger:
    logger = logging.getLogger(LOG_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{LOG_NAME}_{time.strftime('%Y%m%d_%H%M%S')}.log"

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    logger.info("log_file=%s", log_path)
    return logger


def collect_json_files(folder: Path) -> dict[str, Path]:
    if not folder.exists():
        return {}
    return {path.name: path for path in sorted(folder.glob("*.json")) if path.is_file()}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def ensure_section_list(data: dict[str, Any], key: str, file_name: str) -> list[dict[str, Any]]:
    if key not in data:
        raise ValueError(f"{file_name}: missing key '{key}'")
    value = data[key]
    if not isinstance(value, list):
        raise ValueError(f"{file_name}: '{key}' must be a list")
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{file_name}: '{key}[{index}]' must be an object")
    return value


def build_entry_index(entries: list[dict[str, Any]], file_name: str, section_key: str) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for idx, entry in enumerate(entries):
        entry_id = entry.get("entry_id")
        if not isinstance(entry_id, int):
            raise ValueError(f"{file_name}: '{section_key}[{idx}].entry_id' must be int")
        if entry_id in indexed:
            raise ValueError(f"{file_name}: duplicate entry_id={entry_id} in '{section_key}'")
        indexed[entry_id] = entry
    return indexed


def compare_conditions_keys(
    s2_1_entry: dict[str, Any],
    s2_2_entry: dict[str, Any],
) -> tuple[str, str] | None:
    """Compare condition-name sets between section2_1(list) and section2_2(dict keys)."""
    list_conditions = s2_1_entry.get("conditions")
    obj_conditions = s2_2_entry.get("conditions")

    if list_conditions is None and obj_conditions is None:
        return None
    if list_conditions is not None and not isinstance(list_conditions, list):
        return ("section2_1.conditions is not list", "")
    if obj_conditions is not None and not isinstance(obj_conditions, dict):
        return ("", "section2_2.conditions is not object")

    keys_1 = {key for key in list_conditions if isinstance(key, str)} if isinstance(list_conditions, list) else set()
    keys_2 = set(obj_conditions.keys()) if isinstance(obj_conditions, dict) else set()

    only_in_1 = sorted(keys_1 - keys_2)
    only_in_2 = sorted(keys_2 - keys_1)
    if only_in_1 or only_in_2:
        return (str(only_in_1), str(only_in_2))
    return None


# --- Check row helpers ---

def init_check_rows() -> CheckRows:
    return {name: [] for name in FULL_CHECK_SEQUENCE}


def add_check_row(check_rows: CheckRows, check_name: str, row: list[str]) -> None:
    check_rows[check_name].append(row)


# --- Precheck (check-0) ---

def normalize_top_level(
    path: Path,
    expected_key: str,
) -> tuple[Any | None, bool, str | None, str | None]:
    """Normalize top-level key/shape for one JSON file.

    Returns:
    - payload (None if parse failed)
    - modified flag
    - action (short tag)
    - detail (description)
    """
    try:
        payload = load_json(path)
    except Exception as error:
        return None, False, "parse_error", str(error)

    modified = False
    action: str | None = None
    detail: str | None = None

    if not isinstance(payload, dict):
        payload = {expected_key: []}
        modified = True
        action = "fixed_top_level"
        detail = "top-level was not object; replaced with expected key and empty list"
        return payload, modified, action, detail

    if expected_key not in payload:
        alternate_key = "section2_2" if expected_key == "section2_1" else "section2_1"

        if "section2" in payload and isinstance(payload["section2"], list):
            payload[expected_key] = payload.pop("section2")
            modified = True
            action = "renamed_top_key"
            detail = "renamed 'section2' to expected top-level key"
        elif alternate_key in payload and isinstance(payload[alternate_key], list):
            payload[expected_key] = payload.pop(alternate_key)
            modified = True
            action = "renamed_top_key"
            detail = f"renamed '{alternate_key}' to expected top-level key"
        else:
            list_keys = [key for key, value in payload.items() if isinstance(value, list)]
            if len(list_keys) == 1:
                key = list_keys[0]
                payload[expected_key] = payload.pop(key)
                modified = True
                action = "renamed_single_list_key"
                detail = f"renamed '{key}' to expected top-level key"
            else:
                payload[expected_key] = []
                modified = True
                action = "added_missing_top_key"
                detail = "added expected top-level key with empty list"

    value = payload.get(expected_key)
    if not isinstance(value, list):
        if isinstance(value, dict):
            payload[expected_key] = [value]
            modified = True
            action = "normalized_top_value"
            detail = "expected top-level value was object; wrapped into list"
        else:
            payload[expected_key] = []
            modified = True
            action = "normalized_top_value"
            detail = "expected top-level value was not list; replaced with empty list"

    return payload, modified, action, detail


def precheck_and_fix_top_level(
    files: dict[str, Path],
    dataset_name: str,
    expected_key: str,
    check_rows: CheckRows,
    logger: logging.Logger,
) -> set[str]:
    """Pre-scan and auto-fix top-level key issues before all checks.

    Returns filenames that are unrecoverable in this stage (e.g., parse errors).
    """
    fatal_files: set[str] = set()

    for filename, path in files.items():
        payload, modified, action, detail = normalize_top_level(path, expected_key)

        if action == "parse_error":
            # Unrecoverable at precheck stage.
            fatal_files.add(filename)
            add_check_row(check_rows, PRECHECK_NAME, [filename, dataset_name, action, detail or ""])
            add_check_row(check_rows, "check-json", [filename, f"parse_error: {detail}"])
            logger.warning("precheck_fail=%s dataset=%s reason=parse_error", filename, dataset_name)
            continue

        if modified and payload is not None:
            save_json(path, payload)
            add_check_row(check_rows, PRECHECK_NAME, [filename, dataset_name, action or "fixed", detail or ""])
            logger.info("precheck_fixed=%s dataset=%s action=%s", filename, dataset_name, action)

    return fatal_files


# --- File pairing and merge core ---

def record_file_pair_gaps(
    check_rows: CheckRows,
    file_names_s2_1: set[str],
    file_names_s2_2: set[str],
) -> None:
    for missing in sorted(file_names_s2_1 - file_names_s2_2):
        add_check_row(check_rows, "check-1", [missing, "missing in inputs/s2_2"])
    for missing in sorted(file_names_s2_2 - file_names_s2_1):
        add_check_row(check_rows, "check-1", [missing, "missing in inputs/s2_1"])


def merge_one_file(
    file_name: str,
    data_s2_1: dict[str, Any],
    data_s2_2: dict[str, Any],
    check_rows: CheckRows,
) -> list[dict[str, Any]]:
    s2_1_list = ensure_section_list(data_s2_1, "section2_1", file_name)
    s2_2_list = ensure_section_list(data_s2_2, "section2_2", file_name)

    s2_2_by_id = build_entry_index(s2_2_list, file_name, "section2_2")
    merged: list[dict[str, Any]] = []

    for idx, entry_s2_1 in enumerate(s2_1_list):
        entry_id = entry_s2_1.get("entry_id")
        if not isinstance(entry_id, int):
            add_check_row(check_rows, "check-2", [file_name, "", f"section2_1[{idx}].entry_id must be int"])
            continue

        output_entry: dict[str, Any] = {}
        for key in MERGE_FIELDS:
            if key in entry_s2_1:
                output_entry[key] = entry_s2_1[key]
            else:
                add_check_row(check_rows, "check-3", [file_name, str(entry_id), key])

        entry_s2_2 = s2_2_by_id.get(entry_id)
        if entry_s2_2 is None:
            # Keep partial entry when matching entry_id is absent in s2_2.
            add_check_row(check_rows, "check-2", [file_name, str(entry_id), "entry_id missing in section2_2"])
            merged.append(output_entry)
            continue

        # conditions comes from section2_2.
        if "conditions" in entry_s2_2 and isinstance(entry_s2_2["conditions"], dict):
            output_entry["conditions"] = entry_s2_2["conditions"]
        elif "conditions" in entry_s2_2:
            add_check_row(check_rows, "check-2", [file_name, str(entry_id), "section2_2.conditions must be object"])
        else:
            add_check_row(check_rows, "check-3", [file_name, str(entry_id), "conditions"])

        mismatch = compare_conditions_keys(entry_s2_1, entry_s2_2)
        if mismatch is not None:
            add_check_row(check_rows, "check-4", [file_name, str(entry_id), mismatch[0], mismatch[1]])

        merged.append(output_entry)

    return merged


# --- Reporting and pipeline ---

def write_check_reports(check_rows: CheckRows) -> None:
    for check_name in FULL_CHECK_SEQUENCE:
        config = CHECK_CONFIG[check_name]
        path: Path = config["path"]
        headers: list[str] = config["headers"]
        rows = check_rows[check_name]
        with path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(headers)
            if rows:
                writer.writerows(rows)
            else:
                writer.writerow(["ALL FILES PASSED"] + [""] * (len(headers) - 1))


def run_merge(logger: logging.Logger) -> None:
    validate_check_config()

    files_s2_1 = collect_json_files(S2_1_DIR)
    files_s2_2 = collect_json_files(S2_2_DIR)

    check_rows = init_check_rows()

    # Step 0: normalize top-level key/value shape before formal checks.
    fatal_s2_1 = precheck_and_fix_top_level(files_s2_1, "s2_1", "section2_1", check_rows, logger)
    fatal_s2_2 = precheck_and_fix_top_level(files_s2_2, "s2_2", "section2_2", check_rows, logger)

    names_1 = set(files_s2_1)
    names_2 = set(files_s2_2)
    common_names = sorted(names_1 & names_2)

    # Step 1: record filename pairing gaps.
    record_file_pair_gaps(check_rows, names_1, names_2)

    logger.info("s2_1_files=%d s2_2_files=%d common=%d", len(names_1), len(names_2), len(common_names))

    merged_count = 0
    for name in common_names:
        if name in fatal_s2_1 or name in fatal_s2_2:
            logger.warning("skip=%s reason=precheck_parse_error", name)
            continue

        path_1 = files_s2_1[name]
        path_2 = files_s2_2[name]

        try:
            data_1 = load_json(path_1)
            data_2 = load_json(path_2)
        except Exception as error:
            add_check_row(check_rows, "check-json", [name, f"parse_error: {error}"])
            logger.warning("skip=%s reason=parse_error", name)
            continue

        if not isinstance(data_1, dict) or not isinstance(data_2, dict):
            add_check_row(check_rows, "check-json", [name, "top-level JSON must be object"])
            logger.warning("skip=%s reason=top_level_not_object", name)
            continue

        try:
            # Step 2: merge one file pair by entry_id.
            merged_entries = merge_one_file(name, data_1, data_2, check_rows)
            save_json(OUT_DIR / name, {"section2": merged_entries})
            merged_count += 1
            logger.info("merged=%s entries=%d", name, len(merged_entries))
        except Exception as error:
            add_check_row(check_rows, "check-2", [name, "", f"structure error: {error}"])
            logger.warning("skip=%s reason=structure_error", name)

    write_check_reports(check_rows)
    logger.info("merged_files=%d", merged_count)


def main() -> None:
    logger = create_logger(LOG_DIR)
    run_merge(logger)


if __name__ == "__main__":
    main()
