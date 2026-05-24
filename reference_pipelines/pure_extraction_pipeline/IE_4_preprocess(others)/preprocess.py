"""Config-driven preprocessing for section outputs before final merge.

Features:
- Per-part rules are defined in config.yaml (no code changes needed for most updates).
- Supports key renaming, keep/drop filtering, default value filling, and key ordering.
- Supports a lightweight extractor mode for single/primary material keys.
- Writes outputs into outputs/<part>/<filename>.json and a preprocess report.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config.yaml"


def create_logger(log_dir: Path) -> logging.Logger:
    logger = logging.getLogger("preprocess0")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"preprocess_{timestamp}.log"

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    logger.info("log_file=%s", log_path)
    return logger


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for YAML config. Please install: pip install pyyaml"
        ) from exc

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = load_yaml(path)
    elif suffix == ".json":
        data = load_json(path)
    else:
        raise ValueError(f"unsupported config format: {path}")

    if not isinstance(data, dict):
        raise ValueError("config root must be object")
    return data


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def collect_json_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.glob("*.json") if path.is_file())


def deep_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def ordered_from_key_order(payload: dict[str, Any], key_order: list[str]) -> OrderedDict[str, Any]:
    ordered: OrderedDict[str, Any] = OrderedDict()
    for key in key_order:
        if key in payload:
            ordered[key] = payload[key]
    for key, value in payload.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def get_schema_object_keys(
    schema_file: Path,
    schema_root_key: str,
    schema_cache: dict[Path, Any],
) -> list[str]:
    if schema_file not in schema_cache:
        schema_cache[schema_file] = load_json(schema_file)

    schema_data = schema_cache[schema_file]
    if not isinstance(schema_data, dict):
        raise ValueError(f"schema file root must be object: {schema_file}")
    if schema_root_key not in schema_data:
        raise ValueError(f"schema root key '{schema_root_key}' not found in {schema_file}")

    schema_root = schema_data[schema_root_key]
    if not isinstance(schema_root, dict):
        raise ValueError(f"schema root '{schema_root_key}' must be object in {schema_file}")

    return list(schema_root.keys())


def resolve_rule_with_schema(
    part_name: str,
    rule: dict[str, Any],
    schema_root_dir: Path,
    schema_cache: dict[Path, Any],
) -> dict[str, Any]:
    resolved = dict(rule)

    schema_file_name = resolved.get("schema_file")
    if not schema_file_name:
        return resolved

    schema_file = schema_root_dir / schema_file_name

    if "input_root" in resolved:
        schema_root_key = resolved.get(
            "schema_root",
            resolved.get("output_root", resolved.get("input_root")),
        )
        if not schema_root_key:
            raise ValueError(f"part={part_name} missing input_root/output_root for schema resolution")

        schema_keys = get_schema_object_keys(schema_file, schema_root_key, schema_cache)

        if "keep_keys" not in resolved:
            resolved["keep_keys"] = schema_keys
        if "key_order" not in resolved:
            if "field_order" in resolved:
                resolved["key_order"] = resolved["field_order"]
            else:
                resolved["key_order"] = schema_keys

        return resolved

    if schema_file not in schema_cache:
        schema_cache[schema_file] = load_json(schema_file)

    schema_data = schema_cache[schema_file]
    if not isinstance(schema_data, dict):
        raise ValueError(f"schema file root must be object: {schema_file}")

    schema_keys = list(schema_data.keys())

    if "extract_keys" not in resolved and "extract_fields" not in resolved:
        resolved["extract_keys"] = {key: [key] for key in schema_keys}
    if "key_order" not in resolved:
        if "field_order" in resolved:
            resolved["key_order"] = resolved["field_order"]
        else:
            resolved["key_order"] = schema_keys

    return resolved


def normalize_dict_payload(
    payload: dict[str, Any],
    rule: dict[str, Any],
) -> OrderedDict[str, Any]:
    rename_map = rule.get("rename_map", {})
    drop_keys = set(rule.get("drop_keys", []))
    keep_keys = rule.get("keep_keys")
    defaults = rule.get("defaults", {})
    apply_defaults = bool(rule.get("apply_defaults", False))
    key_order = rule.get("key_order", rule.get("field_order", []))

    renamed: dict[str, Any] = {}
    for key, value in payload.items():
        target_key = rename_map.get(key, key)
        renamed[target_key] = value

    if keep_keys is not None:
        keep_set = set(keep_keys)
        renamed = {key: value for key, value in renamed.items() if key in keep_set}

    if drop_keys:
        renamed = {key: value for key, value in renamed.items() if key not in drop_keys}

    if apply_defaults:
        for key, value in defaults.items():
            if key not in renamed:
                renamed[key] = deep_clone(value)

    return ordered_from_key_order(renamed, key_order)


def process_schema_object(data: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    input_root = rule["input_root"]
    output_root = rule.get("output_root", input_root)

    if input_root not in data or not isinstance(data[input_root], dict):
        raise ValueError(f"missing or invalid root key '{input_root}'")

    normalized = normalize_dict_payload(data[input_root], rule)
    return {output_root: normalized}


def extract_by_path(data: dict[str, Any], path_items: list[str]) -> Any:
    current: Any = data
    for key in path_items:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def process_extract_keys(data: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    extract_keys: dict[str, Any] = rule.get("extract_keys", rule.get("extract_fields", {}))
    include_missing = bool(rule.get("include_missing", False))
    apply_defaults = bool(rule.get("apply_defaults", False))

    for target_key, source_path in extract_keys.items():
        if not isinstance(source_path, list) or not source_path:
            raise ValueError(f"extract_keys.{target_key} must be a non-empty list")
        extracted = extract_by_path(data, source_path)
        if extracted is None and not include_missing:
            continue
        output[target_key] = extracted

    defaults = rule.get("defaults", {})
    if apply_defaults:
        for key, value in defaults.items():
            if key not in output or output[key] is None:
                output[key] = deep_clone(value)

    key_order = rule.get("key_order", rule.get("field_order", list(output.keys())))
    ordered = ordered_from_key_order(output, key_order)
    return dict(ordered)


def process_rule(data: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    if "input_root" in rule:
        return process_schema_object(data, rule)
    return process_extract_keys(data, rule)


def clear_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def preprocess(config_path: Path) -> None:
    config = load_config(config_path)

    input_root = BASE_DIR / config.get("input_root", "inputs")
    output_root = BASE_DIR / config.get("output_root", "outputs")
    log_root = BASE_DIR / config.get("log_root", "logs")
    report_path = BASE_DIR / config.get("report_path", "preprocess_report.txt")
    schema_root_dir = BASE_DIR / config.get("schema_root", "schema")

    logger = create_logger(log_root)

    clean_output = bool(config.get("clean_output", True))
    if clean_output:
        clear_output_dir(output_root)
        logger.info("cleaned_output=%s", output_root)
    else:
        output_root.mkdir(parents=True, exist_ok=True)

    parts: dict[str, Any] = config.get("parts", {})
    if not parts:
        raise ValueError("config.parts is empty")

    report_lines: list[str] = []
    total_success = 0
    total_failed = 0
    schema_cache: dict[Path, Any] = {}

    for part_name, rule in parts.items():
        enabled = bool(rule.get("enabled", True))
        if not enabled:
            report_lines.append(f"{part_name}\tSKIPPED\tdisabled")
            continue

        resolved_rule = resolve_rule_with_schema(part_name, rule, schema_root_dir, schema_cache)

        input_subdir = resolved_rule.get("input_subdir", part_name)
        output_subdir = resolved_rule.get("output_subdir", part_name)

        input_dir = input_root / input_subdir
        output_dir = output_root / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        files = collect_json_files(input_dir)
        if not files:
            msg = f"no_json_files_in={input_dir}"
            logger.warning("%s %s", part_name, msg)
            report_lines.append(f"{part_name}\tSKIPPED\t{msg}")
            continue

        logger.info("processing part=%s files=%d", part_name, len(files))

        for path in files:
            rel = f"{input_subdir}/{path.name}"
            try:
                data = load_json(path)
                if not isinstance(data, dict):
                    raise ValueError("json_root_not_object")

                transformed = process_rule(data, resolved_rule)
                save_json(output_dir / path.name, transformed)

                total_success += 1
                report_lines.append(f"{rel}\tSUCCESS")
            except Exception as exc:
                total_failed += 1
                logger.error("failed part=%s file=%s error=%s", part_name, path.name, exc)
                report_lines.append(f"{rel}\tFAILED\t{exc}")

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    logger.info("success=%d failed=%d", total_success, total_failed)
    logger.info("report=%s", report_path)


def main() -> None:
    preprocess(DEFAULT_CONFIG_PATH)


if __name__ == "__main__":
    main()
