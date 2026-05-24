"""Merge inputs into schema-shaped outputs with validation and logging."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"
CHECK_PATH = BASE_DIR / "check.txt"
SCHEMA_PATH = BASE_DIR / "schema" / "schema0_store.json"
SECTION_PATTERN = re.compile(r"^section(\d+)$")
OPTIONAL_INPUT_DIRS = set()


@dataclass(frozen=True)
class MergeRoute:
    input_dir: str
    source_key: str
    target_path: Tuple[str, ...]


def build_routes_from_schema(schema: Dict[str, Any]) -> List[MergeRoute]:
    routes: List[MergeRoute] = []

    for top_key, top_value in schema.items():
        if top_key == "material_info" and isinstance(top_value, dict):
            for child_key in top_value.keys():
                match = SECTION_PATTERN.match(child_key)
                if not match:
                    continue
                routes.append(
                    MergeRoute(
                        input_dir=f"s{match.group(1)}",
                        source_key=child_key,
                        target_path=("material_info", child_key),
                    )
                )
            continue

        if top_key == "paper_info" and isinstance(top_value, dict):
            for child_key in top_value.keys():
                routes.append(
                    MergeRoute(
                        input_dir=child_key,
                        source_key=child_key,
                        target_path=("paper_info", child_key),
                    )
                )
            continue

        if top_key == "primary_signature":
            routes.append(
                MergeRoute(
                    input_dir="single",
                    source_key="primary_signature",
                    target_path=(top_key,),
                )
            )
            continue

        match = SECTION_PATTERN.match(top_key)
        if match:
            routes.append(
                MergeRoute(
                    input_dir=f"s{match.group(1)}",
                    source_key=top_key,
                    target_path=(top_key,),
                )
            )
            continue

        routes.append(
            MergeRoute(
                input_dir=top_key,
                source_key=top_key,
                target_path=(top_key,),
            )
        )

    return routes


def set_nested(data: Dict[str, Any], path: Sequence[str], value: Any) -> None:
    if not path:
        raise ValueError("target path cannot be empty")

    cursor: Dict[str, Any] = data
    for key in path[:-1]:
        current = cursor.get(key)
        if not isinstance(current, dict):
            current = {}
            cursor[key] = current
        cursor = current
    cursor[path[-1]] = value


def get_value_from_source(source: Any, source_key: str) -> Tuple[Optional[Any], Optional[str]]:
    if not isinstance(source, dict):
        return None, "not_a_dict"
    if source_key not in source:
        return None, f"missing_key:{source_key}"
    return source[source_key], None


def collect_file_sets(input_dirs: Sequence[str]) -> Dict[str, Set[str]]:
    return {d: set(collect_names(INPUT_DIR / d)) for d in input_dirs}


def all_candidate_names(names_by_dir: Dict[str, Set[str]]) -> List[str]:
    if not names_by_dir:
        return []
    return sorted(set().union(*names_by_dir.values()))


def find_missing_dirs(name: str, names_by_dir: Dict[str, Set[str]]) -> List[str]:
    return [
        d
        for d, names in names_by_dir.items()
        if name not in names and d not in OPTIONAL_INPUT_DIRS
    ]


def build_merged_output(schema: Dict[str, Any], loaded_parts: Dict[str, Any], routes: Sequence[MergeRoute]) -> Dict[str, Any]:
    merged = copy.deepcopy(schema)
    for route in routes:
        source_obj = loaded_parts[route.input_dir]
        value, err = get_value_from_source(source_obj, route.source_key)
        if err:
            raise ValueError(f"invalid route data: {route.input_dir}.{route.source_key} ({err})")
        set_nested(merged, route.target_path, value)
    return merged


def create_logger() -> logging.Logger:
    logger = logging.getLogger("merge")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"merge_{ts}.log"

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


def load_json(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as exc:
        return None, str(exc)


def collect_names(dir_path: Path) -> List[str]:
    if not dir_path.exists():
        return []
    return sorted(p.name for p in dir_path.glob("*.json") if p.is_file())


def write_check(entries: Dict[str, List[str]]) -> None:
    lines: List[str] = []
    for name in sorted(entries):
        issues = "; ".join(entries[name])
        lines.append(f"{name}\t{issues}")
    CHECK_PATH.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    logger = create_logger()

    schema, err = load_json(SCHEMA_PATH)
    if err or not isinstance(schema, dict):
        raise RuntimeError(f"failed to load schema: {SCHEMA_PATH} error={err}")

    try:
        routes = build_routes_from_schema(schema)
    except Exception as exc:
        logger.error("schema_route_error=%s", exc)
        raise

    if not routes:
        logger.error("schema_route_error=no routes generated")
        raise RuntimeError("no merge routes generated from schema")

    input_dirs = [route.input_dir for route in routes]
    names_by_dir = collect_file_sets(input_dirs)

        # 🌟 核心改进：不再取并集，而是以 single 文件夹为绝对基准
    if "single" in names_by_dir:
        all_names = sorted(names_by_dir["single"])
        logger.info("以 single 文件夹为基准进行合并，共计 %d 篇候选文献。", len(all_names))
    else:
        # 兜底：如果连 single 文件夹都没有，才按旧逻辑报错
        all_names: Iterable[str] = all_candidate_names(names_by_dir)
        logger.warning("未检测到 single 输入目录，采用全量并集模式。")
    
    check_entries: Dict[str, List[str]] = {}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    merged_count = 0
    skipped_count = 0

    for name in all_names:
        missing = find_missing_dirs(name, names_by_dir)
        if missing:
            logger.warning("skip %s: missing_parts=%s", name, ",".join(sorted(missing)))
            check_entries.setdefault(name, []).append(
                f"missing_parts={','.join(sorted(missing))}"
            )
            skipped_count += 1
            continue

        parts: Dict[str, Any] = {}
        invalid = False

        for route in routes:
            d = route.input_dir
            path = INPUT_DIR / d / name
            if d in OPTIONAL_INPUT_DIRS and name not in names_by_dir.get(d, set()):
                continue
            data, load_err = load_json(path)
            if load_err:
                if d in OPTIONAL_INPUT_DIRS:
                    logger.info("optional part missing_or_invalid %s: %s", path, load_err)
                    continue
                logger.warning("skip %s: parse_error_%s=%s", name, d, load_err)
                check_entries.setdefault(name, []).append(
                    f"parse_error_{d}={load_err}"
                )
                invalid = True
                continue

            if not isinstance(data, dict) or route.source_key not in data:
                if d in OPTIONAL_INPUT_DIRS:
                    logger.info("optional part invalid_key %s expected:%s", name, route.source_key)
                    continue
                logger.warning(
                    "skip %s: invalid_key_%s expected:%s",
                    name,
                    d,
                    route.source_key,
                )
                check_entries.setdefault(name, []).append(
                    f"invalid_key_{d}=expected:{route.source_key}"
                )
                invalid = True
                continue

            parts[d] = data

        if invalid:
            skipped_count += 1
            continue

        try:
            merged = build_merged_output(schema, parts, routes)
        except Exception as exc:
            logger.warning("skip %s: merge_error=%s", name, exc)
            check_entries.setdefault(name, []).append(f"merge_error={exc}")
            skipped_count += 1
            continue

        out_path = OUTPUT_DIR / name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
            f.write("\n")
        merged_count += 1
        logger.info("merged: %s", name)

    write_check(check_entries)

    logger.info("merged_count=%d", merged_count)
    logger.info("skipped_count=%d", skipped_count)
    logger.info("check_file=%s", CHECK_PATH)


if __name__ == "__main__":
    main()
