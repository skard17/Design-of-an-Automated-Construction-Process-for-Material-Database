"""Validate and repair JSON files under IE_3_check/inputs, then export to outputs.

Pipeline (per file):
1) Structure repair via ``json_repair`` (optional but recommended)
2) Invalid-escape repair for JSON strings (targeted, minimal)

All files are recorded in ``check.txt`` with clear status, steps, and details.

Path behavior:
- Base directory is this script's folder.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Tuple

try:
    from json_repair import repair_json  # type: ignore[import]
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    repair_json = None  # type: ignore[assignment]
    JSON_REPAIR_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
INPUT_ROOT = BASE_DIR / "inputs"
OUTPUT_ROOT = BASE_DIR / "outputs"
LOG_ROOT = BASE_DIR / "logs"
CHECK_PATH = BASE_DIR / "check.txt"

VALID_ESCAPES = {"\"", "\\", "/", "b", "f", "n", "r", "t", "u"}
STEP_TO_COUNTER = {
    "-": "ok_no_fix",
    "structure": "fixed_structure_only",
    "escape": "fixed_escape_only",
    "structure+escape": "fixed_both",
}
DEFAULT_STATS = {
    "total": 0,
    "ok_no_fix": 0,
    "fixed_structure_only": 0,
    "fixed_escape_only": 0,
    "fixed_both": 0,
    "failed": 0,
}


class FileOutcome(NamedTuple):
    rel_path: str
    status: str
    steps: str
    detail: str
    counter_key: str

    def to_check_line(self) -> str:
        return f"{self.rel_path}\t{self.status}\t{self.steps}\t{self.detail}"


def create_logger() -> logging.Logger:
    logger = logging.getLogger("check_json")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_ROOT / f"check_{ts}.log"

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


def iter_json_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.json") if p.is_file())


def parse_json(text: str) -> Tuple[object | None, str | None]:
    try:
        return json.loads(text), None
    except Exception as exc:
        return None, str(exc)


def apply_structure_repair(text: str) -> Tuple[str, bool, str | None]:
    """Repair JSON structure using json_repair.

    Returns:
        repaired_text: output text (or original text on failure/unavailable)
        changed: whether text changed
        error: error string when repair is unavailable/failed, else None
    """
    if repair_json is None:
        return text, False, "json_repair_not_installed"

    try:
        repaired = repair_json(text)
    except Exception as exc:
        return text, False, f"json_repair_failed: {exc}"

    if not isinstance(repaired, str):
        return text, False, "json_repair_failed: non_string_result"

    return repaired, repaired != text, None


def fix_invalid_escapes(text: str) -> str:
    """Fix only invalid backslash escapes inside JSON string literals."""
    out: List[str] = []
    in_string = False
    escape = False
    i = 0

    while i < len(text):
        ch = text[i]

        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue

        if escape:
            out.append(ch)
            escape = False
            i += 1
            continue

        if ch == '"':
            out.append(ch)
            in_string = False
            i += 1
            continue

        if ch == "\\":
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt not in VALID_ESCAPES:
                out.append("\\\\")
                i += 1
                continue

            if nxt == "u" and not is_valid_unicode_escape(text, i):
                out.append("\\\\")
                i += 1
                continue

            out.append(ch)
            escape = True
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def is_valid_unicode_escape(text: str, backslash_index: int) -> bool:
    hex_part = text[backslash_index + 2 : backslash_index + 6]
    return len(hex_part) == 4 and all(c in "0123456789abcdefABCDEF" for c in hex_part)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_check(entries: List[str]) -> None:
    header = "path\tstatus\tsteps\tdetail"
    lines = [header]
    lines.extend(entries)
    CHECK_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_success_outcome(rel: Path, rel_str: str, data: object, step_key: str, status: str) -> FileOutcome:
    write_json(OUTPUT_ROOT / rel, data)
    return FileOutcome(rel_str, status, step_key, "-", STEP_TO_COUNTER.get(step_key, "ok_no_fix"))


def process_file(path: Path, logger: logging.Logger) -> FileOutcome:
    """Run parse/repair pipeline for one file and return a normalized outcome."""
    rel = path.relative_to(INPUT_ROOT)
    rel_str = rel.as_posix()
    raw_text = path.read_text(encoding="utf-8")

    data, err0 = parse_json(raw_text)
    if err0 is None:
        return build_success_outcome(rel, rel_str, data, "-", "ok")

    steps: List[str] = []
    details: List[str] = [f"initial_parse_error={err0}"]

    text_after_structure, structure_changed, structure_err = apply_structure_repair(raw_text)
    if structure_err is not None:
        details.append(structure_err)
    elif structure_changed:
        steps.append("structure")
        logger.info("structure_fixed: %s", rel_str)

    data, err1 = parse_json(text_after_structure)
    if err1 is None:
        step_key = "structure" if "structure" in steps else "-"
        status = "fixed" if step_key == "structure" else "ok"
        return build_success_outcome(rel, rel_str, data, step_key, status)

    details.append(f"after_structure_parse_error={err1}")

    text_after_escape = fix_invalid_escapes(text_after_structure)
    escape_changed = text_after_escape != text_after_structure
    if escape_changed:
        steps.append("escape")
        logger.info("escape_fixed: %s", rel_str)

    data, err2 = parse_json(text_after_escape)
    if err2 is None:
        step_key = "+".join(steps) if steps else "-"
        status = "ok" if step_key == "-" else "fixed"
        return build_success_outcome(rel, rel_str, data, step_key, status)

    details.append(f"after_escape_parse_error={err2}")
    detail_text = " | ".join(details)
    logger.error("failed: %s details=%s", rel_str, detail_text)
    return FileOutcome(rel_str, "failed", "+".join(steps) if steps else "-", detail_text, "failed")


def main() -> None:
    logger = create_logger()

    if not JSON_REPAIR_AVAILABLE:
        logger.warning("json_repair is not installed; structure repair step will be skipped")
    else:
        logger.info("json_repair is available; structure repair step enabled")

    check_entries: List[str] = []
    stats: Dict[str, int] = dict(DEFAULT_STATS)

    for path in iter_json_files(INPUT_ROOT):
        stats["total"] += 1
        outcome = process_file(path, logger)
        check_entries.append(outcome.to_check_line())
        stats[outcome.counter_key] += 1

    write_check(check_entries)

    logger.info("total=%d", stats["total"])
    logger.info("ok_no_fix=%d", stats["ok_no_fix"])
    logger.info("fixed_structure_only=%d", stats["fixed_structure_only"])
    logger.info("fixed_escape_only=%d", stats["fixed_escape_only"])
    logger.info("fixed_both=%d", stats["fixed_both"])
    logger.info("failed=%d", stats["failed"])
    logger.info("check_file=%s", CHECK_PATH)


if __name__ == "__main__":
    main()
