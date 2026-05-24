from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROMPT_BANK_FILE = ROOT / "multi-example.txt"


SECTION_ORDER = [
    "ie0_multi",
    "section5",
    "resources",
    "section0",
    "section1",
    "section2_1",
    "fig_classify",
    "section2_2",
    "section3",
    "section4",
]

SECTION_MARKERS = {
    "ie0_multi": "IE_0_multi:paper",
    "section5": "section5",
    "resources": "resources",
    "section0": "section0",
    "section1": "section1",
    "section2_1": "section2_1",
    "fig_classify": "fig_classify",
    "section2_2": "section2_2",
    "section3": "section3",
    "section4": "section4",
}


def _find_line_index(lines: list[str], marker: str, start_at: int = 0) -> int:
    for idx in range(start_at, len(lines)):
        if lines[idx].strip() == marker:
            return idx
    raise KeyError(f"prompt marker not found as full line: {marker}")


def load_multi_prompt(name: str) -> str:
    if name not in SECTION_MARKERS:
        raise KeyError(f"unknown prompt section: {name}")

    lines = PROMPT_BANK_FILE.read_text(encoding="utf-8").splitlines()
    ordered_idx = SECTION_ORDER.index(name)
    start_marker = SECTION_MARKERS[name]
    start_line = _find_line_index(lines, start_marker) + 1

    end_line = len(lines)
    for later_name in SECTION_ORDER[ordered_idx + 1 :]:
        marker = SECTION_MARKERS[later_name]
        try:
            candidate = _find_line_index(lines, marker, start_at=start_line)
        except KeyError:
            continue
        end_line = candidate
        break

    return "\n".join(lines[start_line:end_line]).strip()
