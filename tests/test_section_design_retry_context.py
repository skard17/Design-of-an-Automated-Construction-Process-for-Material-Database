import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from section_design_agent_prompt import build_module_redo_prompt


def test_module_retry_preserves_original_task_context():
    original = "TASK_SENTINEL: bind every observation to its sample and evidence span"

    retry = build_module_redo_prompt(
        "subjective_supervisor_module",
        "RuntimeError: empty stream",
        ["RuntimeError: empty stream"],
        '{"requirement_contract": {"concepts": []}}',
        original_prompt=original,
    )

    assert original in retry
    assert "Original task and module context" in retry
    assert "RuntimeError: empty stream" in retry
