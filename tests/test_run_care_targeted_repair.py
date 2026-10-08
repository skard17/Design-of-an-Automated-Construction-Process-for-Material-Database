import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import run_care_targeted_repair as repair


def test_provider_failure_is_reported_instead_of_reusing_seed(tmp_path, monkeypatch):
    target = {
        "target_id": "t1",
        "paper_id": "p1",
        "record_key": "sample-a",
        "concept_id": "transition_temperature",
        "qualifier_keys": [],
    }
    (tmp_path / "targets.json").write_text(
        json.dumps({"targets": [target]}), encoding="utf-8"
    )
    (tmp_path / "contract.json").write_text(
        json.dumps({"concepts": []}), encoding="utf-8"
    )
    (tmp_path / "seed.json").write_text(
        json.dumps({"facts": [{"target_id": "t1", "value": 4}]}),
        encoding="utf-8",
    )
    packs = tmp_path / "packs"
    packs.mkdir()
    (packs / "p1.md").write_text("<!-- SOURCE_LINE: 9 -->\nTc is 4 K.\n", encoding="utf-8")
    monkeypatch.setenv("CODE_AGENT_API_KEY", "test-key")
    monkeypatch.setattr(
        repair.prompt_agent,
        "litellm_chat",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("provider 504")),
    )
    args = repair.build_parser().parse_args(
        [
            "--targets", str(tmp_path / "targets.json"),
            "--concept-contract", str(tmp_path / "contract.json"),
            "--seed-predictions", str(tmp_path / "seed.json"),
            "--pack-dir", str(packs),
            "--run-dir", str(tmp_path / "runs"),
            "--output", str(tmp_path / "predictions.json"),
            "--base-url", "https://example.invalid",
            "--model", "test-model",
            "--max-attempts", "1",
            "--workers", "1",
        ]
    )

    result = repair.run(args)

    assert result["prediction_count"] == 0
    assert result["failure_count"] == 1
    assert result["seed_fallback_disabled"] is True
    assert result["supervisor_decision"]["action"] == "hold_for_review"
    assert result["failures"][0]["supervisor_decision"]["action"] == "hold_failed_paper"
