import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import prompt_quality_code_agent as agent


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_empty_chat_error_reports_safe_completion_metadata(monkeypatch):
    payload = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": "",
                    "reasoning_content": "secret reasoning text",
                },
            }
        ],
        "usage": {
            "completion_tokens": 4096,
            "completion_tokens_details": {"reasoning_tokens": 4096},
        },
    }
    monkeypatch.setattr(agent, "DEFAULT_REQUEST_RETRIES", 1)
    monkeypatch.setattr(agent.urllib.request, "urlopen", lambda *args, **kwargs: _Response(payload))

    with pytest.raises(RuntimeError) as exc_info:
        agent.litellm_chat(
            base_url="https://example.invalid/v1",
            api_key="not-a-real-key",
            model="test-model",
            prompt="test",
        )

    message = str(exc_info.value)
    assert '"finish_reason": "length"' in message
    assert '"reasoning_chars": 21' in message
    assert '"reasoning_tokens": 4096' in message
    assert "secret reasoning text" not in message


def test_chat_includes_bounded_reasoning_effort_when_configured(monkeypatch):
    captured = {}

    def urlopen(request, **kwargs):
        captured.update(json.loads(request.data.decode("utf-8")))
        return _Response(
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": '{"ok": true}'}}
                ]
            }
        )

    monkeypatch.setenv("CODE_AGENT_REASONING_EFFORT", "low")
    monkeypatch.setattr(agent, "DEFAULT_REQUEST_RETRIES", 1)
    monkeypatch.setattr(agent.urllib.request, "urlopen", urlopen)

    result = agent.litellm_chat(
        base_url="https://example.invalid/v1",
        api_key="not-a-real-key",
        model="test-model",
        prompt="test",
    )

    assert result == '{"ok": true}'
    assert captured["reasoning_effort"] == "low"
