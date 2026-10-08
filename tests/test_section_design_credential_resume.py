import sys
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from section_design_agent import get_client


def test_redacted_checkpoint_key_resolves_only_at_runtime():
    runtime_key = "sk-runtime-only-test-value-123456"
    with patch.dict("os.environ", {"SECTION_AGENT_API_KEY": runtime_key}):
        with patch("qiniu_model_client.QiniuModelClient") as client_type:
            get_client(api_key="[REDACTED]", backend="qiniu")
    assert client_type.call_args.kwargs["api_key"] == runtime_key


def test_explicit_key_is_not_replaced_by_environment():
    with patch.dict("os.environ", {"SECTION_AGENT_API_KEY": "sk-unused-test-value-123456"}):
        with patch("qiniu_model_client.QiniuModelClient") as client_type:
            get_client(api_key="sk-explicit-test-value-123456", backend="qiniu")
    assert client_type.call_args.kwargs["api_key"] == "sk-explicit-test-value-123456"
