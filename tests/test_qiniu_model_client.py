import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import qiniu_model_client as client  # noqa: E402


class FakeResponse:
    def __init__(self, payload=None, lines=None):
        self.payload = payload or {}
        self.lines = lines or []

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload

    def iter_lines(self, decode_unicode=True):
        return iter(self.lines)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


class QiniuModelClientTests(unittest.TestCase):
    def test_deepseek_v4_capacity_matches_provider_model_metadata(self):
        self.assertEqual("deepseek/deepseek-v4-pro", client.PREFERRED_MODEL)
        self.assertEqual(1000000, client.MODEL_CONTEXT_LENGTH)
        self.assertEqual(384000, client.MODEL_MAX_OUTPUT_TOKENS)
        self.assertEqual(384000, client.MAX_RETRY_OUTPUT_TOKENS)

    def test_loads_key_without_returning_other_file_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "credentials.txt"
            path.write_text("label: sk-abcdefghijklmnopqrstuvwxyz123456\n", encoding="utf-8")

            key = client.load_api_key(path)

            self.assertEqual("sk-abcdefghijklmnopqrstuvwxyz123456", key)

    def test_selects_latest_deepseek_v4_pro_before_pinned_fallback(self):
        api = client.QiniuModelClient(api_key="sk-abcdefghijklmnopqrstuvwxyz123456")
        response = FakeResponse(
            {"data": [{"id": client.FALLBACK_MODEL}, {"id": client.PREFERRED_MODEL}]}
        )
        with mock.patch.object(client.requests, "get", return_value=response):
            selected = api.select_available_model()

        self.assertEqual(client.PREFERRED_MODEL, selected)

    def test_streaming_chat_parses_json_and_usage(self):
        event = {
            "model": client.PREFERRED_MODEL,
            "choices": [{"delta": {"content": '{"ok":true}'}}],
            "usage": {"total_tokens": 12},
        }
        response = FakeResponse(lines=["data: " + json.dumps(event), "data: [DONE]"])
        api = client.QiniuModelClient(
            api_key="sk-abcdefghijklmnopqrstuvwxyz123456", max_retries=0
        )
        with mock.patch.object(client.requests, "post", return_value=response):
            payload, metadata = api.chat_json(
                [{"role": "user", "content": "test"}], max_tokens=512
            )

        self.assertEqual({"ok": True}, payload)
        self.assertEqual(12, metadata["usage"]["total_tokens"])

    def test_deepseek_rejects_output_budget_above_provider_limit(self):
        api = client.QiniuModelClient(api_key="sk-abcdefghijklmnopqrstuvwxyz123456")

        with self.assertRaises(client.QiniuClientError):
            api.chat_json(
                [{"role": "user", "content": "test"}],
                max_tokens=client.MODEL_MAX_OUTPUT_TOKENS + 1,
            )

    def test_error_redaction_removes_key(self):
        secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
        self.assertNotIn(secret, client.redact_secrets("failed " + secret))

    def test_reasoning_only_retry_doubles_output_budget(self):
        reasoning_event = {
            "model": client.PREFERRED_MODEL,
            "choices": [{"delta": {"reasoning_content": "thinking"}}],
        }
        final_event = {
            "model": client.PREFERRED_MODEL,
            "choices": [{"delta": {"content": '{"ok":true}'}}],
        }
        responses = [
            FakeResponse(lines=["data: " + json.dumps(reasoning_event), "data: [DONE]"]),
            FakeResponse(lines=["data: " + json.dumps(final_event), "data: [DONE]"]),
        ]
        api = client.QiniuModelClient(
            api_key="sk-abcdefghijklmnopqrstuvwxyz123456", max_retries=1
        )

        with mock.patch.object(client.requests, "post", side_effect=responses) as post:
            payload, metadata = api.chat_json(
                [{"role": "user", "content": "test"}], max_tokens=512
            )

        self.assertEqual({"ok": True}, payload)
        self.assertEqual(1024, post.call_args_list[1].kwargs["json"]["max_tokens"])
        self.assertEqual(1024, metadata["requested_max_tokens"])

    def test_empty_response_reports_reasoning_and_finish_reason(self):
        stream_response = FakeResponse(
            lines=[
                "data: "
                + json.dumps(
                    {
                        "model": client.PREFERRED_MODEL,
                        "choices": [
                            {
                                "finish_reason": "length",
                                "delta": {"reasoning_content": "thinking"},
                            }
                        ],
                    }
                ),
                "data: [DONE]",
            ]
        )
        non_stream_response = FakeResponse(
            payload={
                "model": client.PREFERRED_MODEL,
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"reasoning_content": "more thinking"},
                    }
                ],
            }
        )
        api = client.QiniuModelClient(
            api_key="sk-abcdefghijklmnopqrstuvwxyz123456", max_retries=0
        )

        with mock.patch.object(
            client.requests,
            "post",
            side_effect=[stream_response, non_stream_response],
        ):
            with self.assertRaises(client.QiniuClientError) as caught:
                api.chat_json(
                    [{"role": "user", "content": "test"}], max_tokens=512
                )

        message = str(caught.exception)
        self.assertIn("reasoning_chars=13", message)
        self.assertIn("finish_reason=length", message)

    def test_retry_never_reduces_large_module_output_budget(self):
        reasoning_event = {
            "model": client.PREFERRED_MODEL,
            "choices": [{"delta": {"reasoning_content": "thinking"}}],
        }
        final_event = {
            "model": client.PREFERRED_MODEL,
            "choices": [{"delta": {"content": '{"ok":true}'}}],
        }
        responses = [
            FakeResponse(lines=["data: " + json.dumps(reasoning_event), "data: [DONE]"]),
            FakeResponse(lines=["data: " + json.dumps(final_event), "data: [DONE]"]),
        ]
        api = client.QiniuModelClient(
            api_key="sk-abcdefghijklmnopqrstuvwxyz123456", max_retries=1
        )

        with mock.patch.object(client.requests, "post", side_effect=responses) as post:
            payload, metadata = api.chat_json(
                [{"role": "user", "content": "test"}], max_tokens=8192
            )

        self.assertEqual({"ok": True}, payload)
        self.assertEqual(16384, post.call_args_list[1].kwargs["json"]["max_tokens"])
        self.assertEqual(16384, metadata["requested_max_tokens"])

    def test_malformed_stream_falls_back_to_non_stream_response(self):
        stream_response = FakeResponse(lines=['data: {"choices":[{"delta":{"content":"unterminated'])
        non_stream_response = FakeResponse(
            payload={
                "model": client.PREFERRED_MODEL,
                "choices": [{"message": {"content": '{"ok":true}'}}],
                "usage": {"total_tokens": 24},
            }
        )
        api = client.QiniuModelClient(
            api_key="sk-abcdefghijklmnopqrstuvwxyz123456", max_retries=0
        )

        with mock.patch.object(
            client.requests, "post", side_effect=[stream_response, non_stream_response]
        ):
            payload, metadata = api.chat_json(
                [{"role": "user", "content": "test"}], max_tokens=512
            )

        self.assertEqual({"ok": True}, payload)
        self.assertEqual("non_stream_fallback", metadata["transport"])
        self.assertEqual(24, metadata["usage"]["total_tokens"])

    def test_reasoning_timeout_does_not_start_non_stream_fallback(self):
        reasoning_event = {
            "model": client.PREFERRED_MODEL,
            "choices": [{"delta": {"reasoning_content": "thinking"}}],
        }
        response = FakeResponse(
            lines=["data: " + json.dumps(reasoning_event), "data: [DONE]"]
        )
        api = client.QiniuModelClient(
            api_key="sk-abcdefghijklmnopqrstuvwxyz123456",
            max_retries=0,
            max_reasoning_seconds=-1,
        )

        with mock.patch.object(client.requests, "post", return_value=response) as post:
            with self.assertRaises(client.QiniuReasoningTimeout):
                api.chat_json(
                    [{"role": "user", "content": "test"}],
                    max_tokens=512,
                )

        self.assertEqual(post.call_count, 1)

    def test_absolute_wall_timeout_does_not_start_non_stream_fallback(self):
        content_event = {
            "model": client.PREFERRED_MODEL,
            "choices": [{"delta": {"content": '{"ok":'}}],
        }
        response = FakeResponse(
            lines=["data: " + json.dumps(content_event), "data: [DONE]"]
        )
        api = client.QiniuModelClient(
            api_key="sk-abcdefghijklmnopqrstuvwxyz123456",
            max_retries=0,
            max_wall_seconds=-1,
        )

        with mock.patch.object(client.requests, "post", return_value=response) as post:
            with self.assertRaises(client.QiniuWallTimeout):
                api.chat_json(
                    [{"role": "user", "content": "test"}],
                    max_tokens=512,
                )

        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
