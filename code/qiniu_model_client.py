"""Secret-safe OpenAI-compatible client for Qiniu AI models."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

try:
    import requests
except ModuleNotFoundError:
    class _MissingRequests:
        @staticmethod
        def get(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("requests is required for live Qiniu API calls")

        @staticmethod
        def post(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("requests is required for live Qiniu API calls")

    requests = _MissingRequests()


DEFAULT_BASE_URL = "https://api.qnaigc.com/v1"
PREFERRED_MODEL = "deepseek/deepseek-v4-pro"
FALLBACK_MODEL = "deepseek/deepseek-v4-pro-0813"
KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{20,}")
MODEL_CONTEXT_LENGTH = 1000000
MODEL_MAX_OUTPUT_TOKENS = 384000
MAX_RETRY_OUTPUT_TOKENS = 384000


class QiniuClientError(RuntimeError):
    pass


class QiniuReasoningTimeout(QiniuClientError):
    pass


class QiniuWallTimeout(QiniuClientError):
    pass


def redact_secrets(text: str) -> str:
    return KEY_PATTERN.sub("[REDACTED]", str(text))


def load_api_key(path: str | Path) -> str:
    content = Path(path).read_text(encoding="utf-8", errors="ignore")
    match = KEY_PATTERN.search(content)
    if not match:
        raise QiniuClientError("credential file does not contain a recognizable API key")
    return match.group(0)


def parse_json_content(content: str) -> Any:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start_candidates = [index for index in (text.find("{"), text.find("[")) if index >= 0]
        if not start_candidates:
            raise QiniuClientError("model response does not contain JSON")
        start = min(start_candidates)
        closing = "}" if text[start] == "{" else "]"
        end = text.rfind(closing)
        if end <= start:
            raise QiniuClientError("model response contains incomplete JSON")
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise QiniuClientError(f"invalid model JSON: {exc.msg}") from exc


class QiniuModelClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = PREFERRED_MODEL,
        connect_timeout: float = 30,
        read_timeout: float = 600,
        max_retries: int = 2,
        max_reasoning_seconds: float = 300,
        max_wall_seconds: float | None = None,
    ) -> None:
        if not KEY_PATTERN.fullmatch(api_key):
            raise QiniuClientError("invalid API key format")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = (connect_timeout, read_timeout)
        self.max_retries = max_retries
        self.max_reasoning_seconds = max_reasoning_seconds
        self.max_wall_seconds = float(max_wall_seconds or read_timeout)

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def list_models(self) -> list[str]:
        try:
            response = requests.get(
                f"{self.base_url}/models", headers=self.headers, timeout=self.timeout
            )
            response.raise_for_status()
            payload = response.json()
            return sorted(
                str(item["id"])
                for item in payload.get("data", [])
                if isinstance(item, dict) and item.get("id")
            )
        except Exception as exc:
            raise QiniuClientError(redact_secrets(exc)) from exc

    def select_available_model(self) -> str:
        models = set(self.list_models())
        if PREFERRED_MODEL in models:
            self.model = PREFERRED_MODEL
        elif FALLBACK_MODEL in models:
            self.model = FALLBACK_MODEL
        else:
            raise QiniuClientError(
                f"neither {PREFERRED_MODEL} nor {FALLBACK_MODEL} is available"
            )
        return self.model

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = MODEL_MAX_OUTPUT_TOKENS,
        temperature: float = 0,
    ) -> tuple[Any, dict[str, Any]]:
        if max_tokens <= 0:
            raise QiniuClientError("max_tokens must be positive")
        if self.model in {PREFERRED_MODEL, FALLBACK_MODEL} and max_tokens > MODEL_MAX_OUTPUT_TOKENS:
            raise QiniuClientError(
                f"max_tokens exceeds the model limit of {MODEL_MAX_OUTPUT_TOKENS}"
            )
        last_error: Exception | None = None
        attempt_max_tokens = max_tokens
        for attempt in range(1, self.max_retries + 2):
            started = time.monotonic()
            request_payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": attempt_max_tokens,
                "stream": True,
            }
            try:
                with requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={**self.headers, "Accept": "text/event-stream"},
                    json=request_payload,
                    timeout=self.timeout,
                    stream=True,
                ) as response:
                    response.raise_for_status()
                    content_parts: list[str] = []
                    reasoning_chars = 0
                    first_content_seconds = None
                    returned_model = None
                    usage = None
                    finish_reason = None
                    for raw in response.iter_lines(decode_unicode=True):
                        if not raw or not raw.startswith("data:"):
                            continue
                        data = raw[5:].strip()
                        if data == "[DONE]":
                            break
                        event = json.loads(data)
                        if time.monotonic() - started > self.max_wall_seconds:
                            raise QiniuWallTimeout(
                                "model exceeded absolute request wall-time limit"
                            )
                        returned_model = event.get("model") or returned_model
                        usage = event.get("usage") or usage
                        choices = event.get("choices") or [{}]
                        choice = choices[0]
                        finish_reason = choice.get("finish_reason") or finish_reason
                        delta = choice.get("delta") or {}
                        reasoning_chars += len(delta.get("reasoning_content") or "")
                        piece = delta.get("content") or ""
                        if piece:
                            if first_content_seconds is None:
                                first_content_seconds = round(time.monotonic() - started, 3)
                            content_parts.append(piece)
                        elif (
                            not content_parts
                            and time.monotonic() - started > self.max_reasoning_seconds
                        ):
                            raise QiniuReasoningTimeout(
                                "model exceeded reasoning-only wall-time limit"
                            )
                    content = "".join(content_parts).strip()
                    if not content:
                        raise QiniuClientError(
                            "model returned no final content "
                            f"(reasoning_chars={reasoning_chars}, "
                            f"finish_reason={finish_reason or 'unknown'})"
                        )
                    parsed = parse_json_content(content)
                    return parsed, {
                        "requested_model": self.model,
                        "requested_max_tokens": attempt_max_tokens,
                        "returned_model": returned_model,
                        "attempt": attempt,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "first_content_seconds": first_content_seconds,
                        "reasoning_chars": reasoning_chars,
                        "content_length": len(content),
                        "usage": usage,
                    }
            except Exception as exc:
                last_error = exc
                if attempt > self.max_retries:
                    break
                if isinstance(exc, QiniuClientError):
                    attempt_max_tokens = max(
                        attempt_max_tokens,
                        min(attempt_max_tokens * 2, MAX_RETRY_OUTPUT_TOKENS),
                    )
                time.sleep(min(2**attempt, 8))
        if isinstance(last_error, (QiniuReasoningTimeout, QiniuWallTimeout)):
            raise last_error
        if isinstance(last_error, (json.JSONDecodeError, QiniuClientError)):
            started = time.monotonic()
            request_payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": attempt_max_tokens,
                "stream": False,
            }
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=request_payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                envelope = response.json()
                choices = envelope.get("choices") or [{}]
                choice = choices[0]
                message = choice.get("message") or {}
                content = str(message.get("content") or "").strip()
                if not content:
                    raise QiniuClientError(
                        "non-stream fallback returned no final content "
                        f"(reasoning_chars={len(message.get('reasoning_content') or '')}, "
                        f"finish_reason={choice.get('finish_reason') or 'unknown'})"
                    )
                parsed = parse_json_content(content)
                return parsed, {
                    "requested_model": self.model,
                    "requested_max_tokens": attempt_max_tokens,
                    "returned_model": envelope.get("model"),
                    "attempt": self.max_retries + 2,
                    "transport": "non_stream_fallback",
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "first_content_seconds": None,
                    "reasoning_chars": len(message.get("reasoning_content") or ""),
                    "content_length": len(content),
                    "usage": envelope.get("usage"),
                }
            except Exception as exc:
                last_error = exc
        raise QiniuClientError(redact_secrets(last_error or "unknown API error"))
