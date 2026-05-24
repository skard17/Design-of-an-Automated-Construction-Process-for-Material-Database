from __future__ import annotations

import os
import threading
import time

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LOCAL_BASE_URL = os.getenv(
    "SILICONFLOW_BASE_URL",
    os.getenv("LOCAL_DEEPSEEK_BASE_URL", "https://api.siliconflow.cn/v1/chat/completions"),
)
LOCAL_API_KEY = os.getenv(
    "SILICONFLOW_API_KEY",
    os.getenv("LOCAL_DEEPSEEK_API_KEY", ""),
)
LOCAL_MODEL = os.getenv("SILICONFLOW_MODEL", os.getenv("LOCAL_DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V3.2"))
VERIFY_SSL = os.getenv("SILICONFLOW_VERIFY_SSL", os.getenv("LOCAL_DEEPSEEK_VERIFY_SSL", "true")).lower() in {"1", "true", "yes"}
MAX_RETRIES = max(1, int(os.getenv("SILICONFLOW_MAX_RETRIES", "4")))
REQUEST_INTERVAL_SEC = max(0.0, float(os.getenv("SILICONFLOW_REQUEST_INTERVAL_SEC", "0")))
_RATE_LIMIT_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0


def _split_slot(api_key: str, model: str | None = None) -> tuple[str, str]:
    slot = (api_key or "").strip()
    if "|||" in slot:
        key_part, model_part = slot.split("|||", 1)
        return key_part.strip(), (model_part.strip() or model or LOCAL_MODEL)
    return slot, (model or LOCAL_MODEL)


def _resolve_chat_url(base_url: str) -> str:
    url = (base_url or "").strip() or LOCAL_BASE_URL
    if url.endswith("/chat/completions"):
        return url
    return url.rstrip("/") + "/chat/completions"


def chat(
    prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.0,
    timeout_sec: int = 60,
    system_prompt: str | None = None,
) -> str:
    global _LAST_REQUEST_AT
    resolved_key, resolved_model = _split_slot(api_key, model)
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if REQUEST_INTERVAL_SEC > 0:
                with _RATE_LIMIT_LOCK:
                    now = time.monotonic()
                    wait_sec = REQUEST_INTERVAL_SEC - (now - _LAST_REQUEST_AT)
                    if wait_sec > 0:
                        time.sleep(wait_sec)
                    _LAST_REQUEST_AT = time.monotonic()
            response = requests.post(
                _resolve_chat_url(base_url),
                json={
                    "model": resolved_model,
                    "messages": [
                        {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "stream": False,
                    "enable_thinking": True,
                },
                headers={
                    "Authorization": f"Bearer {resolved_key or (os.getenv('SILICONFLOW_API_KEY') or os.getenv('LOCAL_DEEPSEEK_API_KEY') or LOCAL_API_KEY).strip()}",
                    "Content-Type": "application/json",
                },
                verify=VERIFY_SSL,
                timeout=timeout_sec,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_error = e
            message = str(e).lower()
            should_retry = (
                attempt < MAX_RETRIES
                and (
                    "503" in message
                    or "502" in message
                    or "504" in message
                    or "429" in message
                    or "timed out" in message
                    or "timeout" in message
                    or "service unavailable" in message
                )
            )
            if not should_retry:
                break
            time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"stage=ds_client/chat, error: {str(last_error)}")
