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
LOCAL_API_KEY = os.getenv("SILICONFLOW_API_KEY", os.getenv("LOCAL_DEEPSEEK_API_KEY", ""))
LOCAL_MODEL = os.getenv("SILICONFLOW_MODEL", os.getenv("LOCAL_DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V3.2"))
VERIFY_SSL = os.getenv("SILICONFLOW_VERIFY_SSL", os.getenv("LOCAL_DEEPSEEK_VERIFY_SSL", "true")).lower() in {"1", "true", "yes"}
MAX_RETRIES = max(1, int(os.getenv("SILICONFLOW_MAX_RETRIES", "4")))
REQUEST_INTERVAL_SEC = max(0.0, float(os.getenv("SILICONFLOW_REQUEST_INTERVAL_SEC", "0")))
_RATE_LIMIT_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0


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
    effective_model = os.getenv("SILICONFLOW_MODEL", os.getenv("LOCAL_DEEPSEEK_MODEL", model or LOCAL_MODEL))
    effective_base_url = os.getenv("SILICONFLOW_BASE_URL", os.getenv("LOCAL_DEEPSEEK_BASE_URL", base_url or LOCAL_BASE_URL))
    effective_api_key = os.getenv("SILICONFLOW_API_KEY", os.getenv("LOCAL_DEEPSEEK_API_KEY", api_key or LOCAL_API_KEY)).strip()
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
                _resolve_chat_url(effective_base_url),
                json={
                    "model": effective_model,
                    "messages": [
                        {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "stream": False,
                },
                headers={
                    "Authorization": f"Bearer {effective_api_key}",
                    "Content-Type": "application/json",
                },
                verify=VERIFY_SSL,
                timeout=timeout_sec,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            should_retry = attempt < MAX_RETRIES and any(x in message for x in ("429", "502", "503", "504", "timeout"))
            if not should_retry:
                break
            time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"stage=ds_client/chat error={last_error}")
