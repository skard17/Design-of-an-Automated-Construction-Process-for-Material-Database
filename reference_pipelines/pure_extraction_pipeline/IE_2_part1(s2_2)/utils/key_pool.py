from __future__ import annotations

import os
import queue
import threading
from contextlib import contextmanager
from typing import Iterable


LOCAL_API_KEY = os.getenv(
    "SILICONFLOW_API_KEY",
    os.getenv("LOCAL_DEEPSEEK_API_KEY", ""),
)
LOCAL_API_KEY_2 = os.getenv("SILICONFLOW_API_KEY_2", "").strip()
LOCAL_MODEL = os.getenv("SILICONFLOW_MODEL", os.getenv("LOCAL_DEEPSEEK_MODEL", "Pro/deepseek-ai/DeepSeek-V3.2"))
LOCAL_MODEL_2 = os.getenv("SILICONFLOW_MODEL_2", "deepseek-ai/DeepSeek-V3.2")
LOCAL_PER_KEY_PARALLEL = max(1, int(os.getenv("SILICONFLOW_PER_KEY_PARALLEL", "4")))
LOCAL_PARALLEL = max(1, int(os.getenv("SILICONFLOW_MAX_PARALLEL", os.getenv("LOCAL_DEEPSEEK_MAX_PARALLEL", "8"))))


def _pack_slot(api_key: str, model: str) -> str:
    return f"{api_key}|||{model}"


class KeyPool:
    _keys: list[str] = []
    _available: "queue.Queue[str] | None" = None
    _lock = threading.Lock()
    _initialized = False

    @classmethod
    def initialize(cls, config_keys: Iterable[str] | None = None) -> None:
        with cls._lock:
            if cls._initialized:
                return

            keys: list[str] = []
            local_key = LOCAL_API_KEY.strip()
            if local_key:
                if LOCAL_API_KEY_2:
                    keys = [_pack_slot(local_key, LOCAL_MODEL)] * LOCAL_PER_KEY_PARALLEL
                    keys += [_pack_slot(LOCAL_API_KEY_2, LOCAL_MODEL_2)] * LOCAL_PER_KEY_PARALLEL
                else:
                    keys = [_pack_slot(local_key, LOCAL_MODEL)] * LOCAL_PARALLEL
            elif config_keys:
                keys = [str(k).strip() for k in config_keys if str(k).strip()]
            elif (keys_str := os.getenv("LLM_API_KEYS", "")).strip():
                keys = [k.strip() for k in keys_str.split(",") if k.strip()]
            elif (single := os.getenv("LLM_API_KEY", "")).strip():
                keys = [single.strip()]

            cls._keys = keys
            cls._available = queue.Queue()
            for key in cls._keys:
                cls._available.put(key)
            cls._initialized = True

    @classmethod
    def acquire_key(cls, config_keys: Iterable[str] | None = None, timeout_sec: float | None = None) -> str:
        if not cls._initialized:
            cls.initialize(config_keys=config_keys)
        if cls._available is None or not cls._keys:
            raise RuntimeError("No API key configured for local DeepSeek service.")
        try:
            if timeout_sec is None:
                return cls._available.get(block=True)
            return cls._available.get(block=True, timeout=timeout_sec)
        except queue.Empty as e:
            raise RuntimeError("Timed out while waiting for a local DeepSeek worker slot.") from e

    @classmethod
    def release_key(cls, api_key: str) -> None:
        if api_key and cls._available is not None:
            cls._available.put(api_key)

    @classmethod
    @contextmanager
    def leased_key(cls, config_keys: Iterable[str] | None = None, timeout_sec: float | None = None):
        api_key = cls.acquire_key(config_keys=config_keys, timeout_sec=timeout_sec)
        try:
            yield api_key
        finally:
            cls.release_key(api_key)

    @classmethod
    def key_count(cls, config_keys: Iterable[str] | None = None) -> int:
        if not cls._initialized:
            cls.initialize(config_keys=config_keys)
        return len(cls._keys)
