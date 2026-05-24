from __future__ import annotations

import itertools
import os
import threading


class KeyPool:
    _lock = threading.Lock()
    _cycle = itertools.cycle([""])
    _initialized = False

    @classmethod
    def initialize(cls, config_keys: list[str] | None = None) -> None:
        with cls._lock:
            keys = [k for k in (config_keys or []) if str(k).strip()]
            env_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
            if env_key:
                keys.insert(0, env_key)
            if not keys:
                keys = [""]
            cls._cycle = itertools.cycle(keys)
            cls._initialized = True

    @classmethod
    def get_next_key(cls, config_keys: list[str] | None = None) -> str:
        if not cls._initialized:
            cls.initialize(config_keys)
        with cls._lock:
            return next(cls._cycle)
