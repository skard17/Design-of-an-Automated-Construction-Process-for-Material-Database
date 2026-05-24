from __future__ import annotations

import logging
import os
import time
from threading import Lock

from utils.io_utils import ensure_dir

_LOGGER_LOCK = Lock()


def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    with _LOGGER_LOCK:
        logger = logging.getLogger(name)
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)
        ensure_dir(log_dir)

        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"{name}_{ts}.log")

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        logger.propagate = False
        logger.info(f"log_file={log_path}")
        return logger
