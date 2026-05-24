"""Schema-driven preprocessing with mode 0/1/2/3 field transforms.

Modes:
- 0: keep original value (no transform)
- 1: replace underscores with spaces
- 2: apply built-in unicode/LaTeX normalization
- 3: apply LLM-based transform (prompt + config driven)

Behavior:
- Only fields declared in schema_unicode_transfer/schema_<part>.json are transformed.
- If a schema key is assigned a mode number (0/1/2/3), that mode applies to the whole value subtree under
    that key (all descendant child keys inherit the same mode).
- Fields not declared in schema are kept unchanged (default no-op).
- JSON files with effectively empty payload (all-null/blank leaves, or empty list/dict trees) are skipped
    from transform and directly mirrored to outputs.
- Input/output directory trees are mirrored: inputs/<part>/<file>.json -> outputs/<part>/<file>.json
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
import time
import requests
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Clear broken CA-bundle env vars inherited from Windows shells.
for _env_name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    os.environ.pop(_env_name, None)

BASE_DIR = Path(__file__).resolve().parent
INPUT_ROOT = BASE_DIR / "inputs"
OUTPUT_ROOT = BASE_DIR / "outputs"
LOG_ROOT = BASE_DIR / "logs"
LIST_PATH = BASE_DIR / "preprocess_list.txt"
CONFIG_PATH = BASE_DIR / "config.yaml"
SCHEMA_ROOT_DEFAULT = BASE_DIR / "schema_unicode_transfer"
LOCAL_BASE_URL = os.getenv(
    "SILICONFLOW_BASE_URL",
    os.getenv("LOCAL_DEEPSEEK_BASE_URL", "https://api.siliconflow.cn/v1/chat/completions"),
)
LOCAL_API_KEY = os.getenv(
    "SILICONFLOW_API_KEY",
    os.getenv("LOCAL_DEEPSEEK_API_KEY", ""),
)
LOCAL_MODEL = os.getenv("SILICONFLOW_MODEL", os.getenv("LOCAL_DEEPSEEK_MODEL", "Pro/deepseek-ai/DeepSeek-V3.2"))
LOCAL_MAX_PARALLEL = max(1, int(os.getenv("SILICONFLOW_MAX_PARALLEL", os.getenv("LOCAL_DEEPSEEK_MAX_PARALLEL", "6"))))
VERIFY_SSL = os.getenv("SILICONFLOW_VERIFY_SSL", os.getenv("LOCAL_DEEPSEEK_VERIFY_SSL", "true")).lower() in {"1", "true", "yes"}

MODE_KEEP = 0
MODE_UNDERSCORE_TO_SPACE = 1
MODE_BUILTIN_UNICODE = 2
MODE_LLM = 3
VALID_MODES = {MODE_KEEP, MODE_UNDERSCORE_TO_SPACE, MODE_BUILTIN_UNICODE, MODE_LLM}
MISSING = object()

STATUS_INVALID_JSON = "invalid_json"
STATUS_TRANSFORM_FAILED = "transform_failed"
STATUS_SKIPPED_NO_SCHEMA = "skipped_no_schema"
STATUS_SKIPPED_EMPTY_INPUT = "skipped_empty_input"
STATUS_SUCCESS = "success"

Status = Literal[
    "invalid_json",
    "transform_failed",
    "skipped_no_schema",
    "skipped_empty_input",
    "success",
]


# -------------------------- Logging / IO --------------------------

def create_logger() -> logging.Logger:
    logger = logging.getLogger("preprocess_json")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_ROOT / f"preprocess_{ts}.log"

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    logger.info("log_file=%s", log_path)
    return logger


def iter_json_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.json") if p.is_file())


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_list(entries: list[str], target_path: Path = LIST_PATH) -> None:
    content = "\n".join(entries) + ("\n" if entries else "")
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(target_path)


def is_effectively_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (int, float, bool)):
        return False
    if isinstance(value, list):
        return (not value) or all(is_effectively_empty(item) for item in value)
    if isinstance(value, dict):
        return (not value) or all(is_effectively_empty(item) for item in value.values())
    return False


def format_list_entry(rel_path: str, status: Status, detail: str | None = None) -> str:
    if detail:
        return f"{rel_path}\t{status}\t{detail}"
    return f"{rel_path}\t{status}"


ENTRY_STATUS_ORDER = {
    STATUS_INVALID_JSON: 0,
    STATUS_TRANSFORM_FAILED: 1,
    STATUS_SKIPPED_NO_SCHEMA: 2,
    STATUS_SKIPPED_EMPTY_INPUT: 3,
    STATUS_SUCCESS: 4,
}


def entry_sort_key(entry: str) -> tuple[int, str]:
    parts = entry.split("\t", 2)
    status = parts[1] if len(parts) > 1 else ""
    return ENTRY_STATUS_ORDER.get(status, 99), entry


def normalize_field_key(key: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", key.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def resolve_chat_url(base_url: str) -> str:
    url = (base_url or "").strip() or LOCAL_BASE_URL
    if url.endswith("/chat/completions"):
        return url
    return url.rstrip("/") + "/chat/completions"


# -------------------------- Config --------------------------

@dataclass(frozen=True)
class LLMConfig:
    model: str
    base_url: str
    api_keys: tuple[str, ...]
    prompt_file: Path
    system_prompt: str
    temperature: float
    timeout_sec: int
    max_workers: int
    max_llm_inflight: int
    max_retries: int
    retry_base_delay_sec: float
    retry_max_delay_sec: float


@dataclass(frozen=True)
class AppConfig:
    schema_root: Path
    llm: LLMConfig


def _normalize_key_list(raw_keys: list[str]) -> tuple[str, ...]:
    dedup: list[str] = []
    seen: set[str] = set()
    for key in raw_keys:
        value = (key or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        dedup.append(value)
    return tuple(dedup)


def _resolve_api_keys(raw: dict[str, Any]) -> tuple[str, ...]:
    local_api_key = (os.getenv("LOCAL_DEEPSEEK_API_KEY") or LOCAL_API_KEY).strip()
    if local_api_key:
        return tuple([local_api_key] * LOCAL_MAX_PARALLEL)

    keys: list[str] = []

    list_from_config = raw.get("api_keys")
    if isinstance(list_from_config, list):
        keys.extend(str(k) for k in list_from_config)

    single = str(raw.get("api_key", "") or "").strip()
    if single:
        keys.append(single)

    env_multi = os.getenv("LLM_API_KEYS", "") or ""
    if env_multi.strip():
        keys.extend(x.strip() for x in env_multi.split(","))

    env_single = (os.getenv("LLM_API_KEY", "") or "").strip()
    if env_single:
        keys.append(env_single)

    return _normalize_key_list(keys)


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        raise RuntimeError(f"missing config file: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("config root must be an object")

    llm_raw = raw.get("llm", {})
    if not isinstance(llm_raw, dict):
        raise RuntimeError("config.llm must be an object")

    prompt_file_raw = str(llm_raw.get("prompt_file", "prompts/mode3_transform.md"))
    llm = LLMConfig(
        model=str(os.getenv("LOCAL_DEEPSEEK_MODEL", llm_raw.get("model", LOCAL_MODEL))),
        base_url=str(os.getenv("LOCAL_DEEPSEEK_BASE_URL", llm_raw.get("base_url", LOCAL_BASE_URL))),
        api_keys=_resolve_api_keys(llm_raw) or tuple([LOCAL_API_KEY] * LOCAL_MAX_PARALLEL),
        prompt_file=(BASE_DIR / prompt_file_raw).resolve(),
        system_prompt=str(llm_raw.get("system_prompt", "You are a precise text normalization assistant.")),
        temperature=float(llm_raw.get("temperature", 0.0)),
        timeout_sec=int(llm_raw.get("timeout_sec", 60)),
        max_workers=max(1, min(int(llm_raw.get("max_workers", LOCAL_MAX_PARALLEL)), LOCAL_MAX_PARALLEL)),
        max_llm_inflight=max(1, min(int(llm_raw.get("max_llm_inflight", LOCAL_MAX_PARALLEL)), LOCAL_MAX_PARALLEL)),
        max_retries=max(0, int(llm_raw.get("max_retries", 3))),
        retry_base_delay_sec=max(0.0, float(llm_raw.get("retry_base_delay_sec", 1.0))),
        retry_max_delay_sec=max(0.0, float(llm_raw.get("retry_max_delay_sec", 8.0))),
    )

    return AppConfig(schema_root=SCHEMA_ROOT_DEFAULT.resolve(), llm=llm)


# -------------------------- Mode 2: built-in normalization --------------------------

def decode_unicode(text: str) -> str:
    if "\\" not in text:
        return text
    try:
        return text.encode("utf-8").decode("unicode_escape", errors="replace")
    except Exception:
        return text


LATEX_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "zeta": "ζ", "eta": "η",
    "theta": "θ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "omicron": "ο", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
    "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    "times": "×", "cdot": "·", "pm": "±", "mp": "∓", "approx": "≈", "simeq": "≃", "sim": "∼",
    "equiv": "≡", "neq": "≠", "leq": "≤", "geq": "≥", "to": "→", "rightarrow": "→", "leftarrow": "←",
    "leftrightarrow": "↔", "partial": "∂", "nabla": "∇", "infty": "∞", "degree": "°", "circ": "°",
    "hbar": "ℏ", "ell": "ℓ", "ohm": "Ω", "angstrom": "Å", "ldots": "…", "cdots": "⋯",
}

SUPERSCRIPTS = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾", "n": "ⁿ", "i": "ⁱ",
    "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ", "e": "ᵉ", "f": "ᶠ", "g": "ᵍ", "h": "ʰ", "j": "ʲ", "k": "ᵏ",
    "l": "ˡ", "m": "ᵐ", "o": "ᵒ", "p": "ᵖ", "r": "ʳ", "s": "ˢ", "t": "ᵗ", "u": "ᵘ", "v": "ᵛ",
    "w": "ʷ", "x": "ˣ", "y": "ʸ", "z": "ᶻ",
}

SUBSCRIPTS = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎", "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
    "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ", "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
    "v": "ᵥ", "x": "ₓ",
}


def _map_chars(text: str, table: dict[str, str]) -> str:
    return "".join(table.get(ch, ch) for ch in text)


def replace_latex_commands(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return LATEX_SYMBOLS.get(match.group(1), match.group(0))

    return re.sub(r"\\([A-Za-z]+)", repl, text)


def replace_supersubs(text: str) -> str:
    placeholder = "__P_UNDERSCORE__"

    def _protect(match: re.Match[str]) -> str:
        return match.group(1) + placeholder + match.group(2)

    text = re.sub(r"([A-Za-z]{2,})_([A-Za-z]{2,})", _protect, text)
    text = re.sub(r"\^\{([^{}]+)\}", lambda m: _map_chars(m.group(1), SUPERSCRIPTS), text)
    text = re.sub(r"_\{([^{}]+)\}", lambda m: _map_chars(m.group(1), SUBSCRIPTS), text)
    text = re.sub(r"\^([+\-]?[0-9]+)", lambda m: _map_chars(m.group(1), SUPERSCRIPTS), text)
    text = re.sub(r"_([+\-]?[0-9]+)", lambda m: _map_chars(m.group(1), SUBSCRIPTS), text)
    text = re.sub(r"\^([A-Za-z0-9+\-])", lambda m: _map_chars(m.group(1), SUPERSCRIPTS), text)
    text = re.sub(r"_([A-Za-z0-9+\-])", lambda m: _map_chars(m.group(1), SUBSCRIPTS), text)
    return text.replace(placeholder, "_")


def normalize_text_builtin(text: str) -> str:
    decoded = decode_unicode(text)
    return replace_supersubs(replace_latex_commands(decoded))


# -------------------------- Transformers (mode 0/1/2/3) --------------------------

def transform_strings_deep(value: Any, text_fn) -> Any:
    if isinstance(value, str):
        return text_fn(value)
    if isinstance(value, list):
        return [transform_strings_deep(v, text_fn) for v in value]
    if isinstance(value, dict):
        return {k: transform_strings_deep(v, text_fn) for k, v in value.items()}
    return value


class LLMTransformer:
    def __init__(self, cfg: LLMConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.prompt_template = self._load_prompt_template(cfg.prompt_file)
        self._key_pool = APIKeyPool(cfg.api_keys)
        self._request_slots = threading.BoundedSemaphore(cfg.max_llm_inflight)
        self._cache: dict[str, str] = {}
        self._cache_lock = threading.Lock()
        self._inflight: dict[str, threading.Event] = {}
        self.request_count = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        message = str(exc).lower()
        retry_tokens = (
            "429",
            "rate limit",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "connection",
            "internal server error",
            "server error",
            "503",
            "502",
            "500",
        )
        return any(token in message for token in retry_tokens)

    def _wait_or_claim(self, cache_key: str) -> bool:
        while True:
            with self._cache_lock:
                if cache_key in self._cache:
                    self.cache_hit_count += 1
                    return False
                pending = self._inflight.get(cache_key)
                if pending is None:
                    self._inflight[cache_key] = threading.Event()
                    self.cache_miss_count += 1
                    return True
            pending.wait()

    @staticmethod
    def _load_prompt_template(path: Path) -> str:
        if not path.exists():
            raise RuntimeError(f"mode3 prompt file not found: {path}")
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            raise RuntimeError(f"mode3 prompt file is empty: {path}")
        return content

    @staticmethod
    def _extract_key_name(field_path: str) -> str:
        if not field_path:
            return "value"
        tail = field_path.split(".")[-1]
        tail = re.sub(r"\[\d+\]", "", tail)
        return tail or "value"

    @staticmethod
    def _parse_llm_json_pair(content: str, expected_key: str) -> str:
        text = (content or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            obj = json.loads(text)
        except Exception as exc:
            raise RuntimeError(f"mode3 response is not valid JSON: {exc}; content={text!r}") from exc

        if not isinstance(obj, dict):
            raise RuntimeError(f"mode3 response must be a JSON object, got {type(obj).__name__}")
        if len(obj) != 1:
            raise RuntimeError(
                f"mode3 response must contain exactly one key-value pair, got keys={list(obj.keys())}"
            )
        if expected_key not in obj:
            raise RuntimeError(f"mode3 response missing expected key '{expected_key}'")

        value = obj[expected_key]
        if not isinstance(value, str):
            raise RuntimeError(
                f"mode3 response value for key '{expected_key}' must be string, got {type(value).__name__}"
            )
        return value

    def _build_prompt(self, part: str, rel_path: str, field_path: str, text: str) -> str:
        key_name = self._extract_key_name(field_path)
        input_pair_json = json.dumps({key_name: text}, ensure_ascii=False)
        prompt = self.prompt_template
        prompt = prompt.replace("{{part}}", part)
        prompt = prompt.replace("{{rel_path}}", rel_path)
        prompt = prompt.replace("{{field_path}}", field_path)
        prompt = prompt.replace("{{key_name}}", key_name)
        prompt = prompt.replace("{{input_pair_json}}", input_pair_json)
        prompt = prompt.replace("{{text}}", text)
        return prompt

    def transform_text(self, text: str, part: str, rel_path: str, field_path: str) -> str:
        cache_key = f"{part}|{field_path}|{text}"

        is_leader = self._wait_or_claim(cache_key)
        if not is_leader:
            with self._cache_lock:
                cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
            is_leader = self._wait_or_claim(cache_key)
            if not is_leader:
                with self._cache_lock:
                    cached = self._cache.get(cache_key)
                if cached is not None:
                    return cached

        key_name = self._extract_key_name(field_path)
        prompt = self._build_prompt(
            part=part,
            rel_path=rel_path,
            field_path=field_path,
            text=text,
        )

        try:
            last_error: Exception | None = None
            for attempt in range(self.cfg.max_retries + 1):
                try:
                    with self._request_slots:
                        with self._key_pool.lease() as api_key:
                            with self._cache_lock:
                                self.request_count += 1
                            response = requests.post(
                                resolve_chat_url(self.cfg.base_url),
                                json={
                                    "model": self.cfg.model,
                                    "messages": [
                                        {"role": "system", "content": self.cfg.system_prompt},
                                        {"role": "user", "content": prompt},
                                    ],
                                    "temperature": self.cfg.temperature,
                                    "stream": False,
                                },
                                headers={
                                    "Authorization": f"Bearer {api_key}",
                                    "Content-Type": "application/json",
                                },
                                verify=VERIFY_SSL,
                                timeout=self.cfg.timeout_sec,
                            )
                            response.raise_for_status()
                            payload = response.json()
                    content = payload["choices"][0]["message"]["content"] if payload.get("choices") else ""
                    result = self._parse_llm_json_pair(content=content, expected_key=key_name)
                    with self._cache_lock:
                        self._cache[cache_key] = result
                    return result
                except Exception as exc:
                    last_error = exc
                    is_retryable = self._is_retryable_error(exc)
                    is_last_attempt = attempt >= self.cfg.max_retries
                    if (not is_retryable) or is_last_attempt:
                        break
                    delay = min(
                        self.cfg.retry_max_delay_sec,
                        self.cfg.retry_base_delay_sec * (2 ** attempt),
                    )
                    self.logger.warning(
                        "mode3_retry rel_path=%s field=%s attempt=%d/%d wait=%.2fs reason=%s",
                        rel_path,
                        field_path,
                        attempt + 1,
                        self.cfg.max_retries + 1,
                        delay,
                        exc,
                    )
                    if delay > 0:
                        time.sleep(delay)

            raise RuntimeError(f"mode3 llm request failed: {last_error}")
        finally:
            with self._cache_lock:
                pending = self._inflight.pop(cache_key, None)
            if pending is not None:
                pending.set()


class FieldTransformer:
    def __init__(self, llm_transformer: LLMTransformer | None):
        self.llm_transformer = llm_transformer
        self.mode_hit_counts: Counter[int] = Counter()
        self._mode_lock = threading.Lock()

    def apply_mode(self, value: Any, mode: int, part: str, rel_path: str, field_path: str) -> Any:
        with self._mode_lock:
            self.mode_hit_counts[mode] += 1
        if mode == MODE_KEEP:
            return value
        if mode == MODE_UNDERSCORE_TO_SPACE:
            return transform_strings_deep(value, lambda s: s.replace("_", " "))
        if mode == MODE_BUILTIN_UNICODE:
            return transform_strings_deep(value, normalize_text_builtin)
        if mode == MODE_LLM:
            if self.llm_transformer is None:
                raise RuntimeError("mode3 requested but llm is not configured")
            return transform_strings_deep(
                value,
                lambda s: self.llm_transformer.transform_text(
                    text=s,
                    part=part,
                    rel_path=rel_path,
                    field_path=field_path,
                ),
            )
        raise RuntimeError(f"unsupported mode: {mode}")


class APIKeyPool:
    def __init__(self, keys: tuple[str, ...]):
        if not keys:
            raise RuntimeError("mode3 requires at least one API key")
        self._queue: queue.Queue[str] = queue.Queue()
        for key in keys:
            self._queue.put(key)

    @contextmanager
    def lease(self) -> Iterable[str]:
        key = self._queue.get()
        try:
            yield key
        finally:
            self._queue.put(key)


# -------------------------- Schema-driven application --------------------------

def load_part_schema(schema_root: Path, part: str) -> Any:
    path = schema_root / f"schema_{part}.json"
    if not path.exists():
        raise RuntimeError(f"schema file not found for part={part}: {path}")

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise RuntimeError(f"schema file is empty for part={part}: {path}")

    try:
        return json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"invalid schema file {path}: {exc}") from exc


def contains_mode3(spec: Any) -> bool:
    if type(spec) is int:
        return spec == MODE_LLM
    if isinstance(spec, dict):
        return any(contains_mode3(v) for v in spec.values())
    if isinstance(spec, list):
        return any(contains_mode3(v) for v in spec)
    return False


def validate_schema_modes(spec: Any, path: str = "$") -> None:
    if type(spec) is int:
        if spec not in VALID_MODES:
            raise RuntimeError(f"invalid mode at {path}: {spec}, expected one of {sorted(VALID_MODES)}")
        return
    if isinstance(spec, dict):
        for key, value in spec.items():
            validate_schema_modes(value, f"{path}.{key}")
        return
    if isinstance(spec, list):
        for idx, value in enumerate(spec):
            validate_schema_modes(value, f"{path}[{idx}]")
        return
    raise RuntimeError(
        f"invalid schema node at {path}: type={type(spec).__name__}, expected int/dict/list"
    )


def apply_schema_transform(
    value: Any,
    spec: Any,
    transformer: FieldTransformer,
    part: str,
    rel_path: str,
    field_path: str,
) -> Any:
    # Leaf mode (applies to the entire value subtree; child keys inherit this mode).
    # This is value-type agnostic: string/list/dict under this key all follow the same mode.
    if type(spec) is int:
        return transformer.apply_mode(value=value, mode=spec, part=part, rel_path=rel_path, field_path=field_path)

    # Dict schema: only keys declared in schema are transformed; others untouched.
    if isinstance(spec, dict):
        if not isinstance(value, dict):
            return value
        normalized_spec_map: dict[str, tuple[str, Any]] = {}
        for schema_key, schema_value in spec.items():
            norm = normalize_field_key(schema_key)
            if norm and norm not in normalized_spec_map:
                normalized_spec_map[norm] = (schema_key, schema_value)

        out: dict[str, Any] = {}
        for key, child_value in value.items():
            child_spec = spec.get(key, MISSING)
            matched_schema_key = key

            if child_spec is MISSING:
                norm_key = normalize_field_key(key)
                matched = normalized_spec_map.get(norm_key)
                if matched is not None:
                    matched_schema_key, child_spec = matched

            if child_spec is MISSING:
                out[key] = child_value
            else:
                child_path = f"{field_path}.{matched_schema_key}" if field_path else matched_schema_key
                out[key] = apply_schema_transform(
                    value=child_value,
                    spec=child_spec,
                    transformer=transformer,
                    part=part,
                    rel_path=rel_path,
                    field_path=child_path,
                )
        return out

    # List schema uses first element as item template.
    if isinstance(spec, list):
        if not isinstance(value, list) or not spec:
            return value
        item_spec = spec[0]
        out_list: list[Any] = []
        for idx, item in enumerate(value):
            child_path = f"{field_path}[{idx}]"
            out_list.append(
                apply_schema_transform(
                    value=item,
                    spec=item_spec,
                    transformer=transformer,
                    part=part,
                    rel_path=rel_path,
                    field_path=child_path,
                )
            )
        return out_list

    # Unknown schema node -> no-op for safety.
    return value


# -------------------------- Main --------------------------

def _extract_part(rel: Path) -> str:
    if not rel.parts:
        return ""
    return rel.parts[0]


@dataclass(frozen=True)
class FileProcessResult:
    rel_path: str
    status: Status
    detail: str | None
    error_kind: str | None

    @property
    def success(self) -> bool:
        return self.status in {STATUS_SUCCESS, STATUS_SKIPPED_EMPTY_INPUT}

    def as_entry(self) -> str:
        return format_list_entry(self.rel_path, self.status, self.detail)


@dataclass
class RunStats:
    total: int = 0
    ok: int = 0
    failed: int = 0
    skipped_no_schema: int = 0
    skipped_empty_input: int = 0
    error_counts: Counter[str] = field(default_factory=Counter)

    def record_result(self, result: FileProcessResult) -> None:
        if result.status == STATUS_SUCCESS:
            self.ok += 1
            return
        if result.status == STATUS_SKIPPED_EMPTY_INPUT:
            self.skipped_empty_input += 1
            return
        self.failed += 1
        if result.error_kind:
            self.error_counts[result.error_kind] += 1


def build_schema_cache(
    *,
    json_files: list[Path],
    cfg: AppConfig,
    logger: logging.Logger,
) -> tuple[dict[str, Any], set[str]]:
    schema_cache: dict[str, Any] = {}
    missing_schema_parts: set[str] = set()

    part_names = sorted({_extract_part(p.relative_to(INPUT_ROOT)) for p in json_files})
    for part in part_names:
        schema_path = cfg.schema_root / f"schema_{part}.json"
        if not schema_path.exists():
            missing_schema_parts.add(part)
            logger.warning("schema_missing part=%s path=%s action=skip_part", part, schema_path)
            continue
        schema_spec = load_part_schema(cfg.schema_root, part)
        validate_schema_modes(schema_spec, path=f"schema_{part}")
        schema_cache[part] = schema_spec

    return schema_cache, missing_schema_parts


def collect_runnable_files(
    *,
    json_files: list[Path],
    missing_schema_parts: set[str],
    entries: list[str],
    stats: RunStats,
    logger: logging.Logger,
) -> list[Path]:
    runnable_files: list[Path] = []
    for input_path in json_files:
        rel = input_path.relative_to(INPUT_ROOT)
        rel_posix = rel.as_posix()
        part = _extract_part(rel)
        if part in missing_schema_parts:
            stats.skipped_no_schema += 1
            entries.append(format_list_entry(rel_posix, STATUS_SKIPPED_NO_SCHEMA, f"missing schema_{part}.json"))
            logger.warning("file_skipped_no_schema rel_path=%s part=%s schema=%s", rel_posix, part, f"schema_{part}.json")
            continue
        runnable_files.append(input_path)
    return runnable_files


def compute_worker_count(*, llm_needed: bool, runnable_count: int, llm_cfg: LLMConfig) -> int:
    if runnable_count <= 0:
        return 1
    llm_worker_cap = max(1, len(llm_cfg.api_keys)) if llm_needed else None
    return min(
        llm_cfg.max_workers if llm_needed else 4,
        runnable_count,
        llm_worker_cap if llm_worker_cap is not None else 4,
    )


def process_runnable_files(
    *,
    runnable_files: list[Path],
    schema_cache: dict[str, Any],
    field_transformer: FieldTransformer,
    workers: int,
    entries: list[str],
    stats: RunStats,
    logger: logging.Logger,
) -> None:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_one_file, input_path, schema_cache, field_transformer): input_path
            for input_path in runnable_files
        }

        for future in as_completed(futures):
            result = future.result()
            entries.append(result.as_entry())
            stats.record_result(result)
            if result.status == STATUS_SUCCESS:
                logger.info("processed %s", result.rel_path)
            elif result.status == STATUS_SKIPPED_EMPTY_INPUT:
                logger.info("skipped_empty_input %s", result.rel_path)
            else:
                logger.error("failed %s kind=%s", result.rel_path, result.error_kind)


def process_one_file(
    input_path: Path,
    schema_cache: dict[str, Any],
    field_transformer: FieldTransformer,
) -> FileProcessResult:
    rel = input_path.relative_to(INPUT_ROOT)
    rel_posix = rel.as_posix()
    part = _extract_part(rel)

    try:
        data = load_json(input_path)
    except Exception as exc:
        return FileProcessResult(
            rel_path=rel_posix,
            status=STATUS_INVALID_JSON,
            detail=str(exc),
            error_kind=STATUS_INVALID_JSON,
        )

    if is_effectively_empty(data):
        out_path = OUTPUT_ROOT / rel
        write_json(out_path, data)
        return FileProcessResult(
            rel_path=rel_posix,
            status=STATUS_SKIPPED_EMPTY_INPUT,
            detail="input payload is effectively empty",
            error_kind=None,
        )

    try:
        schema_spec = schema_cache[part]
        transformed = apply_schema_transform(
            value=data,
            spec=schema_spec,
            transformer=field_transformer,
            part=part,
            rel_path=rel_posix,
            field_path="",
        )
        out_path = OUTPUT_ROOT / rel
        write_json(out_path, transformed)
        return FileProcessResult(
            rel_path=rel_posix,
            status=STATUS_SUCCESS,
            detail=None,
            error_kind=None,
        )
    except Exception as exc:
        return FileProcessResult(
            rel_path=rel_posix,
            status=STATUS_TRANSFORM_FAILED,
            detail=str(exc),
            error_kind=STATUS_TRANSFORM_FAILED,
        )


def main() -> None:
    logger = create_logger()
    cfg = load_config(CONFIG_PATH)

    if not INPUT_ROOT.exists():
        raise RuntimeError(f"inputs folder not found: {INPUT_ROOT}")

    start_ts = time.perf_counter()
    entries: list[str] = []
    stats = RunStats()

    json_files = list(iter_json_files(INPUT_ROOT))
    schema_cache, missing_schema_parts = build_schema_cache(json_files=json_files, cfg=cfg, logger=logger)

    llm_needed = any(contains_mode3(spec) for spec in schema_cache.values())

    if llm_needed:
        if not cfg.llm.api_keys:
            raise RuntimeError("schema contains mode=3 fields, but no API key found (config.llm.api_key/api_keys or env LLM_API_KEY/LLM_API_KEYS)")

    llm_transformer = LLMTransformer(cfg.llm, logger) if llm_needed else None
    field_transformer = FieldTransformer(llm_transformer=llm_transformer)

    runnable_files = collect_runnable_files(
        json_files=json_files,
        missing_schema_parts=missing_schema_parts,
        entries=entries,
        stats=stats,
        logger=logger,
    )
    workers = compute_worker_count(llm_needed=llm_needed, runnable_count=len(runnable_files), llm_cfg=cfg.llm)

    logger.info(
        "run_workers=%d llm_inflight=%d files_total=%d files_runnable=%d skipped_no_schema=%d llm_needed=%s llm_key_count=%d",
        workers,
        cfg.llm.max_llm_inflight,
        len(json_files),
        len(runnable_files),
        stats.skipped_no_schema,
        llm_needed,
        len(cfg.llm.api_keys),
    )

    stats.total = len(json_files)

    process_runnable_files(
        runnable_files=runnable_files,
        schema_cache=schema_cache,
        field_transformer=field_transformer,
        workers=workers,
        entries=entries,
        stats=stats,
        logger=logger,
    )

    entries.sort(key=entry_sort_key)
    write_list(entries)

    duration = time.perf_counter() - start_ts
    logger.info(
        "summary total=%d ok=%d failed=%d skipped_no_schema=%d skipped_empty_input=%d duration=%.3fs list=%s errors=%s",
        stats.total,
        stats.ok,
        stats.failed,
        stats.skipped_no_schema,
        stats.skipped_empty_input,
        duration,
        LIST_PATH,
        dict(stats.error_counts),
    )
    logger.info(
        "mode_hits mode0=%d mode1=%d mode2=%d mode3=%d",
        field_transformer.mode_hit_counts.get(MODE_KEEP, 0),
        field_transformer.mode_hit_counts.get(MODE_UNDERSCORE_TO_SPACE, 0),
        field_transformer.mode_hit_counts.get(MODE_BUILTIN_UNICODE, 0),
        field_transformer.mode_hit_counts.get(MODE_LLM, 0),
    )
    if llm_transformer is not None:
        logger.info(
            "llm_usage requests=%d cache_hit=%d cache_miss=%d cache_size=%d",
            llm_transformer.request_count,
            llm_transformer.cache_hit_count,
            llm_transformer.cache_miss_count,
            len(llm_transformer._cache),
        )


if __name__ == "__main__":
    main()
