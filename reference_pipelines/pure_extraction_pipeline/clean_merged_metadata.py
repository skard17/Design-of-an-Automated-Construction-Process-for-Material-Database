from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "IE_5_unicode" / "prompts" / "mode3_transform.md"

LOCAL_BASE_URL = os.getenv(
    "LOCAL_DEEPSEEK_BASE_URL",
    "https://10.140.158.153:1020/dsr1/all/v1/chat/completions",
)
LOCAL_API_KEY = os.getenv(
    "LOCAL_DEEPSEEK_API_KEY",
    "",
)
LOCAL_MODEL = os.getenv("LOCAL_DEEPSEEK_MODEL", "deepseek-r1-huawei-910b")
VERIFY_SSL = os.getenv("LOCAL_DEEPSEEK_VERIFY_SSL", "false").lower() in {"1", "true", "yes"}

SYSTEM_PROMPT = (
    "You are a strict text normalizer.\n"
    "Return only transformed text content, with no explanation."
)


def resolve_chat_url(base_url: str) -> str:
    url = (base_url or "").strip() or LOCAL_BASE_URL
    if url.endswith("/chat/completions"):
        return url
    return url.rstrip("/") + "/chat/completions"


def iter_json_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for file_path in sorted(path.rglob("*.json")):
        if file_path.is_file():
            yield file_path


def extract_key_name(field_path: str) -> str:
    tail = field_path.split(".")[-1]
    tail = re.sub(r"\[\d+\]", "", tail)
    return tail or "value"


def load_prompt_template() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")
    content = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not content:
        raise RuntimeError(f"Prompt file is empty: {PROMPT_PATH}")
    return content


def build_prompt(template: str, field_path: str, text: str) -> str:
    key_name = extract_key_name(field_path)
    input_pair_json = json.dumps({key_name: text}, ensure_ascii=False)
    prompt = template
    prompt = prompt.replace("{{part}}", "metadata")
    prompt = prompt.replace("{{rel_path}}", "paper_info.metadata")
    prompt = prompt.replace("{{field_path}}", field_path)
    prompt = prompt.replace("{{key_name}}", key_name)
    prompt = prompt.replace("{{input_pair_json}}", input_pair_json)
    prompt = prompt.replace("{{text}}", text)
    return prompt


def parse_llm_json_pair(content: str, expected_key: str) -> str:
    text = (content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Response must be a JSON object, got {type(obj).__name__}")
    if len(obj) != 1:
        raise RuntimeError(f"Response must contain exactly one key-value pair, got keys={list(obj.keys())}")
    if expected_key not in obj:
        raise RuntimeError(f"Response missing expected key '{expected_key}'")
    value = obj[expected_key]
    if not isinstance(value, str):
        raise RuntimeError(f"Response value for key '{expected_key}' must be string, got {type(value).__name__}")
    return value


def transform_text(text: str, field_path: str, prompt_template: str, max_retries: int, timeout_sec: int) -> str:
    key_name = extract_key_name(field_path)
    prompt = build_prompt(prompt_template, field_path, text)
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                resolve_chat_url(LOCAL_BASE_URL),
                json={
                    "model": LOCAL_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                    "stream": False,
                },
                headers={
                    "Authorization": f"Bearer {LOCAL_API_KEY}",
                    "Content-Type": "application/json",
                },
                verify=VERIFY_SSL,
                timeout=timeout_sec,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"] if payload.get("choices") else ""
            return parse_llm_json_pair(content, key_name)
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(2 ** attempt, 20))

    raise RuntimeError(f"metadata mode3 transform failed for {field_path}: {last_error}")


def transform_metadata(value: Any, field_path: str, prompt_template: str, max_retries: int, timeout_sec: int) -> Any:
    if isinstance(value, str):
        return transform_text(value, field_path, prompt_template, max_retries, timeout_sec)
    if isinstance(value, list):
        return [
            transform_metadata(item, f"{field_path}[{idx}]", prompt_template, max_retries, timeout_sec)
            for idx, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: transform_metadata(sub_value, f"{field_path}.{key}", prompt_template, max_retries, timeout_sec)
            for key, sub_value in value.items()
        }
    return value


def clean_one(input_path: Path, output_path: Path, prompt_template: str, max_retries: int, timeout_sec: int) -> bool:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return False

    paper_info = data.get("paper_info")
    if not isinstance(paper_info, dict):
        return False

    metadata = paper_info.get("metadata")
    if not isinstance(metadata, dict):
        return False

    paper_info["metadata"] = transform_metadata(
        metadata,
        "paper_info.metadata",
        prompt_template,
        max_retries,
        timeout_sec,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean only paper_info.metadata in merged JSON files with mode3 LLM logic.")
    parser.add_argument("--input", required=True, help="Merged JSON file or directory.")
    parser.add_argument("--output", required=True, help="Output file or directory.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per metadata string field.")
    parser.add_argument("--timeout-sec", type=int, default=60, help="Request timeout for each field.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    prompt_template = load_prompt_template()

    if input_path.is_file():
        clean_one(input_path, output_path, prompt_template, args.max_retries, args.timeout_sec)
        return

    output_path.mkdir(parents=True, exist_ok=True)
    for file_path in iter_json_files(input_path):
        relative = file_path.relative_to(input_path)
        clean_one(file_path, output_path / relative, prompt_template, args.max_retries, args.timeout_sec)


if __name__ == "__main__":
    main()
