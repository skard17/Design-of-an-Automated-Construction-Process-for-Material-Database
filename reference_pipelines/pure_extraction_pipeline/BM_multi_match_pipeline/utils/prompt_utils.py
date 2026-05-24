from __future__ import annotations

import os
import re
from typing import Mapping


def load_prompt_template(prompts_dir: str, filename: str) -> str:
    path = os.path.join(prompts_dir, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _replace_placeholder(text: str, key: str, value: str) -> str:
    out = text.replace(f"{{{key}}}", value)
    out = out.replace(f"{{{{{key}}}}}", value)
    return out


def inject_blocks(template: str, blocks: Mapping[str, str]) -> str:
    prompt = template
    for key, value in blocks.items():
        prompt = _replace_placeholder(prompt, key, str(value or ""))
    return prompt


def clean_llm_output(text: str) -> str:
    if not text:
        return ""
    value = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", value, re.DOTALL | re.IGNORECASE)
    if match:
        value = match.group(1).strip()
    value = re.sub(r"^```json", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"^```", "", value).strip()
    value = re.sub(r"```$", "", value).strip()
    return value
