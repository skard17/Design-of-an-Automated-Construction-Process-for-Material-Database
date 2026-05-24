from __future__ import annotations
import re
import os
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
    for k, v in blocks.items():
        if v is None:
            v = ""
        prompt = _replace_placeholder(prompt, k, str(v))
    return prompt


def inject_paper_text(template: str, paper_text: str, tag: str = "paper_text") -> str:
    prompt = template
    if f"{{{tag}}}" in prompt or f"{{{{{tag}}}}}" in prompt:
        return _replace_placeholder(prompt, tag, paper_text)
    return prompt + "\n\n" + f"<paper>\n{paper_text}\n</paper>"


def inject_fabrication_extraction1(template: str, fabrication_json: str) -> str:
    """注入第一层粗抽结果到模板。"""
    prompt = template
    keys = ["fabrication_extracton1", "fabrication_extraction1"]
    if any((f"{{{k}}}" in prompt or f"{{{{{k}}}}}" in prompt) for k in keys):
        for k in keys:
            prompt = _replace_placeholder(prompt, k, fabrication_json)
        return prompt
    return prompt + "\n\n" + f"<fabrication_extraction1>\n{fabrication_json}\n</fabrication_extraction1>"


def inject_supplementary_information(template: str, supplementary_information: str | None) -> str:
    """
    注入补充材料到模板。
    如果没有补充材料，删除整个 supplementary information 部分（包括标题）。
    """
    import re

    value = (supplementary_information or "").strip()

    if value:
        return inject_blocks(template, {"supplementary_information": value})

    patterns = [
        r"\n+###\s*Supplementary\s+information\s*\(if\s+provided\)\s*\n*\{\{supplementary_information\}\}",
        r"\n+###\s*Supplementary\s+information\s*\(if\s+provided\)\s*\n*\{supplementary_information\}",
        r"\n+###\s*Supplementary\s+information\s*\n*\{\{supplementary_information\}\}",
        r"\n+###\s*Supplementary\s+information\s*\n*\{supplementary_information\}",
        r"\{\{supplementary_information\}\}",
        r"\{supplementary_information\}",
    ]

    result = template
    for pattern in patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)

    return result.rstrip()


def clean_llm_output(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    
    if t.startswith("```json"):
        t = t[7:].strip()
    elif t.startswith("```"):
        t = t[3:].strip()
    
    if t.endswith("```"):
        t = t[:-3].strip()

    # Additional cleanup for markdown/LaTeX artifacts that might break JSON
    import re
    # Remove any remaining markdown code blocks or inline code
    t = re.sub(r'```.*?```', '', t, flags=re.DOTALL)
    t = re.sub(r'`([^`]*)`', r'\1', t)  # Convert inline code to plain text

    return t.strip()