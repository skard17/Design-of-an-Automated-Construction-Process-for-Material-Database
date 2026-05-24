from __future__ import annotations
import re
import os
from typing import Mapping


def load_prompt_template(prompts_dir: str, filename: str) -> str:
    path = os.path.join(prompts_dir, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _replace_placeholder(text: str, key: str, value: str) -> str:
    # 同时支持 {x} 和 {{x}} 两种占位符。
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
    # 如果 prompt 没有定义占位符，则直接在末尾追加一个带标签的文本块。
    return prompt + "\n\n" + f"<paper>\n{paper_text}\n</paper>"


def inject_section0_json(template: str, section0_json: str) -> str:
    # 允许多个可能的占位符名字，降低 prompt 与代码的耦合。
    prompt = template
    keys = ["section0", "section0_json", "section0_output", "s0_json", "s0_output"]
    if any((f"{{{k}}}" in prompt or f"{{{{{k}}}}}" in prompt) for k in keys):
        for k in keys:
            prompt = _replace_placeholder(prompt, k, section0_json)
        return prompt
    return prompt + "\n\n" + f"<section0>\n{section0_json}\n</section0>"


def inject_single_system_result(template: str, single_system_result: str) -> str:
    prompt = template
    keys = ["single_system_result", "single_system_json", "single_system_output"]
    if any((f"{{{k}}}" in prompt or f"{{{{{k}}}}}" in prompt) for k in keys):
        for k in keys:
            prompt = _replace_placeholder(prompt, k, single_system_result)
        return prompt
    return prompt + "\n\n" + f"<single_system>\n{single_system_result}\n</single_system>"


def inject_supplementary_information(template: str, supplementary_information: str | None) -> str:
    value = str(supplementary_information or "")
    if value.strip():
        return inject_blocks(template, {"supplementary_information": value})
    return _strip_empty_supplementary_section(template)


def _strip_empty_supplementary_section(text: str) -> str:
    import re

    out = text
    out = re.sub(
        r"(?m)^[ \t]*#+\s*Supplementary information.*\n[ \t]*\{\{supplementary_information\}\}\s*\n?",
        "",
        out,
    )
    out = re.sub(
        r"(?m)^[ \t]*#+\s*Supplementary information.*\n[ \t]*\{supplementary_information\}\s*\n?",
        "",
        out,
    )
    out = re.sub(r"(?m)^[ \t]*\{\{supplementary_information\}\}\s*\n?", "", out)
    out = re.sub(r"(?m)^[ \t]*\{supplementary_information\}\s*\n?", "", out)
    return out


def clean_llm_output(text: str) -> str:
    if not text: return ""
    t = text.strip()
    json_block_pattern = r'```(?:json)?\s*\n(.*?)\n```'
    match = re.search(json_block_pattern, t, re.DOTALL | re.IGNORECASE)
    if match:
        t = match.group(1).strip()
    else:
        if t.startswith("```json"): t = t[7:].strip()
        elif t.startswith("```"): t = t[3:].strip()
    if t.endswith("```"): t = t[:-3].strip()
    t = re.sub(r'```.*?```', '', t, flags=re.DOTALL)
    t = re.sub(r'`([^`]*)`', r'\1', t)
    return t.strip()