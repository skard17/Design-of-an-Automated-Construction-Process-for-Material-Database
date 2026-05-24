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


def inject_figure_classification(template: str, figure_classification: str) -> str:
    prompt = template
    keys = ["figure_classification", "figure_classify", "fig_classify"]
    if any((f"{{{k}}}" in prompt or f"{{{{{k}}}}}" in prompt) for k in keys):
        for k in keys:
            prompt = _replace_placeholder(prompt, k, figure_classification)
        return prompt
    return prompt + "\n\n" + f"<figure_classification>\n{figure_classification}\n</figure_classification>"


def inject_supplementary_information(template: str, supplementary_information: str | None) -> str:
    value = str(supplementary_information or "")
    if value.strip():
        return inject_blocks(template, {"supplementary_information": value})  # ✅ 添加4个空格缩进
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
    if not text:
        return ""
    t = text.strip()

    # First, try to extract JSON from code blocks
    import re
    json_block_pattern = r'```(?:json)?\s*\n(.*?)\n```'
    match = re.search(json_block_pattern, t, re.DOTALL | re.IGNORECASE)
    if match:
        # Found a JSON code block, extract its content
        t = match.group(1).strip()
    else:
        # No code block found, remove code block markers if they exist at start/end
        if t.startswith("```json"):  # ✅ 添加8个空格缩进（在else块内）
            t = t[7:].strip()
        elif t.startswith("```"):
            t = t[3:].strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    return t.strip()


def _remove_json_comments(text: str) -> str:
    out: list[str] = []
    i = 0
    in_str = False
    escape = False
    length = len(text)
    while i < length:
        c = text[i]
        if in_str:
            out.append(c)
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            i += 1
            continue

        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue

        if c == "/" and i + 1 < length:
            n = text[i + 1]
            if n == "/":
                i += 2
                while i < length and text[i] not in "\r\n":
                    i += 1
                continue
            if n == "*":
                i += 2
                while i + 1 < length and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                i += 2 if i + 1 < length else 0
                continue

        out.append(c)
        i += 1
    return "".join(out)


def _extract_json_fragment(text: str) -> str:
    first_obj = text.find("{")
    first_arr = text.find("[")
    if first_obj == -1 and first_arr == -1:
        return ""
    if first_obj == -1:
        start = first_arr
    elif first_arr == -1:
        start = first_obj
    else:
        start = min(first_obj, first_arr)

    stack: list[str] = []
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue

        if c == '"':
            in_str = True
            continue

        if c in "{[":
            stack.append(c)
        elif c in "}]":
            if not stack:
                return ""
            open_c = stack.pop()
            if (open_c == "{" and c != "}") or (open_c == "[" and c != "]"):
                return ""
            if not stack:
                return text[start : i + 1].strip()
    return ""


def repair_json_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip().lstrip("\ufeff")

    import re

    t = re.sub(r"[^\x20-\x7E\n\r\t]", "", t)
    t = _remove_json_comments(t)
    frag = _extract_json_fragment(t)
    if frag:
        return frag
    return t.strip()

