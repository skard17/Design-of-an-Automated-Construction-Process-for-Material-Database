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
    if not text:
        return ""
    t = text.strip()

    # Additional cleanup for markdown/LaTeX artifacts that might break JSON
    import re
    # Remove any remaining markdown code blocks or inline code
    t = re.sub(r'```.*?```', '', t, flags=re.DOTALL)
    t = re.sub(r'`([^`]*)`', r'\1', t)  # Convert inline code to plain text

    return t.strip()