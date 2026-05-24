from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.ds_client import chat
from utils.io_utils import ensure_dir, load_config, load_supplementary_information, read_text, select_paper_file, write_text
from utils.key_pool import KeyPool
from utils.log_utils import setup_logger
from utils.prompt_utils import clean_llm_output, inject_blocks
from multi_prompt_bank import load_multi_prompt

MAX_PAPER_CHARS = 45000
MAX_SUPP_CHARS = 12000
SCOPE_DIR = Path(os.getenv("BM_MULTI_SCOPE_DIR", str(ROOT / "outputs" / "multi_scope")))


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[Text truncated for API safety]"


def run_ie0_multi(paper_id: str | None = None) -> str:
    logger = setup_logger("ie0_multi")
    config = load_config(str(ROOT / "config.yaml"))
    provider = config["provider"]
    papers_dir = str((ROOT / config["paths"]["papers_dir"]).resolve())

    selected_paper_id, paper_path = select_paper_file({"papers_dir": papers_dir}, paper_id)
    paper_text = read_text(paper_path)
    supplementary_information = load_supplementary_information(paper_path, selected_paper_id)
    paper_text = truncate_text(paper_text, MAX_PAPER_CHARS)
    supplementary_information = truncate_text(supplementary_information, MAX_SUPP_CHARS)
    template = load_multi_prompt("ie0_multi")
    prompt = inject_blocks(
        template,
        {
            "paper_text": paper_text,
            "supplementary_information": supplementary_information,
        },
    )

    api_key = KeyPool.get_next_key(config_keys=provider.get("api_keys"))
    response = chat(
        prompt=prompt,
        model=provider["model"],
        base_url=provider["base_url"],
        api_key=api_key,
        temperature=0.0,
        timeout_sec=int(os.getenv("BM_MULTI_TIMEOUT_SEC", str(provider.get("timeout_sec", 60)))),
        system_prompt="You are an expert in superconducting materials information extraction.",
    )
    out_text = clean_llm_output(response)
    out_dir = SCOPE_DIR / selected_paper_id
    ensure_dir(str(out_dir))
    out_path = out_dir / "multi_system_result.json"
    write_text(str(out_path), json.dumps(json.loads(out_text), ensure_ascii=False, indent=2))
    logger.info("done output=%s", out_path)
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IE_0_multi prompt and write paper-level multi_system_result.json")
    parser.add_argument("--paper_id", default=None)
    args = parser.parse_args()
    print(run_ie0_multi(args.paper_id))


if __name__ == "__main__":
    main()
