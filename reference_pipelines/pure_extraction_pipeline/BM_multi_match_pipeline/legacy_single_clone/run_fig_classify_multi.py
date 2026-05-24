from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_prompt_bank import load_multi_prompt
from utils.ds_client import chat
from utils.io_utils import ensure_dir, load_config, load_json, load_supplementary_information, read_text, select_paper_file, write_text
from utils.key_pool import KeyPool
from utils.log_utils import setup_logger
from utils.prompt_utils import clean_llm_output, inject_blocks

SCOPE_DIR = Path(os.getenv("BM_MULTI_SCOPE_DIR", str(ROOT / "outputs" / "multi_scope")))
FIG_DIR = Path(os.getenv("BM_MULTI_FIG_DIR", str(ROOT / "outputs" / "multi_fig_classify")))


def run_fig_classify_multi(paper_id: str, multi_system_json: str | None = None) -> str:
    logger = setup_logger("fig_classify_multi")
    config = load_config(str(ROOT / "config.yaml"))
    provider = config["provider"]
    papers_dir = str((ROOT / config["paths"]["papers_dir"]).resolve())

    selected_paper_id, paper_path = select_paper_file({"papers_dir": papers_dir}, paper_id)
    paper_text = read_text(paper_path)
    supplementary_information = load_supplementary_information(paper_path, selected_paper_id)
    if multi_system_json:
        multi_system_result = load_json(multi_system_json)
    else:
        multi_system_result = load_json(str(SCOPE_DIR / selected_paper_id / "multi_system_result.json"))

    prompt = inject_blocks(
        load_multi_prompt("fig_classify"),
        {
            "multi_system_result": json.dumps(multi_system_result, ensure_ascii=False, indent=2),
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
    out_dir = FIG_DIR / selected_paper_id
    ensure_dir(str(out_dir))
    out_path = out_dir / "figure_classification.json"
    write_text(str(out_path), json.dumps(json.loads(out_text), ensure_ascii=False, indent=2))
    logger.info("done output=%s", out_path)
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-system figure classification prompt from multi-example.txt")
    parser.add_argument("--paper_id", required=True)
    parser.add_argument("--multi_system_json", default=None)
    args = parser.parse_args()
    print(run_fig_classify_multi(args.paper_id, args.multi_system_json))


if __name__ == "__main__":
    main()
