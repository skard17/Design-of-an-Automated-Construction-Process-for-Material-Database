"""Generate a frozen external knowledge pack, then launch a fresh Step8 campaign."""

import argparse
import json
import os
from pathlib import Path
import sys

import run_three_version_parallel_step8 as runner
import task_domain_knowledge as knowledge
from qiniu_model_client import QiniuModelClient


def build_parser():
    parser = runner.build_parser()
    parser.add_argument("--research-output", required=True)
    parser.add_argument("--research-provider", choices=["crossref", "brave"], default="crossref")
    parser.add_argument("--research-approved-host", action="append", default=[])
    parser.add_argument("--research-max-concepts", type=int, default=40)
    parser.add_argument("--allow-partial-knowledge", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.launch_prepared or args.domain_knowledge_pack:
        raise ValueError("Automatic research requires fresh preparation, not a supplied pack or prepared launch")
    credentials = runner.load_runtime_credentials(args, {})
    client = QiniuModelClient(api_key=credentials["CODE_AGENT_API_KEY"],
                              base_url=credentials["CODE_AGENT_BASE_URL"], model=credentials["CODE_AGENT_MODEL"],
                              read_timeout=180, max_retries=1)
    searcher = knowledge.crossref_search
    if args.research_provider == "brave":
        token = os.environ.get("BRAVE_SEARCH_API_KEY", "")
        if not token or not args.research_approved_host:
            raise ValueError("Brave research requires a configured search key and approved hosts")
        searcher = lambda query: knowledge.brave_search(query, args.research_approved_host, token)
    task = {"database_goal": runner.TASK["objective"], "discipline": runner.TASK["discipline"],
            "query_requirements": [runner.TASK["query_requirements"]]}
    pack = knowledge.research(task, client, searcher=searcher, max_concepts=args.research_max_concepts, output=args.research_output)
    if pack["status"] != "ready_for_design" and not args.allow_partial_knowledge:
        print(json.dumps({"status": "knowledge_review_required", "knowledge_pack": str(Path(args.research_output) / "knowledge_pack.json")}))
        return 2
    args.domain_knowledge_pack = str(Path(args.research_output).resolve() / "knowledge_pack.json")
    os.environ["FIELD_DEFINITIONS_REQUIRED"] = "1"
    campaign = runner.prepare_campaign(args)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "output_root": args.output_root}))
        return 0
    return runner.launch_and_wait(campaign, Path(args.output_root).resolve(), credentials, sys.executable)


if __name__ == "__main__":
    raise SystemExit(main())
