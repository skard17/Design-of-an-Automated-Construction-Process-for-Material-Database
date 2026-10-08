import argparse
import json
import time
from pathlib import Path

from section_design_langgraph_human_gate import (
    write_schema_revision_snapshot,
    write_transition_snapshot,
)


TERMINAL_STATUSES = {"complete", "completed", "failed", "needs_review"}


def archive_current_revision(state_path):
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    schema_archive = write_schema_revision_snapshot(
        state,
        state.get("current_node") or "external_checkpoint_archive",
        state.get("next_node"),
    )
    event_archive = write_transition_snapshot(
        state,
        state.get("current_node") or "external_checkpoint_archive",
        state.get("next_node"),
    )
    return state, schema_archive, event_archive


def main():
    parser = argparse.ArgumentParser(
        description="Archive Step8 schema revisions from a live checkpoint without mutating it."
    )
    parser.add_argument("--state", required=True)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()

    last_signature = None
    while True:
        try:
            state, archived, event_archived = archive_current_revision(args.state)
            signature = (
                int(state.get("schema_revision", 0) or 0),
                str(archived or ""),
                str(event_archived or ""),
            )
            if signature != last_signature:
                print(
                    json.dumps(
                        {
                            "schema_revision": signature[0],
                            "archive": signature[1],
                            "event_archive": signature[2],
                            "status": state.get("status"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                last_signature = signature
            if str(state.get("status") or "").lower() in TERMINAL_STATUSES:
                return
        except (OSError, json.JSONDecodeError) as exc:
            print(f"checkpoint_read_retry:{type(exc).__name__}", flush=True)
        time.sleep(max(args.poll_seconds, 0.2))


if __name__ == "__main__":
    main()
