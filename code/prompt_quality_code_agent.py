import argparse
import json
import os
import re
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_BASE_URL = "https://chat.iphy.ac.cn/litellm/v1"
DEFAULT_MODEL = os.getenv("CODE_AGENT_MODEL", "kimi-k2.6")
DEFAULT_SYSTEM_PROMPT = (
    "You are a senior Python coding agent for scientific extraction workflows. "
    "Return valid JSON only."
)


MOJIBAKE_MARKERS = ("é", "ç", "è", "å", "æ", "Â", "â", "î", "Ð", "Ñ", "�")
QUALITY_KEYWORDS = {
    "json_contract": ("json", "valid json", "json only"),
    "evidence_provenance": ("evidence", "provenance", "source", "quote", "span", "caption", "ocr"),
    "anti_hallucination": ("do not invent", "never invent", "hallucination", "unsupported", "missing_reason"),
    "field_path_control": ("field_path", "field paths", "allowed field"),
    "section_ownership": ("section ownership", "owning section", "allowed_sections", "outside this section"),
    "uncertainty_handling": ("confidence", "uncertain", "low-confidence", "missing_reason"),
}
REQUIRED_MODULES = (
    "prompt_supervisor_module",
    "classification_prompt_module",
    "section_extraction_prompt_module",
    "figure_extraction_prompt_module",
    "postprocess_repair_prompt_module",
    "prompt_package_aggregation",
)
MODEL_ALIASES = {
    "deepseek-v4": "deepseek-v4-pro",
}
BLOCKING_JUDGEMENT_STATUSES = {
    "blocked_by_quality_review",
    "quality_review_failed",
}


def json_dumps(value, max_chars=None):
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if max_chars and len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n...[truncated]"
    return text


def load_json(path):
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(
            f"Input JSON not found: {json_path}. "
            "Pass a Step 9 prompt package with --prompt-output."
        )
    return json.loads(json_path.read_text(encoding="utf-8"))


def walk_strings(value, path="$"):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from walk_strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk_strings(item, f"{path}[{index}]")


def looks_mojibake(text):
    if not text:
        return False
    marker_count = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    if marker_count >= 3 and marker_count / max(len(text), 1) > 0.015:
        return True
    suspicious_pairs = ("é", "éŽ", "ç’", "æµ", "é", "â‚", "î†", "ã„")
    return any(pair in text for pair in suspicious_pairs)


def collect_prompt_texts(prompt_output):
    texts = []
    modules = prompt_output.get("module_outputs") or {}
    for module_name, module in modules.items():
        for path, text in walk_strings(module, f"module_outputs.{module_name}"):
            leaf = path.rsplit(".", 1)[-1]
            if leaf in {"prompt", "output_contract"}:
                texts.append({"path": path, "chars": len(text), "text": text})
    return texts


def collect_field_paths(prompt_output):
    context_paths = {
        field.get("field_path")
        for field in (prompt_output.get("shared_prompt_context") or {}).get("field_index", [])
        if isinstance(field, dict) and field.get("field_path")
    }
    covered_paths = set()
    modules = prompt_output.get("module_outputs") or {}
    section_module = modules.get("section_extraction_prompt_module") or {}
    for item in section_module.get("field_coverage_index", []) or []:
        if isinstance(item, dict) and item.get("field_path"):
            covered_paths.add(item["field_path"])
    figure_module = modules.get("figure_extraction_prompt_module") or {}
    for item in figure_module.get("figure_field_coverage_index", []) or []:
        if isinstance(item, dict) and item.get("field_path"):
            covered_paths.add(item["field_path"])
    return context_paths, covered_paths


def make_check(check_id, category, severity, passed, summary, evidence=None, recommended_code_action=""):
    return {
        "id": check_id,
        "category": category,
        "severity": severity,
        "passed": passed,
        "summary": summary,
        "evidence": evidence if evidence is not None else [],
        "recommended_code_action": recommended_code_action,
    }


def keyword_presence(text, keywords):
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def get_prompt_module(prompt_output, name):
    modules = prompt_output.get("module_outputs") or {}
    module = modules.get(name)
    return module if isinstance(module, dict) else {}


def run_format_checks(prompt_output):
    checks = []
    status = prompt_output.get("status")
    validation_errors = prompt_output.get("validation_errors") or []
    modules = prompt_output.get("module_outputs") or {}

    checks.append(
        make_check(
            "prompt_status_success",
            "format",
            "high",
            status == "success",
            f"Prompt generation status is {status!r}.",
            validation_errors,
            "Stop downstream code generation unless --allow-needs-review is set.",
        )
    )
    checks.append(
        make_check(
            "no_validation_errors",
            "format",
            "high",
            not validation_errors,
            "Step 9 validation_errors should be empty.",
            validation_errors,
            "Add a preflight validator before extraction agents run.",
        )
    )

    missing_modules = [name for name in REQUIRED_MODULES if name not in modules]
    checks.append(
        make_check(
            "required_modules_present",
            "format",
            "high",
            not missing_modules,
            "All required prompt modules should be present.",
            missing_modules,
            "Regenerate Step 9 or add deterministic fallback module builders.",
        )
    )

    context_paths, covered_paths = collect_field_paths(prompt_output)
    uncovered = sorted(context_paths - covered_paths)
    checks.append(
        make_check(
            "all_fields_covered",
            "format",
            "high",
            not uncovered,
            "Every schema field should be covered by a section or figure extraction prompt.",
            uncovered[:100],
            "Generate coverage tests and fail extraction startup on uncovered fields.",
        )
    )

    result = prompt_output.get("result") or {}
    expected_order = ["classification", "section_extraction", "figure_extraction", "postprocess_repair"]
    execution_order = result.get("execution_order") if isinstance(result, dict) else None
    checks.append(
        make_check(
            "expected_execution_order",
            "format",
            "medium",
            execution_order == expected_order,
            "Execution order should match the downstream runner order.",
            execution_order,
            "Normalize execution order in the downstream runner.",
        )
    )

    package_modules = (result.get("prompt_modules") or {}) if isinstance(result, dict) else {}
    missing_package_modules = [
        name
        for name in ("classification", "section_extraction", "figure_extraction", "postprocess_repair")
        if name not in package_modules
    ]
    checks.append(
        make_check(
            "aggregation_contains_prompt_modules",
            "format",
            "medium",
            not missing_package_modules,
            "Aggregated package should expose all downstream prompt families.",
            missing_package_modules,
            "Make aggregation fallback copy validated module outputs into result.prompt_modules.",
        )
    )
    return checks, context_paths, covered_paths


def run_quality_checks(prompt_output):
    checks = []
    prompt_texts = collect_prompt_texts(prompt_output)
    combined_prompt_text = "\n".join(item["text"] for item in prompt_texts)

    for quality_id, keywords in QUALITY_KEYWORDS.items():
        checks.append(
            make_check(
                quality_id,
                "quality",
                "medium",
                keyword_presence(combined_prompt_text, keywords),
                f"Prompt texts should include controls for {quality_id}.",
                {"keywords": list(keywords)},
                "Add deterministic prompt lint rules and strengthen the prompt template.",
            )
        )

    long_prompts = [{"path": item["path"], "chars": item["chars"]} for item in prompt_texts if item["chars"] > 8000]
    checks.append(
        make_check(
            "prompt_length_budget",
            "quality",
            "medium",
            not long_prompts,
            "Generated prompts should stay within a practical length budget.",
            long_prompts[:50],
            "Add prompt compaction before calling downstream extraction agents.",
        )
    )

    empty_prompts = [
        {"path": item["path"], "chars": item["chars"]}
        for item in prompt_texts
        if len(item["text"].strip()) < 80
    ]
    checks.append(
        make_check(
            "non_empty_actionable_prompts",
            "quality",
            "medium",
            not empty_prompts,
            "Prompt fields should contain actionable instructions, not placeholders.",
            empty_prompts[:50],
            "Reject prompt modules with empty prompt strings or placeholder output contracts.",
        )
    )

    mojibake_samples = []
    for path, text in walk_strings(prompt_output):
        if looks_mojibake(text):
            mojibake_samples.append({"path": path, "sample": text[:180]})
        if len(mojibake_samples) >= 20:
            break
    checks.append(
        make_check(
            "no_mojibake_text",
            "quality",
            "medium",
            not mojibake_samples,
            "Prompt package strings should not look encoding-corrupted.",
            mojibake_samples,
            "Add an encoding quality gate and keep corrupted source strings out of prompts.",
        )
    )

    repair_module = get_prompt_module(prompt_output, "postprocess_repair_prompt_module")
    repair_text = json.dumps(repair_module, ensure_ascii=False).lower()
    missing_repair_topics = [
        topic
        for topic in ("json", "missing", "ownership", "figure", "provenance")
        if topic not in repair_text
    ]
    checks.append(
        make_check(
            "repair_prompt_topic_coverage",
            "quality",
            "medium",
            not missing_repair_topics,
            "Repair prompts should cover JSON, missing fields, ownership, figure conflicts, and provenance.",
            missing_repair_topics,
            "Generate a deterministic repair prompt fallback with required repair types.",
        )
    )

    figure_fields = [
        field
        for field in (prompt_output.get("shared_prompt_context") or {}).get("figure_fields", [])
        if isinstance(field, dict)
    ]
    figure_module = get_prompt_module(prompt_output, "figure_extraction_prompt_module")
    figure_text = json.dumps(figure_module, ensure_ascii=False).lower()
    figure_quality_ok = not figure_fields or (
        "allowed_figure_categories" in figure_text
        and ("allowed_sections" in figure_text or "ownership" in figure_text)
    )
    checks.append(
        make_check(
            "figure_constraint_quality",
            "quality",
            "medium",
            figure_quality_ok,
            "Figure prompts should enforce allowed categories and ownership boundaries when figure fields exist.",
            {"figure_field_count": len(figure_fields)},
            "Inject figure_constraint rules into every figure extraction prompt.",
        )
    )
    return checks


def split_checks(checks):
    failed = [check for check in checks if not check["passed"]]
    warnings = [check for check in failed if check["severity"] in {"low", "medium"}]
    issues = [check for check in failed if check["severity"] == "high"]
    return issues, warnings


def judge_prompt_package(prompt_output):
    recommendations = []
    format_checks, context_paths, covered_paths = run_format_checks(prompt_output)
    quality_checks = run_quality_checks(prompt_output)
    issues, warnings = split_checks(format_checks + quality_checks)

    if not issues:
        recommendations.append("Prompt package has no high-severity format blockers.")
    recommendations.extend(
        [
            "Persist this judgement file next to the generated code so future runs are reproducible.",
            "Require downstream agents to emit JSON with evidence spans and missing_reason fields.",
            "Add a dry-run mode that validates prompt coverage without calling the LLM API.",
        ]
    )

    return {
        "step": "step10_prompt_quality_judgement",
        "status": "needs_code_action" if issues else "ready_for_code_generation",
        "summary": {
            "issue_count": len(issues),
            "warning_count": len(warnings),
            "format_failed_count": len([check for check in format_checks if not check["passed"]]),
            "quality_failed_count": len([check for check in quality_checks if not check["passed"]]),
            "field_count": len(context_paths),
            "covered_field_count": len(covered_paths),
        },
        "format_checks": format_checks,
        "quality_checks": quality_checks,
        "issues": issues,
        "warnings": warnings,
        "recommendations": recommendations,
    }


def build_quality_review_prompt(prompt_output, judgement):
    modules = prompt_output.get("module_outputs") or {}
    excerpt = {
        "status": prompt_output.get("status"),
        "validation_errors": prompt_output.get("validation_errors"),
        "summary": judgement.get("summary"),
        "format_failed_checks": [check for check in judgement.get("format_checks", []) if not check.get("passed")],
        "local_quality_failed_checks": [check for check in judgement.get("quality_checks", []) if not check.get("passed")],
        "prompt_supervisor_module": modules.get("prompt_supervisor_module"),
        "classification_prompt_module": modules.get("classification_prompt_module"),
        "section_extraction_prompt_samples": (modules.get("section_extraction_prompt_module") or {}).get(
            "section_extraction_prompts", []
        )[:3],
        "figure_extraction_prompt_module": modules.get("figure_extraction_prompt_module"),
        "postprocess_repair_prompt_module": modules.get("postprocess_repair_prompt_module"),
        "aggregation_result": prompt_output.get("result"),
    }
    return textwrap.dedent(
        f"""
        You are a Prompt Quality Review Agent for a scientific extraction workflow.

        Review the Step 9 prompt package as a downstream-agent designer. Local checks already
        verified hard format constraints; your job is to judge semantic quality and give concrete
        improvement advice that will guide the next code-generation agent.

        Prompt package excerpt:
        {json_dumps(excerpt, max_chars=26000)}

        Return JSON only with this schema:
        {{
          "status": "pass" | "needs_improvement" | "blocked",
          "overall_score": 0,
          "dimension_scores": {{
            "clarity": 0,
            "schema_alignment": 0,
            "evidence_grounding": 0,
            "anti_hallucination": 0,
            "section_ownership": 0,
            "figure_handling": 0,
            "repairability": 0,
            "downstream_code_readiness": 0
          }},
          "quality_findings": [
            {{
              "id": "string",
              "severity": "high|medium|low",
              "dimension": "string",
              "finding": "string",
              "evidence": "string",
              "recommended_prompt_change": "string",
              "recommended_code_change": "string"
            }}
          ],
          "must_fix_before_code_generation": ["string"],
          "nice_to_have_improvements": ["string"],
          "code_generation_guidance": [
            {{
              "target": "validator|runner|repair|logging|tests|prompt_template",
              "instruction": "string",
              "rationale": "string"
            }}
          ]
        }}

        Scoring rule:
        - Scores are integers from 0 to 10.
        - Use "blocked" only when downstream extraction code should not be generated.
        - Use "needs_improvement" when code can be generated but should include safeguards.
        - Be specific: name the exact missing guard, test, validator, or prompt template change.
        """
    ).strip()


def run_quality_review_agent(prompt_output, judgement, args):
    prompt = build_quality_review_prompt(prompt_output, judgement)
    raw_response = litellm_chat(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        prompt=prompt,
        temperature=args.temperature,
        max_tokens=args.quality_review_max_tokens,
    )
    review = parse_llm_json(raw_response)
    return review


def merge_quality_review(judgement, review):
    merged = dict(judgement)
    merged["llm_quality_review"] = review
    findings = review.get("quality_findings", []) if isinstance(review, dict) else []
    high_findings = [
        item
        for item in findings
        if isinstance(item, dict) and item.get("severity") == "high"
    ]
    blocked = isinstance(review, dict) and review.get("status") == "blocked"
    needs_improvement = isinstance(review, dict) and review.get("status") in {"needs_improvement", "blocked"}
    summary = dict(merged.get("summary") or {})
    summary["llm_quality_status"] = review.get("status") if isinstance(review, dict) else "invalid_review"
    summary["llm_quality_finding_count"] = len(findings)
    summary["llm_high_quality_finding_count"] = len(high_findings)
    merged["summary"] = summary
    if blocked:
        merged["status"] = "blocked_by_quality_review"
    elif needs_improvement and merged.get("status") == "ready_for_code_generation":
        merged["status"] = "ready_with_quality_safeguards"
    return merged


def litellm_chat(base_url, api_key, model, prompt, temperature=0, max_tokens=4096):
    if not api_key:
        raise RuntimeError("Missing API key. Use --api-key or set CODE_AGENT_API_KEY.")
    resolved_model = MODEL_ALIASES.get(model, model)
    payload = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
    }
    for attempt in range(3):
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"].get("content") or ""
            if content.strip():
                return content
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError("LiteLLM returned empty message content after retries.")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in {502, 503, 504} and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"LiteLLM HTTP error {exc.code}: {body}") from exc


def build_code_agent_prompt(prompt_output, judgement, target_files):
    return textwrap.dedent(
        f"""
        Build the next-step code generation sub-agent plan for this repository.

        The system already produced a Step 9 prompt package. A local judgement pass found:
        {json_dumps(judgement, max_chars=12000)}

        Relevant prompt package excerpt:
        {json_dumps({
            "status": prompt_output.get("status"),
            "validation_errors": prompt_output.get("validation_errors"),
            "result": prompt_output.get("result"),
            "module_output_keys": list((prompt_output.get("module_outputs") or {}).keys()),
        }, max_chars=10000)}

        Target files that may be changed by a human/Codex after reviewing your plan:
        {json_dumps(target_files)}

        Return JSON only with this schema:
        {{
          "status": "ready" | "blocked",
          "diagnosis": ["string"],
          "code_change_plan": [
            {{
              "file_path": "string",
              "change_type": "add|modify|test",
              "reason": "string",
              "implementation_notes": ["string"]
            }}
          ],
          "generated_files": [
            {{
              "file_path": "string",
              "purpose": "string",
              "content": "full UTF-8 file content"
            }}
          ],
          "manual_review_notes": ["string"]
        }}

        Requirements:
        - Prefer small Python files that use argparse and JSON, matching the current repo style.
        - Do not include secrets or API keys in generated file content.
        - Generated code must read prompt packages and judgement files from paths.
        - If judgement.status is needs_code_action, generated code should include preflight validation.
        - Keep any generated file self-contained unless it can import existing code safely.
        """
    ).strip()


def parse_llm_json(raw_text):
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.S)
        if match:
            return json.loads(match.group(0))
        raise


def write_generated_files(response, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for item in response.get("generated_files", []) or []:
        if not isinstance(item, dict) or not item.get("file_path") or "content" not in item:
            continue
        relative_path = Path(item["file_path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"Refusing unsafe generated file path: {item['file_path']}")
        target = output_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(item["content"], encoding="utf-8")
        written.append(str(target))
    return written


def build_blocked_code_response(judgement, reason):
    return {
        "status": "blocked",
        "diagnosis": [reason],
        "judgement_status": judgement.get("status"),
        "code_change_plan": [],
        "generated_files": [],
        "manual_review_notes": [
            "Resolve the judgement blocker before running the code-generation agent.",
            "Use --allow-code-generation-on-failed-judgement only for debugging."
        ],
    }


def run(args):
    prompt_output = load_json(args.prompt_output)
    judgement = judge_prompt_package(prompt_output)
    model = MODEL_ALIASES.get(args.model, args.model)
    args.model = model

    quality_review = None
    if args.run_quality_review:
        try:
            quality_review = run_quality_review_agent(prompt_output, judgement, args)
            judgement = merge_quality_review(judgement, quality_review)
        except Exception as exc:
            judgement["llm_quality_review_error"] = {
                "model": args.model,
                "base_url": args.base_url,
                "error": str(exc),
            }
            judgement["status"] = "quality_review_failed"
            summary = dict(judgement.get("summary") or {})
            summary["llm_quality_status"] = "api_error"
            judgement["summary"] = summary

    judgement_path = Path(args.judgement_output)
    judgement_path.write_text(json_dumps(judgement), encoding="utf-8")
    print(f"Saved judgement to {judgement_path}")
    print(f"Judgement status: {judgement['status']}")

    if args.judge_only:
        return {"judgement": judgement, "code_agent": None}

    if (
        judgement.get("status") in BLOCKING_JUDGEMENT_STATUSES
        and not args.allow_code_generation_on_failed_judgement
    ):
        blocked_response = build_blocked_code_response(
            judgement,
            f"Code generation blocked because judgement.status is {judgement.get('status')!r}.",
        )
        response_path = Path(args.code_agent_output)
        response_path.write_text(json_dumps(blocked_response), encoding="utf-8")
        print(f"Saved blocked code-agent response to {response_path}")
        return {
            "judgement": judgement,
            "quality_review": quality_review,
            "code_agent": blocked_response,
            "written_files": [],
        }

    target_files = args.target_files or ["code/prompt_quality_code_agent.py", "code/downstream_extraction_runner.py"]
    prompt = build_code_agent_prompt(prompt_output, judgement, target_files)
    raw_response = litellm_chat(
        base_url=args.base_url,
        api_key=args.api_key,
        model=model,
        prompt=prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    code_response = parse_llm_json(raw_response)

    response_path = Path(args.code_agent_output)
    response_path.write_text(json_dumps(code_response), encoding="utf-8")
    print(f"Saved code-agent response to {response_path}")

    written = []
    if args.write_generated_files:
        written = write_generated_files(code_response, args.generated_output_dir)
        print(f"Wrote {len(written)} generated file(s) under {args.generated_output_dir}")

    return {
        "judgement": judgement,
        "quality_review": quality_review,
        "code_agent": code_response,
        "written_files": written,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Judge a Step 9 prompt package and run a code-generation sub-agent.")
    parser.add_argument("--prompt-output", default="skyrmion_prompt_generation_output.json")
    parser.add_argument("--judgement-output", default="prompt_quality_judgement.json")
    parser.add_argument("--code-agent-output", default="code_generation_agent_output.json")
    parser.add_argument("--base-url", default=os.getenv("CODE_AGENT_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.getenv("CODE_AGENT_API_KEY"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--quality-review-max-tokens", type=int, default=4096)
    parser.add_argument("--target-files", nargs="*", default=[])
    parser.add_argument("--judge-only", action="store_true")
    parser.add_argument(
        "--run-quality-review",
        action="store_true",
        help="Call the LLM quality review agent and merge its suggestions into the judgement file.",
    )
    parser.add_argument("--write-generated-files", action="store_true")
    parser.add_argument("--generated-output-dir", default="generated_code")
    parser.add_argument(
        "--allow-code-generation-on-failed-judgement",
        action="store_true",
        help="Debug escape hatch: run code generation even when Step 10 judgement has a blocking status.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
