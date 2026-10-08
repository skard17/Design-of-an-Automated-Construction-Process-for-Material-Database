import inspect
import sys
import unittest
import json
import threading
import time
import urllib.error
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import downstream_extraction_runner as runner  # noqa: E402
import prompt_quality_code_agent as prompt_agent  # noqa: E402
import run_artifact_guard  # noqa: E402
import step9_extraction_build_graph as extraction_graph  # noqa: E402


def test_official_extraction_defaults_match_verified_provider_capacity():
    parameters = inspect.signature(runner.run_extraction_bench).parameters

    assert parameters["max_tokens"].default == 384000
    assert parameters["max_document_chars"].default == 200000
    assert parameters["max_workers"].default == 4


def test_code_agent_prompt_requires_bounded_live_concurrency():
    prompt = prompt_agent.build_code_agent_prompt({}, {}, ["code/downstream_extraction_runner.py"])

    assert "process independent documents within the same stage concurrently" in prompt
    assert "bounded, configurable concurrency" in prompt
    assert "do not emit dummy, mock, or placeholder" in prompt


def test_litellm_chat_retries_http_429_with_backoff():
    error = urllib.error.HTTPError(
        "https://example.invalid",
        429,
        "rate limited",
        {"Retry-After": "0"},
        None,
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"status\\":\\"ok\\"}"}}]}'

    with (
        mock.patch.object(prompt_agent, "DEFAULT_REQUEST_RETRIES", 2),
        mock.patch.object(prompt_agent.urllib.request, "urlopen", side_effect=[error, Response()]) as urlopen,
        mock.patch.object(prompt_agent.time, "sleep") as sleep,
        mock.patch.object(prompt_agent.random, "uniform", return_value=0),
    ):
        result = prompt_agent.litellm_chat(
            "https://example.invalid",
            "unused",
            "model",
            "prompt",
            max_tokens=100,
        )

    assert result == '{"status":"ok"}'
    assert urlopen.call_count == 2
    sleep.assert_called_once_with(15.0)


def test_litellm_chat_enforces_total_response_deadline():
    class StalledResponse:
        def __init__(self):
            self.close_started = threading.Event()
            self.release = threading.Event()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def close(self):
            self.close_started.set()
            self.release.wait(1)

        def read(self):
            self.release.wait(1)
            return b"partial"

    response = StalledResponse()
    started = time.monotonic()
    with (
        mock.patch.object(prompt_agent, "DEFAULT_REQUEST_RETRIES", 1),
        mock.patch.object(prompt_agent, "DEFAULT_REQUEST_TOTAL_TIMEOUT_SECONDS", 0.01),
        mock.patch.object(prompt_agent.urllib.request, "urlopen", return_value=response),
    ):
        try:
            prompt_agent.litellm_chat(
                "https://example.invalid",
                "unused",
                "model",
                "prompt",
                max_tokens=100,
            )
        except RuntimeError as exc:
            assert "total wall-clock limit" in str(exc)
        else:
            raise AssertionError("expected the total response deadline to fail the request")

    assert response.close_started.is_set()
    assert time.monotonic() - started < 0.5


def test_litellm_chat_enforces_total_deadline_while_opening_connection():
    release = threading.Event()

    def stalled_urlopen(*_args, **_kwargs):
        release.wait(1)
        raise AssertionError("connection worker should remain detached after timeout")

    started = time.monotonic()
    with (
        mock.patch.object(prompt_agent, "DEFAULT_REQUEST_RETRIES", 1),
        mock.patch.object(prompt_agent, "DEFAULT_REQUEST_TOTAL_TIMEOUT_SECONDS", 0.01),
        mock.patch.object(prompt_agent.urllib.request, "urlopen", side_effect=stalled_urlopen),
    ):
        try:
            prompt_agent.litellm_chat(
                "https://example.invalid",
                "unused",
                "model",
                "prompt",
                max_tokens=100,
            )
        except RuntimeError as exc:
            assert "total wall-clock limit" in str(exc)
        else:
            raise AssertionError("expected the total request deadline to fail")

    assert time.monotonic() - started < 0.5


def test_transport_failures_are_not_classified_as_invalid_json():
    section = {
        "stage_id": "material_info.section1",
        "section_id": "material_info.section1",
        "status": "transport_error",
        "field_count_tested": 20,
        "document_results": [
            {
                "document": "paper.md",
                "status": "transport_error",
                "checks": {
                    "error": "LiteLLM HTTP error 429: rate limit reached for TPM",
                    "error_type": "rate_limit",
                    "retryable_transport_error": True,
                },
            }
        ],
        "summary": {"extracted_total": 0, "field_coverage_ratio": 0},
    }

    diagnoses = extraction_graph.diagnose_section_result(section)
    judgement = extraction_graph.build_extraction_judgement(
        [section],
        {"status": "completed"},
        {},
        [],
    )

    assert [item["type"] for item in diagnoses] == ["retryable_transport_error"]
    assert judgement["recommended_next_action"] == "transport_retry"


def test_stage_documents_execute_concurrently_with_deterministic_result_order(tmp_path):
    documents = []
    for index in range(3):
        document = tmp_path / f"paper_{index}.md"
        document.write_text(f"Measured value {index}.", encoding="utf-8")
        documents.append(str(document))

    workflow = {
        "section_test_plan": [
            {
                "stage_id": "material_info.section0",
                "section_id": "material_info.section0",
                "purpose": "Extract identity.",
                "depends_on": [],
            }
        ]
    }
    prompt_output = {
        "module_outputs": {
            "section_extraction_prompt_module": {
                "section_extraction_prompts": {
                    "material_info.section0": {
                        "output_contract": {
                            "field_specs": [
                                {
                                    "field_path": "material_info.section0.name",
                                    "description": "Material name.",
                                }
                            ]
                        }
                    }
                }
            }
        }
    }
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_chat(**_kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return json.dumps(
            {
                "stage_id": "material_info.section0",
                "section_id": "material_info.section0",
                "document": "paper.md",
                "extracted_fields": [],
                "missing_fields": [
                    {
                        "field_path": "material_info.section0.name",
                        "missing_reason": "Not stated.",
                    }
                ],
                "unresolved_fields": [],
                "quality_notes": [],
            }
        )

    with mock.patch.object(prompt_agent, "litellm_chat", side_effect=fake_chat):
        result = runner.run_extraction_bench(
            workflow,
            prompt_output,
            documents,
            base_url="https://example.invalid",
            api_key="unused",
            model="unused",
            output_dir=str(tmp_path / "run"),
            max_documents=3,
            max_workers=3,
        )

    stage = result["section_results"][0]
    assert max_active >= 2
    assert stage["parallelism"]["effective_workers"] == 3
    assert [item["document"] for item in stage["document_results"]] == [
        "paper_0.md",
        "paper_1.md",
        "paper_2.md",
    ]


def test_official_extraction_rejects_an_existing_run_directory(tmp_path):
    output_path = tmp_path / "step9.json"
    run_dir = tmp_path / "extraction-run"
    input_identity = run_artifact_guard.build_input_identity(
        "step9_extraction_build",
        {"test": "clean extraction namespace"},
        [],
    )
    run_identity = run_artifact_guard.reserve_fresh_run(
        pipeline="step9_extraction_build",
        output_path=output_path,
        input_identity=input_identity,
        artifact_paths=[output_path, run_dir],
    )
    run_dir.mkdir()

    result = runner.run_extraction_bench(
        {"section_test_plan": []},
        {},
        [],
        base_url="https://example.invalid",
        api_key="unused",
        model="unused",
        output_dir=str(run_dir),
        run_identity=run_identity,
    )

    assert result["status"] == "clean_run_rejected"
    assert "never reused" in result["reason"]


def test_official_stage_artifact_carries_the_step9_run_identity(tmp_path):
    document = tmp_path / "paper.md"
    document.write_text("metadata-only test", encoding="utf-8")
    output_path = tmp_path / "step9.json"
    run_dir = tmp_path / "extraction-run"
    input_identity = run_artifact_guard.build_input_identity(
        "step9_extraction_build",
        {"test": "stage identity binding"},
        [document],
    )
    run_identity = run_artifact_guard.reserve_fresh_run(
        pipeline="step9_extraction_build",
        output_path=output_path,
        input_identity=input_identity,
        artifact_paths=[output_path, run_dir],
    )
    workflow = {
        "section_test_plan": [
            {
                "stage_id": "paper_info",
                "section_id": "paper_info",
                "depends_on": [],
            }
        ]
    }
    prompt_output = {
        "module_outputs": {
            "section_extraction_prompt_module": {
                "field_coverage_index": [
                    {
                        "field_path": "paper_info.title",
                        "section_id": "paper_info",
                        "description": "Paper title from retrieval metadata.",
                        "reason": "Required bibliographic identity.",
                    }
                ]
            }
        }
    }

    result = runner.run_extraction_bench(
        workflow,
        prompt_output,
        [str(document)],
        base_url="https://example.invalid",
        api_key="unused",
        model="unused",
        output_dir=str(run_dir),
        paper_metadata_by_document={str(document): {"title": "Test paper"}},
        run_identity=run_identity,
    )

    artifact = json.loads(
        (run_dir / "paper_info" / "paper.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "completed"
    assert result["run_identity"]["run_id"] == run_identity["run_id"]
    assert artifact["run_identity"]["run_id"] == run_identity["run_id"]


def test_step9_graph_retries_live_extraction_in_fresh_attempt_directories(tmp_path):
    extraction_root = tmp_path / "extraction-run"
    output_path = tmp_path / "step9.json"
    document = tmp_path / "paper.md"
    document.write_text("test", encoding="utf-8")
    calls = []

    def fake_run_extraction_bench(*args, **kwargs):
        calls.append(kwargs["output_dir"])
        return {
            "status": "completed",
            "run_dir": kwargs["output_dir"],
            "documents": [str(document)],
            "selective_reprocess": {},
            "section_results": [],
        }

    state = {
        "args": {
            "test_documents": [str(document)],
            "extraction_results": "",
            "dry_run": False,
            "base_url": "https://example.invalid",
            "api_key": "unused",
            "model": "unused",
            "temperature": 0,
            "extraction_max_tokens": 100,
            "extraction_run_dir": str(extraction_root),
            "max_test_documents": 1,
            "max_document_chars": 1000,
            "max_fields_per_stage": 80,
            "output": str(output_path),
        },
        "workflow_plan": {},
        "prompt_output": {},
        "paper_metadata_by_document": {},
        "impact_manifest_input": {},
    }

    with mock.patch.object(
        extraction_graph.downstream_extraction_runner,
        "run_extraction_bench",
        side_effect=fake_run_extraction_bench,
    ):
        first = extraction_graph.extraction_test_run_node(state)
        second = extraction_graph.extraction_test_run_node(first)

    assert calls == [
        str(extraction_root / "attempt_001"),
        str(extraction_root / "attempt_002"),
    ]
    assert first["extraction_attempt_count"] == 1
    assert second["extraction_attempt_count"] == 2


def test_step9_graph_skips_interrupted_extraction_attempt_directory(tmp_path):
    extraction_root = tmp_path / "extraction"
    (extraction_root / "attempt_001").mkdir(parents=True)
    document = tmp_path / "paper.md"
    document.write_text("test", encoding="utf-8")
    calls = []
    state = {
        "args": {
            "test_documents": [str(document)],
            "extraction_results": "",
            "dry_run": False,
            "base_url": "https://example.invalid",
            "api_key": "unused",
            "model": "unused",
            "temperature": 0,
            "extraction_max_tokens": 100,
            "extraction_run_dir": str(extraction_root),
            "max_test_documents": 1,
            "max_document_chars": 1000,
            "max_fields_per_stage": 80,
            "output": str(tmp_path / "step9.json"),
        },
        "workflow_plan": {},
        "prompt_output": {},
        "paper_metadata_by_document": {},
        "impact_manifest_input": {},
    }

    def fake_run_extraction_bench(*args, **kwargs):
        calls.append(kwargs["output_dir"])
        return {"status": "completed", "section_results": []}

    with mock.patch.object(
        extraction_graph.downstream_extraction_runner,
        "run_extraction_bench",
        side_effect=fake_run_extraction_bench,
    ):
        result = extraction_graph.extraction_test_run_node(state)

    assert calls == [str(extraction_root / "attempt_002")]
    assert result["extraction_attempt_count"] == 2


def test_step9_graph_preserves_each_generated_code_attempt(tmp_path):
    generated_root = tmp_path / "generated"
    output_path = tmp_path / "step9.json"
    calls = []
    state = {
        "args": {
            "skip_code_generation": False,
            "target_files": [],
            "base_url": "https://example.invalid",
            "api_key": "unused",
            "model": "unused",
            "temperature": 0,
            "max_tokens": 100,
            "write_generated_files": True,
            "generated_output_dir": str(generated_root),
            "output": str(output_path),
        },
        "prompt_output": {},
        "judgement": {},
    }

    def fake_write_generated_files(response, output_dir):
        calls.append(str(output_dir))
        return []

    with (
        mock.patch.object(extraction_graph.prompt_agent, "build_code_agent_prompt", return_value="prompt"),
        mock.patch.object(extraction_graph.prompt_agent, "litellm_chat", return_value="{}"),
        mock.patch.object(extraction_graph.prompt_agent, "parse_llm_json", return_value={"status": "success"}),
        mock.patch.object(
            extraction_graph.prompt_agent,
            "write_generated_files",
            side_effect=fake_write_generated_files,
        ),
    ):
        first = extraction_graph.code_generate_or_patch_node(state)
        second = extraction_graph.code_generate_or_patch_node(first)

    assert calls == [
        str(generated_root / "attempt_001"),
        str(generated_root / "attempt_002"),
    ]
    assert first["code_generation_attempt_count"] == 1
    assert second["code_generation_attempt_count"] == 2


def test_step9_graph_skips_interrupted_generated_code_attempt_directory(tmp_path):
    generated_root = tmp_path / "generated"
    (generated_root / "attempt_001").mkdir(parents=True)
    calls = []
    state = {
        "args": {
            "skip_code_generation": False,
            "target_files": [],
            "base_url": "https://example.invalid",
            "api_key": "unused",
            "model": "unused",
            "temperature": 0,
            "max_tokens": 100,
            "write_generated_files": True,
            "generated_output_dir": str(generated_root),
            "output": str(tmp_path / "step9.json"),
        },
        "prompt_output": {},
        "judgement": {},
    }

    with (
        mock.patch.object(extraction_graph.prompt_agent, "build_code_agent_prompt", return_value="prompt"),
        mock.patch.object(extraction_graph.prompt_agent, "litellm_chat", return_value="{}"),
        mock.patch.object(extraction_graph.prompt_agent, "parse_llm_json", return_value={"status": "success"}),
        mock.patch.object(
            extraction_graph.prompt_agent,
            "write_generated_files",
            side_effect=lambda response, output_dir: calls.append(str(output_dir)),
        ),
    ):
        result = extraction_graph.code_generate_or_patch_node(state)

    assert calls == [str(generated_root / "attempt_002")]
    assert result["code_generation_attempt_count"] == 2


def test_stage_batch_size_never_truncates_the_schema_inventory():
    field_specs = [
        {"field_path": f"material_info.section1.property_{index}"}
        for index in range(239)
    ]
    prompt_output = {
        "module_outputs": {
            "section_extraction_prompt_module": {
                "section_extraction_prompts": {
                    "material_info.section1": {
                        "output_contract": {"field_specs": field_specs}
                    }
                }
            }
        }
    }

    fields = runner.stage_fields(prompt_output, "material_info.section1")
    batches = runner.chunk_fields(fields, 80)

    assert len(fields) == 239
    assert [len(batch) for batch in batches] == [80, 80, 79]
    assert [item for batch in batches for item in batch] == fields


def test_step9_builds_stages_for_dynamic_schema_owners():
    fields = [
        {
            "field_path": "material_info.section0.material_name",
            "section_id": "material_info.section0",
            "description": "Material identity.",
        },
        {
            "field_path": "sample_info.sample_dimensions",
            "section_id": "sample_info",
            "description": "Dimensions of the owning sample.",
        },
        {
            "field_path": "measurement_info.measured_value",
            "section_id": "measurement_info",
            "description": "Measured value bound to its sample and conditions.",
        },
        {
            "field_path": "property_observation_info.phase_label",
            "section_id": "property_observation_info",
            "description": "Observed property or phase label.",
        },
    ]
    step8 = {"result": {"schema_definition": {"field_registry": fields}}}

    workflow = extraction_graph.build_workflow_plan(step8, {}, {})
    package = extraction_graph.build_prompt_package_from_step8(step8, workflow)
    stage_ids = [stage["stage_id"] for stage in workflow["section_test_plan"]]

    assert "sample_info" in stage_ids
    assert "measurement_info" in stage_ids
    assert "property_observation_info" in stage_ids
    assert stage_ids.index("sample_info") < stage_ids.index("measurement_info")
    assert stage_ids.index("measurement_info") < stage_ids.index("property_observation_info")
    checks, _, _ = prompt_agent.run_format_checks(package)
    assert {check["id"]: check for check in checks}["all_fields_have_extraction_specs"]["passed"]


def test_prompt_contract_sync_repairs_missing_dynamic_stage_and_is_idempotent():
    fields = [
        {
            "field_path": "material_info.section0.material_name",
            "section_id": "material_info.section0",
            "description": "Material identity.",
        },
        {
            "field_path": "measurement_info.measured_value",
            "section_id": "measurement_info",
            "description": "Measured value bound to its sample and conditions.",
        },
    ]
    step8 = {"result": {"schema_definition": {"field_registry": fields}}}
    workflow = extraction_graph.build_workflow_plan(step8, {}, {})
    package = extraction_graph.build_prompt_package_from_step8(step8, workflow)
    workflow["section_test_plan"] = [
        stage for stage in workflow["section_test_plan"] if stage["section_id"] != "measurement_info"
    ]
    prompts = package["module_outputs"]["section_extraction_prompt_module"]["section_extraction_prompts"]
    prompts.pop("measurement_info")

    repaired, repaired_workflow, report = extraction_graph.synchronize_prompt_field_contracts(
        package,
        workflow,
    )
    repaired_again, workflow_again, second_report = extraction_graph.synchronize_prompt_field_contracts(
        repaired,
        repaired_workflow,
    )
    checks, _, _ = prompt_agent.run_format_checks(repaired_again)

    assert report["added_stages"] == ["measurement_info"]
    assert second_report["added_stages"] == []
    assert [stage["stage_id"] for stage in workflow_again["section_test_plan"]].count("measurement_info") == 1
    specs = runner.stage_field_specs(repaired_again, "measurement_info")
    assert [spec["field_path"] for spec in specs] == ["measurement_info.measured_value"]
    assert {check["id"]: check for check in checks}["all_fields_have_extraction_specs"]["passed"]


class DownstreamExtractionQualityTests(unittest.TestCase):
    def test_step9_preserves_complete_field_semantics_in_output_contract(self):
        field = {
            "field_path": "material_info.section1.ionic_conductivity",
            "section_id": "material_info.section1",
            "field_name": "ionic_conductivity",
            "description": "Measured ionic conductivity with its conditions.",
            "extraction_notes": "Do not merge records measured at different temperatures.",
            "data_type": "array of objects",
            "required": True,
            "source_basis": ["text", "table"],
            "concept_ids": ["ionic_conductivity"],
            "object_contract": {
                "object_kind": "measurement",
                "required_subfields": ["value", "unit", "conditions", "evidence"],
            },
            "contract_refs": ["measurement_context_v1"],
            "reason": "Required by the materials database task.",
        }
        step8 = {
            "result": {
                "schema_definition": {
                    "field_registry": [field],
                    "shared_record_contracts": {
                        "measurement_context_v1": {
                            "optional_slots": ["temperature", "pressure", "protocol"]
                        }
                    },
                }
            }
        }
        workflow = extraction_graph.build_workflow_plan(
            step8,
            {},
            {"available": False, "strategies": []},
        )

        package = extraction_graph.build_prompt_package_from_step8(step8, workflow)
        specs = runner.stage_field_specs(package, "material_info.section1")

        self.assertEqual(1, len(specs))
        self.assertEqual("Measured ionic conductivity with its conditions.", specs[0]["description"])
        self.assertIn("different temperatures", specs[0]["extraction_notes"])
        self.assertEqual("measurement", specs[0]["object_contract"]["object_kind"])
        self.assertEqual(["measurement_context_v1"], specs[0]["contract_refs"])
        self.assertIn("temperature", specs[0]["shared_contracts"]["measurement_context_v1"]["optional_slots"])
        checks, _, _ = prompt_agent.run_format_checks(package)
        by_id = {check["id"]: check for check in checks}
        self.assertTrue(by_id["all_fields_have_extraction_specs"]["passed"])
        self.assertTrue(by_id["field_specs_have_semantic_guidance"]["passed"])

    def test_stage_prompt_embeds_binding_field_definitions(self):
        field_specs = [
            {
                "field_path": "material_info.section1.band_gap",
                "description": "Experimentally reported electronic band gap.",
                "data_type": "number with unit",
                "source_basis": ["text", "table"],
            }
        ]

        prompt = runner.build_stage_prompt(
            "The optical band gap is 2.1 eV.",
            "paper.md",
            {
                "stage_id": "material_info.section1",
                "section_id": "material_info.section1",
                "purpose": "Extract electronic properties.",
                "depends_on": [],
            },
            ["material_info.section1.band_gap"],
            {},
            field_specs,
        )

        self.assertIn("Authoritative field definitions", prompt)
        self.assertIn("Experimentally reported electronic band gap.", prompt)
        self.assertIn("number with unit", prompt)
        self.assertIn("binding", prompt)

    def test_llm_json_parser_repairs_stray_latex_backslashes(self):
        payload = prompt_agent.parse_llm_json(
            '{"value": "\\mathrm{T}", "valid": true}'
        )

        self.assertEqual(r"\mathrm{T}", payload["value"])
        self.assertTrue(payload["valid"])

    def test_llm_json_parser_preserves_valid_escaped_backslashes(self):
        payload = prompt_agent.parse_llm_json(
            '{"value": "\\\\mathrm{T}", "valid": true}'
        )

        self.assertEqual(r"\mathrm{T}", payload["value"])
        self.assertTrue(payload["valid"])

    def test_negative_placeholder_is_moved_to_missing(self):
        payload = {
            "stage_id": "core_parameters",
            "section_id": "material_info.section1",
            "extracted_fields": [
                {
                    "field_path": "material_info.section1.coercive_field",
                    "value": "Not explicitly extracted; no coercive field was reported.",
                    "evidence_text": "The paper does not report a coercive field.",
                }
            ],
            "missing_fields": [],
        }

        sanitized = runner.sanitize_payload(payload)

        self.assertEqual([], sanitized["extracted_fields"])
        self.assertEqual(
            "material_info.section1.coercive_field",
            sanitized["missing_fields"][0]["field_path"],
        )

    def test_scientific_negative_observation_is_preserved(self):
        self.assertFalse(runner.is_null_like("No magnetic ordering was observed above 2 K."))

    def test_duplicate_values_merge_evidence(self):
        payload = {
            "stage_id": "core_parameters",
            "section_id": "material_info.section1",
            "extracted_fields": [
                {
                    "field_path": "material_info.section1.transition_temperature",
                    "value": 14.6,
                    "unit": "K",
                    "material_system": "EuNi2As2",
                    "evidence_text": "A transition occurs at 14.6 K.",
                    "source_hint": "page 2",
                    "confidence": 0.91,
                },
                {
                    "field_path": "material_info.section1.transition_temperature",
                    "value": 14.6,
                    "unit": "K",
                    "material_system": "EuNi2As2",
                    "evidence_text": "TN = 14.6 K.",
                    "source_hint": "Figure 3",
                    "confidence": 0.96,
                },
            ],
            "missing_fields": [],
        }

        sanitized = runner.sanitize_payload(payload)

        self.assertEqual(1, len(sanitized["extracted_fields"]))
        merged = sanitized["extracted_fields"][0]
        self.assertEqual(2, len(merged["evidence_items"]))
        self.assertEqual(0.96, merged["confidence"])

    def test_extracted_value_removes_conflicting_missing_entry(self):
        payload = {
            "stage_id": "paper_info",
            "section_id": "paper_info",
            "extracted_fields": [
                {
                    "field_path": "paper_info.metadata.doi",
                    "value": "10.1000/example",
                    "evidence_text": "doi:10.1000/example",
                }
            ],
            "missing_fields": [
                {
                    "field_path": "paper_info.metadata.doi",
                    "missing_reason": "not provided in document",
                }
            ],
        }

        sanitized = runner.sanitize_payload(payload)

        self.assertEqual(1, len(sanitized["extracted_fields"]))
        self.assertEqual([], sanitized["missing_fields"])

    def test_unstructured_inferred_label_is_not_treated_as_fact(self):
        payload = {
            "stage_id": "mechanism",
            "section_id": "section5",
            "extracted_fields": [
                {
                    "field_path": "section5.gap_symmetry",
                    "value": "s-wave (implied conventional)",
                    "evidence_text": "The discussion describes conventional behavior.",
                }
            ],
            "missing_fields": [],
        }

        sanitized = runner.sanitize_payload(payload)

        self.assertEqual([], sanitized["extracted_fields"])
        self.assertIn("structured inference basis", sanitized["missing_fields"][0]["missing_reason"])

    def test_structured_inference_is_preserved(self):
        payload = {
            "stage_id": "mechanism",
            "section_id": "section5",
            "extracted_fields": [
                {
                    "field_path": "section5.state_assignment",
                    "value": "candidate state (inferred)",
                    "assignment_basis": "fit to the explicitly reported model",
                    "source_type": "author_interpretation",
                    "confidence": 0.7,
                    "evidence_text": "The authors assign this state from the fit.",
                }
            ],
            "missing_fields": [],
        }

        sanitized = runner.sanitize_payload(payload)

        self.assertEqual(1, len(sanitized["extracted_fields"]))

    def test_complete_missing_coverage_is_not_low_quality(self):
        checks = {
            "extracted_count": 0,
            "missing_count": 2,
            "addressed_field_paths": ["device_info.forward_current", "device_info.reverse_current"],
        }

        assessment = runner.assess_stage_yield(
            [checks],
            ["device_info.forward_current", "device_info.reverse_current"],
            stage_id="device_response",
        )

        self.assertEqual("passed", assessment["status"])
        self.assertEqual("no_values_found", assessment["extraction_outcome"])
        self.assertEqual(1.0, assessment["field_coverage_ratio"])

    def test_supervisor_accepts_complete_negative_result(self):
        section_result = {
            "stage_id": "device_response",
            "section_id": "device_info",
            "status": "passed",
            "summary": {
                "extracted_total": 0,
                "missing_total": 2,
                "field_coverage_ratio": 1.0,
                "extraction_outcome": "no_values_found",
            },
            "document_results": [],
        }

        self.assertEqual([], extraction_graph.diagnose_section_result(section_result))


if __name__ == "__main__":
    unittest.main()
