import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import downstream_extraction_runner as runner  # noqa: E402
import step9_extraction_build_graph as graph  # noqa: E402


class Step9MetadataHandoffTests(unittest.TestCase):
    def test_auto_discovers_and_matches_archive_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "domain"
            document = root / "local_text_md" / "arxiv__2607.04104v1.md"
            metadata_file = root / "paper_archive" / "metadata" / "arxiv_2607.04104v1.json"
            document.parent.mkdir(parents=True)
            metadata_file.parent.mkdir(parents=True)
            document.write_text("paper text", encoding="utf-8")
            metadata_file.write_text(
                json.dumps(
                    {
                        "paper_id": "arxiv:2607.04104v1",
                        "arxiv_id": "2607.04104v1",
                        "title": "Example title",
                        "authors": ["A. Author"],
                        "source": "arxiv",
                    }
                ),
                encoding="utf-8",
            )

            matches, report = graph.load_paper_metadata_for_documents([str(document)])

            self.assertEqual("Example title", matches[str(document)]["title"])
            self.assertEqual(1, report["matched_documents"])
            self.assertEqual(1, report["records_loaded"])

    def test_step8_registry_gets_canonical_metadata_fields(self):
        fields = graph.field_registry_from_step8(
            {"result": {"schema_definition": {"field_registry": []}}}
        )
        paths = {field["field_path"] for field in fields}

        self.assertIn("paper_info.metadata.title", paths)
        self.assertIn("paper_info.metadata.authors", paths)
        self.assertIn("paper_info.metadata.publication_date", paths)

    def test_runner_prefills_paper_info_without_calling_llm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            document = Path(temp_dir) / "arxiv__2607.04104v1.md"
            document.write_text("PDF text should not be used for metadata.", encoding="utf-8")
            output_dir = Path(temp_dir) / "run"
            workflow_plan = {
                "section_test_plan": [
                    {
                        "stage_id": "paper_info",
                        "section_id": "paper_info",
                        "depends_on": [],
                    }
                ]
            }
            fields = [
                "paper_info.metadata.title",
                "paper_info.metadata.authors",
                "paper_info.metadata.doi",
                "paper_info.metadata.publication_date",
            ]
            prompt_output = {
                "module_outputs": {
                    "section_extraction_prompt_module": {
                        "section_extraction_prompts": {
                            "paper_info": {"output_contract": {"fields": fields}}
                        }
                    }
                }
            }
            metadata = {
                str(document): {
                    "paper_id": "arxiv:2607.04104v1",
                    "title": "Vortex paper",
                    "authors": ["Linghao Huang", "Jing Wang"],
                    "doi": "",
                    "published_date": "2026-07-05T03:56:27+00:00",
                    "source": "arxiv",
                    "_metadata_source_path": "metadata.json",
                }
            }

            with mock.patch.object(
                runner.prompt_agent,
                "litellm_chat",
                side_effect=AssertionError("LLM must not be called for metadata prefill"),
            ):
                result = runner.run_extraction_bench(
                    workflow_plan,
                    prompt_output,
                    [str(document)],
                    base_url="https://example.invalid",
                    api_key="unused",
                    model="unused",
                    output_dir=str(output_dir),
                    paper_metadata_by_document=metadata,
                )

            payload = json.loads(
                (output_dir / "paper_info" / "arxiv__2607.04104v1.json").read_text(
                    encoding="utf-8"
                )
            )
            extracted = {item["field_path"]: item["value"] for item in payload["extracted_fields"]}
            missing = {item["field_path"] for item in payload["missing_fields"]}

            self.assertEqual("Vortex paper", extracted["paper_info.metadata.title"])
            self.assertEqual(
                ["Linghao Huang", "Jing Wang"],
                extracted["paper_info.metadata.authors"],
            )
            self.assertIn("paper_info.metadata.doi", missing)
            self.assertEqual(
                ["arxiv__2607.04104v1.md"],
                result["paper_metadata_handoff"]["prefilled_documents"],
            )

    def test_missing_metadata_is_blocking_and_does_not_fall_back_to_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            document = Path(temp_dir) / "paper.md"
            document.write_text("Title and authors appear here.", encoding="utf-8")
            workflow_plan = {
                "section_test_plan": [
                    {"stage_id": "paper_info", "section_id": "paper_info", "depends_on": []}
                ]
            }
            prompt_output = {
                "module_outputs": {
                    "section_extraction_prompt_module": {
                        "section_extraction_prompts": {
                            "paper_info": {
                                "output_contract": {
                                    "fields": ["paper_info.metadata.title"]
                                }
                            }
                        }
                    }
                }
            }

            with mock.patch.object(
                runner.prompt_agent,
                "litellm_chat",
                side_effect=AssertionError("PDF fallback must not run for paper metadata"),
            ):
                result = runner.run_extraction_bench(
                    workflow_plan,
                    prompt_output,
                    [str(document)],
                    base_url="https://example.invalid",
                    api_key="unused",
                    model="unused",
                    output_dir=str(Path(temp_dir) / "run"),
                    paper_metadata_by_document={},
                )

            section = result["section_results"][0]
            self.assertEqual("metadata_missing", section["status"])
            diagnoses = graph.diagnose_section_result(section)
            self.assertEqual("metadata_handoff_missing", diagnoses[0]["type"])
            self.assertEqual("blocking", diagnoses[0]["severity"])


if __name__ == "__main__":
    unittest.main()
