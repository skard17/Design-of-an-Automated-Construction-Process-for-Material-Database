import importlib.util
import json
import os
from pathlib import Path

import pytest


PROMPT_PATH = Path(os.environ.get("FIELD_DESIGN_PROMPT_PATH", str(
    Path(__file__).resolve().parents[1] / "code" / "section_design_agent_prompt.py"
)))
spec = importlib.util.spec_from_file_location("transferable_prompts", PROMPT_PATH)
prompts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompts)


@pytest.mark.parametrize("domain,quantity", [
    ("battery materials", "specific_capacity"),
    ("catalyst materials", "turnover_frequency"),
    ("structural materials", "yield_strength"),
])
def test_design_rules_reach_all_stages_without_a_domain_catalog(domain, quantity):
    context = {"discipline": domain, "database_goal": f"Compare {quantity}"}
    plan = prompts.build_field_planning_prompt(context, *[{} for _ in range(7)])
    schema = prompts.build_schema_design_prompt(context, *[{} for _ in range(10)])
    critic = prompts.build_specialization_critic_prompt(context, *[{} for _ in range(7)])
    for rendered in (plan, schema, critic):
        assert prompts.TRANSFERABLE_FIELD_DESIGN_RULES in rendered
        assert domain in rendered
        assert quantity in rendered
    policy = prompts.TRANSFERABLE_FIELD_DESIGN_RULES.lower()
    for forbidden in ("superconduct", "schema2_store", "critical_temperature", "tc[]"):
        assert forbidden not in policy


def test_critic_receives_repeatability_and_semantic_contracts():
    field = {
        "field_path": "material_info.section1.yield_strength[].value",
        "section_id": "material_info.section1",
        "data_type": "number",
        "description": "Stress at the declared yield criterion",
        "extraction_notes": "Bind strain rate and temperature to this observation",
        "relation_constraints": {"separate_instances": False, "condition_binding_required": True},
        "evidence_requirements": {"locator_required": True},
    }
    rendered = prompts.build_specialization_critic_prompt(
        {}, {}, {}, {}, {}, {}, {}, {"field_registry": [field]}
    )
    if "Compact schema projection:" not in rendered:
        assert '"separate_instances": false' in rendered
        assert field["description"] in rendered
        assert field["extraction_notes"] in rendered
        return
    projection = rendered.split("Compact schema projection:", 1)[1].split(
        "Supervisor validation feedback", 1
    )[0].strip()
    projected = json.loads(projection)["field_registry"][0]
    for key in ("description", "extraction_notes", "relation_constraints", "evidence_requirements"):
        assert projected[key] == field[key]
