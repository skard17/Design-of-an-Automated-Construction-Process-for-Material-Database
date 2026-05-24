import json
import textwrap


STEP8_FRAMEWORK = textwrap.dedent(
    """
    Step 8: Section Design Agent

    Overall workflow position:
    1. Receive upstream database target, discipline, query requirements, and key-description reference.
    2. Let a subjective supervisor define the modeling position, domain boundaries, and anti-generic constraints.
    3. Infer the database's retrieval unit, comparison granularity, mechanism prerequisites, and evidence structure.
    4. Start from the default six-section materials-database backbone, then adapt fields inside each section to the target domain.
    5. Split sections into core and non-core sections only after deciding whether the default backbone needs target-specific extension.
    6. Let a figure supervisor review figure-conflict risk after the section and field plans are available.
    7. If needed, classify figures by section first and only then by subsection.
    8. Assemble the schema from section architecture, field planning, and figure-ownership rules.
    9. Let a specialization critic check whether the design is still too generic, structurally weak, or missing must-have concepts.
    10. Aggregate and validate the result as a supervisor-managed loop.

    Required outputs:
    - section design plan
    - schema definition

    Mandatory design constraints:
    - A subjective supervisor must explicitly define the modeling stance before section design starts.
    - The default material database layout is the six-section backbone: material system/tuning, core parameters/performance, fabrication/processing, microscopic characterization/electronic structure, macroscopic property curves, and theory/mechanism.
    - The section layout must preserve this backbone for most materials domains unless the supervisor explicitly justifies a domain that does not fit it.
    - Core sections and non-core sections must be explicitly separated.
    - Domain-critical concepts must be surfaced as explicit schema constraints rather than left as descriptive suggestions.
    - A figure supervisor must explicitly review figure-heavy outputs and decide whether a figure classification repair agent should be enabled.
    - If figure-heavy sections may interfere with each other, figures must be split by section before field design.
    - Each field must indicate whether it depends on text, tables, figures, or multiple sources.
    - The design must avoid generic umbrella fields when parameter-level evidence linkage is needed.
    - Domain claims, labels, and scalar values must be separated from the evidence used to assign or measure them.
    - Any indirect evidence must be marked as supporting evidence rather than silently promoted to a direct assignment.
    - Mechanism, model, and fitted parameters must record whether they come from experiment, fitted simulation, first-principles calculation, literature assumption, or author interpretation.
    - The no-figure-classification branch must still produce an explicit stable plan rather than leaving the routing implicit.
    - The agent must explain how the target-domain fields specialize the shared backbone rather than inventing unrelated section semantics.
    """
).strip()


MATERIAL_DATABASE_SECTION_BACKBONE = textwrap.dedent(
    """
    Default materials-database section backbone:
    - material_info.section0: Material System and Tuning. Identity, composition, doping/substitution, defects, strain, interfaces, sample variants, and other descriptors that define what material or sample is being compared.
    - material_info.section1: Core Parameters and Performance. Discrete domain-critical scalar values or structured parameter records with value, unit, conditions, provenance, and confidence. This section changes the most by domain.
    - material_info.section2: Fabrication and Processing. Synthesis, growth, treatment, device fabrication, processing route, method, description, geometry, and conditions such as temperature, time, pressure, atmosphere, and annealing.
    - material_info.section3: Microscopic Characterization and Electronic/Structural Evidence. Figure-linked evidence from structure, microstructure, phase, spectroscopy, microscopy, diffraction, and electronic-structure characterization.
    - material_info.section4: Macroscopic Property Curves. Figure-linked curves and maps used to read or support physical-property trends, such as transport, magnetization, thermodynamic, optical, electrochemical, or mechanical curves.
    - section5: Theory and Mechanism. Mechanism interpretation, model assumptions, fitting, simulation, first-principles calculation, theory figures, and calculated/fitted parameters kept separate from experimental scalar values.

    Use paper_info as a separate top-level owner for bibliographic metadata and resources; do not model paper metadata as one of material_info.section0-section4.
    If the target is device-, reaction-, dataset-, or interface-centric, the supervisor may add a separate top-level owner such as device_info, reaction_info, dataset_info, or interface_info, but should still preserve the six-section material_info backbone when material records are present.
    """
).strip()


FINAL_OUTPUT_SCHEMA_DESCRIPTION = textwrap.dedent(
    """
    Return valid JSON only. Use this schema:
    {
      "database_positioning": {
        "database_goal": "string",
        "discipline": "string",
        "query_requirements": ["string"],
        "retrieval_unit": "string",
        "design_rationale": "string"
      },
      "section_design": {
        "core_sections": [
          {
            "section_id": "material_info.section0",
            "section_name": "string",
            "purpose": "string",
            "why_core": "string",
            "included_information": ["string"],
            "excluded_information": ["string"]
          }
        ],
        "non_core_sections": [
          {
            "section_id": "sectionX",
            "section_name": "string",
            "purpose": "string",
            "why_non_core": "string",
            "included_information": ["string"],
            "excluded_information": ["string"]
          }
        ]
      },
      "schema_definition": {
        "top_level_keys": [
          {
            "key": "string",
            "description": "string"
          }
        ],
        "field_registry": [
          {
            "field_path": "string",
            "section_id": "string",
            "field_name": "string",
            "data_type": "string",
            "required": true,
            "source_basis": ["text", "table", "figure"],
            "figure_constraint": {
              "uses_figure_classification": true,
              "allowed_sections": ["material_info.section4"],
              "allowed_figure_categories": ["R_T"],
              "why_needed": "string"
            },
            "reason": "string"
          }
        ]
      },
      "quality_check": {
        "topic_specific_adjustments": ["string"],
        "coverage_check": ["string"],
        "redo_needed": false,
        "redo_reason": ""
      }
    }
    """
).strip()


def _render_query_requirements(query_requirements):
    if isinstance(query_requirements, list):
        return "\n".join(f"- {item}" for item in query_requirements)
    return str(query_requirements)


def build_shared_context_block(
    database_goal,
    discipline,
    query_requirements,
    key_description_text,
    reference_paper_context="",
):
    reference_paper_block = ""
    if reference_paper_context:
        reference_paper_block = textwrap.dedent(
            f"""

            Reference papers:
            Use the following paper excerpts as concrete evidence for deciding section layout and field granularity. Prefer fields that are directly supported by recurring entities, measurement methods, conditions, and figure-bearing results in these papers.
            {reference_paper_context}
            """
        )
    return textwrap.dedent(
        f"""
        Shared context:
        - Database goal: {database_goal}
        - Discipline: {discipline}
        - Query requirements:
        {_render_query_requirements(query_requirements)}

        Domain reminder:
        In materials science, most database tasks should start from a stable six-section backbone and specialize the fields inside each section for the target domain.
        The backbone is shared, but the material_info.section1 performance fields, material_info.section3 evidence types, material_info.section4 curve types, and section5 mechanisms must be domain-specific.
        Do not invent a new section numbering scheme unless the supervisor explicitly justifies why the target cannot fit the backbone.
        For figure-heavy section pairs such as material_info.section4 and section5, do not let one section freely consume figures from the whole paper before figure ownership is clarified.

        {MATERIAL_DATABASE_SECTION_BACKBONE}

        Reference key descriptions:
        {key_description_text}
        {reference_paper_block}
        """
    ).strip()


def build_locating_prompt(
    database_goal,
    discipline,
    query_requirements,
    key_description_text,
    reference_paper_context="",
):
    return textwrap.dedent(
        f"""
        You are the locating module inside Step 8: Section Design Agent.
        Your job is to understand the database target before any section is designed.

        {build_shared_context_block(database_goal, discipline, query_requirements, key_description_text, reference_paper_context)}

        Return valid JSON only:
        {{
          "database_goal": "string",
          "discipline": "string",
          "query_requirements": ["string"],
          "retrieval_unit": "string",
          "organization_focus": "string",
          "design_rationale": "string"
        }}
        """
    ).strip()


def build_mechanism_requirement_prompt(shared_context, locating_result):
    return textwrap.dedent(
        f"""
        You are the mechanism requirement module inside Step 8: Section Design Agent.
        Your job is to identify the domain-critical mechanism concepts that must become explicit schema constraints.

        Shared context:
        {json.dumps(shared_context, ensure_ascii=False, indent=2)}

        Locating result:
        {json.dumps(locating_result, ensure_ascii=False, indent=2)}

        Requirements:
        - Focus on concepts that must appear as explicit fields or structured objects.
        - Avoid generic advice such as "consider more mechanism details".
        - Name domain-specific concepts directly.
        - Include a red-flag list describing what a too-generic schema would miss.

        Return valid JSON only:
        {{
          "domain_focus": "string",
          "must_have_concepts": ["string"],
          "recommended_objects": ["string"],
          "red_flag_patterns": ["string"]
        }}
        """
    ).strip()


def build_query_semantics_prompt(shared_context, locating_result):
    return textwrap.dedent(
        f"""
        You are the query semantics module inside Step 8: Section Design Agent.
        Your job is to convert the user's query requirements into structured retrieval objects, comparison axes, and filterable field groups.

        Shared context:
        {json.dumps(shared_context, ensure_ascii=False, indent=2)}

        Locating result:
        {json.dumps(locating_result, ensure_ascii=False, indent=2)}

        Requirements:
        - Translate each query requirement into schema-friendly objects, not prose summaries.
        - Explicitly distinguish scalar filters, range filters, state objects, and evidence-linked comparisons.
        - Flag any requirement that should not be compressed into one generic field.

        Return valid JSON only:
        {{
          "query_objects": [
            {{
              "query_requirement": "string",
              "object_type": "string",
              "recommended_field_groups": ["string"],
              "comparison_axes": ["string"],
              "anti_generic_warning": "string"
            }}
          ]
        }}
        """
    ).strip()


def build_evidence_model_prompt(shared_context, locating_result):
    return textwrap.dedent(
        f"""
        You are the evidence model module inside Step 8: Section Design Agent.
        Your job is to define the evidence hierarchy that the schema must preserve.

        Shared context:
        {json.dumps(shared_context, ensure_ascii=False, indent=2)}

        Locating result:
        {json.dumps(locating_result, ensure_ascii=False, indent=2)}

        Requirements:
        - Distinguish direct evidence, indirect evidence, reciprocal-space evidence, and theory/simulation support when relevant.
        - Explain which result types should own figures and which parameter fields should only reference evidence instead of owning whole figures.
        - Highlight generic evidence designs that should be avoided.
        - When the database includes inferred labels or states, define an evidence-confidence ladder appropriate to that domain.
        - Indirect measurements, proxy signals, or fitted interpretations should not be treated as direct evidence unless the domain justifies that mapping.
        - Require evidence records to preserve method, observed object, extracted parameter, source figure/table/text, measurement conditions, and interpretation risk.

        Return valid JSON only:
        {{
          "evidence_layers": [
            {{
              "layer_name": "string",
              "description": "string",
              "typical_methods": ["string"],
              "schema_implication": "string"
            }}
          ],
          "figure_ownership_principles": ["string"],
          "anti_generic_evidence_patterns": ["string"]
        }}
        """
    ).strip()


def build_subjective_supervisor_prompt(
    shared_context,
    locating_result,
    mechanism_result,
    query_result,
    evidence_result,
):
    return textwrap.dedent(
        f"""
        You are the subjective supervisor module inside Step 8: Section Design Agent.
        You are not a neutral summarizer. You must take a modeling position and decide what this database fundamentally is and is not.

        Shared context:
        {json.dumps(shared_context, ensure_ascii=False, indent=2)}

        Locating result:
        {json.dumps(locating_result, ensure_ascii=False, indent=2)}

        Mechanism requirement result:
        {json.dumps(mechanism_result, ensure_ascii=False, indent=2)}

        Query semantics result:
        {json.dumps(query_result, ensure_ascii=False, indent=2)}

        Evidence model result:
        {json.dumps(evidence_result, ensure_ascii=False, indent=2)}

        Requirements:
        - Explicitly state what kind of database this should be treated as.
        - Explicitly state what kind of generic template this should not collapse into.
        - List must-have concepts that downstream section and field design must preserve.
        - Identify concrete red flags such as umbrella fields, missing mechanism prerequisites, collapsed phase-window logic, or vague evidence ownership.
        - Treat domain labels and mechanism claims as claims with evidence grade, not just labels.
        - Treat mechanism and model parameters as provenance-sensitive values: record whether they are measured, fitted, simulated, calculated, assumed, or only discussed.
        - Produce actionable redesign directives, not abstract comments.

        Return valid JSON only:
        {{
          "database_nature": "string",
          "modeling_position": "string",
          "must_have_concepts": ["string"],
          "must_not_become": ["string"],
          "red_flags": ["string"],
          "approved_section_strategy": ["string"],
          "redo_directives": ["string"]
        }}
        """
    ).strip()


def build_topic_adaptation_prompt(shared_context, locating_result):
    return textwrap.dedent(
        f"""
        You are the topic adaptation module inside Step 8: Section Design Agent.
        Your job is to explain how this database theme should differ from a generic materials template under the subjective supervisor's modeling position.

        Shared context:
        {json.dumps(shared_context, ensure_ascii=False, indent=2)}

        Locating result:
        {json.dumps(locating_result, ensure_ascii=False, indent=2)}

        Return valid JSON only:
        {{
          "topic_type": "string",
          "adaptation_principles": ["string"],
          "avoid_generic_template": ["string"],
          "topic_specific_adjustments": ["string"]
        }}
        """
    ).strip()


def build_section_partition_prompt(
    shared_context,
    locating_result,
    mechanism_result,
    query_result,
    evidence_result,
    subjective_result,
    topic_result,
):
    return textwrap.dedent(
        f"""
        You are the section architecture module inside Step 8: Section Design Agent.
        Design sections that match the database topic, query goals, mechanism requirements, and evidence structure.

        Shared context:
        {json.dumps(shared_context, ensure_ascii=False, indent=2)}

        Locating result:
        {json.dumps(locating_result, ensure_ascii=False, indent=2)}

        Mechanism requirement result:
        {json.dumps(mechanism_result, ensure_ascii=False, indent=2)}

        Query semantics result:
        {json.dumps(query_result, ensure_ascii=False, indent=2)}

        Evidence model result:
        {json.dumps(evidence_result, ensure_ascii=False, indent=2)}

        Subjective supervisor result:
        {json.dumps(subjective_result, ensure_ascii=False, indent=2)}

        Topic adaptation result:
        {json.dumps(topic_result, ensure_ascii=False, indent=2)}

        Requirements:
        - Use the default six-section materials-database backbone as the starting section architecture.
        - Preserve the canonical section ids and meanings for most materials targets: material_info.section0, material_info.section1, material_info.section2, material_info.section3, material_info.section4, and section5.
        - Adapt the fields inside each section to the target domain rather than redefining what the section numbers mean.
        - Build additional top-level owners only when the target has a real non-material entity such as device_info, reaction_info, dataset_info, interface_info, or paper_info.
        - Build section boundaries around actual domain objects when extending the backbone, not around generic materials-database habits.
        - Ensure the section split can host the must-have concepts from the subjective supervisor.
        - Keep paper metadata under paper_info, not under material_info.section0-section4.
        - Separate state objects, mechanism prerequisites, evidence objects, and paper metadata when they serve different retrieval logic.

        Default backbone to apply unless explicitly overridden:
        {MATERIAL_DATABASE_SECTION_BACKBONE}

        Return valid JSON only:
        {{
          "core_sections": [
            {{
              "section_id": "material_info.section0",
              "section_name": "string",
              "purpose": "string",
              "why_core": "string",
              "included_information": ["string"],
              "excluded_information": ["string"]
            }}
          ],
          "non_core_sections": [
            {{
              "section_id": "sectionX",
              "section_name": "string",
              "purpose": "string",
              "why_non_core": "string",
              "included_information": ["string"],
              "excluded_information": ["string"]
            }}
          ]
        }}
        """
    ).strip()


def build_field_planning_prompt(
    shared_context,
    locating_result,
    mechanism_result,
    query_result,
    evidence_result,
    subjective_result,
    topic_result,
    section_result,
):
    return textwrap.dedent(
        f"""
        You are the field planning module inside Step 8: Section Design Agent.
        Your job is to plan the schema field objects before final assembly.

        Shared context:
        {json.dumps(shared_context, ensure_ascii=False, indent=2)}

        Locating result:
        {json.dumps(locating_result, ensure_ascii=False, indent=2)}

        Mechanism requirement result:
        {json.dumps(mechanism_result, ensure_ascii=False, indent=2)}

        Query semantics result:
        {json.dumps(query_result, ensure_ascii=False, indent=2)}

        Evidence model result:
        {json.dumps(evidence_result, ensure_ascii=False, indent=2)}

        Subjective supervisor result:
        {json.dumps(subjective_result, ensure_ascii=False, indent=2)}

        Topic adaptation result:
        {json.dumps(topic_result, ensure_ascii=False, indent=2)}

        Section architecture result:
        {json.dumps(section_result, ensure_ascii=False, indent=2)}

        Requirements:
        - Plan field groups and nested objects rather than only flat scalar keys.
        - Avoid umbrella fields such as generic figure catch-alls when parameter-level evidence linkage is more appropriate.
        - Explicitly note which objects require evidence references and which objects should own figures directly.
        - For inferred state or classification fields, plan nested objects with label, assignment_basis, primary_method, supporting_methods, confidence_level, and evidence_links when needed.
        - For mechanism fields, plan source-sensitive fields that preserve value, unit, estimation_method, source_type, and evidence_links when the domain requires them.
        - For proxy measurements, separate raw observations from final domain conclusions; weak or indirect evidence should trigger a risk or confidence field when appropriate.

        Return valid JSON only:
        {{
          "field_groups": [
            {{
              "section_id": "string",
              "group_name": "string",
              "purpose": "string",
              "recommended_fields": ["string"],
              "evidence_strategy": "string"
            }}
          ],
          "red_flag_fixes": ["string"]
        }}
        """
    ).strip()


def build_supervisor_prompt(
    shared_context,
    locating_result,
    topic_result,
    section_result,
    field_plan_result,
    human_advice=None,
):
    human_advice_block = ""
    if human_advice:
        human_advice_block = textwrap.dedent(
            f"""

            Human expert advice:
            The following human advice is mandatory input. The supervisor must explicitly account for it in the routing decision, risk assessment, trigger sections, and expected figure fields.
            {human_advice}
            """
        )

    return textwrap.dedent(
        f"""
        You are the figure supervisor module inside Step 8: Section Design Agent.
        Your job is to review the field and section plans and decide whether a figure classification repair agent is needed before schema assembly.

        Shared context:
        {json.dumps(shared_context, ensure_ascii=False, indent=2)}

        Locating result:
        {json.dumps(locating_result, ensure_ascii=False, indent=2)}

        Topic adaptation result:
        {json.dumps(topic_result, ensure_ascii=False, indent=2)}

        Section architecture result:
        {json.dumps(section_result, ensure_ascii=False, indent=2)}

        Field planning result:
        {json.dumps(field_plan_result, ensure_ascii=False, indent=2)}
        {human_advice_block}

        Decision rules:
        - Focus on whether multiple sections may compete for figure evidence.
        - If material_info.section4/section5 or similar figure-heavy sections can be confused, enable the figure classification repair agent.
        - Use the expected field types, evidence ownership plan, and section purposes to justify the decision.

        Return valid JSON only:
        {{
          "review_stage": "post_section_partition",
          "enable_figure_classification": true,
          "routing_decision": "figure_classification_path",
          "decision_summary": "string",
          "risk_signals": ["string"],
          "risk_assessment": "string",
          "trigger_sections": ["material_info.section4", "section5"],
          "expected_figure_fields": ["material_info.section4.R_T.figure"],
          "skip_reason": ""
        }}
        """
    ).strip()


def build_figure_classification_prompt(shared_context, section_result, field_plan_result, supervisor_result):
    return textwrap.dedent(
        f"""
        You are the figure classification repair agent inside Step 8: Section Design Agent.
        Your job is to prevent figure leakage across sections before schema design.

        Shared context:
        {json.dumps(shared_context, ensure_ascii=False, indent=2)}

        Section architecture result:
        {json.dumps(section_result, ensure_ascii=False, indent=2)}

        Field planning result:
        {json.dumps(field_plan_result, ensure_ascii=False, indent=2)}

        Supervisor result:
        {json.dumps(supervisor_result, ensure_ascii=False, indent=2)}

        Requirements:
        - First classify figures by owning section.
        - Then classify figure types only within that section.
        - Explicitly state which figure categories should be blocked from neighboring sections.
        - The output should guide downstream field design, not extract actual paper figures.
        - If the supervisor disabled this module, return an explicit skipped plan with enable_figure_classification=false and an empty section_figure_plan.

        Return valid JSON only:
        {{
          "enable_figure_classification": true,
          "routing_status": "active",
          "classification_strategy": ["string"],
          "section_figure_plan": [
            {{
              "section_id": "material_info.section4",
              "section_name": "string",
              "figure_scope": "string",
              "allowed_figure_categories": ["R_T", "R_H"],
              "blocked_neighbor_sections": ["section5"],
              "blocked_figure_categories": ["band_structure"],
              "routing_rule": "string"
            }}
          ],
          "skip_reason": ""
        }}
        """
    ).strip()


def build_schema_design_prompt(
    shared_context,
    locating_result,
    mechanism_result,
    query_result,
    evidence_result,
    subjective_result,
    topic_result,
    section_result,
    field_plan_result,
    supervisor_result,
    figure_result,
):
    return textwrap.dedent(
        f"""
        You are the schema assembly module inside Step 8: Section Design Agent.
        Build the schema skeleton and field registry from the chosen sections, field groups, and figure rules.

        Shared context:
        {json.dumps(shared_context, ensure_ascii=False, indent=2)}

        Locating result:
        {json.dumps(locating_result, ensure_ascii=False, indent=2)}

        Mechanism requirement result:
        {json.dumps(mechanism_result, ensure_ascii=False, indent=2)}

        Query semantics result:
        {json.dumps(query_result, ensure_ascii=False, indent=2)}

        Evidence model result:
        {json.dumps(evidence_result, ensure_ascii=False, indent=2)}

        Subjective supervisor result:
        {json.dumps(subjective_result, ensure_ascii=False, indent=2)}

        Topic adaptation result:
        {json.dumps(topic_result, ensure_ascii=False, indent=2)}

        Section architecture result:
        {json.dumps(section_result, ensure_ascii=False, indent=2)}

        Field planning result:
        {json.dumps(field_plan_result, ensure_ascii=False, indent=2)}

        Supervisor result:
        {json.dumps(supervisor_result, ensure_ascii=False, indent=2)}

        Figure classification result:
        {json.dumps(figure_result, ensure_ascii=False, indent=2)}

        Requirements:
        - Keep the schema assembly compact enough for reliable execution. Prefer 60-90 high-value fields over exhaustive expansion.
        - Convert each field-planning group into representative database fields; do not enumerate every possible synonym, subsection, or measurement variant when one contextualized array/object field can cover them.
        - Use concise descriptions and reasons. Long domain explanations belong in earlier modules, not in the field registry.
        - field_registry.section_id must point to either a declared canonical section id such as material_info.section1 or a top-level owner key such as paper_info, device_info, reaction_info, dataset_info, or interface_info.
        - Do not assign paper_info.* fields to material_info.section0-section4; bibliographic metadata and downloadable resources belong under paper_info.
        - Use material_info.section0 for material identity and tuning, material_info.section1 for core domain parameters/performance, material_info.section2 for processing, material_info.section3 for microscopic/electronic/structural evidence, material_info.section4 for macroscopic property curves, and section5 for theory/mechanism.
        - For a materials-literature task, schema roots must come from the approved section architecture and field_planning_result: paper_info, primary_signature, primary_signature_normalized, material_info, section5, normalization_aliases, material_name_aliases, formula_aliases, sample_id_aliases, plus explicitly justified non-material owners from the supervisor.
        - Do not introduce any schema root that is absent from both the section architecture and field_planning_result. If a prior attempt contains unrelated template roots, discard that attempt and rebuild from field_planning_result.
        - If figure classification is enabled, figure-linked fields must respect the allowed section and category boundaries from the figure classification result.
        - Avoid umbrella figure fields when object-level evidence references are more precise.
        - Reflect must-have concepts from the subjective supervisor and mechanism requirement module in explicit fields.
        - Include explicit confidence and evidence-link fields for inferred labels or state assignments when the target domain needs them.
        - Include theory or model provenance fields where needed so experimental values are not mixed with fitted, simulated, or calculated parameters.
        - Do not model proxy evidence as a direct high-confidence domain conclusion unless the target domain explicitly supports that inference.
        - Prefer the term figure classification over source attribution when justifying figure-linked fields.

        Return valid JSON only:
        {{
          "top_level_keys": [
            {{
              "key": "string",
              "description": "string"
            }}
          ],
          "field_registry": [
            {{
              "field_path": "string",
              "section_id": "string",
              "field_name": "string",
              "data_type": "string",
              "required": true,
              "source_basis": ["text", "table", "figure"],
              "figure_constraint": {{
                "uses_figure_classification": true,
                "allowed_sections": ["material_info.section4"],
                "allowed_figure_categories": ["R_T"],
                "why_needed": "string"
              }},
              "reason": "string"
            }}
          ]
        }}
        """
    ).strip()


def build_specialization_critic_prompt(
    shared_context,
    mechanism_result,
    query_result,
    evidence_result,
    subjective_result,
    section_result,
    field_plan_result,
    schema_result,
):
    return textwrap.dedent(
        f"""
        You are the specialization critic module inside Step 8: Section Design Agent.
        Your job is to judge whether the current design is still too generic or structurally weak.

        Shared context:
        {json.dumps(shared_context, ensure_ascii=False, indent=2)}

        Mechanism requirement result:
        {json.dumps(mechanism_result, ensure_ascii=False, indent=2)}

        Query semantics result:
        {json.dumps(query_result, ensure_ascii=False, indent=2)}

        Evidence model result:
        {json.dumps(evidence_result, ensure_ascii=False, indent=2)}

        Subjective supervisor result:
        {json.dumps(subjective_result, ensure_ascii=False, indent=2)}

        Section architecture result:
        {json.dumps(section_result, ensure_ascii=False, indent=2)}

        Field planning result:
        {json.dumps(field_plan_result, ensure_ascii=False, indent=2)}

        Schema assembly result:
        {json.dumps(schema_result, ensure_ascii=False, indent=2)}

        Requirements:
        - Judge substance, not formatting.
        - Do not reject a design merely because it uses the shared six-section materials backbone; that backbone is expected.
        - Reject designs that redefine the canonical section meanings without justification, omit the backbone when material records are present, or put paper metadata into material_info.section0-section4.
        - State whether the fields inside the shared backbone are still too generic for the target domain.
        - Flag missing must-have concepts, collapsed objects, weak evidence modeling, or vague field ownership.
        - Reject designs that assign inferred labels from weak proxy evidence without confidence, basis, or risk fields.
        - Reject designs that mention mechanism or model parameters but do not distinguish experimental, fitted, simulated, calculated, literature, or author-interpretation origin.
        - Reject designs that collapse theoretical parameters, simulation parameters, and experimental observables into one generic mechanism summary.
        - If the design is weak, provide targeted redo directives.

        Return valid JSON only:
        {{
          "specialization_status": "pass",
          "is_generic": false,
          "missing_concepts": ["string"],
          "structural_weaknesses": ["string"],
          "redo_needed": false,
          "redo_directives": ["string"]
        }}
        """
    ).strip()


def build_aggregation_prompt(
    shared_context,
    locating_result,
    mechanism_result,
    query_result,
    evidence_result,
    subjective_result,
    topic_result,
    section_result,
    field_plan_result,
    supervisor_result,
    figure_result,
    schema_result,
    critic_result,
):
    return textwrap.dedent(
        f"""
        You are the aggregation module inside Step 8: Section Design Agent.
        Assemble the final result from the upstream module outputs.

        Shared context:
        {json.dumps(shared_context, ensure_ascii=False, indent=2)}

        Locating result:
        {json.dumps(locating_result, ensure_ascii=False, indent=2)}

        Mechanism requirement result:
        {json.dumps(mechanism_result, ensure_ascii=False, indent=2)}

        Query semantics result:
        {json.dumps(query_result, ensure_ascii=False, indent=2)}

        Evidence model result:
        {json.dumps(evidence_result, ensure_ascii=False, indent=2)}

        Subjective supervisor result:
        {json.dumps(subjective_result, ensure_ascii=False, indent=2)}

        Topic adaptation result:
        {json.dumps(topic_result, ensure_ascii=False, indent=2)}

        Section architecture result:
        {json.dumps(section_result, ensure_ascii=False, indent=2)}

        Field planning result:
        {json.dumps(field_plan_result, ensure_ascii=False, indent=2)}

        Supervisor result:
        {json.dumps(supervisor_result, ensure_ascii=False, indent=2)}

        Figure classification result:
        {json.dumps(figure_result, ensure_ascii=False, indent=2)}

        Schema design result:
        {json.dumps(schema_result, ensure_ascii=False, indent=2)}

        Specialization critic result:
        {json.dumps(critic_result, ensure_ascii=False, indent=2)}

        Requirements:
        - database_positioning should be built mainly from the locating result.
        - section_design should be built from the section partition result.
        - schema_definition.top_level_keys should come from the schema design result.
        - schema_definition.field_registry should come from the schema design result.
        - quality_check.topic_specific_adjustments should combine the topic adaptation result, subjective supervisor result, and specialization critic result.
        - quality_check.coverage_check should summarize how the schema supports the query requirements.
        - quality_check should mention whether figure classification was enabled and what conflict it prevents.
        - quality_check should mention whether the specialization critic found generic-template risk.
        - Set redo_needed to true if the specialization critic says the design is still too generic or structurally weak.

        {FINAL_OUTPUT_SCHEMA_DESCRIPTION}
        """
    ).strip()


def build_module_redo_prompt(module_name, previous_result, validation_errors, schema_text):
    error_text = "\n".join(f"- {item}" for item in validation_errors)
    return textwrap.dedent(
        f"""
        Redo the `{module_name}` output.

        The previous result failed validation for these reasons:
        {error_text}

        Previous result:
        {previous_result}

        Return valid JSON only using this schema:
        {schema_text}
        """
    ).strip()


def build_redo_prompt(previous_result, validation_errors):
    error_text = "\n".join(f"- {item}" for item in validation_errors)
    return textwrap.dedent(
        f"""
        Redo the full Step 8 section design.

        The previous result failed validation for these reasons:
        {error_text}

        Previous result:
        {previous_result}

        Fix the problems and return valid JSON only using the same schema.

        {FINAL_OUTPUT_SCHEMA_DESCRIPTION}
        """
    ).strip()
