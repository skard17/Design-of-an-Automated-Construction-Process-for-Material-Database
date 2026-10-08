import json
import re
import textwrap


MAX_REFERENCE_CONTEXT_CHARS = 2000000
MAX_KEY_DESCRIPTION_CHARS = 500000


def _balanced_text_excerpt(text, max_chars):
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    if max_chars < 900:
        return text[:max_chars].rstrip() + "\n...[truncated]"
    marker_budget = 80
    segment_chars = max((max_chars - marker_budget) // 3, 200)
    middle_start = max((len(text) - segment_chars) // 2, segment_chars)
    return (
        text[:segment_chars].rstrip()
        + "\n...[middle excerpt]...\n"
        + text[middle_start : middle_start + segment_chars].strip()
        + "\n...[ending excerpt]...\n"
        + text[-segment_chars:].lstrip()
    )[:max_chars]


def _balanced_reference_paper_context(text, max_chars=MAX_REFERENCE_CONTEXT_CHARS):
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    blocks = [
        block.strip()
        for block in re.split(r"(?=^Paper:\s)", text, flags=re.MULTILINE)
        if block.strip()
    ]
    if len(blocks) <= 1:
        return _balanced_text_excerpt(text, max_chars)
    per_block = max(max_chars // len(blocks), 900)
    return "\n\n".join(
        _balanced_text_excerpt(block, per_block) for block in blocks
    )[:max_chars]


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
    - The task is automated construction of a materials database from scientific literature. The source boundary is research papers, preprints, and their supplementary literature artifacts after document parsing; do not broaden the workflow to textbooks, laboratory notebooks, arbitrary reports, general web text, or generic scientific-text conversion.
    - The output retrieval records must remain materials-database records. Literature structure, bibliographic metadata, evidence units, and lineage support those records; they are not a replacement task or a generic document database.
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
    - Before choosing schema roots, identify the scientific entities that the database must query independently, such as materials, samples, devices, interfaces, reactions, datasets, states, or measurement configurations.
    - Entity-specific quantities must belong to their actual entity owner. Do not force device response, interface behavior, reaction kinetics, directional asymmetry, spectral states, or geometry into the nearest generic material field.
    - A schema may support inferred scientific labels only through an explicit inference object with assignment basis, source type, confidence, and evidence links; otherwise the extraction contract must require directly stated or directly measured facts.
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
    If the target is device-, reaction-, dataset-, or interface-centric, the supervisor may add a separate top-level owner such as device_info, reaction_info, dataset_info, or interface_info, but should still preserve the six-section material_info backbone when material records are present. Choose these owners from the target's real retrieval entities, not from a fixed domain example.
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
      "task_contract": {
        "task_type": "automated_materials_database_construction",
        "source_scope": "scientific_literature_only",
        "source_artifact": "parsed_literature_markdown",
        "record_scope": "queryable material, sample, process, measurement, and mechanism records",
        "scope_version": "materials-literature-v1"
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
            "core_field": true,
            "source_basis": ["text", "table", "figure"],
            "inclusion_rule": "string",
            "absence_rule": "string",
            "field_rule_id": "field.material_info.section1.example",
            "field_rule_version": "sha256:stable contract digest",
            "evidence_requirements": {
              "direct_support_required": true,
              "locator_required": true,
              "allowed_source_types": ["text", "table", "figure"]
            },
            "relation_constraints": {
              "entity_binding_required": true,
              "condition_binding_required": true,
              "separate_instances": true
            },
            "concept_ids": ["ascii_snake_case_id"],
            "object_contract": {
              "object_kind": "measurement",
              "required_subfields": ["value", "conditions", "entity_ref", "source_type", "evidence", "confidence"]
            },
            "figure_constraint": {
              "uses_figure_classification": true,
              "allowed_sections": ["material_info.section4"],
              "allowed_figure_categories": ["domain_specific_curve"],
              "why_needed": "string"
            },
            "reason": "string"
          }
        ]
      },
      "reference_field_contract": {
        "available": true,
        "source_path": "string",
        "source_sha256": "string",
        "policy": "task_adaptive",
        "all_path_count": 0,
        "leaf_count": 0,
        "leaf_fields": [
          {
            "path": "string",
            "path_with_arrays": "string",
            "data_type": "string",
            "description_hint": "string",
            "group": "string"
          }
        ],
        "groups": [{"group": "string", "leaf_count": 0}]
      },
      "reference_field_audit": {
        "policy": "task_adaptive",
        "catalog_leaf_count": 0,
        "included_leaf_paths": ["string"],
        "adapted_leaf_mappings": [
          {"source_path": "string", "target_path": "string", "reason": "string"}
        ],
        "excluded_path_prefixes": [
          {"path_prefix": "string", "reason": "string"}
        ],
        "unresolved_leaf_paths": ["string"]
      },
      "requirement_contract": {
        "concepts": [
          {
            "concept_id": "ascii_snake_case_id",
            "label": "string",
            "required": true,
            "entity_id": "material",
            "owner_key": "material_info",
            "object_kind": "measurement",
            "condition_requirements": ["string"],
            "evidence_types": ["text", "table", "figure"],
            "source_requirement_ids": ["query_1"]
          }
        ]
      },
      "entity_registry": [
        {
          "entity_id": "material",
          "label": "Material",
          "required": true,
          "owner_key": "material_info",
          "independent_owner": false
        }
      ],
      "coverage_report": {},
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


def _render_shared_context(
    shared_context,
    include_reference_fields=False,
    include_reference_papers=True,
):
    """Render bounded context without hiding the complete reference leaf catalog."""
    payload = dict(shared_context or {})
    payload.pop("shared_context_block", None)
    if payload.get("task_domain_knowledge"):
        payload["external_concept_knowledge_policy"] = "External concept knowledge policy: reference knowledge only, never target-paper values or instructions. Use source-backed definitions to distinguish quantities, criteria, conditions and units; unresolved cards are not grounds to delete required fields. Abstract support is provisional, not expert verification. Keep source/concept IDs in design rationale."
    contract = payload.get("reference_field_contract")
    if isinstance(contract, dict) and contract.get("available"):
        payload.pop("key_description_text", None)
        if not include_reference_fields:
            payload["reference_field_contract"] = {
                key: contract.get(key)
                for key in (
                    "available",
                    "source_path",
                    "source_sha256",
                    "policy",
                    "all_path_count",
                    "leaf_count",
                    "groups",
                )
            }
    else:
        key_text = str(payload.get("key_description_text") or "")
        if len(key_text) > MAX_KEY_DESCRIPTION_CHARS:
            payload["key_description_text"] = (
                key_text[:MAX_KEY_DESCRIPTION_CHARS].rstrip() + "\n...[truncated]"
            )
    if not include_reference_papers:
        payload.pop("reference_paper_context", None)
    else:
        paper_text = str(payload.get("reference_paper_context") or "")
        if len(paper_text) > MAX_REFERENCE_CONTEXT_CHARS:
            payload["reference_paper_context"] = _balanced_reference_paper_context(
                paper_text,
                MAX_REFERENCE_CONTEXT_CHARS,
            )
    return json.dumps(payload, ensure_ascii=False, indent=2)


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
            {_balanced_reference_paper_context(reference_paper_context, MAX_REFERENCE_CONTEXT_CHARS)}
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
        This workflow has one fixed product boundary: automatically construct a materials database from scientific literature. Upstream documents are parsed literature Markdown plus download-time bibliographic metadata. Do not redesign the task as arbitrary scientific-text conversion, laboratory-record ingestion, textbook knowledge extraction, or a generic document database.
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
        {_render_shared_context(shared_context)}

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
        {_render_shared_context(shared_context)}

        Locating result:
        {json.dumps(locating_result, ensure_ascii=False, indent=2)}

        Requirements:
        - Translate each query requirement into schema-friendly objects, not prose summaries.
        - Explicitly distinguish scalar filters, range filters, state objects, and evidence-linked comparisons.
        - Flag any requirement that should not be compressed into one generic field.
        - For each shared_context.care_counterfactual_queries item, emit a separate query object using its exact requirement_id as source_requirement_id and its exact query_id as counterfactual_query_id.
        - Preserve every CARE required_distinctions item as an explicit comparison axis or recommended field group. An umbrella field does not satisfy a CARE query.

        Return valid JSON only:
        {{
          "query_objects": [
            {{
              "query_requirement": "string",
              "source_requirement_id": "query_1",
              "counterfactual_query_id": "care_query_id_or_empty",
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
        {_render_shared_context(shared_context)}

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
    care_requirement_ids = set(
        map(str, shared_context.get("care_counterfactual_requirement_ids") or [])
    )
    source_requirement_catalog = [
        {
            "requirement_id": f"query_{index}",
            "source": (
                "care_counterfactual_query"
                if f"query_{index}" in care_requirement_ids
                else "query_requirement"
            ),
            "text": str(requirement),
        }
        for index, requirement in enumerate(shared_context.get("query_requirements", []) or [], start=1)
    ]
    if str(shared_context.get("human_advice") or "").strip():
        source_requirement_catalog.append(
            {
                "requirement_id": "human_advice",
                "source": "human_expert",
                "text": str(shared_context.get("human_advice")).strip(),
            }
        )
    return textwrap.dedent(
        f"""
        You are the subjective supervisor module inside Step 8: Section Design Agent.
        You are not a neutral summarizer. You must take a modeling position and decide what this database fundamentally is and is not.

        Shared context:
        {_render_shared_context(
            shared_context,
            include_reference_fields=True,
            include_reference_papers=False,
        )}

        Locating result:
        {json.dumps(locating_result, ensure_ascii=False, indent=2)}

        Mechanism requirement result:
        {json.dumps(mechanism_result, ensure_ascii=False, indent=2)}

        Query semantics result:
        {json.dumps(query_result, ensure_ascii=False, indent=2)}

        Evidence model result:
        {json.dumps(evidence_result, ensure_ascii=False, indent=2)}

        Source requirement catalog (use these exact requirement_id values):
        {json.dumps(source_requirement_catalog, ensure_ascii=False, indent=2)}

        Requirements:
        - Explicitly state what kind of database this should be treated as.
        - Explicitly state what kind of generic template this should not collapse into.
        - List must-have concepts that downstream section and field design must preserve.
        - Split compound requirements into atomic concepts. Each concept must have a stable ASCII snake_case concept_id that downstream fields can reference exactly.
        - Treat human_advice in shared context as mandatory input to the requirement contract, not as a late figure-routing comment.
        - Inventory independently queryable entities. Materials may use material_info; devices, interfaces, reactions, or datasets need their own owner when their identity, configuration, or response is queried independently.
        - Evidence and provenance records normally remain nested under the scientific material, sample, device, reaction, or dataset they support. Do not create an independent evidence_collection top-level owner unless the user's retrieval unit is explicitly an evidence record.
        - Every query requirement and human-advice requirement must be referenced by at least one concept through source_requirement_ids.
        - Every care_counterfactual_query must be represented by one or more concepts that preserve all required distinctions; do not trace a CARE query to an unrelated generic concept merely to satisfy coverage.
        - There is no target concept count. Stop when the task's independently queryable scientific distinctions are represented; do not expand one distinction into aliases, spelling variants, unit variants, or deterministically derived values.
        - Every concept must state why_needed as a concrete retrieval, comparison, evidence-binding, or provenance need. A broad topic association is not sufficient. Merge concepts that would share the same owner, semantics, conditions, evidence, and extraction rule.
        - Do not create speculative concepts merely because they are common in the discipline. They must be supported by a source requirement, an applicable reference decision, or a recurring distinction in the supplied literature.
        - Identify concrete red flags such as umbrella fields, missing mechanism prerequisites, collapsed phase-window logic, or vague evidence ownership.
        - Treat domain labels and mechanism claims as claims with evidence grade, not just labels.
        - Treat mechanism and model parameters as provenance-sensitive values: record whether they are measured, fitted, simulated, calculated, assumed, or only discussed.
        - Produce actionable redesign directives, not abstract comments.

        Return valid JSON only:
        {{
          "database_nature": "string",
          "modeling_position": "string",
          "must_have_concepts": ["string"],
          "requirement_contract": {{
            "concepts": [
              {{
                "concept_id": "ascii_snake_case_id",
                "label": "string",
                "required": true,
                "entity_id": "material",
                "owner_key": "material_info",
                "object_kind": "scalar|measurement|classification|entity_descriptor|process|evidence_collection",
                "condition_requirements": ["string"],
                "evidence_types": ["text", "table", "figure"],
                "source_requirement_ids": ["query_1", "human_advice"],
                "why_needed": "concrete query, comparison, binding, or provenance need"
              }}
            ]
          }},
          "entity_registry": [
            {{
              "entity_id": "material",
              "label": "Material",
              "required": true,
              "owner_key": "material_info",
              "independent_owner": false
            }}
          ],
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
        {_render_shared_context(shared_context)}

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
        {_render_shared_context(shared_context)}

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
        - Start by inventorying independently queryable entities and their relations. Add an owner when its identity, geometry/configuration, state, input conditions, outputs, or evidence must be queried independently from the material record.
        - Keep entity-specific response quantities with their owner. Direction-dependent responses, operating polarity, geometry, configuration, interfaces, spectra, state populations, kinetics, and protocol-dependent outputs are examples of field families to consider only when relevant to the target, not mandatory template fields.
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


TRANSFERABLE_FIELD_DESIGN_RULES = textwrap.dedent(
    """
    Transferable field-design acceptance rules (apply to the current task, not a fixed domain catalog):
    - Separate reusable database capabilities from task-specific science. Audit document identity, bibliographic identifiers/date/type, material and sample identity, observation ownership, provenance, and reported code/data/structure resources against the task's retrieval requirements. Retain applicable capabilities even when this paper sample has no populated values; missing evidence controls population, not schema inclusion. Use existing canonical owners and keep resource kind, availability, repository, identifier, and link distinct where needed. Do not invent resource values or unapproved roots.
    - Domain extensions need a concrete query and a task requirement or result/method evidence. A rare but relevant result can justify an extension; recurrence is not mandatory. Background citations and plausible discipline-wide candidates are insufficient. Explain retained capabilities and omitted candidates in existing purpose, reason, evidence_strategy, and audit fields. Do not require a particular domain's field names or a numerical field count.
    - Preserve repeatable record boundaries end to end. Use [] at every repeatable ancestor, including nested components; a scalar leaf inside a repeated record remains scalar. Set relation_constraints.separate_instances=true for leaves of repeated observations/components. Preserve distinct instances by owner, quantity definition, method/criterion, conditions, and provenance. An observation_ref does not make a single-valued property container repeatable. Test two measurements of the same quantity under different conditions and two ordered components with different attributes without overwrite or cross-pairing.
    - Shared contracts are acceptable when a typed, resolvable reference preserves an independently queryable distinction. For each named quantity, specify the path from its repeated record to the observation, sample, condition, method, and evidence targets in existing descriptions/contracts. Verify target identity, multiplicity, payload, and ownership. Merely having a global pressure, direction, or locator slot is insufficient. Do not duplicate all common children if a lossless reference and query path already exists.
    - Keep quantity-specific interpretation explicit: criterion, condition value/unit or range, stimulus orientation, reference frame, geometry, and controlled versus swept variables wherever the task or evidence needs them. Distinguish sample orientation, stimulus direction, and current/flow direction. Preserve condition-to-instance and panel/series bindings; unrelated condition values must not form an accidental Cartesian product.
    - A plotted-data record must describe its actual response and axes, applicable conditions, sample, source panel/table, and data or payload reference. A figure locator alone is not a data payload, and a derived scalar is not the original curve. Declare whether a payload is reported, linked, unavailable, or would require a separately authorized digitization step; do not promise unperformed extraction.
    - One field has one stable scientific or bibliographic meaning. Separate dates from identifiers and observation values from interpretations. Keep aliases/unit variants in normalization rules. Distinguish a specific subcase from its broader parent concept; naming a subcase cannot claim full coverage of the parent. Assign concept_ids only where field semantics actually implement that concept; unrelated concepts must not be attached just to satisfy coverage.
    - For an explicitly supplied compatibility catalog, preserve applicable semantics, repeatability, units, and bindings using exact paths or documented lossless mappings. If no catalog is supplied, report reference coverage as unavailable; internal concept coverage and successful execution do not establish external field coverage. Do not import a hidden evaluation schema into design.
    - Before acceptance, execute a paper-to-record thought experiment and a record-to-query round trip using concrete task evidence. Check repeated observations, compound samples, measured versus modeled values, missing data, and referenced resources when applicable. State which distinction and exact path fail; a field list, valid JSON, or concept count alone is insufficient. Use existing red_flag_fixes, audit reasons, and targeted repair channels without adding a new output protocol.
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

        {TRANSFERABLE_FIELD_DESIGN_RULES}

        Shared context:
        {_render_shared_context(shared_context, include_reference_fields=True)}

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

        Per-field definition contract:
        - Semantic definitions must be concise: description <=1200 characters and extraction_notes <=2400 characters. Preserve boundaries and binding rules; do not paste full papers or group inventories into a leaf definition.
        - Emit field_definition_contract_version="materials-field-definitions/v1" and exactly one field_definitions descriptor per recommended_fields path in each group, with identical paths and [] markers. Define fields during planning, before assembly; never leave definitions to heuristics or reuse a group purpose as a leaf description.
        - Define direct meaning, boundaries versus neighboring quantities, owner, conditions, criterion, units and normalization, inclusion/exclusion and absence semantics for each leaf. Explicitly identify reference targets and record multiplicity. Descriptions must be useful to an extractor without guessing from the field name.
        - Object/array-of-objects descriptors additionally need object_contract with object_kind, required_subfields (a string list) and fields (a nonempty typed child dictionary). Scalar leaves inside repeated ancestors retain their scalar type and separate_instances=true.
        - Preserve these definitions on all targeted planner repairs. Update the definition together with any added/moved/removed path; retain unaffected definitions. Do not silently substitute a generic value/evidence template.

        Requirements:
        - Keep the product boundary fixed: design fields for automated materials-database construction from scientific literature, not for arbitrary scientific documents.
        - Plan field groups and nested objects rather than only flat scalar keys.
        - When reference_field_contract is available, treat it as a complete candidate catalog supplied for this task, not as a short illustrative example and not as an unconditional domain template.
        - Audit every reference leaf field. Include it when the database goal, query requirements, or reference papers make it useful; adapt it only when the target schema needs a genuinely different owner or name; exclude it only with a task-specific reason. Do not leave catalog leaves unclassified.
        - Every included reference leaf must appear verbatim in one field_groups[].recommended_fields list. Every adapted source leaf must map to an explicit target leaf that also appears in recommended_fields.
        - Keep independently queryable values, units, criteria, conditions, directions, methods, sample bindings, and evidence links as explicit leaf paths when they are semantically applicable to that quantity. A parent object or generic conditions/evidence catch-all does not count as representing a named distinction required by the task.
        - Avoid Cartesian-product expansion. Do not attach every possible condition, provenance slot, evidence type, unit variant, or method child to every scientific quantity. Reuse object_contract slots for common record mechanics and enumerate a child path only when it is independently queried, changes scientific interpretation, appears in an applicable reference decision, or recurs as a real distinction in the supplied papers.
        - Use excluded_path_prefixes only when one reason applies to the complete subtree. Do not exclude a broad family merely because a small reference-paper sample does not mention every value; distinguish database scope from per-paper sparsity.
        - Avoid umbrella fields such as generic figure catch-alls when parameter-level evidence linkage is more appropriate.
        - Explicitly note which objects require evidence references and which objects should own figures directly.
        - For inferred state or classification fields, plan nested objects with label, assignment_basis, primary_method, supporting_methods, confidence_level, and evidence_links when needed.
        - For mechanism fields, plan source-sensitive fields that preserve value, unit, estimation_method, source_type, and evidence_links when the domain requires them.
        - For proxy measurements, separate raw observations from final domain conclusions; weak or indirect evidence should trigger a risk or confidence field when appropriate.
        - Derive recommended fields from the target questions and reference documents. Examples from any one material domain are illustrative only and must not constrain or populate an unrelated domain.
        - Do not use a generic nearby field as a catch-all for a distinct target quantity. If a quantity has different semantics, conditions, directionality, entity ownership, or evidence type, plan a dedicated contextualized field or object.
        - When the task explicitly names a curve, spectrum, map, or short axis pair such as X-Y, create a separately queryable planned field or collection for that named evidence type. A generic curve_type or figure label alone is not sufficient.
        - Mark every planned field as direct-observation, author-interpretation, model-derived, or explicitly inferred. Inferred fields require assignment_basis, source_type, confidence_level, and evidence_links; otherwise require direct textual or measurement evidence.
        - There is no target field count. Stop adding planned fields after every required concept, applicable reference leaf, entity binding, and evidence contract has an executable home.
        - Keep the inventory compact enough to remain useful as a materials database: merge candidates with the same semantics, owner, conditions, and evidence rule; represent aliases and unit variants through normalization metadata; omit deterministically derivable duplicates unless the reported value is itself a distinct literature claim.
        - Do not add a field only because it is plausible in the wider discipline. Every recommended field must trace to an exact concept_id, an applicable reference-field decision, or an explicit recurring distinction in the supplied papers. More fields are not better when they add no new query or extraction capability.

        Return valid JSON only:
        {{
          "field_definition_contract_version": "materials-field-definitions/v1",
          "field_groups": [
            {{
              "section_id": "string",
              "group_name": "string",
              "purpose": "string",
              "recommended_fields": ["string"],
              "field_definitions": [
                {{
                  "field_path": "exact recommended_fields entry, preserving every [] ancestor",
                  "description": "Independent scientific meaning, physical definition and distinction from neighboring quantities; never the group purpose",
                  "extraction_notes": "Inclusion/exclusion, owner and condition binding, accepted units, normalization limits, criterion and evidence locator rules",
                  "data_type": "number|string|boolean|array|object|array of objects",
                  "required": false,
                  "source_basis": ["text", "table", "figure"],
                  "inclusion_rule": "Explicit evidence that permits population",
                  "absence_rule": "Distinguish unreported, inapplicable and unresolved; do not infer missing values",
                  "relation_constraints": {{"entity_binding_required": true, "condition_binding_required": true, "separate_instances": true}},
                  "evidence_requirements": {{"direct_support_required": true, "locator_required": true, "allowed_source_types": ["text", "table", "figure"]}}
                }}
              ],
              "evidence_strategy": "string"
            }}
          ],
          "reference_field_audit": {{
            "policy": "task_adaptive",
            "catalog_leaf_count": 0,
            "included_leaf_paths": ["exact.reference.leaf_path"],
            "adapted_leaf_mappings": [
              {{
                "source_path": "reference.leaf_path",
                "target_path": "task_specific.leaf_path",
                "reason": "string"
              }}
            ],
            "excluded_path_prefixes": [
              {{"path_prefix": "reference.family", "reason": "string"}}
            ],
            "unresolved_leaf_paths": []
          }},
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
        {_render_shared_context(shared_context)}

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
          "expected_figure_fields": ["material_info.section4.domain_specific_curve.figure"],
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
        {_render_shared_context(shared_context)}

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
        - Every section_figure_plan item must use the fixed keys section_id, section_name, figure_scope, allowed_figure_categories, blocked_neighbor_sections, blocked_figure_categories, and routing_rule.
        - material_info.section2 may receive figures only for fabrication/process schematics, synthesis or growth diagrams, device fabrication flows, processing-condition figures, or processing-condition tables.
        - Do not assign magnetic, transport, thermodynamic, optical, electrochemical, mechanical, or other property curves to material_info.section2; route those to material_info.section4.
        - Do not assign microscopy, diffraction, spectroscopy, composition maps, or electronic-structure characterization figures to material_info.section2; route those to material_info.section3.
        - Do not assign theory, simulation, mechanism schematics, fitted-model plots, or interpretation figures to material_info.section2; route those to section5.
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
              "allowed_figure_categories": ["domain_specific_curve", "domain_specific_map"],
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
        Copy each planned field definition into its matching registry row, preserving scalar type, record multiplicity and extraction semantics. Missing or conflicting definitions require planner repair, not group-purpose or name-based guesses.

        {TRANSFERABLE_FIELD_DESIGN_RULES}

        Shared context:
        {_render_shared_context(shared_context, include_reference_fields=True)}

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
        - Keep the product boundary fixed: the schema consumes parsed scientific literature and constructs queryable materials-database records. Do not add textbook, laboratory-notebook, or generic report-processing roots.
        - Include every field and object contract needed to cover all required concepts. Control downstream context size through staged batching, not by dropping required fields from the schema.
        - There is no fixed target field count. Stop adding fields when every task requirement, applicable reference leaf, entity binding, and evidence contract is represented and additional candidates would only duplicate semantics or add unsupported detail.
        - Do not optimize coverage by generating a huge speculative inventory. Every field must trace to a requirement_contract concept, an applicable reference-field decision, or an explicit recurring distinction in the supplied literature.
        - Store spelling variants, symbols, unit variants, and equivalent names through aliases or normalization rules instead of separate fields. Do not store values that are deterministically computable from already stored fields unless the task explicitly requires the reported value as a distinct literature claim.
        - Convert every selected or adapted leaf path from field_planning_result into its own field_registry row. Arrays and objects may organize records, but independently queryable scientific child values must remain enumerated leaf rows.
        - Do not materialize the complete object_contract required_subfields template as separate field_registry rows under every scientific quantity. The object contract carries shared value/condition/entity/source/evidence/confidence mechanics; add an explicit child row only when that child has quantity-specific semantics or independent query value.
        - Avoid Cartesian-product expansion across quantities and conditions. Temperature, pressure, field, orientation, protocol, method, criterion, and evidence children are included only where applicable, not copied mechanically to every property.
        - Do not compress distinct values, units, criteria, directions, conditions, methods, sample bindings, or evidence locations into one representative umbrella row. Synonyms may share aliases, but semantically distinct measurement variants require separate leaves.
        - Shared object contracts may carry reusable mechanics, but each scientific quantity or distinction required by a CARE counterfactual query must remain an explicit independently queryable leaf or explicit object child.
        - When shared_context.checkpoint_prior_schema is present, audit every prior field path against the repaired requirements. Preserve valid leaves; merge, replace, or remove a prior leaf only with an explicit reason in the replacement field or field-planning decision. A full repair must not silently shrink a valid checkpoint schema.
        - Respect the task-adaptive reference audit: included paths must remain exact unless an explicit adapted mapping is provided; excluded paths must not silently reappear under vague names; unresolved paths are a blocking design error.
        - For reference-catalog leaves, preserve their description_hint and extraction_notes unless an explicit adapted mapping requires a task-specific rewrite. For newly designed leaves, write an equally precise definition and extraction_notes covering inclusion, exclusion, target-entity binding, condition separation, evidence requirements, and normalization limits.
        - Every field row must define inclusion_rule, absence_rule, evidence_requirements, and relation_constraints. These rules are downstream extraction contracts, not narrative suggestions.
        - Mark core_field only when the system considers the field necessary for the target materials database. This system-generated designation must remain distinguishable from any separately supplied human-gold core-field mapping.
        - Keep descriptions concise, but do not shorten extraction_notes until semantic boundaries are lost. Examples are illustrative only and must never become the field inventory for an unrelated materials task.
        - field_registry.section_id must point to either a declared canonical section id such as material_info.section1 or a top-level owner key such as paper_info, device_info, reaction_info, dataset_info, or interface_info.
        - Do not assign paper_info.* fields to material_info.section0-section4; bibliographic metadata and downloadable resources belong under paper_info.
        - Use material_info.section0 for material identity and tuning, material_info.section1 for core domain parameters/performance, material_info.section2 for processing, material_info.section3 for microscopic/electronic/structural evidence, material_info.section4 for macroscopic property curves, and section5 for theory/mechanism.
        - For a materials-literature task, schema roots must come from the approved section architecture and field_planning_result: paper_info, primary_signature, primary_signature_normalized, material_info, section5, normalization_aliases, material_name_aliases, formula_aliases, sample_id_aliases, plus explicitly justified non-material owners from the supervisor.
        - Do not introduce any schema root that is absent from both the section architecture and field_planning_result. If a prior attempt contains unrelated template roots, discard that attempt and rebuild from field_planning_result.
        - If figure classification is enabled, figure-linked fields must respect the allowed section and category boundaries from the figure classification result.
        - Avoid umbrella figure fields when object-level evidence references are more precise.
        - Reflect must-have concepts from the subjective supervisor and mechanism requirement module in explicit fields.
        - Copy the exact concept_id values from subjective_result.requirement_contract into each scientific field's concept_ids. Every required concept must map to at least one field; do not invent new concept ids.
        - Verify that every independently queryable entity identified upstream has an appropriate owner and that its identity, configuration/conditions, response quantities, and evidence can be represented without being folded into an unrelated material field.
        - Do not copy field examples from a reference domain into the final registry unless the target questions or documents justify them.
        - Preserve every explicitly named curve, spectrum, map, or short axis-pair evidence type from the task as its own field or collection under the appropriate task-derived section. Do not collapse named evidence types into only a generic curve_type enum.
        - Every scientific claim field must reference an executable provenance contract covering source text/table/figure locators and confidence. Represent those common mechanics through evidence_requirements or a reusable object_contract; create separate evidence.* field rows only when the database must query those values independently.
        - Measurement fields must preserve applicable temperature, pressure, stimulus, direction/geometry, protocol, method, and criterion context through a reusable measurement-context contract. Create explicit measurement_conditions.* field rows only for conditions with task-specific semantics or independent query value; never copy the full condition template under every quantity.
        - Include explicit confidence and evidence-link fields for inferred labels or state assignments when the target domain needs them.
        - Include theory or model provenance fields where needed so experimental values are not mixed with fitted, simulated, or calculated parameters.
        - Do not model proxy evidence as a direct high-confidence domain conclusion unless the target domain explicitly supports that inference.
        - If an inferred label is not represented by a structured inference object with basis, source type, confidence, and evidence links, omit that label from the schema rather than inviting unsupported extraction.
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
              "description": "precise task-specific field semantics and inclusion boundary",
              "extraction_notes": "precise inclusion, exclusion, condition-binding, evidence, and normalization rules",
              "data_type": "string",
              "required": true,
              "core_field": true,
              "source_basis": ["text", "table", "figure"],
              "inclusion_rule": "when literature evidence permits this field to be populated",
              "absence_rule": "when the field must be missing or unresolved rather than inferred",
              "evidence_requirements": {{
                "direct_support_required": true,
                "locator_required": true,
                "allowed_source_types": ["text", "table", "figure"]
              }},
              "relation_constraints": {{
                "entity_binding_required": true,
                "condition_binding_required": true,
                "separate_instances": true
              }},
              "concept_ids": ["ascii_snake_case_id"],
              "object_contract": {{
                "object_kind": "measurement",
                "required_subfields": ["value", "conditions", "entity_ref", "source_type", "evidence", "confidence"]
              }},
              "figure_constraint": {{
                "uses_figure_classification": true,
                "allowed_sections": ["material_info.section4"],
                "allowed_figure_categories": ["domain_specific_curve"],
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
    repair_feedback=None,
):
    reference_contract = shared_context.get("reference_field_contract") or {}
    critic_context = {
        "task_contract": shared_context.get("task_contract"),
        "database_goal": shared_context.get("database_goal"),
        "discipline": shared_context.get("discipline"),
        "query_requirements": shared_context.get("query_requirements") or [],
        "human_advice": shared_context.get("human_advice") or "",
        "reference_field_contract": {
            key: reference_contract.get(key)
            for key in ("available", "source_path", "source_sha256", "policy", "leaf_count")
        },
    }
    compact_sections = []
    for section in [
        *(section_result.get("core_sections") or []),
        *(section_result.get("non_core_sections") or []),
    ]:
        if not isinstance(section, dict):
            continue
        compact_sections.append(
            {
                key: section.get(key)
                for key in (
                    "section_id",
                    "section_name",
                    "included_information",
                    "excluded_information",
                )
                if section.get(key) not in (None, "", [], {})
            }
        )

    compact_field_groups = []
    for group in field_plan_result.get("field_groups") or []:
        if not isinstance(group, dict):
            continue
        compact_field_groups.append(
            {
                key: group.get(key)
                for key in ("section_id", "group_name", "recommended_fields", "field_definitions")
                if group.get(key) not in (None, "", [], {})
            }
        )

    critic_upstream = {
        "mechanism_requirements": {
            key: mechanism_result.get(key)
            for key in ("domain_focus", "must_have_concepts", "recommended_objects")
        },
        "query_objects": query_result.get("query_objects") or [],
        "evidence_layers": evidence_result.get("evidence_layers") or [],
        "requirement_contract": subjective_result.get("requirement_contract") or {},
        "entity_registry": subjective_result.get("entity_registry") or [],
        "must_not_become": subjective_result.get("must_not_become") or [],
        "section_architecture": compact_sections,
        "field_plan": {
            "field_groups": compact_field_groups,
            "reference_field_audit": field_plan_result.get("reference_field_audit") or {},
        },
    }
    object_contract_registry = {}
    figure_constraint_registry = {}
    object_contract_ids = {}
    figure_constraint_ids = {}

    def intern_contract(value, prefix, identifiers, registry):
        canonical = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if canonical not in identifiers:
            reference = f"{prefix}_{len(identifiers) + 1:03d}"
            identifiers[canonical] = reference
            registry[reference] = value
        return identifiers[canonical]

    compact_registry = []
    for field in schema_result.get("field_registry") or []:
        if not isinstance(field, dict):
            continue
        compact_field = {
            key: field.get(key)
            for key in (
                "field_path",
                "section_id",
                "data_type",
                "required",
                "source_basis",
                "concept_ids",
                "description",
                "extraction_notes",
                "relation_constraints",
                "evidence_requirements",
            )
            if field.get(key) not in (None, "", [], {})
        }
        for semantic_key, limit in (("description", 1200), ("extraction_notes", 2400)):
            if len(str(compact_field.get(semantic_key) or "")) > limit:
                compact_field[semantic_key] = "Oversized field definition: requires a concise field-specific rewrite before acceptance."
        object_contract = field.get("object_contract")
        if object_contract not in (None, "", [], {}):
            compact_field["object_contract_ref"] = intern_contract(
                object_contract,
                "object_contract",
                object_contract_ids,
                object_contract_registry,
            )
        figure_constraint = field.get("figure_constraint")
        if figure_constraint not in (None, "", [], {}):
            compact_field["figure_constraint_ref"] = intern_contract(
                figure_constraint,
                "figure_constraint",
                figure_constraint_ids,
                figure_constraint_registry,
            )
        compact_registry.append(compact_field)
    critic_schema = {
        "top_level_keys": schema_result.get("top_level_keys") or [],
        "field_count": len(compact_registry),
        "field_path_whitelist": [
            field["field_path"]
            for field in compact_registry
            if field.get("field_path")
        ],
        "field_registry": compact_registry,
        "object_contract_registry": object_contract_registry,
        "figure_constraint_registry": figure_constraint_registry,
    }
    retry_feedback = [
        str(item).strip()
        for item in repair_feedback or []
        if str(item).strip()
    ]
    return textwrap.dedent(
        f"""
        You are the specialization critic module inside Step 8: Section Design Agent.
        Your job is to judge whether the current design is still too generic or structurally weak.

        {TRANSFERABLE_FIELD_DESIGN_RULES}

        Compact task context:
        {json.dumps(critic_context, ensure_ascii=False, separators=(",", ":"))}

        Compact upstream contracts:
        {json.dumps(critic_upstream, ensure_ascii=False, separators=(",", ":"))}

        Compact schema projection:
        {json.dumps(critic_schema, ensure_ascii=False, separators=(",", ":"))}

        Supervisor validation feedback from the immediately preceding critic attempt:
        {json.dumps(retry_feedback, ensure_ascii=False, separators=(",", ":"))}

        Requirements:
        - Judge substance, not formatting.
        - Do not reject a design merely because it uses the shared six-section materials backbone; that backbone is expected.
        - Reject designs that redefine the canonical section meanings without justification, omit the backbone when material records are present, or put paper metadata into material_info.section0-section4.
        - `section5.*` is the canonical material-record-owned theory/mechanism namespace. It is not an unowned document-level section. Never move it to `material_info.section5.*`; that spelling is a legacy alias which canonicalizes back to `section5.*`.
        - State whether the fields inside the shared backbone are still too generic for the target domain.
        - Flag missing must-have concepts, collapsed objects, weak evidence modeling, or vague field ownership.
        - If a reference field catalog is available, reject an incomplete audit, broad unsupported exclusions, missing included leaves, or a schema that claims coverage through umbrella objects instead of explicit queryable leaf rows.
        - Judge reference-field decisions against the actual database goal and supplied papers. Do not demand irrelevant fields from another materials task, and do not accept corpus sparsity alone as a reason to shrink a broad-domain database schema.
        - Reject designs that assign inferred labels from weak proxy evidence without confidence, basis, or risk fields.
        - Reject designs that mention mechanism or model parameters but do not distinguish experimental, fitted, simulated, calculated, literature, or author-interpretation origin.
        - Reject designs that collapse theoretical parameters, simulation parameters, and experimental observables into one generic mechanism summary.
        - Audit the utility of every field in the compact schema projection. A field is useful only when removing it would lose an independently queryable value, entity/condition binding, evidence/provenance link, or extraction distinction required by the task or recurring literature.
        - Resolve every object_contract_ref and figure_constraint_ref through the corresponding registry. References deduplicate repeated contracts only; they do not remove fields from the audit.
        - Treat spelling aliases, symbol aliases, unit variants, formatting variants, and deterministically derivable values as normalization metadata or aliases, not separate database fields. A separately reported literature claim may remain distinct only when its provenance or semantics differ.
        - Detect semantic and near-semantic redundancy, not only identical names. Fields sharing the same owner, concept, conditions, evidence rule, and extraction behavior should be merged even when their names differ.
        - Treat mechanical Cartesian expansion of the same generic object slots across unrelated quantities as structural bloat unless each child has a quantity-specific query or extraction distinction.
        - Do not replace independently queryable task-specific scientific quantities with an umbrella `property_observations` array or a generic `property_type` plus `value` record. Keep each task- or literature-required quantity as its own field_registry row and place reusable value/unit/uncertainty/conditions/entity/evidence mechanics in that row's object_contract.
        - Do not infer a numeric ideal from field_count. A compact design may still be incomplete and a large design may still be justified; the decision must come from the per-field utility audit.
        - field_utility_audit.reviewed_field_count must equal the exact field_count shown above. Passing without reviewing all current fields is invalid.
        - Treat field_path_whitelist as authoritative. Every path in a field_utility_audit issue list and every remove_field, update_field, or move_field field_path must be copied exactly from that whitelist. Parent objects, aliases, proposed names, and semantically similar paths are invalid there.
        - A path not yet present in field_path_whitelist may appear only as an add_field field_path or in prose describing a missing concept. Never list a proposed add_field path as an existing-field utility issue.
        - When any utility issue list is non-empty, set field_utility_audit.decision to needs_pruning, set redo_needed=true, and provide an executable remove_field, update_field, or move_field operation for every listed path.
        - If supervisor validation feedback is non-empty, this is a correction attempt. Correct every listed protocol error in the new JSON response; do not repeat or defend the invalid output.
        - If feedback contains `schema_patch_contract_regression`, preserve every listed required concept on at least one top-level field_registry row in the replacement patch. A concept_id nested only inside object_contract.fields does not satisfy field-registry concept coverage.
        - The requirement_contract supplied in this run is the system's own task contract, not a human gold schema. A patch may simplify how a required concept is represented, but it must not leave that concept without any field mapping.
        - concept_ids in every add_field or update_field patch are a closed vocabulary: copy only exact concept_id values present in requirement_contract.concepts. Never invent a narrower concept ID. A field already justified by an exact current field-plan path may keep concept_ids empty; if a genuinely missing distinction is absent from both contracts, report it as an upstream contract gap instead of fabricating an ID.
        - When removing the last field mapped to a required concept_id, add or update its semantically appropriate replacement in the same patch and copy that concept_id to the replacement. Atomic fields may carry a broad task concept_id; preserving the concept mapping does not require retaining a vague umbrella field.
        - If the design is weak, provide targeted redo directives and executable patch_operations.
        - patch_operations are the only machine-executed repair channel. Do not rely on prose for field creation.
        - Use add_field for a missing field, update_field for an existing field, remove_field for one exact redundant field, and move_field for one exact ownership error.
        - Before using add_field, verify that the exact field_path is absent from field_path_whitelist. If it is already present, use update_field, remove_field followed by add_field, or withdraw the operation; add_field on an existing path is invalid.
        - For an object-contract repair, changes may replace the complete object_contract or use only these exact nested keys: object_contract.object_kind, object_contract.required_subfields, object_contract.fields. Do not invent other dotted change keys.
        - In every complete object_contract, `required_subfields` must be a JSON array of strings and `fields` must be a JSON object mapping each subfield name to its descriptor. Never emit `fields` as an array or string, and never emit `required_subfields` as a space-delimited string.
        - object_contract_ref and figure_constraint_ref are read-only compression handles from the input projection. They are forbidden in patch_operations. Resolve a referenced contract and write the complete object_contract when changing it.
        - Never place bare fields or required_subfields inside update_field.changes. Use object_contract.fields or object_contract.required_subfields, respectively.
        - update_field.changes may contain only section_id, field_name, description, extraction_notes, data_type, required, source_basis, concept_ids, figure_constraint, object_contract, reason, inclusion_rule, absence_rule, evidence_requirements, relation_constraints, core_field, or the three exact object_contract.* nested keys listed above.
        - Every add_field.field must include field_path, section_id, field_name, data_type, required, source_basis, concept_ids, figure_constraint, extraction_notes, and reason. Object fields must also include object_contract.
        - Use canonical materials-literature roots and preserve task-specific entity owners approved in entity_registry.
        - Never solve coverage by emitting a large speculative field inventory. Each operation needs a task or evidence reason.
        - Do not use field count as an objective. Add only fields whose absence causes a concrete coverage, query, binding, provenance, or extraction-contract defect; remove exact semantic duplicates.
        - If the design passes, patch_operations must be an empty list.

        Return valid JSON only:
        {{
          "specialization_status": "pass",
          "is_generic": false,
          "missing_concepts": ["string"],
          "structural_weaknesses": ["string"],
          "redo_needed": false,
          "redo_directives": ["string"],
          "field_utility_audit": {{
            "reviewed_field_count": 0,
            "decision": "pass | needs_pruning",
            "unsupported_fields": ["exact.field.path"],
            "redundant_fields": ["exact.field.path"],
            "alias_or_unit_variant_fields": ["exact.field.path"],
            "derivable_duplicate_fields": ["exact.field.path"],
            "rationale": "field utility and stopping decision without a numeric target"
          }},
          "patch_operations": [
            {{
              "op": "add_field | update_field | remove_field | move_field",
              "field_path": "canonical dotted field path",
              "target_path": "canonical dotted field path for move_field only",
              "field": {{}},
              "changes": {{}},
              "reason": "task/evidence justification"
            }}
          ]
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
        {_render_shared_context(shared_context, include_reference_fields=True)}

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
        - Copy reference_field_contract from shared context and reference_field_audit from the field planning result without dropping leaf paths, mappings, exclusions, or unresolved items.
        - quality_check.topic_specific_adjustments should combine the topic adaptation result, subjective supervisor result, and specialization critic result.
        - quality_check.coverage_check should summarize how the schema supports the query requirements.
        - quality_check should mention whether figure classification was enabled and what conflict it prevents.
        - quality_check should mention whether the specialization critic found generic-template risk.
        - Set redo_needed to true if the specialization critic says the design is still too generic or structurally weak.

        {FINAL_OUTPUT_SCHEMA_DESCRIPTION}
        """
    ).strip()


def build_module_redo_prompt(
    module_name,
    previous_result,
    validation_errors,
    schema_text,
    original_prompt=None,
):
    error_text = "\n".join(f"- {item}" for item in validation_errors)
    return textwrap.dedent(
        f"""
        Redo the `{module_name}` output.

        Original task and module context (retain all constraints from this block):
        {original_prompt or "No original prompt was supplied."}

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
