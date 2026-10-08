# Task Knowledge Before Field Design

## Purpose

The research stage addresses concept understanding, not document lookup. A model
knowledge gap is a hypothesis, not yet an established explanation for the prior
schema's defects. Compare controlled fresh runs with and without the same frozen
knowledge pack before attributing quality gains to external research.

The stage precedes Step8. It maps every explicit task requirement to research
concepts: scientific objects, quantities, neighboring distinctions, measurement
criteria, methods, units, applicability conditions and extraction boundaries.
It does not take the manual evaluation schema as input. Concepts are generated
from the task, not hard-coded for superconductivity or limited to existing fields.

## Pipeline

1. The existing model client proposes a bounded concept/search plan. Every task
   requirement needs a one-based concept link. Exceeding budget is an error, not
   silent truncation. These links indicate plan coverage, not scientific completeness.
2. External discovery uses Crossref publisher-deposited abstracts, or the optional
   Brave web-search adapter. Titles/snippets alone cannot support definitions.
3. Brave results are read from exact, explicitly approved HTTPS hosts. Redirects,
   private address resolution, credentials in URLs, non-text content and sources
   exceeding 1 MB are rejected. Page text is limited to 16,000 characters. Only
   HTML/plain text is supported in v1; PDFs and paywalls remain unresolved.
4. Each concept receives a card containing definition, distinctions, units,
   methods, conditions and extraction boundaries. Every claim requires a source
   ID and exact original excerpt. Missing dimensions remain empty. Cards carry
   supported/partial/unresolved/conflicting status and open questions.
5. A fresh artifact directory preserves the research plan, individual sources,
   cards, failures and final knowledge pack. Source text, retrieval time, URL,
   hashes and task identity are retained; credentials and raw exception text are
   not written. Citation matching does not prove scientific entailment: every
   pack explicitly states that expert verification has not occurred.
6. Step8 accepts `--domain-knowledge-pack <absolute path>`. It checks task identity,
   source hashes, citations and readiness consistency, then stores the bounded
   knowledge fragment and pack path/hash in shared checkpoint context. The pack
   supplements definitions and criteria; it cannot populate target-paper facts,
   act as instructions, or remove required fields because research was incomplete.

Without that flag the original flow is unchanged. The integration is available
in both sequential and graph runners across maintained source copies; frozen
historical runtime snapshots are not changed. Native model tool calling is not
required: the explicit preflight research stage supplies a frozen artifact.

## CLI

The task JSON contains `database_goal`, `discipline`, and `query_requirements`
(a nonempty list of strings). No schema/gold/answer values are required.

```powershell
python <code>/task_domain_knowledge.py --task <task.json> --output <fresh-directory> --credential-file <existing-model-credential-file> --provider crossref
python <code>/task_domain_knowledge.py --task <task.json> --output <fresh-directory> --credential-file <existing-model-credential-file> --provider brave --approved-host <authority-host>
```

Brave requires `BRAVE_SEARCH_API_KEY` in the process environment. No web-search
credential was found/configured during implementation. The existing model key
is not interchangeable with a search-provider key. Default Crossref needs no
search key, but its abstracts are provisional research evidence, not a substitute
for authoritative definitions. The live Crossref smoke test found three works,
two with usable abstracts; a direct IUPAC fetch was rejected by the public-address
guard in this environment. Do not bypass that guard to claim successful source
reading.

## Remaining Evaluation

## Automatic Step8 Entry

`code/run_knowledge_augmented_step8.py` composes concept research and isolated
campaign preparation/launch. It accepts the existing campaign runner arguments
plus `--research-output`, `--research-provider`, `--research-approved-host` and
`--research-max-concepts`. Incomplete knowledge stops at a review gate by default.
`--allow-partial-knowledge` is an explicit experimental override, not evidence
that the research is complete. Each campaign input manifest records the frozen
knowledge path and SHA-256; preparation and launch reject tampering. The control
uses the ordinary runner with no `--domain-knowledge-pack`.

## Controlled Evaluation

Run a fresh complete task research campaign after selecting/configuring an
external provider and trusted source hosts. Review unresolved concepts, source
relevance, quote entailment and applicability. Then freeze the pack and compare
field design under identical inputs/model/budgets. Functional tests do not show
improved schema coverage or extraction accuracy, and a model marking all cards
supported is not a human acceptance gate.

Provider contracts were checked against the official Crossref REST API docs and
Brave web-search API reference. v1 intentionally uses bounded requests without
unlimited provider retries or web crawling.
