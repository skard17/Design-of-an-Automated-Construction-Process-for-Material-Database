# Literature Search v1

## Scope

Search only the user-supplied UTF-8 Markdown corpus. No OCR, network, credentials,
embeddings, or manual domain gold is needed. The implementation is shared and
domain-neutral; scientific symbols, units, and multilingual terms remain literal.
It complements full-paper input, rather than replacing that input or deciding
whether a field is necessary from search frequency.

## API

`literature_search.build_index(corpus, index)` creates a fresh SQLite artifact.
`literature_search.search(index, terms, match="all", document_id="", section="", limit=5)`
returns JSON-compatible evidence. Each term is a literal keyword or phrase.
`TOOL_SCHEMA` describes the callable interface for a model tool adapter.
The CLI exposes `build`, `search`, and `tool-schema` subcommands.

Ranking is deterministic lexical coverage plus capped occurrence counts, not
semantic similarity or a relevance probability. Sections are Markdown heading
paths; raw figure/table captions remain searchable text. Windows of 48 lines
overlap by 8 lines inside sections. Overlapping hits are deduplicated. Retrieval
does not cross heading boundaries; cross-section questions should use separate
queries. Substring matching can overmatch short symbols: supply longer phrases
and document/section filters. No automatic domain aliases are injected.

## Evidence Contract

Every hit contains document-relative identity, absolute source path, source
SHA-256, exact excerpt, inclusive line range, heading path, chunk SHA-256 and
chunk ID. Search verifies the complete corpus inventory before returning hits;
changed, deleted, or added papers require a fresh index. Existing indexes are
never overwritten. SQLite access is read-only during search and document filters
use bound parameters. Corpus paths cannot escape the root through symlinks.

## Planner Integration

The current definition-aware Step8 run remains frozen and unchanged. The tool
is callable and tested, but native model tool-call dispatch is not yet wired into
the planner. A future adapter should expose it to field planning and schema
critique with a finite query/result budget, log queries and cited chunk IDs,
and carry corpus hash into the run manifest. A tool response is untrusted source
material, never a system instruction. Define fields from task requirements and
scientific meaning, using retrieved passages to clarify definitions and evidence
binding; no hit does not justify deleting a field.

External retrieval, if added later, must be a separate provider with explicit
URL/date/hash provenance and its own authorization boundary. It may clarify
terminology, but must not populate missing values for a target paper or silently
join external sources into the frozen experimental corpus. Start by evaluating
local lexical recall, then justify synonym or semantic retrieval with held-out
cross-domain queries before expanding dependencies.

## Limits

v1 scans stored chunks after metadata filters and hashes the corpus per request.
This is appropriate for the 30-paper pilot, not a large-scale retrieval engine.
Lexical recall, caption parsing, long-line size limits, and native planner dispatch
are future work; passing functional tests is not evidence of improved field or
extraction accuracy.
