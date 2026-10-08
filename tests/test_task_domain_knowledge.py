from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import task_domain_knowledge as knowledge


TASK = {"database_goal": "Compare battery materials", "discipline": "Materials science",
        "query_requirements": ["Compare capacity with measurement conditions"]}
CONCEPT = {"concept_id": "capacity", "name": "Specific capacity", "why_needed": "Avoid confusing capacity and energy",
           "requirement_ids": [1], "search_queries": ["specific capacity definition measurement criteria"]}
TEXT = "Specific capacity expresses stored charge per unit mass under stated test conditions."
SOURCE = {"source_id": "doi:10.test/example", "text": TEXT, "sha256": knowledge.sha(TEXT.encode()),
          "usable_for_claims": True, "url": "https://doi.org/10.test/example", "title": "A method paper",
          "source_type": "publisher_deposited_abstract"}


def card():
    result = {"concept_id": "capacity", "status": "partial", "open_questions": ["Criterion requires further research"],
              **{d: [] for d in knowledge.DIMENSIONS}}
    result["definition"] = [{"text": "Charge per mass, distinct from energy per mass.",
                             "evidence": [{"source_id": SOURCE["source_id"], "quote": TEXT}]}]
    return result


class Model:
    def __init__(self, answer=None):
        self.calls = []
        self.answer = answer or card()

    def chat_json(self, messages, **kwargs):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return {"concepts": [deepcopy(CONCEPT)]}, {}
        return deepcopy(self.answer), {}


def test_external_research_and_design_handoff(tmp_path):
    model = Model()
    root = tmp_path / "research"
    pack = knowledge.research(TASK, model, searcher=lambda q: [SOURCE], output=root)
    assert pack["status"] == "research_incomplete"
    assert len(model.calls) == 2
    context = deepcopy(TASK)
    knowledge.attach_to_context(context, root / "knowledge_pack.json")
    assert context["task_domain_knowledge"]["cards"][0]["definition"]
    assert "text" not in context["task_domain_knowledge"]["source_catalog"][0]
    assert context["task_domain_knowledge"]["source_sha256"]
    import section_design_agent_prompt as prompts
    rendered = prompts._render_shared_context(context)
    assert "reference knowledge only" in rendered
    assert "Specific capacity" in rendered


def test_no_abstract_no_model_fabrication(tmp_path):
    model = Model()
    source = {**SOURCE, "text": "", "usable_for_claims": False}
    pack = knowledge.research(TASK, model, searcher=lambda q: [source], output=tmp_path / "run")
    assert len(model.calls) == 1
    assert pack["cards"][0]["status"] == "unresolved"


def test_failure_preserved_and_namespace_not_reused(tmp_path):
    def unavailable(query):
        raise knowledge.requests.ConnectionError("secret must not be retained")
    root = tmp_path / "run"
    pack = knowledge.research(TASK, Model(), searcher=unavailable, output=root)
    assert pack["failures"][0]["error"] == "external_search_failed"
    assert "secret" not in json.dumps(pack)
    with pytest.raises(FileExistsError):
        knowledge.research(TASK, Model(), searcher=unavailable, output=root)


@pytest.mark.parametrize("mutation", [
    lambda c: c["definition"][0]["evidence"][0].update(source_id="fake"),
    lambda c: c["definition"][0]["evidence"][0].update(quote="This is not in the source."),
    lambda c: c["definition"][0].update(evidence=[]),
    lambda c: c.update(status="supported"),
    lambda c: c.update(status="unresolved"),
    lambda c: c.update(concept_id="wrong"),
])
def test_invalid_citations_and_false_completeness(mutation):
    item = card()
    mutation(item)
    with pytest.raises(ValueError):
        knowledge.validate_card(item, CONCEPT, [SOURCE])


def test_fabricated_synthesis_becomes_unresolved():
    answer = card()
    answer["definition"][0]["evidence"][0]["quote"] = "Invented definition"
    pack = knowledge.research(TASK, Model(answer), searcher=lambda q: [SOURCE])
    assert pack["cards"][0]["status"] == "unresolved"
    assert pack["failures"][0]["error"] == "knowledge_synthesis_failed"


def test_plan_coverage_and_budget():
    plan = {"concepts": [deepcopy(CONCEPT)]}
    knowledge.validate_plan(plan, TASK["query_requirements"])
    with pytest.raises(ValueError, match="no research concepts"):
        knowledge.validate_plan(plan, ["first", "second"])
    with pytest.raises(ValueError, match="budget"):
        knowledge.validate_plan(plan, ["first"], max_concepts=0)


def test_task_and_source_integrity(tmp_path):
    root = tmp_path / "run"
    knowledge.research(TASK, Model(), searcher=lambda q: [SOURCE], output=root)
    path = root / "knowledge_pack.json"
    with pytest.raises(ValueError, match="another task"):
        knowledge.load_for_design(path, {**TASK, "database_goal": "Compare alloys"})
    pack = json.loads(path.read_text())
    pack["sources"][0]["text"] = "Changed source"
    path.write_text(json.dumps(pack))
    with pytest.raises(ValueError, match="source hash"):
        knowledge.load_for_design(path, TASK)


def test_crossref_provider_uses_abstract_not_title():
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"message": {"items": [{"DOI": "10.test/1", "title": ["Definition"], "abstract": "<jats:p>Stored <b>charge</b> per mass.</jats:p>"}, {"DOI": "10.test/2", "title": ["Definition without source text"]}]}}
    class Session:
        def get(self, url, **kwargs):
            assert url == "https://api.crossref.org/works"
            assert kwargs["timeout"] == (10, 30)
            return Response()
    sources = knowledge.crossref_search("capacity definition", session=Session())
    assert sources[0]["text"] == "Stored charge per mass."
    assert not sources[1]["usable_for_claims"]


def test_disabled_handoff_keeps_legacy_context():
    context = deepcopy(TASK)
    assert knowledge.attach_to_context(context, "") == TASK


def test_web_search_reads_sources_not_snippets():
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"web": {"results": [{"url": "https://standards.example/term", "title": "Official terminology", "description": "Do not trust this snippet"}, {"url": "https://random.example/term"}]}}
    class Session:
        def get(self, url, **kwargs):
            assert url == "https://api.search.brave.com/res/v1/web/search"
            assert "X-Subscription-Token" in kwargs["headers"]
            return Response()
    calls = []
    def fetch(url, hosts):
        calls.append(url)
        return deepcopy(SOURCE)
    result = knowledge.brave_search("capacity definition", ["standards.example"], "test-token", session=Session(), fetcher=fetch)
    assert calls == ["https://standards.example/term"]
    assert result[0]["text"] == TEXT
    assert "snippet" not in json.dumps(result)


@pytest.mark.parametrize("url", ["http://standard.example/", "https://localhost/", "https://user:pass@standard.example/", "https://standard.example:22/"])
def test_reject_unapproved_reference_urls(url):
    with pytest.raises(ValueError):
        knowledge.fetch_reference(url, ["standard.example"])


def test_reject_private_resolution(monkeypatch):
    monkeypatch.setattr(knowledge.socket, "getaddrinfo", lambda *a, **k: [(None, None, None, None, ("127.0.0.1", 443))])
    with pytest.raises(ValueError, match="non-public"):
        knowledge.fetch_reference("https://standard.example/", ["standard.example"])


def test_native_parser_exposes_optional_pack():
    import section_design_agent as agent
    parser = agent.build_parser()
    assert any(a.dest == "domain_knowledge_pack" and a.default == "" for a in parser._actions)
