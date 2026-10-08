import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import literature_search as search


@pytest.fixture
def indexed(tmp_path):
    corpus = tmp_path / "papers"
    corpus.mkdir()
    (corpus / "battery.md").write_text("# Battery\n## Methods\nCapacity measured at 2 C.\n## Results\nSpecific capacity is 140 mAh/g.\n", encoding="utf-8")
    (corpus / "alloy.md").write_text("# Alloy\n## Results\nYield strength is 450 MPa.\nDo not confuse yield strength with tensile strength.\n", encoding="utf-8")
    index = tmp_path / "index.sqlite"
    search.build_index(corpus, index)
    return corpus, index


def test_phrase_and_source_identity(indexed):
    corpus, index = indexed
    hit = search.search(index, ["YIELD STRENGTH"])["results"][0]
    assert hit["document_id"] == "alloy.md"
    assert hit["section"] == "Alloy / Results"
    assert hit["source_sha256"] == search.digest((corpus / "alloy.md").read_bytes())
    lines = (corpus / "alloy.md").read_bytes().decode("utf-8").splitlines(keepends=True)
    assert hit["text"] == "".join(lines[hit["start_line"] - 1:hit["end_line"]])
    assert hit["chunk_sha256"] == search.digest(hit["text"].encode())


def test_filters_and_modes(indexed):
    _, index = indexed
    assert not search.search(index, ["capacity"], document_id="alloy.md")["results"]
    assert not search.search(index, ["capacity"], section="Introduction")["results"]
    assert len(search.search(index, ["capacity"], section="Methods")["results"]) == 1
    assert not search.search(index, ["capacity", "strength"])["results"]
    assert len(search.search(index, ["capacity", "strength"], match="any")["results"]) == 3


def test_literal_symbols_and_untrusted_query(indexed):
    _, index = indexed
    assert search.search(index, ["mAh/g"])["results"]
    assert not search.search(index, ["\" OR 1=1 --"])["results"]


@pytest.mark.parametrize("terms", [[], [""], [None], "capacity", ["x" * 201], ["x"] * 13])
def test_invalid_terms(indexed, terms):
    with pytest.raises(ValueError):
        search.search(indexed[1], terms)


def test_changed_added_and_deleted_corpus(indexed):
    corpus, index = indexed
    (corpus / "new.md").write_text("New paper", encoding="utf-8")
    with pytest.raises(ValueError, match="Corpus changed"):
        search.search(index, ["capacity"])
    (corpus / "new.md").unlink()
    (corpus / "battery.md").write_text("Changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Corpus changed"):
        search.search(index, ["capacity"])
    (corpus / "battery.md").unlink()
    with pytest.raises(ValueError, match="Corpus changed"):
        search.search(index, ["strength"])


def test_index_preservation(indexed):
    corpus, index = indexed
    before = index.read_bytes()
    with pytest.raises(FileExistsError):
        search.build_index(corpus, index)
    assert index.read_bytes() == before


def test_unicode_and_overlap(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    # Domain-neutral multilingual phrases and punctuation remain literal.
    text = "# Results\n" + "ordinary line\n" * 44 + "fracture toughness\n" + "ordinary line\n" * 6 + "\u5f3a\u5ea6 \u03bcm 10^-3\n"
    (root / "paper.md").write_text(text, encoding="utf-8")
    index = tmp_path / "index.sqlite"
    search.build_index(root, index)
    assert len(search.search(index, ["fracture toughness"])["results"]) == 1
    assert search.search(index, ["\u5f3a\u5ea6", "\u03bcm", "10^-3"])["results"]


def test_empty_corpus(tmp_path):
    with pytest.raises(ValueError, match="no UTF-8"):
        search.build_index(tmp_path, tmp_path / "index.sqlite")
