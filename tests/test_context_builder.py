"""Tests for app.context.builder and app.context.models."""
import pytest

from app.context.builder import ContextBuilder, build_context
from app.context.models import ContextPayload, SourceItem
from app.storage.vector_store import SearchResult


def _make_result(
    doc_id: str,
    chunk_index: int,
    text: str,
    page_number: int | None = None,
    distance: float = 0.1,
    score: float = 0.9,
) -> SearchResult:
    return SearchResult(
        text=text,
        document_id=doc_id,
        chunk_index=chunk_index,
        distance=distance,
        score=score,
        page_number=page_number,
    )


def test_build_context_basic_pdf():
    results = [
        _make_result(
            doc_id="report.pdf",
            chunk_index=0,
            text="The operating profit margin rose by 4.2 percent.",
            page_number=3,
        )
    ]
    payload = build_context(results)

    assert isinstance(payload, ContextPayload)
    assert not payload.is_empty
    assert not payload.truncated
    assert len(payload.sources) == 1

    src = payload.sources[0]
    assert isinstance(src, SourceItem)
    assert src.source_id == 1
    assert src.document_id == "report.pdf"
    assert src.page_number == 3
    assert src.citation_tag == "[Source 1]"
    assert src.citation_label == "report.pdf (Page 3)"

    assert "[Source 1]" in payload.formatted_context
    assert "Document: report.pdf" in payload.formatted_context
    assert "Page: 3" in payload.formatted_context
    assert "Content:\nThe operating profit margin rose by 4.2 percent." in payload.formatted_context

    assert payload.total_characters == len(payload.formatted_context)
    assert payload.sources_by_id[1] == src


def test_build_context_txt_omits_page():
    results = [
        _make_result(
            doc_id="server_logs.txt",
            chunk_index=1,
            text="Cluster node synchronized at 12:00 UTC.",
            page_number=None,
        )
    ]
    payload = build_context(results)

    assert not payload.is_empty
    assert "Page:" not in payload.formatted_context
    assert payload.sources[0].page_number is None
    assert payload.sources[0].citation_label == "server_logs.txt"


def test_build_context_preserves_retrieval_ranking():
    results = [
        _make_result("first.pdf", 0, "Highest ranked text.", score=0.95),
        _make_result("second.pdf", 1, "Middle ranked text.", score=0.85),
        _make_result("third.pdf", 2, "Lowest ranked text.", score=0.75),
    ]
    payload = build_context(results)

    assert len(payload.sources) == 3
    assert payload.sources[0].document_id == "first.pdf"
    assert payload.sources[0].source_id == 1
    assert payload.sources[1].document_id == "second.pdf"
    assert payload.sources[1].source_id == 2
    assert payload.sources[2].document_id == "third.pdf"
    assert payload.sources[2].source_id == 3

    assert payload.formatted_context.index("[Source 1]") < payload.formatted_context.index("[Source 2]")
    assert payload.formatted_context.index("[Source 2]") < payload.formatted_context.index("[Source 3]")
    assert "\n\n---\n\n" in payload.formatted_context


def test_build_context_deduplication():
    results = [
        _make_result("doc1", 0, "Initial version of chunk zero.", score=0.9),
        _make_result("doc1", 0, "Duplicate occurrence of chunk zero.", score=0.88),
        _make_result("doc1", 1, "Chunk one content.", score=0.7),
    ]
    payload = build_context(results)

    assert len(payload.sources) == 2
    assert payload.sources[0].document_id == "doc1"
    assert payload.sources[0].chunk_index == 0
    assert payload.sources[0].text == "Initial version of chunk zero."
    assert payload.sources[1].chunk_index == 1


def test_build_context_max_sources_clamp():
    results = [_make_result(f"doc_{i}", 0, f"Text content for document {i}.") for i in range(6)]
    payload = build_context(results, max_sources=3)

    assert len(payload.sources) == 3
    assert payload.truncated is True
    assert [s.source_id for s in payload.sources] == [1, 2, 3]
    assert set(payload.sources_by_id.keys()) == {1, 2, 3}


def test_build_context_max_context_chars_budget_exact():
    # Construct results with known lengths
    r1 = _make_result("docA", 0, "Short text alpha.", page_number=1)
    r2 = _make_result("docB", 0, "Short text beta.", page_number=2)

    # First build to measure exact block size
    single = build_context([r1])
    len_r1 = len(single.formatted_context)

    # Allow room for r1 but not enough for r1 + separator + r2
    budget = len_r1 + 10
    payload = build_context([r1, r2], max_context_chars=budget)

    assert len(payload.sources) == 1
    assert payload.truncated is True
    assert payload.sources[0].document_id == "docA"
    assert payload.total_characters == len(payload.formatted_context)
    assert payload.total_characters <= budget


def test_build_context_empty_results():
    payload = build_context([])

    assert payload.is_empty is True
    assert payload.formatted_context == ""
    assert payload.sources == []
    assert payload.sources_by_id == {}
    assert payload.total_characters == 0
    assert payload.truncated is False


def test_build_context_skips_blank_text():
    results = [
        _make_result("doc", 0, "   "),
        _make_result("doc", 1, "\n\t"),
        _make_result("doc", 2, ""),
    ]
    payload = build_context(results)

    assert payload.is_empty is True
    assert payload.sources == []
    assert payload.formatted_context == ""


def test_build_context_citation_integrity():
    results = [
        _make_result("alpha.pdf", 0, "Alpha text.", page_number=1),
        _make_result("beta.txt", 0, "Beta text.", page_number=None),
    ]
    payload = build_context(results)

    assert set(payload.sources_by_id.keys()) == {1, 2}
    for sid, src in payload.sources_by_id.items():
        assert src.source_id == sid
        assert src.citation_tag == f"[Source {sid}]"

    # Verify lookup of nonexistent source fails cleanly
    assert 99 not in payload.sources_by_id


def test_build_context_fresh_and_deterministic_ids_across_calls():
    results = [
        _make_result("doc1.pdf", 0, "First text.", page_number=1),
        _make_result("doc2.pdf", 0, "Second text.", page_number=2),
    ]
    builder = ContextBuilder()

    first = builder.build(results)
    assert [s.source_id for s in first.sources] == [1, 2]

    # Repeated call on the exact same builder instance must start at 1
    second = builder.build(results)
    assert [s.source_id for s in second.sources] == [1, 2]
    assert first.formatted_context == second.formatted_context
    assert first.sources == second.sources


def test_build_context_explicit_falsy_parameters():
    results = [_make_result("doc.pdf", 0, "Some valid content.", page_number=1)]

    # Explicit max_sources=0 should NOT fall back to config.MAX_SOURCES
    payload_zero_sources = build_context(results, max_sources=0)
    assert payload_zero_sources.is_empty is True
    assert len(payload_zero_sources.sources) == 0
    assert payload_zero_sources.truncated is True

    # Explicit max_context_chars=0 should NOT fall back to config.MAX_CONTEXT_CHARS
    payload_zero_chars = build_context(results, max_context_chars=0)
    assert payload_zero_chars.is_empty is True
    assert len(payload_zero_chars.sources) == 0
    assert payload_zero_chars.truncated is True


def test_build_context_negative_parameters_raise():
    with pytest.raises(ValueError, match="max_context_chars"):
        ContextBuilder(max_context_chars=-1)

    with pytest.raises(ValueError, match="max_sources"):
        ContextBuilder(max_sources=-1)
