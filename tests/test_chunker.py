"""Tests for app.chunking.chunker."""

import re

import pytest

from app.chunking.chunker import Chunk, chunk_pages, chunk_text
from app.ingestion.extractor import ExtractedPage

_SENTENCE_TOKENS = "AAA. BBB. CCC. DDD. EEE. FFF. GGG. HHH."


def _normalized(text):
    return re.sub(r"\s+", " ", text).strip()


def test_empty_input():
    assert chunk_text("") == []


def test_whitespace_only_input():
    assert chunk_text("   \n\t  ") == []


def test_none_input():
    assert chunk_text(None) == []


def test_short_document_is_single_chunk():
    text = "A short paragraph that easily fits inside one chunk."
    chunks = chunk_text(text, document_id="doc-1")
    assert isinstance(chunks, list)
    assert chunks == [Chunk("doc-1", 0, text)]
    assert len(chunks[0].text) <= 500


def test_single_word_shorter_than_chunk_size_is_one_chunk():
    text = "supercalifragilisticexpialidocious"
    chunks = chunk_text(text, chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_no_overlap_reconstructs_text():
    chunks = chunk_text(_SENTENCE_TOKENS, chunk_size=12, overlap=0)
    assert [c.text for c in chunks] == [
        "AAA. BBB.",
        "CCC. DDD.",
        "EEE. FFF.",
        "GGG. HHH.",
    ]
    assert " ".join(c.text for c in chunks) == _SENTENCE_TOKENS


def test_overlap_keeps_tail_in_next_chunk():
    text = " ".join(f"tok{i:02d}" for i in range(30))
    chunks = chunk_text(text, chunk_size=33, overlap=6)
    assert len(chunks) > 1
    for left, right in zip(chunks, chunks[1:]):
        last_token = left.text.rsplit(" ", 1)[-1]
        assert right.text.startswith(last_token)


def test_no_overlap_chunks_do_not_repeat_tail():
    text = " ".join(f"tok{i:02d}" for i in range(30))
    chunks = chunk_text(text, chunk_size=33, overlap=0)
    for left, right in zip(chunks, chunks[1:]):
        last_token = left.text.rsplit(" ", 1)[-1]
        assert not right.text.startswith(last_token)


def test_chunk_size_is_respected():
    sentences = [f"Word number {i} repeats here." for i in range(30)]
    text = " ".join(sentences)
    chunks = chunk_text(text, chunk_size=30, overlap=0)
    assert len(chunks) > 1
    for chunk in chunks:
        assert 0 < len(chunk.text) <= 30


def test_chunk_size_is_configurable():
    text = " ".join([f"Filler sentence number {i}." for i in range(40)])
    small = chunk_text(text, chunk_size=30, overlap=0)
    large = chunk_text(text, chunk_size=200, overlap=0)
    assert len(small) > len(large) > 1


def test_sentence_boundaries_preferred():
    text = " ".join(
        f"This is a nicely punctuated sentence number {i}." for i in range(20)
    )
    chunks = chunk_text(text, chunk_size=120, overlap=0)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.endswith((".", "!", "?"))


def test_invalid_sizes_raise():
    text = "Some valid text."
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text(text, chunk_size=0)
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text(text, chunk_size=-5)
    with pytest.raises(ValueError, match="overlap"):
        chunk_text(text, overlap=-1)
    with pytest.raises(ValueError, match="overlap"):
        chunk_text(text, chunk_size=10, overlap=10)
    with pytest.raises(ValueError, match="overlap"):
        chunk_text(text, chunk_size=10, overlap=15)


def test_deterministic_output():
    text = " ".join(f"Sentence number {i} walks into a bar." for i in range(15))
    first = chunk_text(text, chunk_size=60, overlap=15)
    second = chunk_text(text, chunk_size=60, overlap=15)
    assert first == second


def test_document_id_and_index_metadata():
    text = " ".join("Token {}.".format(i) for i in range(5))
    chunks = chunk_text(
        text,
        chunk_size=20,
        overlap=5,
        document_id="report-2024-001",
    )
    assert len(chunks) > 1
    assert all(c.document_id == "report-2024-001" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_long_word_is_hard_cut():
    chunks = chunk_text("x" * 250, chunk_size=100, overlap=10)
    assert [c.text for c in chunks] == ["x" * 100, "x" * 100, "x" * 70]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_unusual_whitespace_is_safe():
    text = "a\n\n  b \t c    d"
    chunks = chunk_text(text, chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0].text == "a b c d"


def test_whitespace_collapsed_within_chunks():
    chunks = chunk_text("aa   bb\n\ncc dd", chunk_size=4, overlap=0)
    assert [c.text for c in chunks] == ["aa", "bb", "cc", "dd"]


def test_no_empty_chunks():
    text = " ".join(f"Sentence {i} goes on at some length here." for i in range(50))
    chunks = chunk_text(text, chunk_size=40, overlap=8)
    assert chunks
    assert all(c.text.strip() for c in chunks)


def test_default_arguments():
    text = " ".join(f"Sentence {i} fills space nicely." for i in range(60))
    chunks = chunk_text(text)
    assert chunks
    normalized = _normalized(text)
    for chunk in chunks:
        assert len(chunk.text) <= 500
        assert chunk.text in normalized


def test_chunk_text_carries_page_number():
    chunks = chunk_text("Hello world on page 2.", page_number=2)
    assert len(chunks) == 1
    assert chunks[0].page_number == 2


def test_chunk_default_page_number_is_none():
    chunks = chunk_text("Hello unpaginated world.")
    assert len(chunks) == 1
    assert chunks[0].page_number is None


def test_chunk_pages_multi_page_indexes():
    pages = [
        ExtractedPage(1, "Page one text repeats here. Second sentence on page one."),
        ExtractedPage(2, "Page two text starts here. Second sentence on page two."),
    ]
    chunks = chunk_pages(pages, chunk_size=35, overlap=5, document_id="report")
    assert len(chunks) >= 2
    assert all(c.document_id == "report" for c in chunks)
    # Monotonically increasing indices across the document
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Page numbers are preserved for each chunk
    p1_chunks = [c for c in chunks if "page one" in c.text.lower()]
    p2_chunks = [c for c in chunks if "page two" in c.text.lower()]
    assert all(c.page_number == 1 for c in p1_chunks)
    assert all(c.page_number == 2 for c in p2_chunks)


def test_chunk_pages_empty_pages_produce_no_chunks():
    assert chunk_pages([]) == []


def test_overlap_does_not_produce_one_char_sliding_chunks():
    """Regression: _find_break() returning inside the overlap window must not
    cause start to advance by only 1 character, producing dozens of tiny
    overlapping chunks like 'engineering.', 'ngineering.', 'gineering.', ..."""
    # This sentence ends exactly near a chunk boundary, which previously
    # caused _find_break to return a position well inside the overlap window.
    text = (
        "Software engineering. "
        "We build robust and scalable distributed systems. "
        "Data engineering. "
        "We process petabytes of data every single day. "
        "Machine learning engineering. "
        "We train large neural networks at scale."
    )
    chunk_size = 50
    overlap = 20
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    # The total text is ~240 chars; with chunk_size=50 and overlap=20 the
    # expected number of chunks is well under 20.  The bug would produce
    # 50+ chunks (one per character in the word "engineering").
    assert len(chunks) < 20, (
        f"Too many chunks ({len(chunks)}); likely hit the sliding-window bug.\n"
        + "\n".join(repr(c.text) for c in chunks)
    )

    # No two consecutive chunks should be identical or differ by just a single
    # leading character (the signature of the sliding-window bug).
    texts = [c.text for c in chunks]
    for prev, curr in zip(texts, texts[1:]):
        assert prev != curr, f"Duplicate consecutive chunks: {prev!r}"
        # A 1-char slide would mean curr == prev[1:]
        if len(prev) > 1 and len(curr) > 1:
            assert curr != prev[1:], (
                f"One-character slide detected: {prev!r} -> {curr!r}"
            )