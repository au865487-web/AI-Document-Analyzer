"""Tests for app.ingestion.extractor."""

import fitz
import pytest

from app.ingestion.extractor import (
    ExtractedPage,
    ExtractionError,
    extract_pages,
    extract_text,
)


def test_extract_txt_utf8(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_bytes(b"Hello\nWorld")
    assert extract_text(path) == "Hello\nWorld"


def test_extract_txt_strips_bom(tmp_path):
    path = tmp_path / "bom.txt"
    path.write_text("\ufeffHello", encoding="utf-8")
    assert extract_text(path) == "Hello"


def test_extract_txt_falls_back_to_latin1(tmp_path):
    path = tmp_path / "latin.txt"
    path.write_bytes(b"caf\xe9")
    assert extract_text(path) == "caf\xe9"


def test_extract_txt_empty_raises(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ExtractionError, match="no readable text"):
        extract_text(path)


def test_extract_txt_missing_file_raises(tmp_path):
    with pytest.raises(ExtractionError, match="not found"):
        extract_text(tmp_path / "nope.txt")


def test_extract_unsupported_type_raises(tmp_path):
    path = tmp_path / "doc.docx"
    path.write_bytes(b"x")
    with pytest.raises(ExtractionError, match="Unsupported file type"):
        extract_text(path)


def test_extract_accepts_str_path(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("x", encoding="utf-8")
    assert extract_text(str(path)) == "x"


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello PDF")
    page.insert_text((72, 84), "Second line")
    doc.save(str(path))
    doc.close()
    return path


def test_extract_pdf(sample_pdf):
    text = extract_text(sample_pdf)
    assert "Hello PDF" in text
    assert "Second line" in text


def test_extract_pdf_no_text_raises(tmp_path):
    path = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(path))
    doc.close()
    with pytest.raises(ExtractionError, match="no extractable text"):
        extract_text(path)


def test_extract_pdf_password_protected_raises(tmp_path):
    path = tmp_path / "locked.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "secret")
    doc.save(
        str(path),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
    )
    doc.close()
    with pytest.raises(ExtractionError, match="password-protected"):
        extract_text(path)


def test_extract_pages_pdf_multi_page(tmp_path):
    path = tmp_path / "multi.pdf"
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Content of page one")
    p2 = doc.new_page()
    p2.insert_text((72, 72), "Content of page two")
    doc.save(str(path))
    doc.close()

    pages = extract_pages(path)
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "page one" in pages[0].text
    assert pages[1].page_number == 2
    assert "page two" in pages[1].text


def test_extract_pages_txt(tmp_path):
    path = tmp_path / "simple.txt"
    path.write_text("Hello plain text", encoding="utf-8")

    pages = extract_pages(path)
    assert len(pages) == 1
    assert pages[0].page_number is None
    assert pages[0].text == "Hello plain text"


def test_extract_pages_empty_pdf_raises(tmp_path):
    path = tmp_path / "empty.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(path))
    doc.close()

    with pytest.raises(ExtractionError, match="no extractable text"):
        extract_pages(path)


def test_extract_pages_missing_file_raises(tmp_path):
    with pytest.raises(ExtractionError, match="not found"):
        extract_pages(tmp_path / "missing.pdf")