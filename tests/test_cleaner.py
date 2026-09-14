"""Tests for app.cleaning.cleaner."""
from app.cleaning.cleaner import clean_text


def test_collapses_multiple_blank_lines():
    assert clean_text("a\n\n\n\nb") == "a\n\nb"


def test_normalizes_crlf_and_cr():
    assert clean_text("a\r\nb\rc") == "a\nb\nc"


def test_removes_control_characters():
    assert clean_text("a\x00b\x1fc") == "abc"


def test_collapses_horizontal_whitespace():
    assert clean_text("a    b\t\t c") == "a b c"


def test_trims_leading_trailing_whitespace():
    assert clean_text("  hello  \n  ") == "hello"


def test_empty_input():
    assert clean_text("") == ""


def test_whitespace_only_input():
    assert clean_text("   \n\t  ") == ""