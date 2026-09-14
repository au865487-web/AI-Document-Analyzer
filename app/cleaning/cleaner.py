"""Lightweight text cleaning for extracted documents.

Cleaning stays conservative: normalize line endings, drop control
characters, collapse duplicated whitespace, and trim outer whitespace.
Formatting that might matter later (e.g. markdown, code blocks) is left
intact where possible.
"""
import re

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_HORIZONTAL_WS = re.compile(r"[ \t]+")


def clean_text(text: str) -> str:
    """Return a cleaned copy of ``text``, preserving paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _MULTI_BLANK_LINES.sub("\n\n", text)
    text = _HORIZONTAL_WS.sub(" ", text)
    return text.strip()