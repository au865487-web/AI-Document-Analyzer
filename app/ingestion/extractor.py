"""Extract plain text from uploaded documents.

Supported formats: PDF (via PyMuPDF) and TXT (UTF-8 with a Latin-1
fallback). Always operates on a path on disk; callers are responsible for
persisting uploads before extraction.
"""
from dataclasses import dataclass
from pathlib import Path

import fitz

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


class ExtractionError(Exception):
    """Raised when a document cannot be extracted."""


@dataclass(frozen=True)
class ExtractedPage:
    """A single page extracted from a document.

    Attributes:
        page_number: 1-based page number for paginated documents (e.g. PDF),
            or None for unpaginated documents (e.g. TXT).
        text: Plain text content of the page.
    """

    page_number: int | None
    text: str


def extract_pages(path) -> list[ExtractedPage]:
    """Extract pages from a PDF or TXT file.

    Returns:
        List of :class:`ExtractedPage`. For PDFs, ``page_number`` is 1-indexed.
        For TXT files, a single ``ExtractedPage`` with ``page_number=None`` is returned.

    Raises:
        ExtractionError: if the file is missing, of an unsupported type,
            password-protected, or contains no extractable text.
    """
    path = Path(path)
    if not path.is_file():
        raise ExtractionError(f"File not found: {path}")

    extension = path.suffix.lower()
    if extension == ".pdf":
        return _extract_pdf_pages(path)
    if extension == ".txt":
        return _extract_txt_pages(path)
    raise ExtractionError(
        f"Unsupported file type '{path.suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )


def extract_text(path) -> str:
    """Extract plain text from a PDF or TXT file.

    Raises:
        ExtractionError: if the file is missing, of an unsupported type,
            password-protected, or contains no extractable text.
    """
    path = Path(path)
    if not path.is_file():
        raise ExtractionError(f"File not found: {path}")

    extension = path.suffix.lower()
    if extension == ".pdf":
        return _extract_pdf(path)
    if extension == ".txt":
        return _extract_txt(path)
    raise ExtractionError(
        f"Unsupported file type '{path.suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )


def _extract_pdf(path: Path) -> str:
    try:
        doc = fitz.open(str(path))
    except Exception as exc:  # noqa: BLE001 - wrap any PyMuPDF failure
        raise ExtractionError(f"Could not open PDF '{path.name}': {exc}") from exc

    try:
        if doc.needs_pass:
            raise ExtractionError(f"PDF '{path.name}' is password-protected.")
        pages = (page.get_text() for page in doc)
        text = "\n\n".join(pages)
    except ExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"Could not extract text from PDF '{path.name}': {exc}") from exc
    finally:
        doc.close()

    if not text.strip():
        raise ExtractionError(
            f"PDF '{path.name}' contains no extractable text (scanned document?)."
        )
    return text


def _extract_txt(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ExtractionError(f"Could not read file '{path.name}': {exc}") from exc

    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ExtractionError(
            f"Text file '{path.name}' could not be decoded (tried UTF-8 and Latin-1)."
        )

    if not text.strip():
        raise ExtractionError(f"Text file '{path.name}' contains no readable text.")
    return text


def _extract_pdf_pages(path: Path) -> list[ExtractedPage]:
    try:
        doc = fitz.open(str(path))
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"Could not open PDF '{path.name}': {exc}") from exc

    try:
        if doc.needs_pass:
            raise ExtractionError(f"PDF '{path.name}' is password-protected.")
        pages = []
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text()
            if page_text.strip():
                pages.append(ExtractedPage(page_number=page_num, text=page_text))
    except ExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"Could not extract text from PDF '{path.name}': {exc}") from exc
    finally:
        doc.close()

    if not pages:
        raise ExtractionError(
            f"PDF '{path.name}' contains no extractable text (scanned document?)."
        )
    return pages


def _extract_txt_pages(path: Path) -> list[ExtractedPage]:
    text = _extract_txt(path)
    return [ExtractedPage(page_number=None, text=text)]