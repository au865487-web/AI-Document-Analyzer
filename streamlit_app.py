"""Thin Streamlit UI for the AI Document Analyzer.

All retrieval, context building, and generation happen in ``DocumentAnalyzer``.
"""
from pathlib import Path

import streamlit as st

from app.ingestion.extractor import SUPPORTED_EXTENSIONS
from app.pipeline import DocumentAnalyzer, groq_api_key_configured
from app.ui.messages import (
    EMPTY_INDEX,
    GROQ_MISSING,
    GROQ_READY,
    MISSING_QUESTION,
    UNSUPPORTED_UPLOAD,
    empty_ask_status,
    index_success,
    user_message_for_exception,
)

_EXCERPT_CHARS = 280


def _analyzer() -> DocumentAnalyzer:
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = DocumentAnalyzer()
    return st.session_state.analyzer


def _safe_upload_name(name: str) -> str:
    return Path(name).name


def _render_citations(answer) -> None:
    if answer.citations:
        st.subheader("Citations")
        for citation in answer.citations:
            excerpt = citation.text.strip()
            if len(excerpt) > _EXCERPT_CHARS:
                excerpt = excerpt[:_EXCERPT_CHARS].rstrip() + "…"
            title = f"{citation.citation_tag} — {citation.citation_label}"
            with st.expander(title):
                st.write(f"Document: {citation.document_id}")
                if citation.page_number is not None:
                    st.write(f"Page: {citation.page_number}")
                st.write(f"Chunk: {citation.chunk_index}")
                st.write(excerpt)

    if answer.invalid_citations:
        with st.expander("Unverified citation tags"):
            st.caption(
                "These tags appeared in the model output but do not match "
                "indexed sources. They are not used as citations."
            )
            st.write(", ".join(answer.invalid_citations))


def main() -> None:
    st.set_page_config(page_title="AI Document Analyzer", layout="wide")
    st.title("AI Document Analyzer")
    st.write(
        "Upload PDF or TXT documents, index them, and ask questions. "
        "Answers are generated only from retrieved document text and include "
        "verified source citations."
    )

    analyzer = _analyzer()

    with st.sidebar:
        st.subheader("Status")
        if groq_api_key_configured():
            st.success(GROQ_READY)
        else:
            st.warning(GROQ_MISSING)
        st.caption(f"Indexed chunks: {analyzer.chunk_count()}")

    st.header("Documents")
    uploaded = st.file_uploader(
        "Upload a PDF or TXT file",
        type=["pdf", "txt"],
        accept_multiple_files=False,
    )
    index_clicked = st.button("Index document", type="primary")

    if index_clicked:
        if uploaded is None:
            st.warning("Choose a PDF or TXT file to index.")
        else:
            filename = _safe_upload_name(uploaded.name)
            suffix = Path(filename).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                st.error(UNSUPPORTED_UPLOAD)
            else:
                with st.spinner("Indexing document…"):
                    try:
                        path = analyzer.save_upload(filename, uploaded.getvalue())
                        result = analyzer.index_file(path, document_id=filename)
                    except Exception as exc:
                        st.error(user_message_for_exception(exc))
                    else:
                        st.success(index_success(result))

    documents = analyzer.list_documents()
    if not documents:
        st.info(EMPTY_INDEX)
    else:
        st.subheader("Indexed documents")
        for doc in documents:
            label = "chunk" if doc.chunk_count == 1 else "chunks"
            st.write(f"- **{doc.document_id}** — {doc.chunk_count} {label}")

    st.header("Ask a question")
    question = st.text_area("Question", placeholder="Ask about the indexed documents.")
    ask_clicked = st.button("Ask")

    if ask_clicked:
        if not isinstance(question, str) or not question.strip():
            st.warning(MISSING_QUESTION)
        else:
            with st.spinner("Retrieving sources and generating an answer…"):
                try:
                    result = analyzer.ask(question)
                except Exception as exc:
                    st.error(user_message_for_exception(exc))
                else:
                    status = empty_ask_status(result)
                    if status:
                        st.info(status)
                    if result.context.truncated and not result.context.is_empty:
                        st.caption(
                            "Some retrieved chunks were omitted to fit the context budget."
                        )
                    st.subheader("Answer")
                    st.write(result.answer.answer)
                    _render_citations(result.answer)


if __name__ == "__main__":
    main()
