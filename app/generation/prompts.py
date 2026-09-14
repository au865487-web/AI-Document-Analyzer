"""Prompt templates for Groq answer synthesis.

Kept separate from generation logic so instructions can be reviewed
and adjusted without touching the Groq client or citation validation.
"""

SYSTEM_INSTRUCTIONS = """\
You are a document question-answering assistant.

Answer ONLY from the supplied context. Do not speculate and do not use
outside knowledge.

Every factual claim should have supporting [Source N] citations. Multiple
citations are allowed. Only cite source IDs that actually appear in the
supplied context. Never invent citation numbers.

If the context is insufficient to answer the question, respond exactly:
I cannot answer this question based on the provided documents.
"""


def build_user_message(question: str, formatted_context: str) -> str:
    """Assemble a deterministic user message from context and question."""
    return (
        f"Context:\n\n{formatted_context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
