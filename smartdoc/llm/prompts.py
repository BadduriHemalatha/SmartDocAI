"""Prompt templates for grounded RAG and summarization."""

from __future__ import annotations

from smartdoc.models import RetrievedChunk

NOT_FOUND_MESSAGE = (
    "The requested information was not found in the uploaded documents."
)


def format_context(chunks: list[RetrievedChunk], max_chars: int) -> str:
    parts: list[str] = []
    used = 0
    for item in chunks:
        header = (
            f"[Source {item.rank} | Document: {item.chunk.document_name} "
            f"| Page: {item.chunk.page_label()} | Similarity: {item.score:.3f}]"
        )
        block = f"{header}\n{item.chunk.text.strip()}"
        if used + len(block) + 2 > max_chars:
            remaining = max_chars - used - 2
            if remaining > 200:
                parts.append(block[:remaining] + "\n...")
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


def build_qa_prompt(question: str, context: str) -> str:
    return f"""You are SmartDoc AI, a retrieval-augmented assistant.

Answer the user's question using ONLY the document context below.
Rules:
- If the context does not contain enough information, reply exactly:
  {NOT_FOUND_MESSAGE}
- Do not use outside knowledge.
- Do not invent facts, names, numbers, or citations.
- When you answer, cite document names and page numbers from the context tags.
- Be concise and professional.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:"""


def build_summary_prompt(text: str, document_names: list[str]) -> str:
    names = ", ".join(document_names) if document_names else "the uploaded document(s)"
    return f"""You are SmartDoc AI. Write a clear, faithful summary of the following content from {names}.

Rules:
- Use only the provided text.
- Do not invent facts.
- Cover the main topic, purpose, and important details.
- Write 1-3 short paragraphs.

CONTENT:
{text}

SUMMARY:"""


def build_key_points_prompt(text: str, document_names: list[str]) -> str:
    names = ", ".join(document_names) if document_names else "the uploaded document(s)"
    return f"""You are SmartDoc AI. Extract the most important key points from {names}.

Rules:
- Use only the provided text.
- Return 5 to 10 bullet points.
- Each bullet must be a complete, factual statement from the content.
- Do not invent information.

CONTENT:
{text}

KEY POINTS:"""


def build_reduce_prompt(partials: list[str], task: str) -> str:
    joined = "\n\n---\n\n".join(partials)
    return f"""You are SmartDoc AI. Combine the following section-level notes into one {task}.

Rules:
- Use only these notes.
- Remove repetition.
- Keep facts accurate.
- Do not add new information.

SECTION NOTES:
{joined}

FINAL {task.upper()}:"""
