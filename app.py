from __future__ import annotations

import streamlit as st

from smartdoc.config import settings
from smartdoc.exceptions import SmartDocError
from smartdoc.rag.pipeline import RAGPipeline, IngestedCorpus
from smartdoc.rag.summarizer import DocumentSummarizer
from smartdoc.ui.styles import apply_styles


def sidebar() -> None:
    """Render the application sidebar."""
    with st.sidebar:
        st.title("SmartDoc AI")
        st.markdown(
            "Upload documents and ask questions using "
            "retrieval-augmented generation."
        )

        st.divider()

        st.subheader("Settings")
        st.write(f"Embedding model: `{settings.embedding_model}`")
        st.write(f"Gemini model: `{settings.gemini_model}`")
        st.write(f"Top-K retrieval: `{settings.top_k}`")

        if settings.has_gemini_key:
            st.success("Gemini API key configured")
        else:
            st.warning("Gemini API key is not configured")


def qa_tab(
    pipeline: RAGPipeline,
    corpus: IngestedCorpus | None,
) -> None:
    """Question answering tab."""
    st.subheader("Question answering")

    if corpus is None:
        st.info("Upload and process documents first.")
        return

    question = st.text_input(
        "Ask a question about your documents",
        placeholder="Example: What is the main topic of this document?",
    )

    if st.button("Ask question", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Searching documents and generating answer..."):
            try:
                result = pipeline.ask(corpus, question)
            except SmartDocError as exc:
                st.error(exc.user_message)
                return
            except Exception as exc:
                st.error(f"An unexpected error occurred: {exc}")
                return

        st.markdown("### Answer")
        st.write(result.answer)

        if result.sources:
            st.markdown("### Sources")

            for source in result.sources:
                chunk = source.chunk

                st.markdown(
                    f"**{source.rank}. {chunk.document_name}**  \n"
                    f"Page: {chunk.page_label()}  \n"
                    f"Similarity: {source.score:.3f}"
                )

                with st.expander("View source text"):
                    st.write(chunk.text)


def summary_tab(
    pipeline: RAGPipeline,
    corpus: IngestedCorpus | None,
) -> None:
    """Summary tab."""
    st.subheader("Summary")

    if corpus is None:
        st.info("Upload and process documents first.")
        return

    if st.button("Generate summary", type="primary"):
        try:
            summarizer = DocumentSummarizer(
                settings,
                llm=pipeline.llm,
            )

            with st.spinner("Generating document summary..."):
                insights = summarizer.analyze(corpus.chunks)

            st.markdown("### Summary")
            st.write(insights.summary)

            if insights.key_points:
                st.markdown("### Key points")

                for point in insights.key_points:
                    st.markdown(f"- {point}")

            if insights.structured_notes:
                st.markdown("### Structured notes")
                st.write(insights.structured_notes)

        except SmartDocError as exc:
            st.error(exc.user_message)
        except Exception as exc:
            st.error(f"An unexpected error occurred: {exc}")


def sources_tab(
    corpus: IngestedCorpus | None,
) -> None:
    """Sources tab."""
    st.subheader("Sources")

    if corpus is None:
        st.info("Upload and process documents first.")
        return

    st.write(f"Documents processed: **{len(corpus.documents)}**")
    st.write(f"Total chunks: **{corpus.total_chunks}**")

    for document in corpus.documents:
        with st.expander(document.document_name):
            st.write(f"File type: {document.file_type}")
            st.write(f"Pages/sections: {document.page_count}")
            st.write(f"Characters: {document.char_count}")


def main() -> None:
    """Run the SmartDoc AI Streamlit application."""
    st.set_page_config(
        page_title="SmartDoc AI",
        page_icon="📄",
        layout="wide",
    )

    apply_styles()
    sidebar()

    st.title("SmartDoc AI")
    st.caption(
        "RAG-based document summarization and context-aware question answering"
    )

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline(settings)

    if "corpus" not in st.session_state:
        st.session_state.corpus = None

    pipeline = st.session_state.pipeline

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload PDF, DOCX, or TXT files.",
    )

    if st.button("Process documents", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            try:
                with st.spinner(
                    "Extracting, chunking, embedding, and indexing documents..."
                ):
                    st.session_state.corpus = pipeline.ingest(
                        uploaded_files,
                        existing=st.session_state.corpus,
                    )

                corpus = st.session_state.corpus

                st.success(
                    f"Processed {len(corpus.documents)} document(s) "
                    f"into {corpus.total_chunks} chunks."
                )

            except SmartDocError as exc:
                st.error(exc.user_message)
            except Exception as exc:
                st.error(f"An unexpected error occurred: {exc}")

    corpus = st.session_state.corpus

    st.divider()

    tab1, tab2, tab3 = st.tabs(
        [
            "Question answering",
            "Summary",
            "Sources",
        ]
    )

    with tab1:
        qa_tab(pipeline, corpus)

    with tab2:
        summary_tab(pipeline, corpus)

    with tab3:
        sources_tab(corpus)


if __name__ == "__main__":
    main()