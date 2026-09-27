# SmartDoc AI -- Master Specification

**Project title:** SmartDoc AI: RAG-Based Intelligent Document Summarization and Context-Aware Question Answering System

**Audience:** AIML internship -- professional, functional application (not a static demo).

This document is the authoritative source of truth for the implementation.

---

## 1. Problem statement

Organizations accumulate PDF, DOCX, and TXT knowledge that is hard to query. Generic chatbots invent answers. SmartDoc AI must:

1. Ingest one or more user documents.
2. Extract, clean, and chunk text while preserving source metadata (file name, page number when available).
3. Embed chunks with a Sentence Transformer model.
4. Index embeddings in FAISS for semantic similarity search.
5. Retrieve relevant chunks for a user question.
6. Generate an answer with Gemini only from retrieved context.
7. Summarize documents and extract key points without overflowing the LLM context window.
8. Surface sources so every answer is auditable.

---

## 2. Mandatory RAG pipeline

The following pipeline is required and must be actually connected (not simulated):

Document Upload -> Document Text Extraction -> Text Preprocessing -> Text Chunking -> Sentence Transformer Embeddings -> FAISS Vector Store -> Semantic Similarity Retrieval -> Relevant Context -> Gemini LLM -> Context-Aware Answer

The LLM must receive retrieved document context. The application must not be a general chatbot that answers without retrieval.

If the answer cannot be supported by retrieved context, the system must clearly state that the information was not found in the uploaded documents rather than inventing an answer.

---

## 3. Functional requirements

### 3.1 Document ingestion

- Support PDF, DOCX, and TXT.
- Support multiple document uploads in one session.
- Reject unsupported types with a clear error.
- Handle empty documents, corrupted files, and failed extraction.
- Enforce a configurable maximum file size.
- Show processing status and uploaded-document information in the UI.

### 3.2 Text extraction

- PDF: extract text per page; record page numbers.
- DOCX: extract paragraph text; page numbers are typically unavailable -- record N/A rather than inventing pages.
- TXT: extract UTF-8 (with fallback encoding); record line-range metadata where useful.
- Preserve document name and file type on every extracted unit.

### 3.3 Preprocessing

- Normalize whitespace and line breaks.
- Remove excessive repeated characters/noise without destroying meaningful punctuation.
- Preserve enough structure for chunking (paragraph boundaries).

### 3.4 Chunking

- Split into overlapping chunks with configurable size and overlap.
- Prefer splitting on paragraph/sentence boundaries when possible.
- Attach metadata to every chunk: document name, file type, page number(s) when available, chunk index.
- Do not drop metadata during embedding or indexing.

### 3.5 Embeddings

- Use Sentence Transformers (default model: all-MiniLM-L6-v2, overridable via environment).
- Cache/reuse the loaded model.
- Handle embedding failures with user-visible errors.

### 3.6 Vector store

- Use FAISS for vector similarity search.
- Store a side mapping from FAISS integer IDs to chunk metadata.
- Rebuild or replace the index when the user processes a new document set.
- Reuse the in-session index; do not re-embed unchanged documents unnecessarily.

### 3.7 Retrieval

- Embed the user query with the same model.
- Return top-k similar chunks (k configurable).
- Apply a minimum similarity threshold.
- If no chunk meets the threshold, treat the query as unsupported by the documents and skip unnecessary LLM calls.

### 3.8 Question answering (RAG)

- Build a context string from retrieved chunks including source labels.
- Call Gemini with a prompt that:
  - Restricts answers to the provided context.
  - Requires an explicit "not found in uploaded documents" response when context is insufficient.
  - Requests citations (document name and page when available).
- Display the answer plus source information that corresponds to the retrieved chunks.

### 3.9 Summarization and key points

- Produce a document-level (or corpus-level) summary.
- Produce important key points.
- For long documents, use map-reduce (chunk summaries then synthesis) instead of sending unbounded raw text to Gemini.
- Cache summaries in session to avoid duplicate API calls.

### 3.10 Sources / references

For answers, display available:

- Document name
- Page number (when available)
- Relevant source/chunk excerpt
- Retrieval score / rank where useful

Source information must match retrieved content.

### 3.11 User interface (Streamlit)

Professional Streamlit UI with:

- Application title and description
- Document upload
- Processing status
- Uploaded document information
- Question answering
- Chat/history
- Document summary
- Key points
- Sources/references
- Errors and warnings

### 3.12 Configuration and security

- Load configuration from environment variables via python-dotenv.
- Never hard-code API keys.
- Ship .env.example without secrets.
- Ignore .env in git.

### 3.13 Error handling

Handle and surface user-friendly messages for:

- Unsupported file types
- Empty documents
- Corrupted files
- Failed text extraction
- API failures
- Missing API key
- Embedding errors
- FAISS errors
- Invalid user questions
- Long documents
- No relevant retrieved context

### 3.14 Logging

- Structured application logging to console and log file.
- Do not log API keys or full document contents.

---

## 4. Technology stack (required)

- Language: Python 3
- UI: Streamlit
- Embeddings: Sentence Transformers
- Vector search: FAISS
- PDF: pypdf
- DOCX: python-docx
- LLM: Gemini API (google-generativeai)
- Secrets: python-dotenv
- Arrays: NumPy
- Tests: pytest

Do not introduce React, FastAPI, Kubernetes, Docker, microservices, complex databases, fine-tuning, or training a custom LLM.

---

## 5. Architecture (modular)

Responsibilities must be separated across modules:

- Application / UI
- Document processing
- Text cleaning
- Chunking
- Embedding generation
- Vector storage
- Retrieval
- RAG pipeline
- LLM interaction
- Summarization
- Prompt management
- Configuration
- Logging
- Error handling

Do not put the entire application in one Python file.

Package layout:

```
smartdoc/
  config.py
  exceptions.py
  logging_setup.py
  models.py
  document_processing/   extractor, preprocessor, chunker
  embeddings/            encoder
  vectorstore/           faiss_store
  retrieval/             retriever
  llm/                   gemini_client, prompts
  rag/                   pipeline, summarizer
  ui/                    styles
app.py                   Streamlit entrypoint
```

---

## 6. Configuration keys

- GEMINI_API_KEY: Gemini API key (required for LLM features)
- GEMINI_MODEL: Model name (default gemini-2.0-flash)
- EMBEDDING_MODEL: Sentence Transformer id
- CHUNK_SIZE: Target chunk size in characters
- CHUNK_OVERLAP: Overlap in characters
- TOP_K: Number of chunks to retrieve
- MIN_SIMILARITY: Cosine similarity floor
- MAX_FILE_SIZE_MB: Upload size limit
- MAX_CONTEXT_CHARS: Cap on QA context sent to Gemini
- LOG_LEVEL: Logging verbosity

---

## 7. Testing requirements

Automated tests must cover at least:

1. PDF upload / extraction
2. DOCX upload / extraction
3. TXT upload / extraction
4. Multiple documents
5. Text extraction
6. Chunking
7. Embedding generation
8. FAISS indexing
9. Semantic retrieval
10. RAG question answering (context is passed; grounded behavior)
11. Unknown / unavailable information handling
12. Summarization path
13. Key point generation path
14. Source / page display metadata
15. Streamlit app import / structure
16. Error handling

---

## 8. Efficiency

- Reuse embeddings and the FAISS index within a session.
- Use sensible chunk sizes and retrieval parameters.
- Do not make unnecessary Gemini API calls (especially when retrieval finds nothing).
- Keep code readable and maintainable.

---

## 9. Documentation

Ship README.md, requirements.txt, .env.example, and .gitignore.

---

## 10. Out of scope

- Training or fine-tuning an LLM
- Multi-user auth / cloud SaaS
- OCR for scanned image-only PDFs (documented as a limitation)
- Persistent multi-session vector databases
