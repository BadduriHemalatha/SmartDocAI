# SmartDoc AI

RAG-based intelligent document summarization and context-aware question answering.

SmartDoc AI is an AI/ML internship project that allows users to upload PDF, DOCX, and TXT documents, process them using semantic retrieval, ask questions grounded in the uploaded documents, and generate document summaries and key points using Gemini.

The authoritative product specification is [SMARTDOC_AI_MASTER_SPEC.md](SMARTDOC_AI_MASTER_SPEC.md).

## Problem Statement

Long documents can be difficult to search and understand efficiently. Generic AI chatbots may provide information that is not present in a user's documents. SmartDoc AI addresses this using a Retrieval-Augmented Generation (RAG) pipeline that retrieves relevant document content before generating an answer.

The system provides source information and reports when the requested information is not found in the uploaded documents.

## Features

* Upload PDF, DOCX, and TXT documents
* Multi-document processing
* Page-aware PDF text extraction
* Text preprocessing and overlapping chunking
* Sentence Transformer embeddings
* FAISS vector similarity search
* Top-k semantic retrieval with similarity filtering
* Context-aware Gemini question answering
* Document-grounded answers
* Source document, page, and similarity information
* Document summary generation
* Key-point extraction
* Structured document notes
* Streamlit-based user interface
* Environment-variable based configuration
* API key is not hard-coded in the application

## Architecture

```text
app.py                         Streamlit UI
smartdoc/
  config.py                    Environment configuration
  exceptions.py                Typed application errors
  logging_setup.py             File and console logging
  models.py                    Document and RAG data models

  document_processing/
    extractor.py               PDF, DOCX, and TXT extraction
    preprocessor.py            Text preprocessing
    chunker.py                 Overlapping text chunking

  embeddings/
    encoder.py                 Sentence Transformer embeddings

  vectorstore/
    faiss_store.py             FAISS vector index

  retrieval/
    retriever.py               Semantic top-k retrieval

  llm/
    gemini_client.py           Gemini API client
    prompts.py                 RAG prompts and context formatting

  rag/
    pipeline.py                Document ingestion and Q&A pipeline
    summarizer.py              Document summarization

  ui/
    styles.py                  Streamlit UI styling
```

## RAG Workflow

1. Upload one or more PDF, DOCX, or TXT documents.
2. Extract text and document metadata.
3. Preprocess the extracted text.
4. Split the text into overlapping chunks.
5. Generate embeddings using Sentence Transformers.
6. Store the embeddings in a FAISS vector index.
7. Convert the user's question into an embedding.
8. Retrieve the most relevant document chunks using semantic similarity.
9. If no relevant information is available, return a document-not-found response without calling Gemini.
10. If relevant context is found, send the retrieved context and question to Gemini.
11. Display the generated answer together with source document, page information, and similarity scores.

## Technology Stack

| Area                 | Technology                               |
| -------------------- | ---------------------------------------- |
| Programming Language | Python                                   |
| User Interface       | Streamlit                                |
| Embeddings           | Sentence Transformers                    |
| Embedding Model      | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Search        | FAISS                                    |
| PDF Processing       | pypdf                                    |
| DOCX Processing      | python-docx                              |
| LLM                  | Gemini via `google-genai`                |
| Configuration        | python-dotenv                            |
| Testing              | pytest                                   |

## Installation

From the project root:

```cmd
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On the first run, Sentence Transformers may download the configured embedding model. Internet access is required for model download and Gemini API requests.

## Configuration

1. Copy `.env.example` to `.env`.
2. Create a Gemini API key through Google AI Studio.
3. Add the key to `.env`:

```text
GEMINI_API_KEY=your_real_key_here
```

Optional configuration variables include:

```text
GEMINI_MODEL
EMBEDDING_MODEL
CHUNK_SIZE
CHUNK_OVERLAP
TOP_K
MIN_SIMILARITY
MAX_FILE_SIZE_MB
MAX_CONTEXT_CHARS
LOG_LEVEL
```

Never commit or publicly share `.env`.

## How to Run

Activate the virtual environment:

```cmd
.venv\Scripts\activate.bat
```

Start the Streamlit application:

```cmd
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, normally:

```text
http://localhost:8501
```

## How to Use

1. Open SmartDoc AI in the browser.
2. Upload one or more PDF, DOCX, or TXT documents.
3. Click **Process documents**.
4. Open the **Question answering** tab.
5. Enter a question related to the uploaded documents.
6. Review the generated answer and retrieved sources.
7. Open the **Summary** tab to generate a document summary and key points.
8. Open the **Sources** tab to view processed document information and chunk statistics.

If the requested information is not present in the uploaded documents, the application reports:

```text
The requested information was not found in the uploaded documents.
```

## Testing

Run the complete test suite with:

```cmd
python -m pytest -v
```

The test suite covers:

* PDF, DOCX, and TXT extraction
* Multiple-document processing
* Text preprocessing
* Chunking and overlap
* Sentence Transformer embeddings
* FAISS indexing
* Semantic retrieval
* RAG grounding
* Unknown-information handling
* Summarization
* Source metadata
* Index reuse
* Streamlit UI structure
* API-key security
* Error handling

### Current Validation

The complete test suite currently passes:

```text
28 passed
```

The application has also been manually tested with:

* RL Unit 1 PDF
* TXT document
* Document-grounded questions
* Questions outside the uploaded document content
* Document summaries
* Key points
* Source and page information

## Limitations

* Image-only scanned PDFs are not OCR'd.
* Password-protected PDFs cannot be opened.
* DOCX and TXT files do not provide reliable PDF-style page numbers; the UI displays `N/A` where page information is unavailable.
* Answer quality depends on document quality, chunking, embedding quality, retrieval settings, and Gemini.
* The FAISS index is maintained in the Streamlit session and is not a persistent multi-user database.
* Gemini requires a valid API key and network access.

## Future Enhancements

* OCR support for scanned PDFs
* Persistent vector indexes
* Hybrid BM25 and dense retrieval
* Cross-encoder re-ranking
* Export of answers and citations
* Additional document formats
* Improved conversation management
