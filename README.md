# RAG Portfolio

This repository contains a Retrieval-Augmented Generation (RAG) proof-of-concept built with LangChain, Google Gemini embeddings, and a Chroma-based local vector store. It is designed to ingest Markdown documents, split them into chunks, embed the chunks, and prepare them for retrieval and question-answering.

## Project Structure

- `requirements.txt` - Python dependencies for the project.
- `.env` - Local environment variables for Gemini API configuration.
- `context_documents/` - Source Markdown content used for RAG ingestion.
- `RAG/pipeline.ipynb` - Notebook exploring the RAG pipeline, vectorization, retrieval, and LLM usage.
- `RAG/vectorization.py` - Script that loads Markdown docs, splits them into chunks, and initializes Chroma embeddings.
- `RAG/chroma_langchain_db/` - Persisted local Chroma database created during vectorization.
- `src/` - Conceptual module structure for the pipeline:
  - `1.ingestion/loader.py` - Document loader for Markdown files.
  - `2.chunking/chunker.py` - Markdown header splitting and character-based chunking.
  - `3.embeddings/embedder.py` - Embeddings module (currently a placeholder).
  - `4.vectordb/vector_store.py` - Vector store module scaffold.
  - `5.retrieval/retriever.py` - Retrieval module scaffold.
  - `6.prompts/prompt_templates.py` - Prompt template scaffold.
  - `7.llm/llm.py` - LLM invocation scaffold.
  - `8.api/routes.py` - API routes scaffold.
  - `9.utils/helpers.py` - Utility helper scaffold.

## Key Features

- Markdown document ingestion from `context_documents/`
- Header-aware chunking with `MarkdownHeaderTextSplitter`
- Character-based chunking to produce smaller retrieval units
- Google Gemini embedding function via `langchain_google_genai`
- Local Chroma persistence for vector storage
- Notebook-based experimentation and step-by-step pipeline exploration

## Setup

1. Create and activate a Python virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Create or update `.env` with your Gemini API key.

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3-flash-preview
GEMINI_EMBEDDING_MODEL=gemini-embedding-2-preview
```

> Do not commit your actual API keys to source control.

## Running the Vectorization Script

Run the main pipeline script to load Markdown files, split them, and prepare chunked documents for embedding.

```powershell
python RAG\vectorization.py
```

This script will:

- load Markdown files from `context_documents/`
- attach metadata such as filename and category
- split documents by Markdown headings
- further chunk content by character count
- print document metadata and chunk counts

## Notebook Exploration

Open `RAG/pipeline.ipynb` to explore additional workflow steps, including:

- building a Chroma retriever
- hybrid retrieval with BM25 plus vector search
- using Google Gemini for generation
- saving and persisting vector embeddings

## Notes

- The `src/` folder is organized into pipeline stages but several modules are currently placeholders.
- The active implementation is in `RAG/vectorization.py` and the notebook.
- `RAG/chroma_langchain_db/` holds persisted Chroma data and can be deleted if you want to rebuild from scratch.

## Recommended Next Steps

- complete the embedding and vector store modules in `src/`
- implement retrieval and prompt orchestration logic in `src/`
- add API endpoints in `src/8.api/routes.py`
- extend the notebook for full question-answering and RAG application demos

## License

This repository does not include a license file. Add one if you want to share or publish the project.

## Chatbot API (`rag_api/`)

Lightweight, citation-grounded endpoint meant to be called from another website. Uses in-memory BM25 retrieval (no torch/vector DB) and a single free-tier LLM call (Gemini `gemini-2.5-flash-lite`, optional Groq fallback).

```powershell
pip install -r requirements-api.txt
uvicorn rag_api.app:app --host 0.0.0.0 --port 8000
```

Env vars (`.env`): `GEMINI_API_KEY` (required unless `GROQ_API_KEY` is set), `GROQ_API_KEY` (optional fallback), `ALLOWED_ORIGINS` (comma-separated, e.g. `https://yoursite.com`; default `*`), `RATE_LIMIT_PER_MIN` (default 15).

`POST /api/chat` with `{"message": "...", "history": [{"role": "user"|"assistant", "content": "..."}]}` returns `{"reply", "citations": [{"id","source","section","snippet"}], "grounded"}`. `GET /api/health` for checks.

```js
const r = await fetch("https://YOUR-HOST/api/chat", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({message, history})});
```
