# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Merlin RAG is a FastAPI-based Retrieval Augmented Generation system designed for local Docker deployment. It combines Qdrant (vector DB), Ollama (LLM runtime), and SQLite (conversation persistence) into a single-file Python application (`app.py`).

## Architecture

**Single-file monolith**: The entire application lives in `app.py` (~1,300 lines). There are no separate modules or packages.

Key layers within `app.py`:
- **Configuration** (lines 1-80): Environment variable-driven config with defaults suited for Docker networking
- **Database** (`db()` function): SQLite with tables `users`, `conversations`, `messages`. Foreign keys enforced, parameterized queries throughout
- **Auth**: Cookie-based sessions via `itsdangerous`, password hashing with `bcrypt`, optional internal API key (`X-Internal-Key` header)
- **Text Processing Pipeline**: Extracts text from .md, .txt, .docx, .xlsx, .csv, and images (Tesseract OCR with fallback to vision model). Chunks text at 900 chars with 150-char overlap
- **Vector Search**: Qdrant with cosine similarity, embedding via Ollama's `/api/embeddings` endpoint
- **Chat/RAG**: Hybrid mode combines local vector search + DuckDuckGo web search, then generates via Ollama's `/api/chat` endpoint
- **Embedded UI**: The `/` endpoint serves a complete HTML5 single-page app inline

## Build & Run

Build the Docker image:
```bash
docker build -t merlin-rag-api:<version> .
```

Run locally with uvicorn (for development outside Docker):
```bash
pip install -r requirements.txt
uvicorn app:app --host=0.0.0.0 --port=8000
```

Note: Running locally requires Qdrant and Ollama accessible at the URLs in the environment config. Default URLs assume Docker networking (`http://ollama-llm:11434`, `http://qdrant:6333`).

Health check:
```bash
curl -s http://127.0.0.1:8000/health
```

## Docker Service Architecture

Four services communicate over a shared `ragnet` Docker network:
- **Qdrant** (port 6333): Vector database storage
- **Ollama** (port 11434): LLM runtime with models `llama3.1:8b` (chat), `nomic-embed-text` (embeddings), `llava:7b` (vision)
- **Merlin RAG API** (port 8000): This application
- **Open WebUI** (port 3000): Optional external chat frontend

## Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://ollama-llm:11434` | Ollama service URL |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant service URL |
| `CHAT_MODEL` | `llama3.1:8b` | LLM for chat generation |
| `EMBED_MODEL` | `nomic-embed-text` | Model for text embeddings |
| `VISION_MODEL` | `llava:7b` | Model for image description |
| `RAG_DATA_DIR` | `/rag-data/text` | Document storage path |
| `SQLITE_PATH` | `/data/chat.db` | SQLite database location |
| `QDRANT_COLLECTION` | `general_docs` | Qdrant collection name |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `150` | Text chunking parameters |
| `TOP_K` | `6` | Number of vector search results |
| `SESSION_SECRET` | Derived from hostname | HMAC key for session cookies |

## API Endpoint Groups

- **Auth**: `/auth/register`, `/auth/login`, `/auth/logout`, `/auth/me`
- **Conversations**: CRUD at `/conversations` and `/conversations/{id}/messages`
- **Chat**: `POST /chat` — main RAG endpoint (modes: `hybrid`, `rag`, `web`)
- **Files**: `/files`, `/upload`, `/ingest` — document management and vectorization
- **Export**: `/export/{conv_id}?format=csv|txt|docx` — conversation export
- **Generation**: `POST /generate-file` — LLM-generated files (csv, txt, docx, png charts)

## Development Notes

- No test suite exists. Manual testing via API calls or the embedded UI at `/`.
- No linting/formatting configuration is set up.
- The Dockerfile runs as non-root user `appuser` (UID 10001) for security.
- Database schema changes are handled inline in the `db()` function with `ALTER TABLE` try/except blocks (no migration framework).
