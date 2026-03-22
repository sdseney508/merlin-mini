# Merlin RAG API (tiny-rag-api)

Merlin RAG API is a FastAPI-based Retrieval-Augmented Generation (RAG) service.

It uses:
- Ollama → LLMs + embeddings + optional vision
- Qdrant → vector database
- SQLite → users, projects, chat history
- Docker Compose → simple deployment

This app connects to an external Ollama server over HTTP.

---

## Prerequisites

- Docker
- Docker Compose
- Access to an Ollama server

---

## 1. Setup Ollama (on Ollama server)

```bash
ollama serve
ollama pull llama3.1:8b
ollama pull nomic-embed-text
# optional
ollama pull llava:7b
```

---

## 2. Clone repo

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

---

## 3. Create .env

```env
OLLAMA_BASE_URL=http://YOUR-OLLAMA-SERVER:11434
CHAT_MODEL=llama3.1:8b
EMBED_MODEL=nomic-embed-text
VISION_MODEL=llava:7b

QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=general_docs

RAG_ROOT=/rag-data
RAG_DATA_DIR=/rag-data/text
SQLITE_PATH=/data/chat.db

SESSION_SECRET=replace-with-long-random-string
COOKIE_SECURE=false

TAVILY_API_KEY=your-key-here
```

---

## 4. Run

```bash
docker compose up -d --build
```

---

## 5. Access

http://localhost:8000

---

## Notes

- Tavily is required for web search
- Keep Dockerfile, requirements.txt, and app.py in repo (build required)