# Merlin RAG API (tiny-rag-api)

Merlin RAG API is a FastAPI-based Retrieval-Augmented Generation (RAG) service that uses:

- **Ollama** for chat, embeddings, and optional vision
- **Qdrant** for vector storage and retrieval
- **SQLite** for users, projects, and chat history
- **Docker Compose** for simple deployment

This package includes an **all-in-one Docker Compose setup** so your team can launch:

- Merlin API
- Qdrant
- Ollama

with one stack.

---

## What this stack runs

- **Merlin API** on port `8000`
- **Qdrant** on port `6333`
- **Ollama** on port `11434`

Architecture:

```text
[Browser / User]
        |
        v
[ Merlin API :8000 ] ---> [ Qdrant :6333 ]
        |
        v
[ Ollama :11434 ]
```

---

## Prerequisites

You'll need'

- Docker
- Docker Compose
- Enough disk space for models (~50 GB)
- For good performance: an NVIDIA GPU host with NVIDIA Container Toolkit installed or an AMD GPU with at least 8 GB of VRAM (12 is better)

You do **not** need to install Python manually when using this Docker setup.

---

## Files required in the repo

Because this setup uses `build: .`, keep these files together in the repo:

- `app.py`
- `requirements.txt`
- `Dockerfile`
- `docker-compose.yml`
- `.env` (created from `.env.example`)

---

## 1. Clone the repo

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

---

## 2. Create `.env`

Copy the example file:

```bash
cp .env.example .env
```

Default values in `.env.example` are already set for the included Ollama and Qdrant containers.

If you want web search, set your Tavily key:

```env
TAVILY_API_KEY=your-key-here
```

---

## 3. Launch the full stack

From the repo root:

```bash
docker compose up -d --build
```

This starts:

- `merlin-rag`
- `merlin-qdrant`
- `merlin-ollama`

Check status:

```bash
docker compose ps
```

View logs:

```bash
docker compose logs -f
```

---

## 4. Pull the Ollama models (first time only)

After the containers are up, pull the models into the Ollama container.

### Required models

```bash
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull nomic-embed-text
```

### Optional vision model

```bash
docker compose exec ollama ollama pull llava:7b
```

These downloads are saved in the persistent Docker volume `ollama_data`, so they do **not** need to be re-downloaded every restart.

---

## 5. Verify Ollama is running

From the host:

```bash
curl http://localhost:11434
```

List installed models:

```bash
docker compose exec ollama ollama list
```

---

## 6. Open Merlin

Open in a browser:

```text
http://localhost:8000
```

If running on a remote server, replace `localhost` with the server hostname or IP.

Examples:

```text
http://sparky:8000
http://your-server-name:8000
```

---

## 7. Health check

Check application health:

```text
http://localhost:8000/health
```

You should see Merlin report that:

- Qdrant is reachable
- Ollama is reachable
- search is configured only if `TAVILY_API_KEY` is set

---

## Web search

This project uses **Tavily** for web search.

### Required for search

Set this in `.env`:

```env
TAVILY_API_KEY=your-key-here
```

### Important

- DuckDuckGo has been removed
- If `TAVILY_API_KEY` is not set, web search features will not work
- Chat and document retrieval still work without Tavily

---

## Daily launch commands

### Start the stack

```bash
docker compose up -d
```

### Start and rebuild after code changes

```bash
docker compose up -d --build
```

### Stop the stack

```bash
docker compose down
```

### Stop and remove all volumes

```bash
docker compose down -v
```

---

## Useful commands

### Show running containers

```bash
docker compose ps
```

### Merlin logs

```bash
docker compose logs -f merlin-rag
```

### Qdrant logs

```bash
docker compose logs -f qdrant
```

### Ollama logs

```bash
docker compose logs -f ollama
```

### Open a shell in the Merlin container

```bash
docker compose exec merlin-rag sh
```

### Open a shell in the Ollama container

```bash
docker compose exec ollama sh
```

### List installed Ollama models

```bash
docker compose exec ollama ollama list
```

---

## First-time setup summary

```bash
git clone <your-repo-url>
cd <your-repo-folder>
cp .env.example .env
docker compose up -d --build
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull nomic-embed-text
# optional:
docker compose exec ollama ollama pull llava:7b
```

Then open:

```text
http://localhost:8000
```

---

## Updating the app after code changes

If you modify `app.py`, `requirements.txt`, or the Dockerfile:

```bash
docker compose up -d --build
```

If you only restart existing containers:

```bash
docker compose restart
```

---

## Storage and persistence

This stack uses Docker volumes so data survives container restarts:

- `merlin_rag_data` → uploaded and processed RAG data
- `merlin_sqlite_data` → SQLite database
- `qdrant_storage` → Qdrant vectors
- `ollama_data` → downloaded Ollama models

---

## Troubleshooting

### Merlin starts but chat fails

Check Ollama:

```bash
docker compose logs -f ollama
curl http://localhost:11434
```

### Embeddings fail

Make sure the embedding model is installed:

```bash
docker compose exec ollama ollama pull nomic-embed-text
```

### Vision/image features fail

Make sure the vision model is installed:

```bash
docker compose exec ollama ollama pull llava:7b
```

### Qdrant issues

```bash
docker compose logs -f qdrant
```

### Web search fails

Check that `.env` contains:

```env
TAVILY_API_KEY=your-key-here
```

Then restart Merlin:

```bash
docker compose up -d --build
```

### Full clean reset

```bash
docker compose down -v
docker compose up -d --build
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull nomic-embed-text
```

---

## Notes for GPU hosts

If the host has NVIDIA GPU support configured for Docker, Ollama can use it automatically.

If the host does **not** have a supported GPU setup, Ollama may still run on CPU, but it will be much slower.

---

## Optional: use an external Ollama server instead

If you later want to go back to a shared external Ollama server:

1. remove or disable the `ollama` service in `docker-compose.yml`
2. set this in `.env`:

```env
OLLAMA_BASE_URL=http://YOUR-OLLAMA-SERVER:11434
```

---

## Quick answer for your team

### How do we launch everything?

```bash
cp .env.example .env
docker compose up -d --build
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull nomic-embed-text
```

Then browse to:

```text
http://localhost:8000
```