# Merlin RAG (Local Docker Launch)

This guide launches **Merlin** locally on a single machine using Docker:
- **Qdrant** (vector DB)
- **Ollama** (LLM runtime)
- **Merlin RAG API** (FastAPI app)
- **Open WebUI** (UI required for caht histories and continuous conversations)

Everything can be run **local-only**.

To enable hitting your copy oF Merlin remotely, you'll need to establish a VPN (tailscale, wiregurad, etc.) and create a node for the docker container.  

---

## Prereqs

- Docker installed (Docker Desktop on Windows/macOS; Docker Engine on Linux)
- Machine capable of running Ollama (GPU recommended; CPU works but slower)

---

## Project files (for Merlin RAG API)

In a folder (e.g., `merlin-rag/`), you should have:

- `Dockerfile`
- `app.py`
- `requirements.txt`

These are used to build the `merlin-rag-api:local` image.

---

## 1) One-time setup: network + volumes

```bash
docker network create ragnet 2>/dev/null || true

docker volume create rag_data 2>/dev/null || true
docker volume create rag_api_data 2>/dev/null || true
docker volume create qdrant_storage 2>/dev/null || true
docker volume create open_webui_data 2>/dev/null || true
```

---

## 2) Start Qdrant (vector database)

```bash
docker rm -f qdrant 2>/dev/null || true

docker run -d \
  --name qdrant \
  --network ragnet \
  -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  --restart unless-stopped \
  qdrant/qdrant:latest
```

Optional quick check:

```bash
curl -s http://127.0.0.1:6333/ | head
```

---

## 3) Start Ollama (LLM runtime)

```bash
docker rm -f ollama-llm 2>/dev/null || true

docker run -d \
  --name ollama-llm \
  --network ragnet \
  -p 11434:11434 \
  -v ollama:/root/.ollama \
  --restart unless-stopped \
  ollama/ollama
```

### Pull models (first time only)

Defaults used by Merlin:

- Chat model: `llama3.1:8b`
- Embeddings: `nomic-embed-text`
- Vision model: `llava:7b`

```bash
docker exec -it ollama-llm ollama pull llama3.1:8b
docker exec -it ollama-llm ollama pull nomic-embed-text
docker exec -it ollama-llm ollama pull llava:7b
```

---

## 4) Build Merlin RAG API image

From the folder containing `Dockerfile`, `app.py`, and `requirements.txt`:

```bash
docker build -t merlin-rag-api:local .
```

---

## 5) Run Merlin RAG API (FastAPI)

Merlin stores:
- Document data / uploads under `/rag-data`
- SQLite state under `/data/chat.db`

```bash
docker rm -f rag-api 2>/dev/null || true

docker run -d \
  --name rag-api \
  --network ragnet \
  -p 8000:8000 \
  -e OLLAMA_BASE_URL=http://ollama-llm:11434 \
  -e QDRANT_URL=http://qdrant:6333 \
  -e CHAT_MODEL=llama3.1:8b \
  -e EMBED_MODEL=nomic-embed-text \
  -e VISION_MODEL=llava:7b \
  -e RAG_DATA_DIR=/rag-data/text \
  -e SQLITE_PATH=/data/chat.db \
  -v rag_data:/rag-data \
  -v rag_api_data:/data \
  --restart unless-stopped \
  merlin-rag-api:local
```

Open Merlin:

- http://127.0.0.1:8000

Health check:

```bash
curl -s http://127.0.0.1:8000/health
```

---

## 6) (Optional) Run Open WebUI

Open WebUI is useful as a chat UI for Ollama. It should have its **own** volume.
**Do not** mount `rag_data` into Open WebUI.

```bash
docker rm -f open-webui 2>/dev/null || true

docker run -d \
  --name open-webui \
  --network ragnet \
  -p 127.0.0.1:3000:8080 \
  -e OLLAMA_BASE_URL=http://ollama-llm:11434 \
  -v open_webui_data:/app/backend/data \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main
```

Open WebUI:

- http://127.0.0.1:3000

---

# Common operations

## View logs

```bash
docker logs -f rag-api
docker logs -f ollama-llm
docker logs -f qdrant
docker logs -f open-webui
```

## Restart everything

```bash
docker restart qdrant ollama-llm rag-api open-webui
```

## Stop everything

```bash
docker stop qdrant ollama-llm rag-api open-webui
```

## Remove containers (keeps data)

```bash
docker rm -f qdrant ollama-llm rag-api open-webui
```

---

# Reset / wipe all data

⚠️ This deletes chats, uploaded docs, and vectors.

```bash
docker rm -f qdrant ollama-llm rag-api open-webui

docker volume rm rag_data rag_api_data qdrant_storage open_webui_data 2>/dev/null || true
```

If a volume says “in use”, find what’s using it:

```bash
docker ps -a --filter volume=rag_data
```

---

# Troubleshooting

## “Permission denied” writing to /rag-data (uploads fail)

This usually means the `rag_data` volume was created earlier with root-owned files, or another container accidentally mounted it and created root-owned dirs.

Best clean fix (recommended):

1) Remove the containers using the volume  
2) Remove the `rag_data` volume  
3) Recreate volume + relaunch containers

```bash
docker rm -f rag-api open-webui 2>/dev/null || true
docker volume rm rag_data
docker volume create rag_data
```

Then rerun the launch steps above.

---

## Ports

- Merlin RAG API: `http://127.0.0.1:8000`
- Open WebUI: `http://127.0.0.1:3000`
- Qdrant: `http://127.0.0.1:6333`
- Ollama: `http://127.0.0.1:11434`

---

# Notes

- This setup is local-only. If you later want remote access (travel / multiple users), add a reverse proxy (Traefik) and/or a VPN overlay (Tailscale/WireGuard).

## Remote Access (Optional)
- To securely access Merlin while traveling, you'll need to establish a VPN ([Tailscale](https://tailscale.com), [WireGuard](https://www.wireguard.com/)) and configure the rag-api docker container as a 
 node.