# ============================================================
# MERLIN RAG API — Enchanted Forest Edition
# FastAPI + Qdrant + Ollama with auth & conversation history
# ============================================================

import os
import re
import json
import time
import sqlite3
import shutil
import uuid
import hashlib
import hmac
import csv
import io
import base64
import tempfile
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import requests
import bcrypt
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Response, Depends
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from ddgs import DDGS
from docx import Document as DocxDocument
from openpyxl import load_workbook
import pytesseract
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------- Configuration ----------
APP_TITLE = "Merlin RAG API"
DATA_DIR = os.getenv("RAG_DATA_DIR", "/rag-data/text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-llm:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1:8b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "general_docs")
SQLITE_PATH = os.getenv("SQLITE_PATH", "/data/chat.db")
RAG_ROOT = Path(os.getenv("RAG_ROOT", "/rag-data"))
TEXT_DIR = Path(DATA_DIR)
INCOMING_DIR = RAG_ROOT / "incoming"
TEXT_DIR.mkdir(parents=True, exist_ok=True)
INCOMING_DIR.mkdir(parents=True, exist_ok=True)
MIN_CONTEXT_CHARS = int(os.getenv("MIN_CONTEXT_CHARS", "1200"))
MIN_TOP_SCORE = float(os.getenv("MIN_TOP_SCORE", "0.25"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "6"))

VISION_MODEL = os.getenv("VISION_MODEL", "llava:7b")
OCR_MIN_CHARS = int(os.getenv("OCR_MIN_CHARS", "50"))
EXPORTS_DIR = Path(os.getenv("EXPORTS_DIR", "/tmp/merlin-exports"))
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_MAX_AGE = int(os.getenv("EXPORT_MAX_AGE", "3600"))

TEXT_EXTENSIONS = (".md", ".txt", ".docx", ".xlsx", ".csv")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff")

# Auth configuration
SESSION_SECRET = os.getenv(
    "SESSION_SECRET",
    hashlib.sha256(f"merlin-dev-{os.getenv('HOSTNAME', 'local')}".encode()).hexdigest(),
)
SESSION_MAX_AGE = int(os.getenv("SESSION_MAX_AGE", "28800"))  # 8 hours
INTERNAL_KEY = os.getenv("INTERNAL_API_KEY", "")
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"
serializer = URLSafeTimedSerializer(SESSION_SECRET)

# ---------- FastAPI Application ----------
app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000", "http://localhost:8000",
        "http://127.0.0.1:3000", "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Database ----------
def db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    conn.execute("""
      CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at INTEGER NOT NULL
      )
    """)

    conn.execute("""
      CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL DEFAULT 'New Conversation',
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
      )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_user "
        "ON conversations(user_id, updated_at DESC)"
    )

    conn.execute("""
      CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        convo_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        ts INTEGER NOT NULL
      )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_convo_ts ON messages(convo_id, ts)"
    )

    # Migrate: add user_id column if missing
    cursor = conn.execute("PRAGMA table_info(messages)")
    columns = [row[1] for row in cursor.fetchall()]
    if "user_id" not in columns:
        conn.execute("ALTER TABLE messages ADD COLUMN user_id INTEGER")

    return conn


def add_message(convo_id: str, role: str, content: str, user_id: int = None) -> None:
    conn = db()
    with conn:
        conn.execute(
            "INSERT INTO messages (convo_id, role, content, ts, user_id) VALUES (?, ?, ?, ?, ?)",
            (convo_id, role, content, int(time.time()), user_id),
        )
    conn.close()


def get_recent_messages(convo_id: str, limit: int = 12) -> List[Dict[str, str]]:
    conn = db()
    cur = conn.execute(
        "SELECT role, content FROM messages WHERE convo_id=? ORDER BY ts DESC, id DESC LIMIT ?",
        (convo_id, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


# ---------- Authentication ----------
def create_session_cookie(user_id: int, username: str) -> str:
    return serializer.dumps({"user_id": user_id, "username": username})


def verify_session_cookie(cookie_value: str) -> Optional[Dict]:
    try:
        return serializer.loads(cookie_value, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None


def get_current_user(request: Request) -> Dict:
    cookie = request.cookies.get("merlin_session")
    if cookie:
        user = verify_session_cookie(cookie)
        if user:
            return user
    raise HTTPException(status_code=401, detail="Not authenticated")


def get_optional_user(request: Request) -> Dict:
    internal = request.headers.get("X-Internal-Key", "")
    if INTERNAL_KEY and hmac.compare_digest(internal, INTERNAL_KEY):
        return {"user_id": 0, "username": "system"}
    return get_current_user(request)


# ---------- Ollama Helpers ----------
def ollama_embed(texts: List[str]) -> List[List[float]]:
    out = []
    for t in texts:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": t},
            timeout=120,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Ollama embeddings error: {resp.text}")
        out.append(resp.json()["embedding"])
    return out


def ollama_chat(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Call Ollama chat and return content + performance metrics."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": CHAT_MODEL, "messages": messages, "stream": False},
        timeout=600,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Ollama chat error: {resp.text}")
    data = resp.json()
    content = data["message"]["content"]

    # Ollama returns durations in nanoseconds
    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 0)
    prompt_eval_count = data.get("prompt_eval_count", 0)
    prompt_eval_duration_ns = data.get("prompt_eval_duration", 0)
    total_duration_ns = data.get("total_duration", 0)

    eval_duration_s = eval_duration_ns / 1e9 if eval_duration_ns else 0
    prompt_eval_duration_s = prompt_eval_duration_ns / 1e9 if prompt_eval_duration_ns else 0
    total_duration_s = total_duration_ns / 1e9 if total_duration_ns else 0
    tokens_per_sec = round(eval_count / eval_duration_s, 1) if eval_duration_s > 0 else 0
    prompt_tokens_per_sec = round(prompt_eval_count / prompt_eval_duration_s, 1) if prompt_eval_duration_s > 0 else 0

    return {
        "content": content,
        "performance": {
            "tokens": eval_count,
            "prompt_tokens": prompt_eval_count,
            "tokens_per_sec": tokens_per_sec,
            "prompt_tokens_per_sec": prompt_tokens_per_sec,
            "eval_duration": round(eval_duration_s, 2),
            "total_duration": round(total_duration_s, 2),
        },
    }


# ---------- Text Chunking ----------
_ws = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _ws.sub(" ", text).strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + size, n)
        chunks.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks


def list_md_files(root: str) -> List[str]:
    files = []
    if not os.path.isdir(root):
        return files
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isfile(path) and name.lower().endswith((".md", ".txt")):
            files.append(path)
    return sorted(files)


def list_ingestible_files(*dirs: Path) -> List[str]:
    """List all ingestible files (text + image formats) from given directories."""
    files = []
    allowed = TEXT_EXTENSIONS + IMAGE_EXTENSIONS
    for d in dirs:
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in allowed:
                files.append(str(f))
    return files


# ---------- Text Extraction ----------
def extract_text_docx(path: str) -> str:
    """Extract text from a .docx file (paragraphs + table cells)."""
    doc = DocxDocument(path)
    parts = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            parts.append(text)
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append("\t".join(cells))
    return "\n".join(parts)


def extract_text_xlsx(path: str) -> str:
    """Extract text from a .xlsx file, sheet by sheet, row by row."""
    wb = load_workbook(path, read_only=True, data_only=True)
    parts = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        parts.append(f"--- Sheet: {sheet_name} ---")
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            line = "\t".join(cells).strip()
            if line:
                parts.append(line)
    wb.close()
    return "\n".join(parts)


def extract_text_csv(path: str) -> str:
    """Extract text from a .csv file."""
    parts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            line = "\t".join(row).strip()
            if line:
                parts.append(line)
    return "\n".join(parts)


def ollama_vision_describe(image_path: str, filename: str) -> str:
    """Use Ollama vision model to describe an image."""
    img = Image.open(image_path)
    # Resize to max 2048px on longest side
    max_dim = 2048
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    # Convert to PNG bytes and base64 encode
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": f"Describe this image in detail. The filename is '{filename}'. "
                               "Extract any visible text, data, labels, or key information.",
                    "images": [b64],
                }
            ],
            "stream": False,
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama vision error: {resp.text}")
    return resp.json()["message"]["content"]


def extract_text_image(path: str) -> str:
    """Extract text from an image: try Tesseract OCR first, fall back to vision model."""
    filename = os.path.basename(path)
    # Try OCR first
    try:
        img = Image.open(path)
        ocr_text = pytesseract.image_to_string(img).strip()
        if len(ocr_text) >= OCR_MIN_CHARS:
            return f"[OCR from {filename}]\n{ocr_text}"
    except Exception as e:
        print(f"OCR failed for {filename}: {e}")
        ocr_text = ""

    # Fall back to vision model
    try:
        description = ollama_vision_describe(path, filename)
        prefix = f"[Vision description of {filename}]"
        if ocr_text:
            return f"{prefix}\n{description}\n\n[Partial OCR text]\n{ocr_text}"
        return f"{prefix}\n{description}"
    except Exception as e:
        print(f"Vision model failed for {filename}: {e}")
        if ocr_text:
            return f"[Partial OCR from {filename}]\n{ocr_text}"
        return ""


def extract_text_for_file(path: str) -> str:
    """Dispatch to the appropriate text extractor based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".md", ".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif ext == ".docx":
        return extract_text_docx(path)
    elif ext == ".xlsx":
        return extract_text_xlsx(path)
    elif ext == ".csv":
        return extract_text_csv(path)
    elif ext in IMAGE_EXTENSIONS:
        return extract_text_image(path)
    return ""


# ---------- Qdrant Vector Store ----------
def qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(client: QdrantClient, vector_size: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )


def upsert_chunks(
    client: QdrantClient,
    chunks: List[Tuple[str, Dict[str, Any]]],
    vectors: List[List[float]],
) -> int:
    points = []
    for idx, (chunk, meta) in enumerate(chunks):
        points.append(
            qm.PointStruct(
                id=int(time.time() * 1000) + idx,
                vector=vectors[idx],
                payload={"text": chunk, **meta},
            )
        )
    client.upsert(collection_name=COLLECTION, points=points)
    return len(points)


def search(
    client: QdrantClient, query_vec: List[float], top_k: int = TOP_K
) -> List[Dict[str, Any]]:
    hits = client.search(collection_name=COLLECTION, query_vector=query_vec, limit=top_k)
    results = []
    for h in hits:
        payload = h.payload or {}
        results.append({
            "score": h.score,
            "text": payload.get("text", ""),
            "source": payload.get("source", ""),
            "chunk_index": payload.get("chunk_index", -1),
        })
    return results


# ---------- Web Search ----------
def web_search(query: str, max_results: int = 5) -> list:
    try:
        with DDGS() as ddgs:
            # Note: DDGS may return fewer results than requested, depending on the query and availability, also added in the us as the region 
            # to improve relevance for english queries
            return list(ddgs.text(query, max_results=max_results, region="us-en"))
    except Exception as e:
        print(f"Web search error: {e}")
        return []


# ---------- Pydantic Models ----------
class AuthRequest(BaseModel):
    username: str
    password: str


class IngestResponse(BaseModel):
    files: int
    chunks: int
    points_upserted: int


class ChatRequest(BaseModel):
    convo_id: str
    question: str
    history_limit: int = 12
    top_k: int = TOP_K
    mode: str = "hybrid"
    web_search: bool = False
    web_search_max: int = 5


class ChatResponse(BaseModel):
    convo_id: str
    answer: str
    sources: List[Dict[str, Any]]
    performance: Optional[Dict[str, Any]] = None


# ---------- Health ----------
@app.get("/health")
def health():
    return {"ok": True, "collection": COLLECTION, "data_dir": DATA_DIR}


# ---------- Auth Endpoints ----------
@app.post("/auth/register")
def auth_register(req: AuthRequest, response: Response):
    username = req.username.strip()
    password = req.password

    if len(username) < 4:
        raise HTTPException(400, "Username must be at least 4 characters")
    if not re.match(r"^[a-zA-Z0-9._-]+$", username):
        raise HTTPException(400, "Username: letters, numbers, dots, underscores, hyphens only")
    if len(password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    conn = db()
    try:
        with conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, hashed, int(time.time())),
            )
        cur = conn.execute("SELECT id FROM users WHERE username=?", (username,))
        user_id = cur.fetchone()[0]
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(409, "Username already taken")
    conn.close()

    cookie = create_session_cookie(user_id, username)
    response.set_cookie(
        "merlin_session", cookie,
        httponly=True, max_age=SESSION_MAX_AGE, samesite="lax",
        path="/", secure=COOKIE_SECURE,
    )
    return {"ok": True, "user_id": user_id, "username": username}


@app.post("/auth/login")
def auth_login(req: AuthRequest, response: Response):
    username = req.username.strip()
    conn = db()
    cur = conn.execute("SELECT id, password_hash FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()

    if not row or not bcrypt.checkpw(req.password.encode("utf-8"), row[1].encode("utf-8")):
        raise HTTPException(401, "Invalid username or password")

    user_id = row[0]
    cookie = create_session_cookie(user_id, username)
    response.set_cookie(
        "merlin_session", cookie,
        httponly=True, max_age=SESSION_MAX_AGE, samesite="lax",
        path="/", secure=COOKIE_SECURE,
    )
    return {"ok": True, "user_id": user_id, "username": username}


@app.post("/auth/logout")
def auth_logout(response: Response):
    response.delete_cookie("merlin_session", path="/")
    return {"ok": True}


@app.get("/auth/me")
def auth_me(user: Dict = Depends(get_current_user)):
    return {"user_id": user["user_id"], "username": user["username"]}


# ---------- Conversation Endpoints ----------
@app.get("/conversations")
def list_conversations(user: Dict = Depends(get_current_user)):
    conn = db()
    cur = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations "
        "WHERE user_id=? ORDER BY updated_at DESC",
        (user["user_id"],),
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]}
        for r in rows
    ]


@app.post("/conversations")
def create_conversation(user: Dict = Depends(get_current_user)):
    conv_id = str(uuid.uuid4())
    now = int(time.time())
    conn = db()
    with conn:
        conn.execute(
            "INSERT INTO conversations (id, user_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conv_id, user["user_id"], "New Conversation", now, now),
        )
    conn.close()
    return {"id": conv_id, "title": "New Conversation", "created_at": now, "updated_at": now}


class RenameRequest(BaseModel):
    title: str


@app.patch("/conversations/{conv_id}")
def rename_conversation(conv_id: str, req: RenameRequest, user: Dict = Depends(get_current_user)):
    title = req.title.strip()
    if not title:
        raise HTTPException(400, "Title required")

    conn = db()
    cur = conn.execute("SELECT user_id FROM conversations WHERE id=?", (conv_id,))
    row = cur.fetchone()
    if not row or row[0] != user["user_id"]:
        conn.close()
        raise HTTPException(404, "Conversation not found")

    with conn:
        conn.execute(
            "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
            (title[:100], int(time.time()), conv_id),
        )
    conn.close()
    return {"ok": True}


@app.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str, user: Dict = Depends(get_current_user)):
    conn = db()
    cur = conn.execute("SELECT user_id FROM conversations WHERE id=?", (conv_id,))
    row = cur.fetchone()
    if not row or row[0] != user["user_id"]:
        conn.close()
        raise HTTPException(404, "Conversation not found")

    with conn:
        conn.execute("DELETE FROM messages WHERE convo_id=?", (conv_id,))
        conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
    conn.close()
    return {"ok": True}


@app.get("/conversations/{conv_id}/messages")
def get_conversation_messages(conv_id: str, user: Dict = Depends(get_current_user)):
    conn = db()
    cur = conn.execute("SELECT user_id FROM conversations WHERE id=?", (conv_id,))
    row = cur.fetchone()
    if not row or row[0] != user["user_id"]:
        conn.close()
        raise HTTPException(404, "Conversation not found")

    cur = conn.execute(
        "SELECT role, content, ts FROM messages WHERE convo_id=? ORDER BY ts ASC, id ASC",
        (conv_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "ts": r[2]} for r in rows]


# ---------- Chat Endpoint ----------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    # Authenticate: internal key or session cookie
    user = None
    internal = request.headers.get("X-Internal-Key", "")
    if INTERNAL_KEY and hmac.compare_digest(internal, INTERNAL_KEY):
        user = {"user_id": 0, "username": "system"}
    else:
        cookie = request.cookies.get("merlin_session")
        if cookie:
            user = verify_session_cookie(cookie)
        if not user:
            raise HTTPException(401, "Not authenticated")

    # Ensure conversation exists
    conn = db()
    cur = conn.execute(
        "SELECT id, user_id, title FROM conversations WHERE id=?", (req.convo_id,)
    )
    conv_row = cur.fetchone()

    if conv_row:
        if user["user_id"] != 0 and conv_row[1] != user["user_id"]:
            conn.close()
            raise HTTPException(403, "Access denied")
        conv_title = conv_row[2]
    else:
        now = int(time.time())
        with conn:
            conn.execute(
                "INSERT INTO conversations (id, user_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (req.convo_id, user["user_id"], "New Conversation", now, now),
            )
        conv_title = "New Conversation"
    conn.close()

    # Store user message
    add_message(req.convo_id, "user", req.question, user["user_id"])

    # Embed question and search
    qvec = ollama_embed([req.question])[0]
    client = qdrant_client()
    sources = search(client, qvec, top_k=req.top_k)
    top_score = sources[0]["score"] if sources else 0.0

    context_blocks = []
    for s in sources:
        if s["text"]:
            context_blocks.append(f"[{s['source']} #{s['chunk_index']}] {s['text']}")

    # Web search integration
    web_sources = []
    if req.web_search:
        print(f"WEB_SEARCH enabled; query={req.question!r}; max={req.web_search_max}")
        web_results = web_search(req.question, max_results=min(req.web_search_max, 10))
        print(f"WEB_SEARCH got {len(web_results)} results")
        for wr in web_results:
            context_blocks.append(f"[WEB: {wr['title']} | {wr['href']}] {wr['body']}")
            web_sources.append({
                "source": wr["title"],
                "url": wr["href"],
                "text": wr["body"],
                "score": None,
                "chunk_index": -1,
            })

    context = "\n\n".join(context_blocks).strip()
    has_enough_context = (len(context) >= MIN_CONTEXT_CHARS) or (top_score >= MIN_TOP_SCORE)
    if web_sources:
        has_enough_context = True

    system = (
        "You are a helpful assistant.\n"
        "You must answer in two sections:\n"
        "1) DOCUMENT-BASED ANALYSIS: Use the provided CONTEXT first. "
        "Cite sources like [filename #chunk].\n"
        "2) GENERAL KNOWLEDGE: If the context does not fully cover the question, "
        "add a second section using your own training knowledge.\n"
        "   In this section, do NOT invent citations to files. "
        "Clearly label it as general knowledge.\n"
        "Rules:\n"
        "- Never claim you read files that are not in CONTEXT.\n"
        "- If the user asks for something not in the documents, "
        "say so in section 1, then proceed in section 2.\n"
        "- If mode=grounded, ONLY produce section 1 and say what is missing.\n"
    )

    if req.web_search and web_sources:
        system += (
            "\nWEB SEARCH: Context includes web results marked [WEB: title | url].\n"
            "When citing web sources, include the full URL.\n"
            "Format web citations as: [Title](URL)\n"
            "Distinguish between document sources and web sources.\n"
        )

    history = get_recent_messages(req.convo_id, limit=req.history_limit)
    messages = [{"role": "system", "content": system}]

    mode = (req.mode or "hybrid").lower().strip()
    if mode not in ("grounded", "hybrid", "general"):
        mode = "hybrid"

    for m in history:
        if m["role"] in ("user", "assistant"):
            messages.append(m)

    if mode == "general":
        messages.append({"role": "user", "content": req.question})
    elif mode == "grounded":
        ctx = context if context else "(no retrieved context)"
        messages.append({
            "role": "user",
            "content": f"CONTEXT:\n{ctx}\n\nQUESTION:\n{req.question}\n\nMODE: grounded (docs only).",
        })
    else:
        ctx = context if context else "(no retrieved context)"
        coverage_note = (
            "Context appears relevant."
            if has_enough_context
            else "Context likely insufficient; use general knowledge for section 2."
        )
        messages.append({
            "role": "user",
            "content": f"CONTEXT:\n{ctx}\n\nQUESTION:\n{req.question}\n\nMODE: hybrid. {coverage_note}",
        })

    result = ollama_chat(messages)
    answer = result["content"]
    performance = result["performance"]
    add_message(req.convo_id, "assistant", answer, user["user_id"])

    # Auto-title and update timestamp
    conn = db()
    with conn:
        if conv_title == "New Conversation":
            title = req.question[:50].strip()
            if len(req.question) > 50:
                title += "..."
            conn.execute(
                "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                (title, int(time.time()), req.convo_id),
            )
        else:
            conn.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (int(time.time()), req.convo_id),
            )
    conn.close()

    all_sources = sources + web_sources
    return ChatResponse(convo_id=req.convo_id, answer=answer, sources=all_sources, performance=performance)


# ---------- File Endpoints ----------
@app.get("/files")
def files(user: Dict = Depends(get_optional_user)):
    def list_dir(p: Path):
        if not p.exists():
            return []
        out = []
        for f in sorted(p.glob("*")):
            if f.is_file():
                out.append({"name": f.name, "size": f.stat().st_size, "path": str(f)})
        return out

    return {"text": list_dir(TEXT_DIR), "incoming": list_dir(INCOMING_DIR)}


@app.post("/upload")
async def upload(file: UploadFile = File(...), user: Dict = Depends(get_optional_user)):
    filename = (file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    lower = filename.lower()
    ext = os.path.splitext(lower)[1]
    is_text = ext in TEXT_EXTENSIONS
    is_image = ext in IMAGE_EXTENSIONS
    # Text-based formats go to corpus (TEXT_DIR), images and unknown go to INCOMING_DIR
    dest_dir = TEXT_DIR if is_text else INCOMING_DIR
    dest_path = dest_dir / filename

    if dest_path.exists():
        raise HTTPException(status_code=409, detail=f"File already exists: {filename}")

    with dest_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    needs_conversion = not (is_text or is_image)
    return {"ok": True, "saved_to": str(dest_path), "needs_conversion": needs_conversion}


@app.post("/ingest", response_model=IngestResponse)
def ingest(user: Dict = Depends(get_optional_user)):
    all_files = list_ingestible_files(TEXT_DIR, INCOMING_DIR)
    if not all_files:
        raise HTTPException(
            status_code=400,
            detail=f"No ingestible files found in {TEXT_DIR} or {INCOMING_DIR}",
        )

    chunks: List[Tuple[str, Dict[str, Any]]] = []
    errors = []
    for f in all_files:
        try:
            content = extract_text_for_file(f)
            content = normalize(content)
            if not content:
                continue
            ext = os.path.splitext(f)[1].lower()
            pieces = chunk_text(content)
            for i, p in enumerate(pieces):
                chunks.append((
                    p,
                    {"source": os.path.basename(f), "chunk_index": i, "format": ext},
                ))
        except Exception as e:
            errors.append(f"{os.path.basename(f)}: {e}")
            print(f"Ingest error for {f}: {e}")

    if errors:
        print(f"Ingest completed with {len(errors)} error(s): {errors}")

    if not chunks:
        detail = "No chunks produced."
        if errors:
            detail += f" Errors: {'; '.join(errors)}"
        raise HTTPException(status_code=400, detail=detail)

    first_vec = ollama_embed([chunks[0][0]])[0]
    client = qdrant_client()
    ensure_collection(client, vector_size=len(first_vec))

    texts = [c[0] for c in chunks]
    vecs = ollama_embed(texts)
    upserted = upsert_chunks(client, chunks, vecs)
    return IngestResponse(files=len(all_files), chunks=len(chunks), points_upserted=upserted)


# ---------- Conversation Export ----------
def cleanup_exports():
    """Delete export files older than EXPORT_MAX_AGE seconds."""
    now = time.time()
    if not EXPORTS_DIR.exists():
        return
    for f in EXPORTS_DIR.iterdir():
        if f.is_file() and (now - f.stat().st_mtime) > EXPORT_MAX_AGE:
            try:
                f.unlink()
            except OSError:
                pass


def export_messages_csv(messages: List[Dict], title: str) -> Path:
    """Export messages as CSV file."""
    cleanup_exports()
    safe_title = re.sub(r'[^\w\s-]', '', title)[:40].strip() or "conversation"
    filepath = EXPORTS_DIR / f"{safe_title}_{int(time.time())}.csv"
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Role", "Content", "Timestamp"])
        for m in messages:
            ts = datetime.fromtimestamp(m["ts"]).isoformat() if m.get("ts") else ""
            writer.writerow([m["role"], m["content"], ts])
    return filepath


def export_messages_txt(messages: List[Dict], title: str) -> Path:
    """Export messages as plaintext transcript."""
    cleanup_exports()
    safe_title = re.sub(r'[^\w\s-]', '', title)[:40].strip() or "conversation"
    filepath = EXPORTS_DIR / f"{safe_title}_{int(time.time())}.txt"
    lines = [f"Conversation: {title}", f"Exported: {datetime.now().isoformat()}", "=" * 60, ""]
    for m in messages:
        ts = datetime.fromtimestamp(m["ts"]).isoformat() if m.get("ts") else ""
        role_label = "You" if m["role"] == "user" else "Merlin"
        lines.append(f"[{role_label}] ({ts})")
        lines.append(m["content"])
        lines.append("")
    filepath.write_text("\n".join(lines), encoding="utf-8")
    return filepath


def export_messages_docx(messages: List[Dict], title: str) -> Path:
    """Export messages as formatted Word document."""
    cleanup_exports()
    safe_title = re.sub(r'[^\w\s-]', '', title)[:40].strip() or "conversation"
    filepath = EXPORTS_DIR / f"{safe_title}_{int(time.time())}.docx"
    doc = DocxDocument()
    doc.add_heading(title, level=1)
    doc.add_paragraph(f"Exported: {datetime.now().isoformat()}")
    for m in messages:
        ts = datetime.fromtimestamp(m["ts"]).isoformat() if m.get("ts") else ""
        role_label = "You" if m["role"] == "user" else "Merlin"
        doc.add_heading(f"{role_label} ({ts})", level=2)
        doc.add_paragraph(m["content"])
    doc.save(str(filepath))
    return filepath


@app.get("/export/{conv_id}")
def export_conversation(conv_id: str, format: str = "txt", user: Dict = Depends(get_current_user)):
    conn = db()
    cur = conn.execute("SELECT user_id, title FROM conversations WHERE id=?", (conv_id,))
    row = cur.fetchone()
    if not row or row[0] != user["user_id"]:
        conn.close()
        raise HTTPException(404, "Conversation not found")
    title = row[1]

    cur = conn.execute(
        "SELECT role, content, ts FROM messages WHERE convo_id=? ORDER BY ts ASC, id ASC",
        (conv_id,),
    )
    messages = [{"role": r[0], "content": r[1], "ts": r[2]} for r in cur.fetchall()]
    conn.close()

    if not messages:
        raise HTTPException(400, "No messages to export")

    fmt = format.lower().strip()
    if fmt == "csv":
        filepath = export_messages_csv(messages, title)
        media = "text/csv"
    elif fmt == "docx":
        filepath = export_messages_docx(messages, title)
        media = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        filepath = export_messages_txt(messages, title)
        media = "text/plain"

    return FileResponse(
        str(filepath),
        media_type=media,
        filename=filepath.name,
        headers={"Content-Disposition": f'attachment; filename="{filepath.name}"'},
    )


# ---------- LLM File Generation ----------
class FileGenerateRequest(BaseModel):
    convo_id: Optional[str] = None
    prompt: str
    format: str = "txt"  # csv, txt, docx, png
    context_messages: int = 10


def generate_file_content(prompt: str, fmt: str, context: List[Dict[str, str]]) -> Dict:
    """Ask LLM to generate file content as structured JSON."""
    context_text = ""
    if context:
        context_text = "\n\nConversation context:\n"
        for m in context:
            context_text += f"{m['role']}: {m['content']}\n"

    if fmt == "png":
        system_msg = (
            "You generate chart specifications as JSON. "
            "Return ONLY valid JSON with no markdown fences. "
            "Format: {\"chart_type\": \"bar|line|pie|scatter\", \"title\": \"...\", "
            "\"data\": {\"labels\": [...], \"values\": [...]}, \"xlabel\": \"...\", \"ylabel\": \"...\", "
            "\"filename_hint\": \"...\"}"
        )
    else:
        system_msg = (
            "You generate file content as JSON. "
            "Return ONLY valid JSON with no markdown fences. "
            f"Format: {{\"content\": \"the {fmt} content as a string\", \"filename_hint\": \"short_name\"}}"
        )
        if fmt == "csv":
            system_msg += "\nFor CSV, the content should be valid CSV text with headers."

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt + context_text},
    ]
    raw = ollama_chat(messages)["content"]
    # Extract JSON from LLM response which may contain extra text or markdown fences
    raw = raw.strip()

    # Try 1: Strip markdown fences (```json ... ``` anywhere in the response)
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)```', raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()

    # Try 2: If still not valid JSON, find the first { ... } or [ ... ] block
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Find the outermost JSON object in the response
        brace_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if brace_match:
            return json.loads(brace_match.group(0))
        raise


def render_chart_png(spec: Dict) -> Path:
    """Render a chart from a specification dict using matplotlib."""
    cleanup_exports()
    chart_type = spec.get("chart_type", "bar")
    title = spec.get("title", "Chart")
    data = spec.get("data", {})
    labels = data.get("labels", [])
    values = data.get("values", [])
    xlabel = spec.get("xlabel", "")
    ylabel = spec.get("ylabel", "")
    hint = spec.get("filename_hint", "chart")
    safe_hint = re.sub(r'[^\w\s-]', '', hint)[:30].strip() or "chart"

    fig, ax = plt.subplots(figsize=(10, 6))

    if chart_type == "pie":
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
    elif chart_type == "line":
        ax.plot(labels, values, marker='o', linewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    elif chart_type == "scatter":
        ax.scatter(labels, values)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:  # bar
        ax.bar(labels, values)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.set_title(title)
    plt.tight_layout()

    filepath = EXPORTS_DIR / f"{safe_hint}_{int(time.time())}.png"
    fig.savefig(str(filepath), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return filepath


def markdown_to_docx(text: str) -> DocxDocument:
    """Convert basic markdown text to a DOCX document."""
    doc = DocxDocument()
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            doc.add_paragraph("")
            continue
        if stripped.startswith("### "):
            doc.add_heading(stripped[4:], level=3)
        elif stripped.startswith("## "):
            doc.add_heading(stripped[3:], level=2)
        elif stripped.startswith("# "):
            doc.add_heading(stripped[2:], level=1)
        elif stripped.startswith(("- ", "* ", "+ ")):
            doc.add_paragraph(stripped[2:], style='List Bullet')
        elif re.match(r'^\d+\.\s', stripped):
            doc.add_paragraph(re.sub(r'^\d+\.\s', '', stripped), style='List Number')
        else:
            doc.add_paragraph(stripped)
    return doc


@app.post("/generate-file")
def generate_file(req: FileGenerateRequest, user: Dict = Depends(get_current_user)):
    fmt = req.format.lower().strip()
    if fmt not in ("csv", "txt", "docx", "png"):
        raise HTTPException(400, "Format must be csv, txt, docx, or png")

    # Load conversation context if provided
    context = []
    if req.convo_id:
        conn = db()
        cur = conn.execute(
            "SELECT role, content FROM messages WHERE convo_id=? ORDER BY ts DESC, id DESC LIMIT ?",
            (req.convo_id, req.context_messages),
        )
        rows = cur.fetchall()
        conn.close()
        context = [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    # Try up to 3 attempts for JSON parsing
    spec = None
    for attempt in range(3):
        try:
            spec = generate_file_content(req.prompt, fmt, context)
            break
        except json.JSONDecodeError as e:
            print(f"[generate-file] JSON parse attempt {attempt+1} failed: {e}")
            if attempt < 2:
                continue
            raise HTTPException(500, f"Failed to generate valid file content from LLM: {e}")

    if spec is None:
        raise HTTPException(500, "Failed to generate file content")

    cleanup_exports()

    if fmt == "png":
        filepath = render_chart_png(spec)
        return FileResponse(
            str(filepath),
            media_type="image/png",
            filename=filepath.name,
            headers={"Content-Disposition": f'attachment; filename="{filepath.name}"'},
        )

    content = spec.get("content", "")
    hint = spec.get("filename_hint", "generated")
    safe_hint = re.sub(r'[^\w\s-]', '', hint)[:30].strip() or "generated"

    if fmt == "csv":
        filepath = EXPORTS_DIR / f"{safe_hint}_{int(time.time())}.csv"
        filepath.write_text(content, encoding="utf-8")
        return FileResponse(
            str(filepath),
            media_type="text/csv",
            filename=filepath.name,
            headers={"Content-Disposition": f'attachment; filename="{filepath.name}"'},
        )
    elif fmt == "docx":
        filepath = EXPORTS_DIR / f"{safe_hint}_{int(time.time())}.docx"
        doc = markdown_to_docx(content)
        doc.save(str(filepath))
        return FileResponse(
            str(filepath),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=filepath.name,
            headers={"Content-Disposition": f'attachment; filename="{filepath.name}"'},
        )
    else:  # txt
        filepath = EXPORTS_DIR / f"{safe_hint}_{int(time.time())}.txt"
        filepath.write_text(content, encoding="utf-8")
        return FileResponse(
            str(filepath),
            media_type="text/plain",
            filename=filepath.name,
            headers={"Content-Disposition": f'attachment; filename="{filepath.name}"'},
        )


# ---------- UI ----------
@app.get("/", response_class=HTMLResponse)
def ui():
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>MERLIN &mdash; Knowledge Retrieval System</title>
<link href="https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;0,700;1,400&display=swap" rel="stylesheet"/>
<style>
:root{
  --bg:#0f1a14;
  --panel-dark:#132018;
  --panel:#1b2a20;
  --surface:#223527;
  --hover:#2f4a37;
  --scroll:#3c5a44;
  --gold-bright:#e0c26a;
  --gold-warm:#c2a55f;
  --gold-dim:#9b7f3c;
  --gold-glow:rgba(224,194,106,0.16);
  --crimson:#7a2b2b;
  --crimson-soft:#a24a4a;
  --verdigris:#4f8f6b;
  --text:#e7e3d7;
  --text-sec:#b6b1a5;
  --text-dim:#8a8577;
  --text-bright:#f6f2e6;
  --parchment:#f3ead6;
  --border:rgba(194,165,95,0.14);
  --border-sub:rgba(205,187,155,0.08);
  --border-orn:rgba(224,194,106,0.26);
  --shadow:0 10px 36px rgba(0,0,0,0.62);
  --shadow-sm:0 3px 14px rgba(0,0,0,0.46);
  --shadow-glow:0 0 26px rgba(224,194,106,0.10);
  --danger:#a24a4a;
  --success:#4f8f6b;
  --radius:4px;
  --font-h:'Crimson Text','Palatino Linotype','Book Antiqua',Georgia,serif;
  --font-ui:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  --font-mono:'SF Mono','Cascadia Code',Consolas,'Liberation Mono',monospace;
  --tr:0.2s ease;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{font-size:15px;-webkit-font-smoothing:antialiased}
body{font-family:var(--font-h);background:var(--bg);color:var(--text);line-height:1.6;min-height:100vh}
a{color:var(--gold-warm);text-decoration:none}
a:hover{color:var(--gold-bright)}
.hidden{display:none!important}

/* ---- Login ---- */
.login-view{
  display:flex;align-items:center;justify-content:center;min-height:100vh;
  background:radial-gradient(ellipse 800px 600px at 50% 35%,rgba(212,168,67,0.04),transparent),
             radial-gradient(ellipse 400px 300px at 50% 50%,rgba(30,27,24,0.8),transparent),
             var(--bg);
}
.login-card{
  width:100%;max-width:420px;padding:52px 44px 44px;
  background:linear-gradient(180deg,rgba(30,27,24,0.95),rgba(20,18,16,0.98));
  border:1px solid var(--border-orn);border-top:2px solid var(--gold-dim);
  box-shadow:var(--shadow),var(--shadow-glow);position:relative;
}
.login-card::before,.login-card::after{
  content:'';position:absolute;width:20px;height:20px;border-color:var(--gold-dim);opacity:0.5;
}
.login-card::before{top:8px;left:8px;border-top:1px solid;border-left:1px solid}
.login-card::after{top:8px;right:8px;border-top:1px solid;border-right:1px solid}
.corner-bl,.corner-br{position:absolute;width:20px;height:20px;border-color:var(--gold-dim);opacity:0.5}
.corner-bl{bottom:8px;left:8px;border-bottom:1px solid;border-left:1px solid}
.corner-br{bottom:8px;right:8px;border-bottom:1px solid;border-right:1px solid}
.login-brand{text-align:center;margin-bottom:36px}
.login-crest{width:56px;height:56px;margin:0 auto 16px;color:var(--gold-warm);opacity:0.8}
.login-wordmark{
  font-family:var(--font-h);font-size:2.6rem;font-weight:700;
  letter-spacing:0.28em;color:var(--parchment);
  text-shadow:0 0 40px rgba(212,168,67,0.12);
}
.login-accent{width:80px;height:1px;background:linear-gradient(90deg,transparent,var(--gold-warm),transparent);margin:14px auto}
.login-tagline{font-family:var(--font-h);font-size:0.85rem;font-style:italic;color:var(--text-sec);letter-spacing:0.06em}

.auth-tabs{display:flex;gap:0;margin-bottom:28px;border-bottom:1px solid var(--border)}
.auth-tab{
  flex:1;padding:10px;background:none;border:none;border-bottom:2px solid transparent;
  font-family:var(--font-h);font-size:0.95rem;color:var(--text-dim);cursor:pointer;
  transition:all var(--tr);
}
.auth-tab:hover{color:var(--text-sec)}
.auth-tab.active{color:var(--gold-warm);border-bottom-color:var(--gold-warm)}

.form-group{margin-bottom:20px}
.form-label{
  display:block;font-family:var(--font-ui);font-size:0.7rem;font-weight:600;
  color:var(--text-dim);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;
}
.form-input{
  width:100%;padding:11px 14px;background:var(--panel-dark);
  border:1px solid var(--border);border-bottom:1px solid var(--border-orn);
  color:var(--text);font-family:var(--font-h);font-size:1rem;
  outline:none;transition:border-color var(--tr),box-shadow var(--tr);
}
.form-input:focus{border-color:var(--gold-dim);box-shadow:0 1px 0 0 var(--gold-dim)}
.form-input::placeholder{color:var(--text-dim);font-style:italic}
.login-error{font-family:var(--font-ui);font-size:0.8rem;color:var(--crimson-soft);min-height:1.3em;margin-bottom:8px}

/* ---- Buttons ---- */
.btn{
  display:inline-flex;align-items:center;justify-content:center;gap:6px;
  padding:10px 22px;font-family:var(--font-ui);font-size:0.8rem;font-weight:600;
  letter-spacing:0.04em;border:1px solid var(--border);background:var(--surface);
  color:var(--text);cursor:pointer;transition:all var(--tr);white-space:nowrap;
  text-transform:uppercase;border-radius:var(--radius);
}
.btn:hover{background:var(--hover);border-color:var(--border-orn)}
.btn:disabled{opacity:0.4;cursor:not-allowed}
.btn--primary{
  background:linear-gradient(180deg,var(--gold-warm),var(--gold-dim));
  color:var(--bg);border:1px solid var(--gold-warm);
  position:relative;overflow:hidden;
}
.btn--primary:hover{
  background:linear-gradient(180deg,var(--gold-bright),var(--gold-warm));
  box-shadow:0 0 12px rgba(224,194,106,0.35),0 0 30px rgba(224,194,106,0.15);
  transform:translateY(-1px);
}
.btn--primary::after{
  content:'';position:absolute;top:0;left:-75%;width:50%;height:100%;
  background:linear-gradient(120deg,transparent,rgba(255,255,255,0.4),transparent);
  transform:skewX(-20deg);
}
.btn--primary:hover::after{animation:shimmer 0.9s ease forwards}
@keyframes shimmer{100%{left:125%}}
.btn--full{width:100%}
.btn--sm{padding:6px 14px;font-size:0.7rem}
.btn--danger{border-color:var(--danger);color:var(--crimson-soft)}
.btn--danger:hover{background:rgba(122,43,43,0.2)}
.btn-icon{
  display:inline-flex;align-items:center;justify-content:center;
  width:28px;height:28px;border:1px solid transparent;background:transparent;
  color:var(--text-dim);cursor:pointer;transition:all var(--tr);border-radius:var(--radius);
  padding:0;
}
.btn-icon:hover{color:var(--gold-warm);border-color:var(--border-orn);background:var(--surface)}

/* ---- App Layout ---- */
.app-view{
  display:grid;
  grid-template-columns:260px 1fr 300px;
  grid-template-rows:52px 1fr;
  height:100vh;
}
.app-header{
  grid-column:1/-1;display:flex;align-items:center;justify-content:space-between;
  padding:0 24px;height:52px;background:var(--panel-dark);
  border-bottom:1px solid var(--border);position:relative;
}
.app-header::after{
  content:'';position:absolute;bottom:-1px;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--gold-dim),transparent);opacity:0.3;
}
.header-brand{display:flex;align-items:center;gap:12px}
.header-crest{width:24px;height:24px;color:var(--gold-warm);opacity:0.8}
.header-title{font-family:var(--font-h);font-size:1.1rem;font-weight:700;letter-spacing:0.18em;color:var(--parchment)}
.header-sep{color:var(--text-dim);margin:0 4px}
.header-subtitle{font-family:var(--font-h);font-size:0.8rem;font-style:italic;color:var(--text-dim)}
.header-right{display:flex;align-items:center;gap:14px}
.header-user{font-family:var(--font-h);font-size:0.85rem;font-style:italic;color:var(--text-sec)}
.hamburger{display:none;background:none;border:none;color:var(--text-sec);cursor:pointer;padding:4px;font-size:1.2rem}
.panel-toggle{display:none;background:none;border:none;color:var(--text-sec);cursor:pointer;padding:4px;font-size:1rem}

/* ---- Conversation Sidebar ---- */
.conv-sidebar{
  background:linear-gradient(180deg,var(--panel-dark),var(--panel) 50%,var(--panel-dark));
  border-right:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden;
}
.conv-sidebar-header{padding:16px;flex-shrink:0}
.conv-list{flex:1;overflow-y:auto;padding:0 8px 16px}
.conv-group-label{
  font-family:var(--font-ui);font-size:0.6rem;font-weight:700;text-transform:uppercase;
  letter-spacing:0.14em;color:var(--text-dim);padding:14px 12px 6px;
}
.conv-item{
  display:flex;align-items:center;gap:6px;padding:9px 12px;margin-bottom:2px;
  cursor:pointer;transition:all var(--tr);border-radius:var(--radius);
  border-left:2px solid transparent;position:relative;
}
.conv-item:hover{background:var(--surface);color:var(--text)}
.conv-item.active{color:var(--gold-warm);background:rgba(212,168,67,0.06);border-left-color:var(--gold-warm)}
.conv-item.active .conv-title{color:var(--gold-warm)}
.conv-title{
  flex:1;font-family:var(--font-h);font-size:0.9rem;color:var(--text-sec);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.conv-actions{display:none;gap:2px;flex-shrink:0}
.conv-item:hover .conv-actions{display:flex}
.conv-rename-input{
  flex:1;padding:2px 6px;background:var(--panel-dark);border:1px solid var(--gold-dim);
  color:var(--text);font-family:var(--font-h);font-size:0.9rem;outline:none;
  border-radius:var(--radius);
}
.conv-empty{padding:24px 16px;text-align:center;color:var(--text-dim);font-style:italic;font-size:0.85rem}

/* ---- Chat Area ---- */
.chat-main{display:flex;flex-direction:column;overflow:hidden;
  background:radial-gradient(ellipse 600px 400px at 30% 30%,rgba(212,168,67,0.02),transparent),var(--bg);
}
.chat-messages{
  flex:1;overflow-y:auto;padding:24px 28px;scroll-behavior:smooth;
  background:repeating-linear-gradient(180deg,transparent,transparent 3px,rgba(200,184,152,0.008) 3px,rgba(200,184,152,0.008) 4px);
}
.chat-empty{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  height:100%;color:var(--text-dim);text-align:center;gap:12px;
}
.chat-empty-icon{font-size:3rem;opacity:0.3}
.chat-empty-title{font-family:var(--font-h);font-size:1.3rem;color:var(--text-sec)}
.chat-empty-hint{font-size:0.85rem;font-style:italic}
.chat-message{display:flex;gap:14px;margin-bottom:24px;max-width:82%}
.chat-message--user{margin-left:auto;flex-direction:row-reverse}
.chat-avatar{
  flex-shrink:0;width:34px;height:34px;display:flex;align-items:center;justify-content:center;
  font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;
  border-radius:var(--radius);
}
.chat-message--user .chat-avatar{
  background:var(--surface);color:var(--text-sec);border:1px solid var(--border);font-family:var(--font-ui);
}
.chat-message--merlin .chat-avatar{
  background:linear-gradient(135deg,var(--gold-dim),var(--gold-warm));color:var(--bg);
  font-family:var(--font-h);font-size:1rem;font-weight:700;
  box-shadow:0 0 12px rgba(212,168,67,0.15);
}
.chat-body{flex:1;min-width:0}
.chat-bubble{
  padding:14px 18px;line-height:1.65;white-space:pre-wrap;word-break:break-word;
  font-size:0.92rem;border-radius:var(--radius);
}
.chat-message--user .chat-bubble{
  background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--gold-dim);
}
.chat-message--merlin .chat-bubble{
  background:var(--panel);border:1px solid var(--border-sub);
  border-left:3px solid rgba(212,168,67,0.2);color:var(--parchment);
  box-shadow:0 0 25px rgba(224,194,106,0.10);
}
.chat-bubble--error{border-color:var(--crimson)!important;border-left-color:var(--crimson)!important;color:var(--crimson-soft)}

.chat-sources{margin-top:10px;border:1px solid var(--border-sub);border-top:1px solid var(--border);padding:10px 14px;background:rgba(20,18,16,0.6);border-radius:var(--radius)}
.chat-sources summary{font-family:var(--font-h);font-size:0.8rem;font-style:italic;color:var(--text-dim);cursor:pointer;user-select:none}
.chat-sources summary:hover{color:var(--text-sec)}
.chat-sources-list{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.source-pill{
  font-family:var(--font-ui);font-size:0.68rem;padding:4px 10px;
  border:1px solid var(--border);background:var(--panel-dark);color:var(--text-sec);border-radius:var(--radius);
}
.source-pill strong{color:var(--parchment)}
.source-pill a{color:var(--gold-warm);text-decoration:none}
.source-pill a:hover{color:var(--gold-bright);text-decoration:underline}
.source-score{color:var(--gold-warm);margin-left:4px}
.source-pill--web{border-color:var(--verdigris);background:rgba(79,143,107,0.10)}
.source-pill--web strong{color:var(--verdigris)}

/* ---- Chat Toolbar ---- */
.chat-toolbar{
  display:flex;flex-wrap:wrap;align-items:center;gap:10px;
  padding:8px 16px;background:var(--panel-dark);
  border-bottom:1px solid var(--border);
  font-family:var(--font-ui);font-size:0.7rem;
}
.toolbar-group{display:flex;align-items:center;gap:4px}
.toolbar-label{color:var(--text-dim);font-weight:600;text-transform:uppercase;letter-spacing:0.06em;font-size:0.6rem}
.toolbar-select{
  padding:3px 6px;background:var(--panel);border:1px solid var(--border-sub);
  border-bottom:1px solid var(--border);color:var(--text);
  font-family:var(--font-h);font-size:0.75rem;outline:none;
  border-radius:var(--radius);-webkit-appearance:none;
}
.toolbar-select:focus{border-color:var(--gold-dim)}
.toolbar-input{
  width:48px;padding:3px 6px;background:var(--panel);border:1px solid var(--border-sub);
  border-bottom:1px solid var(--border);color:var(--text);
  font-family:var(--font-h);font-size:0.75rem;outline:none;
  border-radius:var(--radius);text-align:center;
}
.toolbar-input:focus{border-color:var(--gold-dim)}
.toolbar-sep{width:1px;height:18px;background:var(--border);margin:0 4px}
.toolbar-check{display:flex;align-items:center;gap:4px;cursor:pointer}
.toolbar-check input[type=checkbox]{
  accent-color:var(--verdigris);width:14px;height:14px;cursor:pointer;
}
.toolbar-check span{color:var(--text-sec);font-size:0.7rem}

@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}
.thinking{animation:pulse 1.5s ease-in-out infinite}

/* ---- Composer ---- */
.chat-composer{
  padding:16px 28px;border-top:1px solid var(--border);background:var(--panel-dark);position:relative;
}
.chat-composer::before{
  content:'';position:absolute;top:-1px;left:20%;right:20%;height:1px;
  background:linear-gradient(90deg,transparent,var(--gold-dim),transparent);opacity:0.3;
}
.chat-input-wrap{display:flex;gap:12px;align-items:flex-end}
.chat-textarea{
  flex:1;min-height:46px;max-height:160px;padding:11px 14px;
  background:var(--panel);border:1px solid var(--border);color:var(--text);
  font-family:var(--font-h);font-size:0.95rem;line-height:1.5;resize:none;outline:none;
  transition:border-color var(--tr);border-radius:var(--radius);
}
.chat-textarea:focus{border-color:var(--gold-dim);box-shadow:0 0 0 3px rgba(224,194,106,0.08)}
.chat-textarea::placeholder{color:var(--text-dim);font-style:italic}
.chat-actions{display:flex;gap:8px;margin-top:8px;align-items:center}
.chat-hint{font-family:var(--font-ui);font-size:0.65rem;color:var(--text-dim);margin-left:auto}
.chat-status{font-family:var(--font-h);font-size:0.8rem;font-style:italic;color:var(--gold-warm)}

/* ---- Right Panel ---- */
.right-panel{
  background:linear-gradient(180deg,var(--panel-dark),var(--panel) 50%,var(--panel-dark));
  border-left:1px solid var(--border);display:flex;flex-direction:column;overflow-y:auto;
}
.panel-section{padding:16px}
.panel-title{
  font-family:var(--font-h);font-size:0.8rem;font-style:italic;color:var(--text-dim);
  margin-bottom:12px;text-align:center;
}
.panel-divider{height:1px;margin:0 16px;background:linear-gradient(90deg,transparent,var(--border),transparent)}
.settings-row{display:flex;gap:8px;margin-bottom:10px}
.settings-row:last-child{margin-bottom:0}
.settings-field{flex:1}
.settings-label{
  display:block;font-family:var(--font-ui);font-size:0.6rem;font-weight:600;
  text-transform:uppercase;letter-spacing:0.08em;color:var(--text-dim);margin-bottom:4px;
}
.settings-input,.settings-select{
  width:100%;padding:6px 10px;background:var(--panel-dark);
  border:1px solid var(--border-sub);border-bottom:1px solid var(--border);
  color:var(--text);font-family:var(--font-h);font-size:0.8rem;outline:none;
  border-radius:var(--radius);-webkit-appearance:none;
}
.settings-input:focus,.settings-select:focus{border-color:var(--gold-dim);box-shadow:0 0 0 3px rgba(224,194,106,0.08)}

.panel-actions{display:flex;gap:8px;margin-bottom:12px}
.drop-zone{
  border:2px dashed var(--border-orn);padding:20px 12px;text-align:center;
  background:rgba(30,27,24,0.5);transition:all var(--tr);cursor:pointer;
  border-radius:var(--radius);font-size:0.8rem;color:var(--text-dim);font-style:italic;
}
.drop-zone.active{border-color:var(--gold-warm);background:rgba(212,168,67,0.04)}
.file-label{color:var(--gold-warm);cursor:pointer;text-decoration:underline}
.upload-status{font-family:var(--font-h);font-size:0.8rem;font-style:italic;color:var(--text-sec);margin-top:8px}
.upload-status--ok{color:var(--success)}
.upload-status--err{color:var(--crimson-soft)}
.file-list{margin-top:12px;max-height:300px;overflow-y:auto}
.file-section-title{
  font-family:var(--font-ui);font-size:0.6rem;font-weight:700;text-transform:uppercase;
  letter-spacing:0.1em;color:var(--text-dim);margin:10px 0 6px;padding-bottom:4px;
  border-bottom:1px solid var(--border-sub);
}
.file-item{
  display:flex;justify-content:space-between;align-items:center;padding:5px 0;
  font-size:0.78rem;border-bottom:1px solid var(--border-sub);
}
.file-item:last-child{border-bottom:none}
.file-name{color:var(--text);font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}
.file-size{color:var(--text-dim);font-family:var(--font-mono);font-size:0.68rem;margin-left:8px}

.health-bar{
  padding:12px 16px;margin-top:auto;border-top:1px solid var(--border-sub);
  font-family:var(--font-ui);font-size:0.65rem;color:var(--text-dim);text-align:center;
}

/* ---- Scrollbar ---- */
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--scroll);border-radius:0}
::-webkit-scrollbar-thumb:hover{background:var(--hover)}

/* ---- Performance Indicator ---- */
.chat-perf{
  display:flex;gap:10px;margin-top:6px;font-family:var(--font-mono);font-size:0.65rem;color:var(--text-dim);
}
.chat-perf span{display:inline-flex;align-items:center;gap:3px}
.perf-value{color:var(--gold-warm);font-weight:600}

/* ---- Export Dropdown ---- */
.export-dropdown{
  position:absolute;z-index:50;background:var(--panel-dark);border:1px solid var(--border-orn);
  padding:6px 0;min-width:120px;box-shadow:var(--shadow-sm);border-radius:var(--radius);
}
.export-dropdown a{
  display:block;padding:6px 14px;font-family:var(--font-ui);font-size:0.75rem;
  color:var(--text);text-decoration:none;transition:background var(--tr);
}
.export-dropdown a:hover{background:var(--surface);color:var(--gold-warm)}

/* ---- Generate Modal ---- */
.generate-overlay{
  position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.6);
  z-index:200;display:flex;align-items:center;justify-content:center;
}
.generate-modal{
  background:var(--panel-dark);border:1px solid var(--border-orn);
  padding:28px;width:90%;max-width:480px;box-shadow:var(--shadow);
  border-radius:var(--radius);
}
.generate-modal h3{
  font-family:var(--font-h);font-size:1.1rem;color:var(--parchment);margin-bottom:16px;
}
.generate-modal textarea{
  width:100%;min-height:80px;padding:10px;background:var(--panel);
  border:1px solid var(--border);color:var(--text);font-family:var(--font-h);
  font-size:0.9rem;resize:vertical;outline:none;margin-bottom:12px;
  border-radius:var(--radius);
}
.generate-modal textarea:focus{border-color:var(--gold-dim)}
.generate-modal select{
  width:100%;padding:8px;background:var(--panel);border:1px solid var(--border);
  color:var(--text);font-family:var(--font-h);font-size:0.85rem;margin-bottom:16px;
  outline:none;border-radius:var(--radius);-webkit-appearance:none;
}
.generate-modal select:focus{border-color:var(--gold-dim)}
.generate-modal .modal-actions{display:flex;gap:10px;justify-content:flex-end}

/* ---- Responsive ---- */
@media(max-width:1024px){
  .app-view{grid-template-columns:220px 1fr 260px}
}
@media(max-width:768px){
  .app-view{grid-template-columns:1fr;grid-template-rows:52px 1fr}
  .conv-sidebar{display:none;position:fixed;top:52px;left:0;bottom:0;width:280px;z-index:100;
    box-shadow:4px 0 20px rgba(0,0,0,0.5)}
  .conv-sidebar.open{display:flex}
  .right-panel{display:none;position:fixed;top:52px;right:0;bottom:0;width:300px;z-index:100;
    box-shadow:-4px 0 20px rgba(0,0,0,0.5)}
  .right-panel.open{display:flex}
  .hamburger{display:block}
  .panel-toggle{display:block}
  .chat-message{max-width:95%}
  .header-subtitle{display:none}
}
</style>
</head>
<body>

<!-- ==================== LOGIN VIEW ==================== -->
<div id="login-view" class="login-view">
  <div class="login-card">
    <div class="corner-bl"></div>
    <div class="corner-br"></div>
    <div class="login-brand">
      <svg class="login-crest" viewBox="0 0 56 56" fill="none">
        <path d="M28 6l5 14h14l-11 8 4 14-12-9-12 9 4-14L9 20h14z" stroke="currentColor" stroke-width="1.5" fill="currentColor" fill-opacity="0.12"/>
        <line x1="28" y1="38" x2="28" y2="52" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/>
      </svg>
      <div class="login-wordmark">MERLIN</div>
      <div class="login-accent"></div>
      <div class="login-tagline">Knowledge Retrieval System</div>
    </div>

    <div class="auth-tabs">
      <button class="auth-tab active" data-tab="login">Login</button>
      <button class="auth-tab" data-tab="register">Register</button>
    </div>

    <form id="login-form">
      <div class="form-group">
        <label class="form-label">Username</label>
        <input class="form-input" type="text" id="login-username" placeholder="Enter username" autocomplete="username"/>
      </div>
      <div class="form-group">
        <label class="form-label">Password</label>
        <input class="form-input" type="password" id="login-password" placeholder="Enter password" autocomplete="current-password"/>
      </div>
      <div class="login-error" id="login-error"></div>
      <button type="submit" class="btn btn--primary btn--full">Enter the Archive</button>
    </form>

    <form id="register-form" class="hidden">
      <div class="form-group">
        <label class="form-label">Username</label>
        <input class="form-input" type="text" id="reg-username" placeholder="Choose a username (4+ chars)" autocomplete="username"/>
      </div>
      <div class="form-group">
        <label class="form-label">Password</label>
        <input class="form-input" type="password" id="reg-password" placeholder="Choose a password (8+ chars)" autocomplete="new-password"/>
      </div>
      <div class="form-group">
        <label class="form-label">Confirm Password</label>
        <input class="form-input" type="password" id="reg-confirm" placeholder="Confirm password" autocomplete="new-password"/>
      </div>
      <div class="login-error" id="register-error"></div>
      <button type="submit" class="btn btn--primary btn--full">Create Account</button>
    </form>
  </div>
</div>

<!-- ==================== APP VIEW ==================== -->
<div id="app-view" class="app-view hidden">

  <!-- Header -->
  <header class="app-header">
    <div class="header-brand">
      <button class="hamburger" id="hamburger-btn" title="Toggle sidebar">&#9776;</button>
      <svg class="header-crest" viewBox="0 0 24 24" fill="none">
        <path d="M12 2l2.2 6.8H21l-5.5 4 2.1 6.5L12 15l-5.6 4.3 2.1-6.5L3 8.8h6.8z" stroke="currentColor" stroke-width="1.2" fill="currentColor" fill-opacity="0.15"/>
        <line x1="12" y1="17" x2="12" y2="22" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
      </svg>
      <span class="header-title">MERLIN</span>
      <span class="header-sep">&middot;</span>
      <span class="header-subtitle">Knowledge Retrieval System</span>
    </div>
    <div class="header-right">
      <span class="header-user" id="display-username"></span>
      <button class="panel-toggle" id="panel-toggle-btn" title="Toggle settings">&#9881;</button>
      <button class="btn btn--sm" id="logout-btn">Logout</button>
    </div>
  </header>

  <!-- Conversation Sidebar -->
  <aside class="conv-sidebar" id="conv-sidebar">
    <div class="conv-sidebar-header">
      <button class="btn btn--primary btn--full" id="new-conv-btn">+ New Conversation</button>
    </div>
    <div class="conv-list" id="conv-list">
      <div class="conv-empty">No conversations yet</div>
    </div>
  </aside>

  <!-- Chat Area -->
  <main class="chat-main">
    <div class="chat-toolbar">
      <div class="toolbar-group">
        <span class="toolbar-label">Mode</span>
        <select class="toolbar-select" id="mode-select">
          <option value="hybrid" selected>Hybrid</option>
          <option value="grounded">Grounded</option>
          <option value="general">General</option>
        </select>
      </div>
      <div class="toolbar-sep"></div>
      <div class="toolbar-group">
        <span class="toolbar-label">Top K</span>
        <input class="toolbar-input" type="number" id="topk-input" value="6" min="1" max="20"/>
      </div>
      <div class="toolbar-sep"></div>
      <div class="toolbar-group">
        <span class="toolbar-label">History</span>
        <input class="toolbar-input" type="number" id="hist-input" value="12" min="0" max="50"/>
      </div>
      <div class="toolbar-sep"></div>
      <label class="toolbar-check">
        <input type="checkbox" id="web-search-cb"/>
        <span>Web Search</span>
      </label>
      <div class="toolbar-group" id="web-max-group" style="display:none">
        <span class="toolbar-label">Results</span>
        <input class="toolbar-input" type="number" id="web-max-input" value="5" min="1" max="10"/>
      </div>
    </div>
    <div class="chat-messages" id="chat-messages">
      <div class="chat-empty" id="chat-empty">
        <div class="chat-empty-icon">&#9733;</div>
        <div class="chat-empty-title">Welcome to Merlin</div>
        <div class="chat-empty-hint">Start a conversation or select one from the sidebar</div>
      </div>
    </div>
    <div class="chat-composer">
      <div class="chat-input-wrap">
        <textarea class="chat-textarea" id="chat-input" placeholder="Ask about your documents..." rows="1"></textarea>
        <button class="btn btn--primary" id="send-btn">Send</button>
        <button class="btn btn--sm" id="generate-btn" title="Generate a file">&#128196; Generate</button>
      </div>
      <div class="chat-actions">
        <span class="chat-status" id="chat-status"></span>
        <span class="chat-hint">Enter to send &middot; Shift+Enter for newline</span>
      </div>
    </div>
  </main>

  <!-- Right Panel -->
  <aside class="right-panel" id="right-panel">
    <div class="panel-section">
      <div class="panel-title">Documents</div>
      <div class="panel-actions">
        <button class="btn btn--sm btn--primary" id="ingest-btn">Re-Ingest</button>
        <button class="btn btn--sm" id="refresh-files-btn">Refresh</button>
      </div>
      <div class="drop-zone" id="drop-zone">
        Drop files here or <label for="file-input" class="file-label">browse</label>
        <input type="file" id="file-input" multiple style="display:none"/>
      </div>
      <div class="upload-status" id="upload-status"></div>
      <div class="file-list" id="file-list"></div>
    </div>

    <div class="health-bar" id="health-bar">checking&hellip;</div>
  </aside>
</div>

<!-- Generate File Modal -->
<div class="generate-overlay hidden" id="generate-overlay">
  <div class="generate-modal">
    <h3>Generate File</h3>
    <textarea id="gen-prompt" placeholder="Describe what you want to generate..."></textarea>
    <select id="gen-format">
      <option value="txt">Text (.txt)</option>
      <option value="csv">Spreadsheet (.csv)</option>
      <option value="docx">Document (.docx)</option>
      <option value="png">Chart (.png)</option>
    </select>
    <div class="modal-actions">
      <button class="btn btn--sm" id="gen-cancel">Cancel</button>
      <button class="btn btn--sm btn--primary" id="gen-submit">Generate</button>
    </div>
  </div>
</div>

<script>
/* ============================================================
   MERLIN RAG — Client SPA
   ============================================================ */
(function(){
'use strict';

// ---- State ----
const S = { user: null, conversations: [], activeConvo: null };

// ---- DOM helpers ----
const $ = id => document.getElementById(id);
function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c] || c);
}
function fmtBytes(n) {
  if (n < 1024) return n + ' B';
  const k = n / 1024;
  if (k < 1024) return k.toFixed(1) + ' KB';
  const m = k / 1024;
  if (m < 1024) return m.toFixed(1) + ' MB';
  return (m / 1024).toFixed(2) + ' GB';
}

// ---- API ----
async function api(method, url, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(url, opts);
  if (r.status === 401) { showLogin(); throw new Error('Session expired'); }
  const text = await r.text();
  let data;
  try { data = JSON.parse(text); } catch { throw new Error(text); }
  if (!r.ok) throw new Error(data.detail || JSON.stringify(data));
  return data;
}
const GET = url => api('GET', url);
const POST = (url, body) => api('POST', url, body);
const PATCH = (url, body) => api('PATCH', url, body);
const DEL = url => api('DELETE', url);

// ---- Views ----
function showLogin() {
  S.user = null;
  $('login-view').classList.remove('hidden');
  $('app-view').classList.add('hidden');
}
function showApp() {
  $('login-view').classList.add('hidden');
  $('app-view').classList.remove('hidden');
  $('display-username').textContent = S.user ? S.user.username : '';
}

// ---- Auth ----
async function checkSession() {
  try {
    const data = await GET('/auth/me');
    S.user = data;
    showApp();
    await loadConversations();
    await loadFiles();
    checkHealth();
  } catch {
    showLogin();
  }
}

async function login(username, password) {
  const data = await POST('/auth/login', { username, password });
  S.user = data;
  showApp();
  await loadConversations();
  await loadFiles();
  checkHealth();
}

async function register(username, password) {
  const data = await POST('/auth/register', { username, password });
  S.user = data;
  showApp();
  await loadConversations();
  await loadFiles();
  checkHealth();
}

async function logout() {
  try { await POST('/auth/logout'); } catch {}
  S.user = null;
  S.conversations = [];
  S.activeConvo = null;
  showLogin();
}

// ---- Conversations ----
function groupConvos(convos) {
  const d = new Date(); d.setHours(0,0,0,0);
  const todayTs = Math.floor(d.getTime() / 1000);
  const weekTs = todayTs - 7 * 86400;
  const today = [], week = [], older = [];
  for (const c of convos) {
    if (c.updated_at >= todayTs) today.push(c);
    else if (c.updated_at >= weekTs) week.push(c);
    else older.push(c);
  }
  return [
    { label: 'Today', items: today },
    { label: 'Previous 7 Days', items: week },
    { label: 'Older', items: older },
  ].filter(g => g.items.length > 0);
}

function renderConversations() {
  const el = $('conv-list');
  if (!S.conversations.length) {
    el.innerHTML = '<div class="conv-empty">No conversations yet</div>';
    return;
  }
  const groups = groupConvos(S.conversations);
  let html = '';
  for (const g of groups) {
    html += '<div class="conv-group-label">' + esc(g.label) + '</div>';
    for (const c of g.items) {
      const active = c.id === S.activeConvo ? ' active' : '';
      html += '<div class="conv-item' + active + '" data-conv="' + esc(c.id) + '">'
        + '<span class="conv-title">' + esc(c.title) + '</span>'
        + '<span class="conv-actions">'
        + '<button class="btn-icon conv-export" title="Export">&#128190;</button>'
        + '<button class="btn-icon conv-rename" title="Rename">&#9998;</button>'
        + '<button class="btn-icon conv-delete" title="Delete">&#128465;</button>'
        + '</span></div>';
    }
  }
  el.innerHTML = html;
}

async function loadConversations() {
  try {
    S.conversations = await GET('/conversations');
    renderConversations();
  } catch (e) {
    console.error('Failed to load conversations:', e);
  }
}

async function createConversation() {
  const conv = await POST('/conversations');
  S.conversations.unshift(conv);
  S.activeConvo = conv.id;
  renderConversations();
  clearChatView();
  $('chat-input').focus();
}

async function selectConversation(id) {
  if (S.activeConvo === id) return;
  S.activeConvo = id;
  renderConversations();
  await loadMessages(id);
}

async function loadMessages(convoId) {
  const msgsEl = $('chat-messages');
  msgsEl.innerHTML = '';
  try {
    const msgs = await GET('/conversations/' + convoId + '/messages');
    if (!msgs.length) {
      msgsEl.innerHTML = '<div class="chat-empty"><div class="chat-empty-icon">&#9733;</div>'
        + '<div class="chat-empty-title">New Conversation</div>'
        + '<div class="chat-empty-hint">Ask a question to begin</div></div>';
      return;
    }
    for (const m of msgs) {
      renderMessage(m.role, m.content);
    }
    msgsEl.scrollTop = msgsEl.scrollHeight;
  } catch (e) {
    msgsEl.innerHTML = '<div class="chat-empty"><div class="chat-empty-hint">Error loading messages: ' + esc(e.message) + '</div></div>';
  }
}

function startRename(convId) {
  const item = document.querySelector('[data-conv="' + convId + '"]');
  if (!item) return;
  const titleEl = item.querySelector('.conv-title');
  const original = titleEl.textContent;
  const input = document.createElement('input');
  input.className = 'conv-rename-input';
  input.value = original;
  titleEl.replaceWith(input);
  input.focus();
  input.select();

  let saved = false;
  async function save() {
    if (saved) return;
    saved = true;
    const title = input.value.trim();
    if (title && title !== original) {
      try { await PATCH('/conversations/' + convId, { title }); } catch {}
    }
    await loadConversations();
  }
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); save(); }
    if (e.key === 'Escape') { saved = true; loadConversations(); }
  });
  input.addEventListener('blur', save);
}

async function deleteConversation(convId) {
  if (!confirm('Delete this conversation? This cannot be undone.')) return;
  try {
    await DEL('/conversations/' + convId);
    if (S.activeConvo === convId) {
      S.activeConvo = null;
      clearChatView();
    }
    await loadConversations();
  } catch (e) {
    alert('Delete failed: ' + e.message);
  }
}

function clearChatView() {
  $('chat-messages').innerHTML = '<div class="chat-empty"><div class="chat-empty-icon">&#9733;</div>'
    + '<div class="chat-empty-title">Welcome to Merlin</div>'
    + '<div class="chat-empty-hint">Start a conversation or select one from the sidebar</div></div>';
  $('chat-status').textContent = '';
}

// ---- Chat ----
function renderMessage(role, content, sources, performance) {
  const msgsEl = $('chat-messages');
  // Remove empty state if present
  const empty = msgsEl.querySelector('.chat-empty');
  if (empty) empty.remove();

  const isUser = role === 'user';
  const wrap = document.createElement('div');
  wrap.className = 'chat-message ' + (isUser ? 'chat-message--user' : 'chat-message--merlin');

  const avatar = document.createElement('div');
  avatar.className = 'chat-avatar';
  avatar.textContent = isUser ? 'YOU' : 'M';

  const body = document.createElement('div');
  body.className = 'chat-body';

  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble';
  if (role === 'error') bubble.classList.add('chat-bubble--error');
  bubble.textContent = content;

  body.appendChild(bubble);

  if (sources && sources.length) {
    const det = document.createElement('details');
    det.className = 'chat-sources';
    const sum = document.createElement('summary');
    sum.textContent = 'Sources (' + sources.length + ')';
    det.appendChild(sum);
    const list = document.createElement('div');
    list.className = 'chat-sources-list';
    sources.forEach(function(s) {
      const pill = document.createElement('span');
      if (s.url) {
        pill.className = 'source-pill source-pill--web';
        pill.innerHTML = '&#128279; <a href="' + esc(s.url) + '" target="_blank" rel="noopener">'
          + '<strong>' + esc(s.source) + '</strong></a>';
      } else {
        pill.className = 'source-pill';
        pill.innerHTML = '<strong>' + esc(s.source) + '</strong> #' + s.chunk_index
          + '<span class="source-score">' + (s.score ? s.score.toFixed(3) : '') + '</span>';
      }
      list.appendChild(pill);
    });
    det.appendChild(list);
    body.appendChild(det);
  }

  // Performance indicator for assistant messages
  if (performance && performance.tokens_per_sec && role === 'assistant') {
    const perf = document.createElement('div');
    perf.className = 'chat-perf';
    perf.innerHTML = '<span><span class="perf-value">' + performance.tokens_per_sec + '</span> tok/s</span>'
      + '<span><span class="perf-value">' + performance.tokens + '</span> tokens</span>'
      + '<span><span class="perf-value">' + performance.total_duration + '</span>s total</span>';
    body.appendChild(perf);
  }

  wrap.appendChild(avatar);
  wrap.appendChild(body);
  msgsEl.appendChild(wrap);
  msgsEl.scrollTop = msgsEl.scrollHeight;
}

function showThinking() {
  const msgsEl = $('chat-messages');
  const empty = msgsEl.querySelector('.chat-empty');
  if (empty) empty.remove();

  const wrap = document.createElement('div');
  wrap.id = 'thinking-msg';
  wrap.className = 'chat-message chat-message--merlin';
  wrap.innerHTML = '<div class="chat-avatar">M</div>'
    + '<div class="chat-body"><div class="chat-bubble thinking">Consulting the archives...</div></div>';
  msgsEl.appendChild(wrap);
  msgsEl.scrollTop = msgsEl.scrollHeight;
}

function hideThinking() {
  const el = $('thinking-msg');
  if (el) el.remove();
}

async function sendMessage() {
  const input = $('chat-input');
  const question = input.value.trim();
  if (!question) return;

  // Auto-create conversation if none selected
  if (!S.activeConvo) {
    try {
      const conv = await POST('/conversations');
      S.activeConvo = conv.id;
      S.conversations.unshift(conv);
      renderConversations();
    } catch (e) {
      $('chat-status').textContent = 'Error: ' + e.message;
      return;
    }
  }

  renderMessage('user', question);
  input.value = '';
  input.style.height = 'auto';
  $('chat-status').textContent = 'Thinking...';
  showThinking();

  const mode = $('mode-select').value;
  const topK = parseInt($('topk-input').value, 10) || 6;
  const histLimit = parseInt($('hist-input').value, 10) || 12;

  try {
    const res = await POST('/chat', {
      convo_id: S.activeConvo,
      question: question,
      mode: mode,
      top_k: topK,
      history_limit: histLimit,
      web_search: $('web-search-cb').checked,
      web_search_max: parseInt($('web-max-input').value, 10) || 5,
    });
    hideThinking();
    renderMessage('assistant', res.answer, res.sources, res.performance);
    $('chat-status').textContent = '';
    // Refresh conversations (title may have updated)
    await loadConversations();
  } catch (e) {
    hideThinking();
    renderMessage('error', 'Error: ' + e.message);
    $('chat-status').textContent = '';
  }
}

// ---- Files ----
async function loadFiles() {
  const el = $('file-list');
  el.innerHTML = '<div style="color:var(--text-dim);font-size:0.78rem;font-style:italic;padding:8px 0">Loading...</div>';
  try {
    const res = await GET('/files');
    const text = res.text || [];
    const incoming = res.incoming || [];
    let html = '';
    html += '<div class="file-section-title">Corpus &middot; ' + text.length + ' file(s)</div>';
    if (text.length) {
      text.forEach(function(f) {
        html += '<div class="file-item"><span class="file-name">' + esc(f.name) + '</span>'
          + '<span class="file-size">' + fmtBytes(f.size) + '</span></div>';
      });
    } else {
      html += '<div style="color:var(--text-dim);font-size:0.75rem;font-style:italic;padding:4px 0">No corpus files</div>';
    }
    html += '<div class="file-section-title" style="margin-top:12px">Incoming &middot; ' + incoming.length + ' file(s)</div>';
    if (incoming.length) {
      incoming.forEach(function(f) {
        html += '<div class="file-item"><span class="file-name">' + esc(f.name) + '</span>'
          + '<span class="file-size">' + fmtBytes(f.size) + '</span></div>';
      });
    } else {
      html += '<div style="color:var(--text-dim);font-size:0.75rem;font-style:italic;padding:4px 0">No incoming files</div>';
    }
    el.innerHTML = html;
  } catch (e) {
    el.innerHTML = '<div style="color:var(--crimson-soft);font-size:0.78rem;padding:8px 0">Error: ' + esc(e.message) + '</div>';
  }
}

async function uploadFiles(files) {
  if (!files || !files.length) return;
  const statusEl = $('upload-status');
  statusEl.textContent = 'Uploading...';
  statusEl.className = 'upload-status';
  try {
    for (const f of files) {
      const form = new FormData();
      form.append('file', f, f.name);
      const r = await fetch('/upload', { method: 'POST', body: form });
      const text = await r.text();
      let data;
      try { data = JSON.parse(text); } catch { throw new Error(text); }
      if (!r.ok) throw new Error(data.detail || text);
    }
    statusEl.textContent = 'Upload complete.';
    statusEl.className = 'upload-status upload-status--ok';
    await loadFiles();
  } catch (e) {
    statusEl.textContent = 'Upload error: ' + e.message;
    statusEl.className = 'upload-status upload-status--err';
  }
}

async function ingestDocs() {
  $('chat-status').textContent = 'Ingesting documents...';
  try {
    const res = await POST('/ingest', {});
    $('chat-status').textContent = 'Ingested: ' + res.files + ' files, ' + res.chunks + ' chunks';
  } catch (e) {
    $('chat-status').textContent = 'Ingest error: ' + e.message;
  }
}

async function checkHealth() {
  try {
    const h = await GET('/health');
    $('health-bar').textContent = 'ok \u00b7 ' + (h.collection || 'unknown');
  } catch {
    $('health-bar').textContent = 'health check failed';
  }
}

// ---- Export ----
function showExportDropdown(convId, anchor) {
  // Remove any existing dropdown
  const old = document.querySelector('.export-dropdown');
  if (old) old.remove();

  const dd = document.createElement('div');
  dd.className = 'export-dropdown';
  const formats = [
    { fmt: 'csv', label: 'Export CSV' },
    { fmt: 'txt', label: 'Export TXT' },
    { fmt: 'docx', label: 'Export DOCX' },
  ];
  formats.forEach(function(f) {
    const a = document.createElement('a');
    a.href = '#';
    a.textContent = f.label;
    a.addEventListener('click', function(e) {
      e.preventDefault();
      window.open('/export/' + encodeURIComponent(convId) + '?format=' + f.fmt, '_blank');
      dd.remove();
    });
    dd.appendChild(a);
  });

  // Position relative to anchor
  const rect = anchor.getBoundingClientRect();
  dd.style.position = 'fixed';
  dd.style.top = (rect.bottom + 4) + 'px';
  dd.style.left = rect.left + 'px';
  document.body.appendChild(dd);

  // Close on outside click
  function closeDropdown(e) {
    if (!dd.contains(e.target)) {
      dd.remove();
      document.removeEventListener('click', closeDropdown);
    }
  }
  setTimeout(function() { document.addEventListener('click', closeDropdown); }, 0);
}

// ---- Generate File ----
function openGenerateModal() {
  $('generate-overlay').classList.remove('hidden');
  $('gen-prompt').value = '';
  $('gen-prompt').focus();
}

function closeGenerateModal() {
  $('generate-overlay').classList.add('hidden');
}

async function submitGeneration() {
  const prompt = $('gen-prompt').value.trim();
  if (!prompt) return;
  const fmt = $('gen-format').value;
  $('gen-submit').disabled = true;
  $('gen-submit').textContent = 'Generating...';

  try {
    const body = { prompt: prompt, format: fmt };
    if (S.activeConvo) body.convo_id = S.activeConvo;

    const resp = await fetch('/generate-file', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const text = await resp.text();
      let detail;
      try { detail = JSON.parse(text).detail; } catch { detail = text; }
      throw new Error(detail || 'Generation failed');
    }

    // Download the file
    const blob = await resp.blob();
    const cd = resp.headers.get('content-disposition') || '';
    const match = cd.match(/filename="?([^"]+)"?/);
    const filename = match ? match[1] : ('generated.' + fmt);
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    closeGenerateModal();
  } catch (e) {
    alert('Generation error: ' + e.message);
  } finally {
    $('gen-submit').disabled = false;
    $('gen-submit').textContent = 'Generate';
  }
}

// ---- Event Listeners ----

// Auth tabs
document.querySelectorAll('.auth-tab').forEach(function(tab) {
  tab.addEventListener('click', function() {
    document.querySelectorAll('.auth-tab').forEach(function(t) { t.classList.remove('active'); });
    tab.classList.add('active');
    if (tab.dataset.tab === 'login') {
      $('login-form').classList.remove('hidden');
      $('register-form').classList.add('hidden');
    } else {
      $('login-form').classList.add('hidden');
      $('register-form').classList.remove('hidden');
    }
  });
});

// Login form
$('login-form').addEventListener('submit', async function(e) {
  e.preventDefault();
  const errEl = $('login-error');
  errEl.textContent = '';
  try {
    await login($('login-username').value, $('login-password').value);
  } catch (err) {
    errEl.textContent = err.message;
  }
});

// Register form
$('register-form').addEventListener('submit', async function(e) {
  e.preventDefault();
  const errEl = $('register-error');
  errEl.textContent = '';
  const pw = $('reg-password').value;
  const confirm = $('reg-confirm').value;
  if (pw !== confirm) { errEl.textContent = 'Passwords do not match'; return; }
  try {
    await register($('reg-username').value, pw);
  } catch (err) {
    errEl.textContent = err.message;
  }
});

// Logout
$('logout-btn').addEventListener('click', logout);

// New conversation
$('new-conv-btn').addEventListener('click', createConversation);

// Conversation list clicks (delegation)
$('conv-list').addEventListener('click', function(e) {
  // Export button
  const exportBtn = e.target.closest('.conv-export');
  if (exportBtn) {
    e.stopPropagation();
    const item = exportBtn.closest('.conv-item');
    if (item) showExportDropdown(item.dataset.conv, exportBtn);
    return;
  }
  // Rename button
  const renameBtn = e.target.closest('.conv-rename');
  if (renameBtn) {
    e.stopPropagation();
    const item = renameBtn.closest('.conv-item');
    if (item) startRename(item.dataset.conv);
    return;
  }
  // Delete button
  const deleteBtn = e.target.closest('.conv-delete');
  if (deleteBtn) {
    e.stopPropagation();
    const item = deleteBtn.closest('.conv-item');
    if (item) deleteConversation(item.dataset.conv);
    return;
  }
  // Select conversation
  const item = e.target.closest('.conv-item');
  if (item) selectConversation(item.dataset.conv);
});

// Send message
$('send-btn').addEventListener('click', sendMessage);

// Textarea: Enter to send, Shift+Enter for newline, auto-resize
$('chat-input').addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
$('chat-input').addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 160) + 'px';
});

// File upload
$('file-input').addEventListener('change', async function() {
  await uploadFiles(this.files);
  this.value = '';
});

// Drop zone
const dz = $('drop-zone');
dz.addEventListener('dragover', function(e) { e.preventDefault(); dz.classList.add('active'); });
dz.addEventListener('dragleave', function() { dz.classList.remove('active'); });
dz.addEventListener('drop', async function(e) {
  e.preventDefault();
  dz.classList.remove('active');
  await uploadFiles(e.dataTransfer.files);
});

// Ingest & refresh
$('ingest-btn').addEventListener('click', ingestDocs);
$('refresh-files-btn').addEventListener('click', loadFiles);

// Web search toggle
$('web-search-cb').addEventListener('change', function() {
  $('web-max-group').style.display = this.checked ? 'flex' : 'none';
});

// Mobile toggles
$('hamburger-btn').addEventListener('click', function() {
  $('conv-sidebar').classList.toggle('open');
  $('right-panel').classList.remove('open');
});
$('panel-toggle-btn').addEventListener('click', function() {
  $('right-panel').classList.toggle('open');
  $('conv-sidebar').classList.remove('open');
});

// Generate file modal
$('generate-btn').addEventListener('click', openGenerateModal);
$('gen-cancel').addEventListener('click', closeGenerateModal);
$('gen-submit').addEventListener('click', submitGeneration);
$('generate-overlay').addEventListener('click', function(e) {
  if (e.target === this) closeGenerateModal();
});

// ---- Init ----
checkSession();

})();
</script>
</body>
</html>"""
