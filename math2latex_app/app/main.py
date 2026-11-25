"""Math → LaTeX personal web app.

Env vars:
  APP_PASSWORD        required — the login password
  ANTHROPIC_API_KEY   optional — enables the "Ask Claude" button
  MODEL_PATH          optional — default: math2latex_model.pt next to this package
  DB_PATH             optional — default: data.db in the working directory

Run locally:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import base64
import hashlib
import hmac
import io
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel

from .model_io import Predictor

APP_DIR = Path(__file__).parent
MODEL_PATH = os.environ.get("MODEL_PATH", str(APP_DIR.parent / "math2latex_model.pt"))
DB_PATH = os.environ.get("DB_PATH", "data.db")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

app = FastAPI(title="Math2LaTeX")

# Loaded once at startup and reused for every request — loading the model per
# request would dominate latency. See the startup handler below.
predictor: Predictor | None = None


# ============================================================ 1. DATABASE
# SQLite, two tables: chats (named projects) and entries (one conversion each).
# entries.thumb is a small base64 PNG of the drawing so the ledger can show
# what you wrote next to what the model read.
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS chats(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS entries(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            latex TEXT NOT NULL,
            thumb TEXT NOT NULL,
            answer TEXT,
            created TEXT NOT NULL);
        """)


@app.on_event("startup")
def startup():
    global predictor
    if not APP_PASSWORD:
        raise RuntimeError("Set the APP_PASSWORD environment variable before starting.")
    init_db()
    predictor = Predictor(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH} on {predictor.device}")


# ================================================================ 2. AUTH
# Single-user password auth. The session cookie is an HMAC of the password, so
# there's no session store to maintain: a valid cookie can only be produced by
# someone who knew the password. httpOnly keeps JavaScript from reading it.
def make_token() -> str:
    return hmac.new(APP_PASSWORD.encode(), b"math2latex-session", hashlib.sha256).hexdigest()


def require_auth(session: str | None = Cookie(default=None)):
    if not session or not hmac.compare_digest(session, make_token()):
        raise HTTPException(status_code=401, detail="Not signed in")


class LoginBody(BaseModel):
    password: str


@app.post("/api/login")
def login(body: LoginBody, response: Response):
    if not hmac.compare_digest(body.password, APP_PASSWORD):
        raise HTTPException(status_code=401, detail="Wrong password")
    response.set_cookie("session", make_token(), httponly=True, samesite="lax",
                        max_age=60 * 60 * 24 * 90)
    return {"ok": True, "claude_enabled": bool(ANTHROPIC_API_KEY)}


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie("session")
    return {"ok": True}


@app.get("/api/me")
def me(_: None = Depends(require_auth)):
    return {"ok": True, "claude_enabled": bool(ANTHROPIC_API_KEY)}


# =============================================================== 3. CHATS
# CRUD for the sidebar. Deleting a chat cascades to its entries.
class ChatBody(BaseModel):
    title: str


@app.get("/api/chats")
def list_chats(_: None = Depends(require_auth)):
    with db() as conn:
        rows = conn.execute("SELECT * FROM chats ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


@app.post("/api/chats")
def create_chat(body: ChatBody, _: None = Depends(require_auth)):
    title = body.title.strip() or "Untitled"
    with db() as conn:
        cur = conn.execute("INSERT INTO chats(title, created) VALUES(?, ?)",
                           (title, datetime.now(timezone.utc).isoformat()))
        chat_id = cur.lastrowid
        row = conn.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
    return dict(row)


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: int, _: None = Depends(require_auth)):
    with db() as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    return {"ok": True}


@app.get("/api/chats/{chat_id}/entries")
def list_entries(chat_id: int, _: None = Depends(require_auth)):
    with db() as conn:
        rows = conn.execute("SELECT * FROM entries WHERE chat_id=? ORDER BY id",
                            (chat_id,)).fetchall()
    return [dict(r) for r in rows]


# ========================================================== 4. PREDICTION
# The core endpoint: canvas PNG in, LaTeX out, saved to the ledger.
def thumbnail_data_url(img: Image.Image, height: int = 56) -> str:
    w, h = img.size
    img = img.convert("L").resize((max(1, int(w * height / h)), height))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/api/predict")
async def predict(chat_id: int = Form(...), image: UploadFile = File(...),
                  beam: int = Form(default=5),
                  _: None = Depends(require_auth)):
    """Canvas PNG -> LaTeX. `beam` selects the decoding strategy:
    1 = greedy (fastest), >1 = beam search (slower, usually more accurate)."""
    raw = await image.read()
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the image")
    latex = predictor.predict(img, beam=max(1, min(beam, 10)))
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO entries(chat_id, latex, thumb, created) VALUES(?,?,?,?)",
            # thumbnail shows the AUTOCROPPED image — i.e. what the model saw
            (chat_id, latex, thumbnail_data_url(predictor.autocrop(img)),
             datetime.now(timezone.utc).isoformat()))
        row = conn.execute("SELECT * FROM entries WHERE id=?", (cur.lastrowid,)).fetchone()
    return dict(row)


class EditBody(BaseModel):
    latex: str


@app.patch("/api/entries/{entry_id}")
def edit_entry(entry_id: int, body: EditBody, _: None = Depends(require_auth)):
    """Correct a prediction. These corrections are also your future fine-tuning data."""
    with db() as conn:
        conn.execute("UPDATE entries SET latex=? WHERE id=?", (body.latex, entry_id))
        row = conn.execute("SELECT * FROM entries WHERE id=?", (entry_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="No such entry")
    return dict(row)


@app.delete("/api/entries/{entry_id}")
def delete_entry(entry_id: int, _: None = Depends(require_auth)):
    """Remove a single entry from the ledger (e.g. a bad prediction you don't want kept)."""
    with db() as conn:
        cur = conn.execute("DELETE FROM entries WHERE id=?", (entry_id,))
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="No such entry")
    return {"ok": True}


# ============================================================== 5. CLAUDE
# Optional. The API key stays server-side and is never sent to the browser;
# the frontend only ever calls THIS endpoint, which proxies to Anthropic.
@app.post("/api/entries/{entry_id}/ask")
async def ask_claude(entry_id: int, _: None = Depends(require_auth)):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=400,
                            detail="Set ANTHROPIC_API_KEY on the server to enable this")
    with db() as conn:
        row = conn.execute("SELECT * FROM entries WHERE id=?", (entry_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="No such entry")
    prompt = (f"Here is a math statement in LaTeX: {row['latex']}\n"
              "If it's a question or asks to solve/prove something, answer it concisely. "
              "If it's just an expression, briefly explain what it represents.")
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY,
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": "claude-sonnet-4-6", "max_tokens": 1024,
                  "messages": [{"role": "user", "content": prompt}]})
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Claude API error: {r.text[:200]}")
    answer = "".join(b.get("text", "") for b in r.json().get("content", []))
    with db() as conn:
        conn.execute("UPDATE entries SET answer=? WHERE id=?", (answer, entry_id))
    return {"answer": answer}


# =========================================================== 6. STATIC/OPS
@app.get("/health")
def health():
    return {"ok": True, "model_loaded": predictor is not None}


@app.get("/")
def index():
    return FileResponse(APP_DIR / "static" / "index.html")
