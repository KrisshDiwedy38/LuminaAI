"""Lumina API. Per message: rule gate + ML -> safety decision -> (crisis | memory -> Gemini).

    uvicorn backend.app:app --reload
"""
import json
import os
import socket
import sqlite3
import sys
import threading
import time
import traceback
import webbrowser
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from backend import llm, memory, safety  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DB = Path(os.getenv("LUMINA_DB") or ROOT / "data" / "lumina.db")
DB.parent.mkdir(parents=True, exist_ok=True)

try:
    from backend import ml
    analyzer = ml.load()
except Exception:  # torch missing or a broken checkpoint: run on the rule gate alone
    traceback.print_exc()
    analyzer = None
print(f"[lumina] ML model: {'loaded' if analyzer else 'not trained yet, rule-based safety only'}"
      f" | Gemini: {'key set' if llm.available() else 'GEMINI_API_KEY missing, using fallback replies'}")


def q(sql, *args):
    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute(sql, args)]
    return rows


def init_db():
    q("""CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY, started_at TEXT, ended_at TEXT,
         summary TEXT, stats TEXT)""")
    q("""CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, session_id INTEGER, role TEXT, text TEXT,
         analysis TEXT, created_at TEXT)""")


init_db()

def open_browser():
    """Open the app in the browser once the server accepts connections.
    ponytail: reads the port from `--port N` only, and --reload reopens a tab on every reload."""
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000

    def wait_then_open():
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                webbrowser.open(f"http://localhost:{port}")
                return
            except OSError:
                time.sleep(0.2)

    threading.Thread(target=wait_then_open, daemon=True).start()


@asynccontextmanager
async def lifespan(app):
    if not os.getenv("LUMINA_NO_BROWSER"):  # set it for headless checks and scripted runs
        open_browser()
    yield


app = FastAPI(title="Lumina", lifespan=lifespan)


def now():
    return datetime.now().isoformat(timespec="seconds")


def get_session(sid):
    rows = q("SELECT * FROM sessions WHERE id = ?", sid)
    if not rows:
        raise HTTPException(404, "Session not found")
    return rows[0]


def history_for_llm(sid, limit=12):
    """Recent turns for context. Crisis turns are replaced by a marker, so crisis wording never reaches Gemini."""
    rows = q("SELECT role, text, analysis FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?", sid, limit)
    out = []
    for r in reversed(rows):
        if json.loads(r["analysis"] or "{}").get("level") == safety.CRITICAL:
            text = "[A message here was routed to crisis support.]" if r["role"] == "user" else "[Lumina shared crisis resources.]"
        else:
            text = r["text"]
        out.append({"role": r["role"], "text": text})
    return out


def run_pipeline(sid, text, genz):
    recent = q("SELECT text, analysis FROM messages WHERE session_id = ? AND role = 'user' ORDER BY id DESC LIMIT ?",
               sid, safety.RECENT_MESSAGES)
    rule = safety.check(text, [r["text"] for r in recent])
    signals = None
    if analyzer:
        try:
            s = analyzer.predict(text)
            if rule["ml_text"] != text:  # denial removed: risk is judged on what's left, if anything
                rest = analyzer.predict(rule["ml_text"]) if rule["ml_text"] else {}
                s.update(risk=rest.get("risk"), risk_confidence=rest.get("risk_confidence"), risk_text=rule["ml_text"])
            signals = s  # only once complete: a crash above must not leave the full-text risk in place
        except Exception:  # a model failure must never block the rule gate
            traceback.print_exc()
    level = safety.decide(rule["level"], signals and signals["risk"], signals["risk_confidence"] if signals else 1.0)
    analysis = {"rule": rule, "ml": signals, "level": level, "memory": [], "llm": None}

    if level == safety.CRITICAL:  # short-circuit: no memory, no Gemini, fixed wording
        analysis["llm"] = "skipped"
        return safety.CRISIS_MESSAGE, analysis

    analysis["after_crisis"] = bool(recent) and json.loads(recent[0]["analysis"] or "{}").get("level") == safety.CRITICAL
    try:
        analysis["memory"] = memory.search(text, exclude_session=sid)
    except Exception:
        traceback.print_exc()
    try:
        reply = llm.reply(text, signals, level, analysis["memory"], history_for_llm(sid), genz,
                          after_crisis=analysis["after_crisis"])
        analysis["llm"] = "ok"
    except Exception as e:
        print(f"[lumina] Gemini failed: {e}")
        reply, analysis["llm"] = llm.FALLBACK, "unavailable"
    cleaned = safety.strip_unverified_numbers(reply)
    analysis["stripped"] = cleaned != reply.strip()
    reply = cleaned or llm.FALLBACK
    analysis["support"] = safety.needs_support(level, reply)
    return reply, analysis


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    genz: bool = False


@app.get("/api/status")
def status():
    return {"ml": analyzer is not None, "llm": llm.available(), "model": llm.MODEL,
            "ml_config": {k: v for k, v in analyzer.config.items() if k != "val"} if analyzer else None}


@app.get("/api/sessions")
def list_sessions():
    rows = q("SELECT * FROM sessions ORDER BY id DESC LIMIT 50")
    for r in rows:
        r["stats"] = json.loads(r["stats"] or "null")
    return rows


@app.post("/api/sessions")
def create_session():
    q("INSERT INTO sessions (started_at) VALUES (?)", now())
    return q("SELECT * FROM sessions ORDER BY id DESC LIMIT 1")[0]


@app.get("/api/sessions/{sid}/messages")
def list_messages(sid: int):
    get_session(sid)
    rows = q("SELECT id, role, text, analysis, created_at FROM messages WHERE session_id = ? ORDER BY id", sid)
    for r in rows:
        r["analysis"] = json.loads(r["analysis"] or "null")
        if r["role"] == "assistant" and r["analysis"] and r["analysis"]["level"] == safety.CRITICAL:
            r["resources"] = safety.CRISIS_RESOURCES
        elif r["role"] == "assistant" and r["analysis"] and r["analysis"].get("support"):
            r["support"] = safety.CRISIS_RESOURCES
    return rows


@app.post("/api/sessions/{sid}/messages")
def send_message(sid: int, msg: Message):
    if get_session(sid)["ended_at"]:
        raise HTTPException(409, "This session has ended. Start a new one to keep talking.")
    text = msg.text.strip()
    if not text:
        raise HTTPException(422, "Message is empty")
    reply, analysis = run_pipeline(sid, text, msg.genz)
    a = json.dumps(analysis)
    q("INSERT INTO messages (session_id, role, text, analysis, created_at) VALUES (?, 'user', ?, ?, ?)", sid, text, a, now())
    q("INSERT INTO messages (session_id, role, text, analysis, created_at) VALUES (?, 'assistant', ?, ?, ?)", sid, reply, a, now())
    out = {"reply": reply, "analysis": analysis}
    if analysis["level"] == safety.CRITICAL:
        out["resources"] = safety.CRISIS_RESOURCES
    elif analysis.get("support"):
        out["support"] = safety.CRISIS_RESOURCES
    return out


@app.post("/api/sessions/{sid}/end")
def end_session(sid: int):
    """Summarise into de-identified memory, keep aggregate mood stats, then delete the raw transcript."""
    s = get_session(sid)
    if s["ended_at"]:
        return s
    rows = q("SELECT role, text, analysis FROM messages WHERE session_id = ? ORDER BY id", sid)
    analyses = [json.loads(r["analysis"]) for r in rows if r["role"] == "user"]
    ml_rows = [a["ml"] for a in analyses if a.get("ml")]
    stats = {
        "messages": len(analyses),
        "highest_level": max([a["level"] for a in analyses] or [safety.SAFE], key=safety.SEVERITY.get),
        "intensity": [m["intensity"] for m in ml_rows],
        "top_emotion": Counter(m["emotion"] for m in ml_rows).most_common(1)[0][0] if ml_rows else None,
    }
    summary = None
    transcript = [{"role": r["role"], "text": r["text"]} for r in rows
                  if json.loads(r["analysis"]).get("level") != safety.CRITICAL]
    if transcript and llm.available():
        try:
            summary = llm.summarize(transcript)
            memory.add(sid, summary, s["started_at"][:10])
        except Exception as e:
            print(f"[lumina] summary failed: {e}")
    q("UPDATE sessions SET ended_at = ?, summary = ?, stats = ? WHERE id = ?", now(), summary, json.dumps(stats), sid)
    q("DELETE FROM messages WHERE session_id = ?", sid)
    return get_session(sid)


dist = ROOT / "frontend" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
