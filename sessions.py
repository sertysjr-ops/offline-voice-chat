#!/usr/bin/env python3
"""Server-side conversation memory for the voice chat.

Sessions are keyed by an opaque client-generated id. History lives in
memory and is mirrored to data/sessions.json (atomic writes) so a server
restart doesn't wipe conversations. Capped at MAX_MESSAGES so the LLM
context stays small and fast — old turns drop off the front.

Usage:
    from sessions import get_history, append_turn, clear, turns
    hist = get_history(sid)               # [] if new
    append_turn(sid, user_text, reply)    # adds both messages, persists
    clear(sid)                            # wipe one session
"""
import json
import os
import threading
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_FILE = os.path.join(DATA_DIR, "sessions.json")

MAX_MESSAGES = 20  # 10 turns — plenty for voice context, cheap on tokens
MAX_SESSIONS = 50  # LRU-ish: drop oldest when exceeded

_lock = threading.Lock()
_sessions: dict[str, list[dict]] = {}


def _load() -> None:
    global _sessions
    if not os.path.exists(DATA_FILE):
        return
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # back-compat: either {sid: [...]} or {"sessions": {...}}
        _sessions = raw.get("sessions", raw) if isinstance(raw, dict) else {}
    except Exception:
        _sessions = {}


def _save() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = DATA_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"sessions": _sessions}, f)
    os.replace(tmp, DATA_FILE)


def _trim() -> None:
    while len(_sessions) > MAX_SESSIONS:
        oldest = min(_sessions, key=lambda k: _session_mtime(k))
        del _sessions[oldest]


def _session_mtime(sid: str) -> str:
    msgs = _sessions.get(sid) or []
    if msgs:
        return msgs[-1].get("_ts", "")
    return ""


_load()


def get_history(sid: str) -> list[dict]:
    """Messages for the LLM, internal fields stripped."""
    with _lock:
        msgs = list(_sessions.get(sid) or [])
    return [
        {"role": m["role"], "content": m["content"]}
        for m in msgs
        if m.get("role") in ("user", "assistant")
    ]


def append_turn(sid: str, user_text: str, assistant_text: str) -> int:
    """Append one Q/A turn; returns total messages kept for the session."""
    if not sid:
        return 0
    with _lock:
        msgs = _sessions.setdefault(sid, [])
        now = datetime.now(timezone.utc).isoformat()
        msgs.append({"role": "user", "content": user_text, "_ts": now})
        msgs.append({"role": "assistant", "content": assistant_text, "_ts": now})
        del msgs[:-MAX_MESSAGES]  # keep last N
        _trim()
        _save()
        return len(msgs)


def clear(sid: str) -> bool:
    with _lock:
        existed = sid in _sessions
        _sessions.pop(sid, None)
        _save()
        return existed


def turns(sid: str) -> int:
    with _lock:
        return len(_sessions.get(sid) or []) // 2
