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

    # voice profiles — per-session TTS prefs, persisted with sessions
    p = get_profile(sid)                  # {"speed": 1.0, "voice": "lessac", "speaker": 0}
    set_profile(sid, speed=1.2, voice="libritts", speaker=121)
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
_profiles: dict[str, dict] = {}

DEFAULT_PROFILE = {"speed": 1.0, "voice": "lessac", "speaker": 0}
SPEED_MIN, SPEED_MAX = 0.5, 2.5


def _load() -> None:
    global _sessions, _profiles
    if not os.path.exists(DATA_FILE):
        return
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            # back-compat: bare {sid: [...]}, or wrapped {"sessions": {...}}
            _sessions = raw.get("sessions", raw if "profiles" not in raw else {})
            _profiles = raw.get("profiles", {}) if isinstance(raw.get("profiles"), dict) else {}
    except Exception:
        _sessions = {}
        _profiles = {}


def _save() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = DATA_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"sessions": _sessions, "profiles": _profiles}, f)
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


def clamp_speed(v) -> float:
    """Coerce anything into a sane speed value."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 1.0
    return max(SPEED_MIN, min(SPEED_MAX, f))


def get_profile(sid: str) -> dict:
    """Session voice prefs merged over defaults. Unknown sid → defaults."""
    if not sid:
        return dict(DEFAULT_PROFILE)
    with _lock:
        stored = _profiles.get(sid) or {}
    p = dict(DEFAULT_PROFILE)
    p.update({k: stored[k] for k in DEFAULT_PROFILE if k in stored})
    p["speed"] = clamp_speed(p["speed"])
    try:
        p["speaker"] = max(0, int(p["speaker"]))
    except (TypeError, ValueError):
        p["speaker"] = 0
    return p


def set_profile(sid: str, speed=None, voice=None, speaker=None) -> dict:
    """Merge provided fields into the session's profile; returns the result."""
    if not sid:
        return dict(DEFAULT_PROFILE)
    with _lock:
        p = dict(_profiles.get(sid) or DEFAULT_PROFILE)
        if speed is not None:
            p["speed"] = clamp_speed(speed)
        if voice is not None and isinstance(voice, str) and voice.strip():
            p["voice"] = voice.strip()[:32]
        if speaker is not None:
            try:
                p["speaker"] = max(0, int(speaker))
            except (TypeError, ValueError):
                pass
        _profiles[sid] = p
        _save()
        return dict(p)
