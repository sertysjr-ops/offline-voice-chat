#!/usr/bin/env python3
"""KWS / wake-gate telemetry — append-only JSONL event log.

Every wake-related decision the server makes lands in
data/kws-telemetry.jsonl: raw KWS calls (/api/wake) and transcript-gate
decisions (hands-free wake-word matching). That's the data you need to
tune the wake word: too many KWS hits that the transcript gate then
rejects = false positives; gate rejects with transcripts that clearly
CONTAIN the wake word = the STT is mangling it.

Events (JSONL, one per line):
  {"ts": ..., "event": "wake",     "awake": bool, "keyword": str|null,
   "seconds": float, "tookMs": int}
  {"ts": ..., "event": "stt-gate", "hit": bool, "word": str|null,
   "transcript": str (first 120 chars), "source": "talk"|"ask"|...}

The file is rotated at ~512KB (last 1000 lines kept) so a long-running
always-on session can't grow it unboundedly.
"""
import json
import os
import threading
import time

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LOG_PATH = os.path.join(_DIR, "kws-telemetry.jsonl")
_MAX_BYTES = 512 * 1024
_KEEP_LINES = 1000

_lock = threading.Lock()


def log(event, **fields):
    """Append one event. Never raises — telemetry must not break the API."""
    try:
        rec = {"ts": round(time.time(), 3), "event": event, **fields}
        line = json.dumps(rec, ensure_ascii=False)
        with _lock:
            os.makedirs(_DIR, exist_ok=True)
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            _maybe_rotate()
    except Exception:
        pass


def _maybe_rotate():
    try:
        if os.path.getsize(LOG_PATH) <= _MAX_BYTES:
            return
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.writelines(lines[-_KEEP_LINES:])
    except Exception:
        pass


def stats(recent=20):
    """Roll up the log for GET /api/kws/stats.

    Returns counts + hit rates + the most recent events (newest first).
    A missing/empty log is a valid zero-state, not an error.
    """
    events = []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a torn final line shouldn't kill the read
    except FileNotFoundError:
        pass

    wake = [e for e in events if e.get("event") == "wake"]
    gate = [e for e in events if e.get("event") == "stt-gate"]
    wake_hits = sum(1 for e in wake if e.get("awake"))
    gate_hits = sum(1 for e in gate if e.get("hit"))
    by_keyword: dict[str, int] = {}
    for e in wake:
        kw = e.get("keyword")
        if kw:
            by_keyword[kw] = by_keyword.get(kw, 0) + 1

    return {
        "total": len(events),
        "wake": {
            "calls": len(wake),
            "hits": wake_hits,
            "hitRate": round(wake_hits / len(wake), 3) if wake else None,
            "byKeyword": by_keyword,
        },
        "sttGate": {
            "calls": len(gate),
            "hits": gate_hits,
            "rejectRate": (
                round(1 - gate_hits / len(gate), 3) if gate else None
            ),
        },
        "recent": list(reversed(events[-recent:])),
    }
