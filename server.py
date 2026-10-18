#!/usr/bin/env python3
"""Offline voice chat — web UI server. Stdlib only, no pip installs.

Serves a single-page browser UI and the voice pipeline API:

    GET  /                 → index.html
    POST /api/talk         → JSON {audio: base64 16kHz mono wav, history?: [...]}
                             → {transcript, reply, audio: base64 wav}

Models load lazily on first request (whisper + piper take a few seconds).

Usage:
    .venv/bin/python server.py [--port 8787]
"""
import argparse
import base64
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_lock = threading.Lock()
_recognizer = None
_tts = None


def get_recognizer():
    global _recognizer
    if _recognizer is None:
        import stt
        _recognizer = stt.build_recognizer()
    return _recognizer


def synthesize(text, out_path):
    global _tts
    if _tts is None:
        import tts as tts_mod
        _tts = tts_mod.build_tts()
    import tts as tts_mod
    return tts_mod.synthesize(text, out_path)


def talk(audio_b64, history):
    """Full pipeline: wav → STT → LLM → TTS. Returns dict."""
    import tempfile
    import numpy as np
    from brain import ask

    wav_bytes = base64.b64decode(audio_b64)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(wav_bytes)
    tmp.close()

    import wave
    with wave.open(tmp.name, "rb") as f:
        assert f.getframerate() == 16000 and f.getnchannels() == 1, "need 16kHz mono wav"
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)

    with _lock:  # sherpa recognizers are not thread-safe
        recognizer = get_recognizer()
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples.astype("float32") / 32768.0)
        t0 = time.time()
        recognizer.decode_stream(stream)
        transcript = stream.result.text.strip()

    if not transcript:
        return {"transcript": "", "reply": "I didn't catch that — try again.", "audio": None}

    reply = ask(transcript, history=history or None)

    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out.close()
    with _lock:
        path, _dur = synthesize(reply, out.name)
    with open(path, "rb") as f:
        audio_b64_out = base64.b64encode(f.read()).decode()

    return {"transcript": transcript, "reply": reply, "audio": audio_b64_out}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(os.path.dirname(__file__), "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/api/health":
            self._send(200, json.dumps({"ok": True, "models": {"stt": _recognizer is not None, "tts": _tts is not None}}))
        else:
            self._send(404, '{"error": "not found"}')

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
        except Exception:
            self._send(400, '{"error": "bad json"}')
            return

        if self.path == "/api/ask":
            # text-only path: skip STT, LLM → TTS
            text = (payload.get("text") or "").strip()
            if not text:
                self._send(400, '{"error": "text required"}')
                return
            try:
                import tempfile
                from brain import ask
                reply = ask(text, history=payload.get("history") or None)
                out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                out.close()
                with _lock:
                    path, _dur = synthesize(reply, out.name)
                with open(path, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode()
                self._send(200, json.dumps({"reply": reply, "audio": audio_b64}))
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}))
            return

        if self.path != "/api/talk":
            self._send(404, '{"error": "not found"}')
            return
        try:
            result = talk(payload["audio"], payload.get("history") or [])
            self._send(200, json.dumps(result))
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    print(f"🎙  Hermes voice chat → http://localhost:{args.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
