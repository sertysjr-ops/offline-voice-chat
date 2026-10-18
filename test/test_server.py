#!/usr/bin/env python3
"""Integration test: boot server.py on an ephemeral port, hit /api/health and
/api/talk with the sample wav, assert a transcript + reply + audio come back.

Usage: .venv/bin/python test/test_server.py
"""
import base64
import json
import os
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_up(port, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def main():
    port = free_port()
    proc = subprocess.Popen(
        [PY, os.path.join(ROOT, "server.py"), "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        if not wait_up(port):
            out = proc.stdout.read() if proc.stdout else ""
            print("FAIL: server never came up\n" + out)
            return 1

        health = json.load(os.popen(
            f'curl -s -m 5 http://127.0.0.1:{port}/api/health'))
        print("health:", health)
        assert health.get("ok") is True, "health check failed"

        with open(os.path.join(ROOT, "test", "test.wav"), "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        t0 = time.time()
        raw = subprocess.run(
            ["curl", "-s", "-m", "120", "-X", "POST",
             f"http://127.0.0.1:{port}/api/talk",
             "-H", "Content-Type: application/json",
             "-d", json.dumps({"audio": b64, "history": []})],
            capture_output=True, text=True).stdout
        dt = time.time() - t0

        data = json.loads(raw)
        print(f"transcript: {data.get('transcript')!r}")
        print(f"reply:      {(data.get('reply') or '')[:160]!r}")
        print(f"audio b64:  {len(data.get('audio') or '')} chars, roundtrip {dt:.1f}s")

        assert data.get("transcript"), "empty transcript"
        assert data.get("reply"), "empty reply"
        assert data.get("audio"), "no audio returned"
        wav = base64.b64decode(data["audio"])
        assert wav[:4] == b"RIFF", "reply is not a wav"
        print("\n✅ /api/talk PASS — full STT→LLM→TTS pipeline over HTTP works")

        # text-only path
        raw = subprocess.run(
            ["curl", "-s", "-m", "120", "-X", "POST",
             f"http://127.0.0.1:{port}/api/ask",
             "-H", "Content-Type: application/json",
             "-d", json.dumps({"text": "Say OK if you can hear me.",
                               "history": []})],
            capture_output=True, text=True).stdout
        d2 = json.loads(raw)
        print(f"/api/ask reply: {(d2.get('reply') or '')[:120]!r}")
        assert d2.get("reply") and d2.get("audio"), "ask endpoint broken"
        # index.html served
        html = subprocess.run(
            ["curl", "-s", "-m", "5", f"http://127.0.0.1:{port}/"],
            capture_output=True, text=True).stdout
        assert "<title>Hermes Voice" in html, "index.html not served"
        print("✅ /api/ask + index.html PASS")

        # --- server-side session memory ---
        sid = "test-session-1"
        r1 = subprocess.run(
            ["curl", "-s", "-m", "120", "-X", "POST",
             f"http://127.0.0.1:{port}/api/ask",
             "-H", "Content-Type: application/json",
             "-d", json.dumps({"text": "Remember this codeword: ZEBRA.",
                               "session": sid})],
            capture_output=True, text=True).stdout
        d1 = json.loads(r1)
        assert d1.get("reply") and d1.get("turns") == 1, f"turn 1 failed: {r1[:200]}"
        print(f"✅ session turn 1 — turns={d1['turns']}")

        r2 = subprocess.run(
            ["curl", "-s", "-m", "120", "-X", "POST",
             f"http://127.0.0.1:{port}/api/ask",
             "-H", "Content-Type: application/json",
             "-d", json.dumps({"text": "What is the codeword I gave you? Answer with just the codeword.",
                               "session": sid})],
            capture_output=True, text=True).stdout
        d2 = json.loads(r2)
        assert d2.get("turns") == 2, f"turn count wrong: {r2[:200]}"
        assert "zebra" in d2["reply"].lower(), \
            f"memory failed — reply has no codeword: {d2['reply'][:160]!r}"
        print(f"✅ session memory works — reply recalls codeword: {d2['reply'][:60]!r}")

        # /api/clear wipes it
        rc = subprocess.run(
            ["curl", "-s", "-m", "10", "-X", "POST",
             f"http://127.0.0.1:{port}/api/clear",
             "-H", "Content-Type: application/json",
             "-d", json.dumps({"session": sid})],
            capture_output=True, text=True).stdout
        dc = json.loads(rc)
        assert dc.get("ok") is True, f"clear failed: {rc[:200]}"

        r3 = subprocess.run(
            ["curl", "-s", "-m", "120", "-X", "POST",
             f"http://127.0.0.1:{port}/api/ask",
             "-H", "Content-Type: application/json",
             "-d", json.dumps({"text": "What is the codeword I gave you? Answer with just the codeword.",
                               "session": sid})],
            capture_output=True, text=True).stdout
        d3 = json.loads(r3)
        assert d3.get("turns") == 1, f"clear did not wipe history: {r3[:200]}"
        print("✅ /api/clear wipes session memory")
        return 0
    finally:
        proc.kill()
        proc.wait()


if __name__ == "__main__":
    sys.exit(main())
