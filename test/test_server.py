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
sys.path.insert(0, ROOT)


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
    # wake-word fixtures are gitignored (regenerable) — auto-build if missing
    need = [os.path.join(ROOT, "test", n) for n in ("wake.wav", "nowake.wav")]
    if any(not os.path.exists(p) for p in need):
        print("generating wake-word fixtures via test/gen_wake_fixtures.py …")
        subprocess.run(
            [PY, os.path.join(ROOT, "test", "gen_wake_fixtures.py")],
            check=True, cwd=ROOT,
        )

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
        # unique sid per run — data/sessions.json persists across runs, so a
        # hardcoded id would leak turns from previous test runs
        sid = "test-session-" + os.urandom(4).hex()
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

        # --- hands-free wake-word gate ---
        # unit-level checks first (no server roundtrip)
        from wakeword import match_wake_word
        m, rest = match_wake_word("Hey Hermes, what time is it?")
        assert m and "time" in rest and "hermes" not in rest.lower(), (m, rest)
        m2, rest2 = match_wake_word("Hermes. turn the lights off")
        assert m2 and "lights" in rest2, (m2, rest2)
        m3, rest3 = match_wake_word("what time is it?")
        assert not m3, (m3, rest3)
        m4, rest4 = match_wake_word("Hey Hermes")
        assert m4 and rest4 == "", (m4, rest4)
        print("✅ match_wake_word unit checks (strip, punctuation, reject, wake-only)")

        def talk_wav(path, **extra):
            with open(os.path.join(ROOT, "test", path), "rb") as f:
                b = base64.b64encode(f.read()).decode()
            t0 = time.time()
            raw = subprocess.run(
                ["curl", "-s", "-m", "120", "-X", "POST",
                 f"http://127.0.0.1:{port}/api/talk",
                 "-H", "Content-Type: application/json",
                 "-d", json.dumps({"audio": b, **extra})],
                capture_output=True, text=True).stdout
            return json.loads(raw), time.time() - t0

        # hands-free ON + no wake word → gated, fast, silent
        g, gdt = talk_wav("nowake.wav", handsFree=True)
        assert g.get("wakeWordHit") is False, f"should be gated: {g}"
        assert g.get("reply") is None and g.get("audio") is None, "gated call must skip LLM+TTS"
        assert gdt < 10, f"gated path should be sub-LLM latency, took {gdt:.1f}s"
        print(f"✅ hands-free gate: no wake word → silent ignore ({gdt:.1f}s, no LLM)")

        # hands-free ON + wake word → full pipeline, wake word stripped
        w, wdt = talk_wav("wake.wav", handsFree=True)
        assert w.get("wakeWordHit") is not False and w.get("reply"), f"gated wrongly: {w}"
        assert "hermes" not in (w.get("transcript") or "").lower(), \
            f"wake word should be stripped before LLM: {w.get('transcript')!r}"
        assert "time" in (w.get("transcript") or "").lower(), \
            f"command should survive stripping: {w.get('transcript')!r}"
        print(f"✅ hands-free wake word → full reply ({wdt:.1f}s), transcript={w['transcript']!r}")

        # hands-free OFF + no wake word → legacy behavior, full pipeline
        n, _ = talk_wav("nowake.wav")
        assert n.get("reply") and n.get("audio"), f"legacy mode broke: {n}"
        print("✅ hands-free off → legacy always-answer mode intact")

        # custom wake words are honored
        c, _ = talk_wav("wake.wav", handsFree=True, wakeWords=["computer"])
        assert c.get("wakeWordHit") is False, f"custom wakeWords ignored: {c}"
        print("✅ custom wakeWords honored")

        # --- streaming TTS: /api/ask + /api/talk with stream:true ---
        import http.client

        def stream_post(path, payload):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
            conn.request("POST", path, body=json.dumps(payload),
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            ctype = resp.getheader("Content-Type") or ""
            raw = resp.read().decode()
            conn.close()
            return ctype, [json.loads(l) for l in raw.splitlines() if l.strip()]

        ctype, msgs = stream_post("/api/ask", {
            "text": "Give me three short tips for better sleep.",
            "session": "test-session-" + os.urandom(4).hex(),
            "stream": True,
        })
        assert "ndjson" in ctype, f"expected ndjson, got {ctype}"
        types = [m["type"] for m in msgs]
        assert types[0] == "meta" and msgs[0].get("reply"), f"bad meta: {types}"
        chunks = [m for m in msgs if m["type"] == "chunk"]
        assert len(chunks) == msgs[0].get("sentences"), \
            f"chunk count {len(chunks)} != declared {msgs[0].get('sentences')}"
        assert len(chunks) >= 2, "multi-sentence reply should yield multiple chunks"
        for c in chunks:
            wav = base64.b64decode(c["audio"])
            assert wav[:4] == b"RIFF", "chunk is not a wav"
        assert types[-1] == "done" and "turns" in msgs[-1], f"no done line: {types}"
        print(f"✅ /api/ask stream: meta → {len(chunks)} wav chunks → done")

        # talk streaming carries the transcript in meta
        with open(os.path.join(ROOT, "test", "test.wav"), "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ctype, msgs = stream_post("/api/talk", {"audio": b64, "history": [], "stream": True})
        assert "ndjson" in ctype, f"talk stream expected ndjson, got {ctype}"
        assert msgs[0]["type"] == "meta" and msgs[0].get("transcript"), \
            f"talk stream meta missing transcript: {msgs[0]}"
        assert any(m["type"] == "chunk" for m in msgs), "talk stream had no audio"
        print(f"✅ /api/talk stream: transcript={msgs[0]['transcript']!r}, "
              f"{sum(1 for m in msgs if m['type'] == 'chunk')} chunks")

        # gated hands-free request stays plain JSON even with stream:true
        with open(os.path.join(ROOT, "test", "nowake.wav"), "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        conn.request("POST", "/api/talk", body=json.dumps({
            "audio": b64, "handsFree": True, "stream": True,
        }), headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        ctype = resp.getheader("Content-Type") or ""
        g = json.loads(resp.read().decode())
        conn.close()
        assert "json" in ctype and "ndjson" not in ctype, f"gate should be JSON: {ctype}"
        assert g.get("wakeWordHit") is False, f"gated stream call answered: {g}"
        print("✅ stream + hands-free gate → plain JSON, still silent")
        return 0
    finally:
        proc.kill()
        proc.wait()


if __name__ == "__main__":
    sys.exit(main())
