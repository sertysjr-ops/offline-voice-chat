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
        assert health.get("kws", {}).get("available") is True, \
            f"health must report KWS availability: {health}"

        # --- KWS pre-warm: a background thread loads the spotter at boot,
        # so /api/wake never pays model-load latency on its first call ---
        t0 = time.time()
        ready = health.get("kws", {}).get("ready") is True
        while not ready and time.time() - t0 < 60:
            time.sleep(0.5)
            h2 = json.load(os.popen(
                f'curl -s -m 5 http://127.0.0.1:{port}/api/health'))
            ready = h2.get("kws", {}).get("ready") is True
        assert ready, "KWS pre-warm never finished (spotter not loaded)"
        print(f"✅ KWS pre-warm: spotter ready {time.time()-t0:.1f}s after boot")

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
        # index.html served + push-to-talk wiring present (client-side feature,
        # so we assert the key handlers exist in the served page)
        html = subprocess.run(
            ["curl", "-s", "-m", "5", f"http://127.0.0.1:{port}/"],
            capture_output=True, text=True).stdout
        assert "<title>Hermes Voice" in html, "index.html not served"
        for token in ("pttActive", "e.code !== 'Space'", "window.addEventListener('keyup'",
                      "release SPACE to send", "window.addEventListener('blur'"):
            assert token in html, f"PTT wiring missing: {token!r}"
        print("✅ /api/ask + index.html + push-to-talk wiring PASS")

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

        # --- streaming TTS v2: /api/ask + /api/talk with stream:true ---
        # v2: meta carries NO reply (not generated yet); each chunk line
        # (audio + sentence text) is written the moment its TTS finishes;
        # done carries the full reply + sentence count.
        import http.client

        def stream_post(path, payload):
            """POST and read the NDJSON stream line by line.
            Returns (ctype, msgs, arrival_times)."""
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
            t0 = time.time()
            conn.request("POST", path, body=json.dumps(payload),
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            ctype = resp.getheader("Content-Type") or ""
            msgs, times = [], []
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if line:
                    msgs.append(json.loads(line))
                    times.append(time.time() - t0)
            conn.close()
            return ctype, msgs, times

        ctype, msgs, times = stream_post("/api/ask", {
            "text": "Give me three short tips for better sleep.",
            "session": "test-session-" + os.urandom(4).hex(),
            "stream": True,
        })
        assert "ndjson" in ctype, f"expected ndjson, got {ctype}"
        types = [m["type"] for m in msgs]
        assert types[0] == "meta", f"bad first line: {types}"
        assert msgs[0].get("voice") and "reply" not in msgs[0], \
            f"v2 meta must carry voice but no reply yet: {msgs[0]}"
        done = msgs[-1]
        chunks = [m for m in msgs if m["type"] == "chunk"]
        assert done["type"] == "done" and done.get("reply") and "turns" in done, \
            f"done must carry the full reply: {done}"
        assert len(chunks) == done.get("sentences"), \
            f"chunk count {len(chunks)} != done.sentences {done.get('sentences')}"
        # true streaming: meta must hit the wire well before generation ends
        t_meta, t_done = times[0], times[-1]
        assert t_done - t_meta > 0.5, \
            f"meta→done gap {t_done - t_meta:.2f}s — stream waited for full generation?"
        print(f"✅ /api/ask stream v2: meta at {t_meta:.1f}s, done at {t_done:.1f}s "
              f"(gap {t_done - t_meta:.1f}s proves live streaming)")
        if len(chunks) < 2:
            # small models sometimes answer in one sentence — force the issue
            # once; the structural invariant above already held either way.
            ctype, msgs, _ = stream_post("/api/ask", {
                "text": "List three colors. Use exactly three separate sentences, "
                        "one per color, each ending with a period.",
                "session": "test-session-" + os.urandom(4).hex(),
                "stream": True,
            })
            chunks = [m for m in msgs if m["type"] == "chunk"]
            done = msgs[-1]
        assert len(chunks) >= 2, "multi-sentence reply should yield multiple chunks"
        for c in chunks:
            wav = base64.b64decode(c["audio"])
            assert wav[:4] == b"RIFF", "chunk is not a wav"
            assert c.get("text"), "chunk missing its sentence text"
        assert chunks[0]["i"] == 0 and chunks[-1]["i"] == len(chunks) - 1, \
            "chunk indexes must be sequential"
        # progressive text: chunk sentences are reflected in the done reply
        assert chunks[0]["text"].split()[0] in done["reply"], \
            f"chunk text not reflected in final reply: {chunks[0]['text']!r}"
        print(f"✅ /api/ask stream: {len(chunks)} wav chunks, progressive text ok")

        # --- streaming TTS v3: stream:"pcm" → raw s16le fragments emitted
        # WHILE a sentence synthesizes (sherpa callback), not after ---
        ctype, msgs, times = stream_post("/api/ask", {
            "text": "List three colors. Use exactly three separate sentences, "
                    "one per color, each ending with a period.",
            "session": "test-session-" + os.urandom(4).hex(),
            "stream": "pcm",
        })
        assert "ndjson" in ctype, f"pcm stream expected ndjson, got {ctype}"
        meta = msgs[0]
        assert meta["type"] == "meta" and meta.get("pcm", {}).get("rate") == 22050, \
            f"v3 meta missing pcm rate: {meta}"
        sbegins = [m for m in msgs if m["type"] == "sbegin"]
        sends = [m for m in msgs if m["type"] == "send"]
        pcms = [m for m in msgs if m["type"] == "pcm"]
        done3 = msgs[-1]
        assert done3["type"] == "done" and done3.get("reply"), f"no done: {done3}"
        assert len(sbegins) >= 2, \
            f"multi-sentence reply → multiple sbegin, got {len(sbegins)}"
        assert len(sends) == len(sbegins) == done3.get("sentences"), \
            f"sbegin/send/done.sentences mismatch: {len(sbegins)}/{len(sends)}/{done3.get('sentences')}"
        assert len(pcms) >= len(sbegins), \
            "every sentence needs at least one pcm fragment"
        total_pcm = 0
        for m in pcms:
            raw = base64.b64decode(m["data"])
            assert len(raw) > 0 and len(raw) % 2 == 0, "pcm fragment not s16le"
            total_pcm += len(raw) // 2
        secs = sum(m["seconds"] for m in sends)
        assert abs(total_pcm / 22050 - secs) < 0.2, \
            f"pcm samples {total_pcm} ≉ send seconds {secs}"
        # ordering: fragments stream BETWEEN sbegin and send of a sentence
        i0 = sbegins[0]["i"]
        pos_sb = next(n for n, m in enumerate(msgs)
                      if m["type"] == "sbegin" and m["i"] == i0)
        pos_pcm = next(n for n, m in enumerate(msgs)
                       if m["type"] == "pcm" and m["i"] == i0)
        pos_send = next(n for n, m in enumerate(msgs)
                        if m["type"] == "send" and m["i"] == i0)
        assert pos_sb < pos_pcm < pos_send, \
            "fragments must arrive between sbegin and send"
        print(f"✅ /api/ask stream v3 pcm: {len(sbegins)} sentences, "
              f"{len(pcms)} fragments, {secs:.1f}s audio, rate={meta['pcm']['rate']}")

        # --- pre-warm: LLM tokens must arrive incrementally (that overlap
        # is what lets TTS start on sentence 1 mid-generation) ---
        from brain import ask_stream
        t0 = time.time()
        first_tok_at = None
        toks = []
        for c in ask_stream("Count from one to five, one number per sentence."):
            if first_tok_at is None:
                first_tok_at = time.time()
            toks.append(c)
        total = time.time() - t0
        assert toks and "".join(toks).strip(), "ask_stream yielded nothing"
        assert first_tok_at is not None
        assert len(toks) > 1 and first_tok_at - t0 < total * 0.9, \
            f"no incremental yield: first token {first_tok_at-t0:.1f}s of {total:.1f}s ({len(toks)} toks)"
        print(f"✅ ask_stream pre-warm: first token {first_tok_at-t0:.1f}s, "
              f"total {total:.1f}s, {len(toks)} token chunks")

        # talk streaming carries the transcript in meta; full reply in done
        with open(os.path.join(ROOT, "test", "test.wav"), "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ctype, msgs, _ = stream_post("/api/talk", {"audio": b64, "history": [], "stream": True})
        assert "ndjson" in ctype, f"talk stream expected ndjson, got {ctype}"
        assert msgs[0]["type"] == "meta" and msgs[0].get("transcript"), \
            f"talk stream meta missing transcript: {msgs[0]}"
        assert "reply" not in msgs[0], "v2 meta must not wait for the reply"
        assert msgs[-1]["type"] == "done" and msgs[-1].get("reply"), \
            f"talk stream done missing reply: {msgs[-1]}"
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

        # --- voice prefs: unit-level profile checks (no server needed) ---
        import sessions
        assert sessions.clamp_speed(99) == 2.5, "speed clamp high"
        assert sessions.clamp_speed(0.1) == 0.5, "speed clamp low"
        assert sessions.clamp_speed("junk") == 1.0, "speed clamp junk"
        pu = sessions.set_profile("unit-prof", speed=1.6, voice="libritts", speaker=121)
        assert pu["speed"] == 1.6 and pu["voice"] == "libritts" and pu["speaker"] == 121
        gu = sessions.get_profile("unit-prof")
        assert gu["speaker"] == 121 and gu["speed"] == 1.6, f"profile roundtrip: {gu}"
        gd = sessions.get_profile("never-seen")
        assert gd == {"speed": 1.0, "voice": "lessac", "speaker": 0}, f"defaults: {gd}"
        print("✅ profile unit checks (clamp bounds, set/get roundtrip, defaults)")

        # --- /api/voices lists installed models + presets ---
        rv = subprocess.run(
            ["curl", "-s", "-m", "10", f"http://127.0.0.1:{port}/api/voices"],
            capture_output=True, text=True).stdout
        dv = json.loads(rv)
        voices = {v["name"]: v for v in dv.get("voices", [])}
        assert "lessac" in voices, f"lessac missing: {voices}"
        assert "libritts" in voices, "libritts (multi-speaker) missing"
        assert voices["libritts"]["multiSpeaker"] is True
        assert voices["libritts"]["numSpeakers"] == 904
        assert "nova" in voices["libritts"]["presets"]
        print("✅ /api/voices: lessac + libritts(904 spk, presets) listed")

        # --- TTS speed: faster speed → shorter wav, same text (unit-level) ---
        import tts as tts_mod
        probe = "The quick brown fox jumps over the lazy dog."
        eng = tts_mod.build_tts("lessac")
        tmp_slow, d_slow = tts_mod.synthesize_with(probe, os.path.join(ROOT, "test", "_sp_slow.wav"), eng, 0.7, 0)
        tmp_fast, d_fast = tts_mod.synthesize_with(probe, os.path.join(ROOT, "test", "_sp_fast.wav"), eng, 1.8, 0)
        os.unlink(tmp_slow); os.unlink(tmp_fast)
        assert d_fast < d_slow * 0.75, f"speed had no effect: 0.7x={d_slow:.2f}s 1.8x={d_fast:.2f}s"
        print(f"✅ TTS speed control: 0.7× → {d_slow:.1f}s, 1.8× → {d_fast:.1f}s")

        # --- API-level: preset resolution + per-session profile persistence ---
        def ask_json(payload):
            raw = subprocess.run(
                ["curl", "-s", "-m", "180", "-X", "POST",
                 f"http://127.0.0.1:{port}/api/ask",
                 "-H", "Content-Type: application/json",
                 "-d", json.dumps(payload)],
                capture_output=True, text=True).stdout
            return json.loads(raw)

        vsid = "test-voice-" + os.urandom(4).hex()
        a1 = ask_json({"text": "Reply with exactly: OK.",
                       "session": vsid, "voice": "libritts:forge", "speed": 1.4})
        assert a1.get("reply") and a1.get("audio"), f"voice ask failed: {a1}"
        v1 = a1.get("voice") or {}
        assert v1.get("voice") == "libritts" and v1.get("speaker") == 121, \
            f"preset 'forge' should resolve to speaker 121: {v1}"
        assert v1.get("speed") == 1.4, f"speed echo: {v1}"
        print(f"✅ preset resolution: libritts:forge → speaker {v1['speaker']}, speed echo {v1['speed']}×")

        # second call, NO voice fields → session profile applies
        a2 = ask_json({"text": "Reply with exactly: OK again.", "session": vsid})
        v2 = a2.get("voice") or {}
        assert v2.get("voice") == "libritts" and v2.get("speaker") == 121 and \
            v2.get("speed") == 1.4, f"profile should persist across calls: {v2}"
        wav = base64.b64decode(a2["audio"])
        assert wav[:4] == b"RIFF", "profiled reply is not a wav"
        print("✅ per-session voice profile persists (libritts #121 @ 1.4× on 2nd call)")

        # unknown voice name falls back to lessac instead of 500ing
        a3 = ask_json({"text": "Reply with exactly: OK.", "voice": "nonexistent-model"})
        v3 = a3.get("voice") or {}
        assert a3.get("audio") and v3.get("voice") == "lessac", \
            f"unknown voice should fall back to lessac: {v3}"
        print("✅ unknown voice name → graceful lessac fallback")

        # index.html carries the voice controls
        html2 = subprocess.run(
            ["curl", "-s", "-m", "5", f"http://127.0.0.1:{port}/"],
            capture_output=True, text=True).stdout
        for token in ("voiceSel", "voicePayload", "/api/voices", 'id="speed"'):
            assert token in html2, f"voice UI wiring missing: {token!r}"
        print("✅ index.html voice controls (selector + speed slider) wired")

        # --- live partial STT: stream PCM chunks, transcript converges ---
        def partial(payload):
            raw = subprocess.run(
                ["curl", "-s", "-m", "60", "-X", "POST",
                 f"http://127.0.0.1:{port}/api/stt/partial",
                 "-H", "Content-Type: application/json",
                 "-d", json.dumps(payload)],
                capture_output=True, text=True).stdout
            return json.loads(raw)

        # wake.wav is the longest fixture (2.25s) — split its raw PCM at a
        # 2-byte boundary and feed it in two chunks, like the browser does
        import wave as _wave
        with _wave.open(os.path.join(ROOT, "test", "wake.wav"), "rb") as f:
            raw_pcm = f.readframes(f.getnframes())
        half = (len(raw_pcm) // 2) & ~1  # keep the split sample-aligned
        psid = "partial-" + os.urandom(4).hex()

        p1 = partial({"stream": psid,
                      "pcm": base64.b64encode(raw_pcm[:half]).decode()})
        assert "partial" in p1 and p1.get("seconds", 0) > 1.0, \
            f"first partial chunk bad: {p1}"
        # 1.1s of new audio ≥ the 0.7s threshold → a real decode happened
        assert p1.get("decoded") is True, f"expected decode on ≥0.7s new audio: {p1}"
        print(f"✅ partial chunk 1: {p1['seconds']}s buffered, decoded={p1['decoded']}, "
              f"partial={p1['partial']!r}")

        p2 = partial({"stream": psid,
                      "pcm": base64.b64encode(raw_pcm[half:]).decode(),
                      "final": True})
        assert p2.get("final") is True and p2.get("transcript"), \
            f"final partial missing transcript: {p2}"
        # determinism check: whisper on the same concatenated audio must
        # match the /api/talk transcript of the whole wav
        assert "time" in p2["transcript"].lower(), \
            f"partial-final transcript off: {p2['transcript']!r}"
        print(f"✅ partial final: transcript={p2['transcript']!r} "
              f"(deltas concatenated == whole-file decode)")

        # buffer freed after final — same stream id starts empty
        p3 = partial({"stream": psid, "pcm": "", "final": True})
        assert p3.get("transcript") == "", \
            f"buffer should be freed after final: {p3}"
        print("✅ partial buffer freed after final decode")

        # tiny appends below the 0.7s threshold don't trigger a re-decode
        p4 = partial({"stream": psid + "b",
                      "pcm": base64.b64encode(raw_pcm[:1600]).decode()})  # 0.05s
        assert p4.get("decoded") is False, f"tiny append should skip decode: {p4}"
        print("✅ sub-threshold append skips re-decode (decoded=false)")

        # validation
        import urllib.request
        import urllib.error
        def partial_status(payload):
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/stt/partial",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status
            except urllib.error.HTTPError as e:
                return e.code
        assert partial_status({"pcm": "eA=="}) == 400, "missing stream id → 400"
        assert partial_status({"stream": "x", "pcm": "!!!notb64"}) == 400, \
            "bad base64 → 400"
        print("✅ /api/stt/partial validation (stream id + base64)")

        # --- hands-free gate on /api/ask (the live-partial fast path) ---
        g1 = ask_json({"text": "what time is it right now", "handsFree": True})
        assert g1.get("wakeWordHit") is False and g1.get("reply") is None, \
            f"/api/ask hands-free gate failed: {g1}"
        print("✅ /api/ask + handsFree: no wake word → silent gate")

        g2 = ask_json({"text": "hey hermes, reply with exactly: OK.",
                       "handsFree": True,
                       "session": "test-hf-" + os.urandom(4).hex()})
        assert g2.get("reply") and g2.get("audio"), \
            f"/api/ask hands-free wake word should answer: {g2}"
        print(f"✅ /api/ask + handsFree: wake word → answer {g2['reply'][:40]!r}")

        # --- /api/wake: audio-native KWS wake-word detection (no STT) ---
        import kws
        assert kws.available(), "KWS model missing — see README"

        def wake_wav(path):
            with open(os.path.join(ROOT, "test", path), "rb") as f:
                b = base64.b64encode(f.read()).decode()
            t0 = time.time()
            raw = subprocess.run(
                ["curl", "-s", "-m", "30", "-X", "POST",
                 f"http://127.0.0.1:{port}/api/wake",
                 "-H", "Content-Type: application/json",
                 "-d", json.dumps({"audio": b})],
                capture_output=True, text=True).stdout
            return json.loads(raw), time.time() - t0

        kw, kdt = wake_wav("wake.wav")
        assert kw.get("awake") is True and kw.get("keyword"), \
            f"KWS missed the wake word: {kw}"
        print(f"✅ /api/wake: wake.wav → awake, keyword={kw['keyword']!r} ({kdt:.1f}s)")

        kn, ndt = wake_wav("nowake.wav")
        assert kn.get("awake") is False and kn.get("keyword") is None, \
            f"KWS false positive: {kn}"
        print(f"✅ /api/wake: nowake.wav → silent ({ndt:.1f}s)")

        # KWS is the always-on gate — it must be cheaper than the STT gate
        assert kdt < 5 and ndt < 5, \
            f"KWS too slow for always-on: {kdt:.1f}s/{ndt:.1f}s"
        print("✅ /api/wake latency within always-on budget")

        # unit-level: detect() on raw float32 agrees with the endpoint
        samples, sr = kws.read_wav(os.path.join(ROOT, "test", "wake.wav"))
        assert kws.detect(kws.resample_16k(samples, sr)), "unit detect() missed"
        print("✅ kws.detect unit-level (wav → float32 → keyword)")

        # --- /api/wake/keywords: runtime wake-phrase management ---
        # keywords.txt is repo-tracked — snapshot and restore byte-exactly
        kw_file = os.path.join(ROOT, "keywords.txt")
        with open(kw_file, "rb") as f:
            kw_orig = f.read()

        def kw_req(method, payload=None):
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/wake/keywords",
                data=(json.dumps(payload).encode()
                      if payload is not None else None),
                headers={"Content-Type": "application/json"},
                method=method)
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    return r.status, json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read().decode() or "{}")

        try:
            st, body = kw_req("GET")
            assert st == 200 and body.get("keywords") == ["HEY HERMES"], \
                f"GET keywords: {st} {body}"
            print(f"✅ GET /api/wake/keywords: {body['keywords']}")

            st, body = kw_req("POST", {"phrase": "computer"})
            assert st == 201 and body.get("added") == "COMPUTER" and \
                len(body.get("keywords", [])) == 2, f"POST add: {st} {body}"
            print(f"✅ POST add 'computer' → {body['keywords']}")

            st, _ = kw_req("POST", {"phrase": "hey hermes"})
            assert st == 409, f"duplicate add should be 409, got {st}"
            st, _ = kw_req("POST", {"phrase": "  "})
            assert st == 400, f"empty phrase should be 400, got {st}"
            print("✅ duplicate → 409, empty phrase → 400")

            # the rebuilt spotter (done in-request) still detects
            kw2, _ = wake_wav("wake.wav")
            assert kw2.get("awake") is True, \
                f"spotter broken after keyword add: {kw2}"
            print("✅ spotter rebuilt live — wake.wav still triggers")

            st, body = kw_req("DELETE", {"phrase": "computer"})
            assert st == 200 and body.get("keywords") == ["HEY HERMES"], \
                f"DELETE: {st} {body}"
            st, _ = kw_req("DELETE", {"phrase": "computer"})
            assert st == 404, f"absent phrase should be 404, got {st}"
            st, _ = kw_req("DELETE", {"phrase": "hey hermes"})
            assert st == 400, f"removing the last phrase should be 400, got {st}"
            print("✅ DELETE: removed → 200, absent → 404, last-one-left → 400")
        finally:
            with open(kw_file, "wb") as f:
                f.write(kw_orig)
            kws.reset_spotter()
        with open(kw_file, "rb") as f:
            assert f.read() == kw_orig, "keywords.txt not restored"
        print("✅ keywords.txt restored byte-identical (repo-tracked file)")

        # index.html live-partial wiring
        html3 = subprocess.run(
            ["curl", "-s", "-m", "5", f"http://127.0.0.1:{port}/"],
            capture_output=True, text=True).stdout
        for token in ("sendPartial", "/api/stt/partial", "recStreamId",
                      "askTranscript"):
            assert token in html3, f"live-partial wiring missing: {token!r}"
        print("✅ index.html live-partial STT wiring present")

        # index.html hands-free KWS pre-gate wiring (browser calls /api/wake
        # concurrently with the final decode; 503 disables the check)
        for token in ("/api/wake", "kwsAvailable", "wake.awake === false",
                      "not for me"):
            assert token in html3, f"hands-free KWS wiring missing: {token!r}"
        print("✅ index.html hands-free KWS pre-gate wiring present")

        # --- always-on watch: a 2.0s rolling window must still trigger KWS ---
        # (the browser ships exactly such windows from startWatch/pollWake)
        import wave, struct
        samples, sr = kws.read_wav(os.path.join(ROOT, "test", "wake.wav"))
        s16 = kws.resample_16k(samples, sr)
        win = s16[: int(16000 * 2.0)]
        assert len(win) < len(s16), "fixture shorter than the watch window?"
        tmp = os.path.join(ROOT, "test", ".watchwin.wav")
        with wave.open(tmp, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(b"".join(
                struct.pack("<h", max(-32768, min(32767, int(x * 32767))))
                for x in win))
        try:
            ww, wdt = wake_wav(".watchwin.wav")
        finally:
            os.unlink(tmp)
        assert ww.get("awake") is True, \
            f"2s watch window missed the wake word: {ww}"
        print(f"✅ always-on: 2.0s watch window still triggers KWS ({wdt:.1f}s)")

        # 2s window of the no-wake fixture must NOT false-positive (the watch
        # loop polls these constantly — a false hit would open the recorder)
        samples2, sr2 = kws.read_wav(os.path.join(ROOT, "test", "nowake.wav"))
        s162 = kws.resample_16k(samples2, sr2)
        win2 = s162[: int(16000 * 2.0)]
        tmp2 = os.path.join(ROOT, "test", ".watchwin2.wav")
        with wave.open(tmp2, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(b"".join(
                struct.pack("<h", max(-32768, min(32767, int(x * 32767))))
                for x in win2))
        try:
            wn, _ = wake_wav(".watchwin2.wav")
        finally:
            os.unlink(tmp2)
        assert wn.get("awake") is False, f"2s no-wake window false-positive: {wn}"
        print("✅ always-on: 2.0s no-wake window stays silent (no false hit)")

        # index.html always-on wiring (watch loop, gate-skip, resume)
        for token in ("startWatch", "pollWake", "skipGateOnce",
                      "maybeResumeWatch", "WATCH_WINDOW_S", "stopWatch"):
            assert token in html3, f"always-on wiring missing: {token!r}"
        print("✅ index.html always-on wake-watch wiring present")

        # index.html barge-in wiring: playback tracking, stream abort, the
        # interrupt path in pollWake, and the watch-during-turn hook
        for token in ("stopPlayback", "liveSources", "liveAudioEls",
                      "bargeIn", "activeAbort", "bargeFlag",
                      "watchDuringTurn", "interrupted",
                      "ac.signal", "echoCancellation"):
            assert token in html3, f"barge-in wiring missing: {token!r}"
        print("✅ index.html barge-in wiring present (abort + playback kill)")

        # barge-in during the THINKING phase: all three request sites start
        # the watch at request time (not just when the first audio lands)
        assert html3.count("watchDuringTurn()") >= 5, \
            "watch should start at request time in ask/sendAudio/sendText"
        print("✅ barge-in covers the thinking phase (watch starts at request time)")

        # --- KWS telemetry: every wake decision is logged + rolled up ---
        # unit-level: log/stats roundtrip against an isolated temp log, and
        # stats() must survive a torn/garbage line in the file
        import tempfile
        import telemetry
        orig_path = telemetry.LOG_PATH
        tmp_log = os.path.join(tempfile.mkdtemp(), "tel.jsonl")
        telemetry.LOG_PATH = tmp_log
        try:
            marker = "telemetry-probe-" + os.urandom(3).hex()
            telemetry.log("stt-gate", hit=True, word="hey hermes",
                          transcript=marker, source="unit")
            telemetry.log("wake", awake=False, keyword=None,
                          seconds=2.0, tookMs=12)
            with open(tmp_log, "a", encoding="utf-8") as f:
                f.write("{torn json line\n")  # a crashed write mid-line
            ust = telemetry.stats()
            assert ust["total"] == 2, f"torn line must be skipped: {ust}"
            assert ust["sttGate"]["calls"] == 1 and \
                ust["sttGate"]["hits"] == 1, f"stt-gate rollup: {ust}"
            assert ust["wake"]["calls"] == 1 and ust["wake"]["hits"] == 0, \
                f"wake rollup: {ust}"
            assert ust["recent"][0]["event"] == "wake", \
                f"recent must be newest-first: {ust['recent']}"
            assert any(e.get("transcript") == marker for e in ust["recent"]), \
                f"marker event missing from recent: {ust['recent']}"
        finally:
            telemetry.LOG_PATH = orig_path
        print("✅ telemetry unit: log→stats roundtrip, torn line tolerated")

        # endpoint rollup: the probes above (4 /api/wake calls, several
        # hands-free gate decisions) must all be reflected
        raw = subprocess.run(
            ["curl", "-s", "-m", "10",
             f"http://127.0.0.1:{port}/api/kws/stats"],
            capture_output=True, text=True).stdout
        ks = json.loads(raw)
        assert ks.get("wake", {}).get("calls", 0) >= 4, \
            f"wake calls not logged: {ks.get('wake')}"
        assert ks["wake"].get("hits", 0) >= 2 and ks["wake"].get("byKeyword"), \
            f"wake hits + per-keyword counts missing: {ks.get('wake')}"
        assert ks.get("sttGate", {}).get("calls", 0) >= 4, \
            f"stt-gate decisions not logged: {ks.get('sttGate')}"
        rejects = [e for e in ks.get("recent", [])
                   if e.get("event") == "stt-gate" and e.get("hit") is False]
        assert rejects and all("transcript" in e and "source" in e
                               for e in rejects), \
            f"gate rejects must carry transcript+source: {ks.get('recent')}"
        sources = {e.get("source") for e in ks.get("recent", [])
                   if e.get("event") == "stt-gate"}
        assert "ask" in sources and "talk" in sources, \
            f"gate sources must distinguish ask/talk: {sources}"
        wake_events = [e for e in ks.get("recent", [])
                       if e.get("event") == "wake"]
        assert wake_events and all(
            "seconds" in e and "tookMs" in e for e in wake_events), \
            f"wake events must carry duration+latency: {wake_events}"
        print(f"✅ /api/kws/stats: wake {ks['wake']['hits']}/{ks['wake']['calls']} hits, "
              f"gate rejectRate={ks['sttGate']['rejectRate']}, sources={sorted(sources)}")

        # --- pipeline.py --kws: audio-native pre-gate in the CLI ---
        # no-wake fixture → gate rejects, STT+LLM never run (fast exit)
        t0 = time.time()
        r = subprocess.run(
            [PY, os.path.join(ROOT, "pipeline.py"),
             os.path.join(ROOT, "test", "nowake.wav"), "--text-only", "--kws"],
            capture_output=True, text=True, timeout=60)
        kws_dt = time.time() - t0
        assert r.returncode == 0, f"pipeline --kws crashed: {r.stderr[-300:]}"
        assert "no wake word" in r.stderr.lower(), \
            f"gate should report the reject: {r.stderr[-300:]}"
        assert "Hermes:" not in r.stdout, f"gated run answered: {r.stdout[-200:]}"
        assert kws_dt < 20, f"gated run too slow (STT leaked in?): {kws_dt:.1f}s"
        print(f"✅ pipeline --kws: no-wake fixture gated in {kws_dt:.1f}s (no STT/LLM)")

        # wake fixture → gate opens, full STT→LLM reply comes out
        r2 = subprocess.run(
            [PY, os.path.join(ROOT, "pipeline.py"),
             os.path.join(ROOT, "test", "wake.wav"), "--text-only", "--kws"],
            capture_output=True, text=True, timeout=180)
        assert r2.returncode == 0, f"pipeline --kws wake run crashed: {r2.stderr[-300:]}"
        assert "Hermes:" in r2.stdout, \
            f"wake fixture should produce a reply: {r2.stdout[-200:]}"
        print(f"✅ pipeline --kws: wake fixture → reply {r2.stdout.strip()[-60:]!r}")

        # kws_gate fails open when the model dir is unreachable
        import pipeline as pl
        import kws as kws_mod
        orig = kws_mod.MODEL_DIR
        kws_mod.MODEL_DIR = os.path.join(ROOT, "models", "definitely-missing")
        try:
            assert pl.kws_gate(os.path.join(ROOT, "test", "nowake.wav")) is True, \
                "kws_gate must fail open when the model is missing"
        finally:
            kws_mod.MODEL_DIR = orig
        print("✅ kws_gate fails open on missing model (never locks the user out)")
        return 0
    finally:
        proc.kill()
        proc.wait()


if __name__ == "__main__":
    sys.exit(main())
