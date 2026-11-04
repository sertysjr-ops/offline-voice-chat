#!/usr/bin/env python3
"""Offline voice chat — web UI server. Stdlib only, no pip installs.

Serves a single-page browser UI and the voice pipeline API:

    GET  /                 → index.html
    POST /api/talk         → JSON {audio: base64 16kHz mono wav, session?: "id",
                             handsFree?: bool, wakeWords?: string[],
                             speed?: 0.5–2.5, voice?: "lessac"|"libritts"|"libritts:preset",
                             speaker?: int}
                             → {transcript, reply, audio, turns, voice}
                             handsFree on + no wake word →
                             {transcript, wakeWordHit: false, reply: null, audio: null}
                             add "stream": true → NDJSON v2 (true streaming):
                             meta line (transcript, voice — no reply yet),
                             one chunk line per spoken sentence (base64 WAV +
                             sentence text) sent THE MOMENT its TTS finishes —
                             while the LLM is still generating later sentences —
                             then a done line with the full reply + sentence
                             count. Playback starts mid-generation.
    POST /api/ask          → JSON {text, session?, speed?, voice?, speaker?}
                             → {reply, audio, turns, voice}
                             "stream": true → same NDJSON protocol minus transcript.
    POST /api/clear        → JSON {session} → {ok}   (wipe conversation)
    POST /api/stt/partial  → JSON {stream: id, pcm: base64 s16le 16k mono,
                             final?: bool}
                             → {partial, seconds, decoded}
                             final=True → +{transcript, final:true}, buffer freed.
                             The browser streams chunks WHILE recording; the
                             growing buffer is re-decoded so the transcript
                             converges mid-speech, and the final decode is
                             ready the moment you stop talking.
    GET  /api/voices       → installed voice models + libritts speaker presets

    Voice prefs: per-request fields win; otherwise the session's saved
    profile applies (persisted in data/sessions.json alongside history).
    Speed is clamped to 0.5–2.5; unknown voice names fall back to lessac.

Models load lazily on first request (whisper + piper take a few seconds).

Usage:
    .venv/bin/python server.py [--port 8787]
"""
import argparse
import base64
import json
import os
import queue
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# RLock: TTS synthesis runs under this lock and itself calls get_tts(),
# which acquires it again — reentrant so same-thread nesting is safe.
_lock = threading.RLock()
_recognizer = None
_tts_cache: dict[str, object] = {}


def get_recognizer():
    global _recognizer
    if _recognizer is None:
        import stt
        _recognizer = stt.build_recognizer()
    return _recognizer


def get_tts(voice="lessac"):
    """One TTS instance per voice model, built lazily under the lock."""
    import tts as tts_mod
    key = voice if voice in tts_mod.list_voices() else "lessac"
    with _lock:
        if key not in _tts_cache:
            _tts_cache[key] = tts_mod.build_tts(key)
    return key, _tts_cache[key]


def resolve_voice(payload, sid=None):
    """Pick (voice, speaker, speed) for a request.

    Payload fields win; otherwise the session profile applies; otherwise
    defaults. When a session id is present and the payload carried any
    voice fields, they are persisted into the profile.
    """
    import sessions

    prof = sessions.get_profile(sid) if sid else dict(sessions.DEFAULT_PROFILE)
    speed = prof["speed"]
    voice = prof["voice"]
    speaker = prof["speaker"]

    if payload.get("speed") is not None:
        speed = sessions.clamp_speed(payload.get("speed"))
    if isinstance(payload.get("voice"), str) and payload["voice"].strip():
        voice = payload["voice"].strip()[:32]
    if payload.get("speaker") is not None:
        try:
            speaker = max(0, int(payload["speaker"]))
        except (TypeError, ValueError):
            pass

    # multi-speaker preset names → concrete speaker ids ("libritts:nova")
    if ":" in voice:
        model, _, tag = voice.partition(":")
        import tts as tts_mod
        presets = tts_mod.list_voices().get(model, {}).get("presets", {})
        if tag in presets:
            voice, speaker = model, presets[tag]

    # validate against installed models, fall back gracefully
    import tts as tts_mod
    voices = tts_mod.list_voices()
    if voice not in voices:
        voice = "lessac" if "lessac" in voices else next(iter(voices), "lessac")
        speaker = 0
    elif not voices[voice].get("multi_speaker"):
        speaker = 0
    else:
        speaker = min(speaker, voices[voice]["num_speakers"] - 1)

    if sid and (
        payload.get("speed") is not None
        or isinstance(payload.get("voice"), str)
        or payload.get("speaker") is not None
    ):
        sessions.set_profile(sid, speed=speed, voice=voice, speaker=speaker)

    return voice, speaker, speed


def decode_samples(samples_f32):
    """float32 16kHz mono samples → text. The one whisper call site."""
    with _lock:  # sherpa recognizers are not thread-safe
        recognizer = get_recognizer()
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples_f32)
        recognizer.decode_stream(stream)
        return stream.result.text.strip()


def transcribe(audio_b64):
    """wav → text. Shared by the streaming and buffered talk paths."""
    import tempfile
    import numpy as np

    wav_bytes = base64.b64decode(audio_b64)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(wav_bytes)
    tmp.close()

    import wave
    with wave.open(tmp.name, "rb") as f:
        assert f.getframerate() == 16000 and f.getnchannels() == 1, "need 16kHz mono wav"
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)

    return decode_samples(samples.astype("float32") / 32768.0)


# --- live partial STT ---------------------------------------------------
# The browser streams raw s16le PCM chunks WHILE recording; each chunk is
# appended to a per-stream buffer and the whole buffer is re-decoded, so
# the transcript converges as you speak. On final=True the decode is the
# definitive one and the client can go straight to /api/ask with the text,
# skipping the record-then-decode wait entirely.
_partial_lock = threading.Lock()
_partials: dict[str, dict] = {}  # stream_id -> {pcm, text, decoded}
PARTIAL_MAX_BYTES = 16000 * 2 * 30       # 30s of s16le 16kHz mono
PARTIAL_MIN_NEW = int(16000 * 2 * 0.7)   # don't re-decode for <0.7s of new audio
PARTIAL_MAX_STREAMS = 20


def partial_transcribe(stream_id, pcm_bytes, final=False):
    """Append PCM to the stream's buffer, re-decode if enough new audio.

    Returns (text, seconds_buffered, did_decode). final=True always decodes
    and drops the buffer afterwards.
    """
    with _partial_lock:
        st = _partials.setdefault(stream_id, {"pcm": bytearray(), "text": "", "decoded": 0})
        st["pcm"] += pcm_bytes
        if len(st["pcm"]) > PARTIAL_MAX_BYTES:  # sliding window, keep the tail
            overflow = len(st["pcm"]) - PARTIAL_MAX_BYTES
            del st["pcm"][:overflow]
            st["decoded"] = max(0, st["decoded"] - overflow)
        if not final and (len(st["pcm"]) - st["decoded"]) < PARTIAL_MIN_NEW:
            return st["text"], len(st["pcm"]) / 32000.0, False
        pcm = bytes(st["pcm"])
        while len(_partials) > PARTIAL_MAX_STREAMS:  # evict oldest
            _partials.pop(next(iter(_partials)))

    import numpy as np
    if pcm:
        samples = np.frombuffer(pcm, dtype=np.int16).astype("float32") / 32768.0
        text = decode_samples(samples)
    else:
        text = ""

    with _partial_lock:
        st = _partials.get(stream_id)
        if st is not None:
            st["text"] = text
            st["decoded"] = len(pcm)  # snapshot length, not live length
            if final:
                del _partials[stream_id]
    return text, len(pcm) / 32000.0, True


def apply_wake_gate(transcript, hands_free=False, wake_words=None):
    """Hands-free gate. Returns (gated_response_or_None, cleaned_transcript)."""
    from wakeword import WAKE_WORDS, match_wake_word

    if not hands_free:
        return None, transcript
    matched, command = match_wake_word(transcript, wake_words or WAKE_WORDS)
    if not matched:
        return {
            "transcript": transcript,
            "wakeWordHit": False,
            "reply": None,
            "audio": None,
        }, None
    # LLM hears the command, not the wake word
    return None, (command or transcript)


def synthesize(text, out_path, speed=1.0, voice="lessac", speaker=0):
    import tts as tts_mod
    key, engine = get_tts(voice)
    return tts_mod.synthesize_with(text, out_path, engine, speed, speaker)


def talk(audio_b64, history, hands_free=False, wake_words=None,
         speed=1.0, voice="lessac", speaker=0):
    """Full pipeline: wav → STT → [wake-word gate] → LLM → TTS.

    With hands_free on, a transcript without a wake word returns early:
    {transcript, wakeWordHit: false, reply: null, audio: None} — no LLM call,
    no TTS, sub-second response. The wake word is stripped before the LLM
    sees the command.
    """
    import tempfile
    from brain import ask

    transcript = transcribe(audio_b64)
    if not transcript:
        return {"transcript": "", "reply": "I didn't catch that — try again.", "audio": None}

    gated, transcript = apply_wake_gate(transcript, hands_free, wake_words)
    if gated:
        return gated

    reply = ask(transcript, history=history or None)

    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out.close()
    with _lock:
        path, _dur = synthesize(reply, out.name, speed, voice, speaker)
    with open(path, "rb") as f:
        audio_b64_out = base64.b64encode(f.read()).decode()

    return {"transcript": transcript, "reply": reply, "audio": audio_b64_out}


def _resolve_history(payload):
    """Server-side session memory wins; client-sent history is the fallback.

    Returns (history_for_llm, session_id_or_None).
    """
    import sessions
    sid = (payload.get("session") or "").strip() or None
    if sid:
        return sessions.get_history(sid), sid
    return payload.get("history") or None, None


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

    # --- NDJSON v2 streaming (HTTP/1.0 close-delimited: no Content-Length,
    # the client reads until EOF). Line order:
    #   meta  {type, transcript?, voice}                — sent immediately,
    #                                                     reply not known yet
    #   chunk {type, i, text, audio}                    — per spoken sentence,
    #                                                     sent the moment its
    #                                                     TTS finishes
    #   done  {type, reply, sentences, turns, voice}    — full reply lands here
    #   error {type, message}                           — mid-stream failure
    # ---
    def _stream_begin(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.close_connection = True

    def _stream_line(self, obj):
        self.wfile.write(json.dumps(obj).encode() + b"\n")
        self.wfile.flush()

    # Matches a sentence boundary: terminal punctuation followed by whitespace.
    _SENT_BOUNDARY = re.compile(r"(?<=[.!?;])\s")

    def _stream_pipeline(self, text, history, speed=1.0, voice="lessac",
                         speaker=0, max_chunks=8):
        """True-streaming LLM→TTS: consumes the token stream, hands each
        completed sentence to a TTS worker, and writes its chunk line to the
        wire IMMEDIATELY — chunk 1 typically hits the client while the model
        is still generating sentence 2+.

        Callers must have sent meta already. Returns (full_reply, n_sent).
        Mid-stream TTS failures raise; the caller turns them into an
        NDJSON error line (the 200 header is already on the wire, so an
        HTTP 500 is no longer possible).
        """
        import tts as tts_mod
        from brain import ask_stream

        _key, engine = get_tts(voice)
        ready: "queue.Queue" = queue.Queue()   # (i, sentence) → worker
        out: "queue.Queue" = queue.Queue()     # (i, sentence, wav) | error ← worker

        def synth_worker():
            while True:
                item = ready.get()
                if item is None:
                    return
                i, sentence = item
                try:
                    with _lock:
                        pcm, sr = tts_mod.synthesize_pcm(engine, sentence, speed, speaker)
                    out.put((i, sentence, base64.b64encode(
                        tts_mod.pcm_to_wav_bytes(pcm, sr)).decode()))
                except BaseException as e:
                    out.put(e)

        worker = threading.Thread(target=synth_worker, daemon=True)
        worker.start()

        emitted = 0

        def drain(block=False):
            """Write whatever the TTS worker has finished to the wire.

            block=True waits for at least one finished chunk, then keeps
            draining NON-blocking. The block must not persist past the
            first get: with a blocking get on every iteration, drain
            looped back onto the emptied queue after the final chunk and
            hung forever — the worker had already exited via the None
            sentinel, so nothing would ever arrive. That deadlocked the
            post-generation `while emitted < n_sent` loop and the done
            line never shipped.
            """
            nonlocal emitted
            while True:
                try:
                    item = out.get(timeout=120) if block else out.get(block=False)
                except queue.Empty:
                    if block:
                        # 120s with no chunk while we were told more are
                        # coming — surface as a stream error, not a hang.
                        raise RuntimeError("timed out waiting for TTS chunk")
                    return
                if isinstance(item, BaseException):
                    raise item
                i, sentence, wav = item
                self._stream_line({"type": "chunk", "i": i,
                                   "text": sentence, "audio": wav})
                emitted += 1
                block = False  # only the first get may block

        parts: list[str] = []
        pending = ""
        n_sent = 0
        for tok in ask_stream(text, history=history):
            parts.append(tok)
            pending += tok
            while n_sent < max_chunks:
                m = self._SENT_BOUNDARY.search(pending)
                if not m:
                    break
                sentence = pending[:m.start()].strip()
                pending = pending[m.end():]
                if sentence:
                    ready.put((n_sent, sentence))
                    n_sent += 1
            drain()  # flush anything TTS finished between tokens
        tail = pending.strip()
        if tail and n_sent < max_chunks:
            ready.put((n_sent, tail))
            n_sent += 1
        ready.put(None)
        while emitted < n_sent:  # worker may still be synthesizing the tail
            drain(block=True)
        worker.join()
        return "".join(parts).strip(), n_sent

    def _stream_reply(self, text, history, sid, voice_info, transcript=None):
        """Shared NDJSON v2 flow for /api/ask and /api/talk: meta → live
        chunks (voice + session turn) → done. Errors mid-stream become an
        NDJSON error line instead of an HTTP 500."""
        voice, speaker, speed = voice_info["voice"], voice_info["speaker"], voice_info["speed"]
        self._stream_begin()
        meta = {"type": "meta", "voice": voice_info}
        if transcript is not None:
            meta["transcript"] = transcript
        self._stream_line(meta)
        try:
            reply, n_sent = self._stream_pipeline(
                text, history, speed=speed, voice=voice, speaker=speaker)
        except Exception as e:
            self._stream_line({"type": "error", "message": str(e)})
            return
        import sessions
        n = sessions.append_turn(sid, text, reply) if sid else 0
        self._stream_line({"type": "done", "reply": reply, "sentences": n_sent,
                           "turns": n // 2, "voice": voice_info})

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(os.path.dirname(__file__), "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/api/health":
            self._send(200, json.dumps({"ok": True, "models": {"stt": _recognizer is not None, "tts": bool(_tts_cache)}}))
        elif self.path == "/api/voices":
            import tts as tts_mod
            voices = tts_mod.list_voices()
            self._send(200, json.dumps({
                "voices": [
                    {
                        "name": name,
                        "multiSpeaker": cfg["multi_speaker"],
                        "numSpeakers": cfg["num_speakers"],
                        "presets": cfg["presets"],
                    }
                    for name, cfg in voices.items()
                ]
            }))
        else:
            self._send(404, '{"error": "not found"}')

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
        except Exception:
            self._send(400, '{"error": "bad json"}')
            return

        if self.path == "/api/clear":
            sid = (payload.get("session") or "").strip()
            if not sid:
                self._send(400, '{"error": "session required"}')
                return
            import sessions
            sessions.clear(sid)
            self._send(200, json.dumps({"ok": True, "session": sid, "turns": 0}))
            return

        if self.path == "/api/stt/partial":
            # Live partial STT: the browser feeds raw s16le PCM (16kHz mono)
            # while recording; we re-decode the growing buffer so the
            # transcript converges mid-speech. final=True returns the
            # definitive transcript and frees the buffer.
            stream_id = (payload.get("stream") or "").strip()[:64]
            if not stream_id:
                self._send(400, '{"error": "stream id required"}')
                return
            try:
                pcm = base64.b64decode(payload.get("pcm") or "", validate=True)
            except Exception:
                self._send(400, '{"error": "pcm must be base64 s16le"}')
                return
            final = bool(payload.get("final"))
            try:
                text, secs, did_decode = partial_transcribe(stream_id, pcm, final)
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}))
                return
            resp = {"partial": text, "seconds": round(secs, 2), "decoded": did_decode}
            if final:
                resp.update({"final": True, "transcript": text})
            self._send(200, json.dumps(resp))
            return

        if self.path == "/api/ask":
            # text-only path: skip STT, LLM → TTS
            text = (payload.get("text") or "").strip()
            if not text:
                self._send(400, '{"error": "text required"}')
                return
            # hands-free clients that already have a transcript (live partial
            # STT path) still get the wake-word gate, server-side
            if payload.get("handsFree"):
                gated, cleaned = apply_wake_gate(
                    text, True, payload.get("wakeWords") or None)
                if gated:
                    self._send(200, json.dumps(gated))
                    return
                text = cleaned
            try:
                history, sid = _resolve_history(payload)
                voice, speaker, speed = resolve_voice(payload, sid)
                voice_info = {"voice": voice, "speaker": speaker, "speed": speed}

                if payload.get("stream"):
                    # true streaming: meta goes out now, chunks hit the wire
                    # as each sentence's TTS finishes (mid-generation)
                    self._stream_reply(text, history, sid, voice_info)
                    return

                import tempfile
                from brain import ask
                reply = ask(text, history=history)
                import sessions
                n = sessions.append_turn(sid, text, reply) if sid else 0

                out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                out.close()
                with _lock:
                    path, _dur = synthesize(reply, out.name, speed, voice, speaker)
                with open(path, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode()
                self._send(200, json.dumps({
                    "reply": reply, "audio": audio_b64, "turns": n // 2,
                    "voice": voice_info,
                }))
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}))
            return

        if self.path != "/api/talk":
            self._send(404, '{"error": "not found"}')
            return
        try:
            history, sid = _resolve_history(payload)
            voice, speaker, speed = resolve_voice(payload, sid)
            voice_info = {"voice": voice, "speaker": speaker, "speed": speed}

            if payload.get("stream"):
                # streaming variant: STT + wake gate happen before the stream
                # starts; if gated we fall back to a plain JSON response
                transcript = transcribe(payload["audio"])
                if not transcript:
                    self._send(200, json.dumps({
                        "transcript": "",
                        "reply": "I didn't catch that — try again.",
                        "audio": None,
                    }))
                    return
                gated, transcript = apply_wake_gate(
                    transcript,
                    bool(payload.get("handsFree")),
                    payload.get("wakeWords") or None,
                )
                if gated:
                    self._send(200, json.dumps(gated))
                    return
                assert transcript is not None  # not gated ⇒ transcript survived

                self._stream_reply(transcript, history or None, sid, voice_info,
                                   transcript=transcript)
                return

            result = talk(
                payload["audio"],
                history,
                hands_free=bool(payload.get("handsFree")),
                wake_words=payload.get("wakeWords") or None,
                speed=speed, voice=voice, speaker=speaker,
            )
            result["voice"] = voice_info
            if sid and result.get("transcript") and result.get("reply"):
                import sessions
                n = sessions.append_turn(sid, result["transcript"], result["reply"])
                result["turns"] = n // 2
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
