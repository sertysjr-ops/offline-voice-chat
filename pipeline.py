#!/usr/bin/env python3
"""End-to-end offline voice pipeline:

    mic (or wav) → Whisper STT → Ollama LLM → Piper TTS → speaker

Usage:
    python pipeline.py                  # one-shot: record 5s, answer, speak
    python pipeline.py --loop           # keep listening until Ctrl+C
    python pipeline.py input.wav        # from file, speaks the reply
    python pipeline.py input.wav --text-only   # print reply, no TTS
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

from stt import transcribe, record_mic
from brain import ask
from tts import synthesize

WAKE_WORDS = ("hey hermes", "hermes", "ok hermes")


def speak(text, play=True):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    path, dur = synthesize(text, tmp.name)
    print(f"  🔊 ({dur:.1f}s)")
    if play:
        subprocess.run(["afplay", path])
    os.unlink(path)


def run_once(audio_path, text_only=False, verbose=True):
    if verbose:
        print("🎙️  Transcribing…", file=sys.stderr)
    heard = transcribe(audio_path)
    if not heard.strip():
        print("🤷 Heard nothing — try again.", file=sys.stderr)
        return None
    print(f"🗣️  You: {heard}")

    if verbose:
        print("🧠 Thinking…", file=sys.stderr)
    reply = ask(heard)
    print(f"🤖 Hermes: {reply}")

    if not text_only:
        speak(reply)
    return reply


def loop():
    print("Voice loop ready. Speak after the beep. Ctrl+C to quit.")
    history = []
    while True:
        path = record_mic(5)
        heard = transcribe(path)
        os.unlink(path)
        if not heard.strip():
            continue
        lower = heard.lower()
        if not any(lower.startswith(w) for w in WAKE_WORDS):
            # still answer everything in v0 — wake-word filtering comes later
            pass
        print(f"🗣️  You: {heard}")
        history.append({"role": "user", "content": heard})
        reply = ask(heard, history=history[-6:])
        history.append({"role": "assistant", "content": reply})
        print(f"🤖 Hermes: {reply}")
        speak(reply)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?", help="16kHz mono wav (skip to use mic)")
    ap.add_argument("--loop", action="store_true", help="continuous mode")
    ap.add_argument("--text-only", action="store_true", help="skip TTS")
    ap.add_argument("--seconds", type=int, default=5, help="mic record length")
    args = ap.parse_args()

    if args.loop:
        loop()
        return

    path = args.audio
    if not path:
        print(f"Recording {args.seconds}s — speak now!", file=sys.stderr)
        path = record_mic(args.seconds)
        print("Done recording.", file=sys.stderr)
    run_once(path, text_only=args.text_only)


if __name__ == "__main__":
    main()
