#!/usr/bin/env python3
"""End-to-end offline voice pipeline:

    mic (or wav) → Whisper STT → Ollama LLM → Piper TTS → speaker

Usage:
    python pipeline.py                  # one-shot: record 5s, answer, speak
    python pipeline.py --loop           # keep listening until Ctrl+C
    python pipeline.py --loop --kws     # always-on: KWS pre-gate skips STT
                                        # unless the wake word is in the audio
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
from wakeword import WAKE_WORDS, match_wake_word


def kws_gate(path):
    """Audio-native wake-word pre-gate. True = proceed to STT+LLM.

    Runs the 3.3M KWS model on the raw audio BEFORE the (much pricier)
    whisper decode: reject is ~0.0-0.5s vs 2-4s for STT+gated transcript.
    Fails open when the KWS model is missing (falls back to the
    transcript gate downstream), so it can never lock the user out.
    """
    try:
        import kws
        if not kws.available():
            return True
        return kws.detect_wav(path) is not None
    except Exception as e:  # fail open — transcript gate still applies
        print(f"  ⚠ KWS gate error ({e}) — proceeding without it", file=sys.stderr)
        return True


def speak(text, play=True):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    path, dur = synthesize(text, tmp.name)
    print(f"  🔊 ({dur:.1f}s)")
    if play:
        subprocess.run(["afplay", path])
    os.unlink(path)


def run_once(audio_path, text_only=False, verbose=True, use_kws=False):
    if use_kws and not kws_gate(audio_path):
        print("  …(no wake word in audio — KWS gate, STT skipped)", file=sys.stderr)
        return None
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


def loop(use_kws=False):
    print("Voice loop ready. Say 'hey hermes' to talk. Ctrl+C to quit.")
    if use_kws:
        print("KWS pre-gate on: STT only runs when the wake word is heard.")
    history = []
    while True:
        path = record_mic(5)
        if use_kws and not kws_gate(path):
            os.unlink(path)
            print("  …(no wake word — KWS, STT skipped)", file=sys.stderr)
            continue
        heard = transcribe(path)
        os.unlink(path)
        if not heard.strip():
            continue
        matched, command = match_wake_word(heard)
        if not matched:
            print(f"  …({heard}) — no wake word, ignored", file=sys.stderr)
            continue
        if not command:
            print("  …wake word only — what's the command?", file=sys.stderr)
            speak("Yes? I'm listening.")
            continue
        print(f"🗣️  You: {command}")
        history.append({"role": "user", "content": command})
        reply = ask(command, history=history[-6:])
        history.append({"role": "assistant", "content": reply})
        print(f"🤖 Hermes: {reply}")
        speak(reply)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?", help="16kHz mono wav (skip to use mic)")
    ap.add_argument("--loop", action="store_true", help="continuous mode")
    ap.add_argument("--kws", action="store_true",
                    help="KWS pre-gate: skip STT unless the wake word is in the audio")
    ap.add_argument("--text-only", action="store_true", help="skip TTS")
    ap.add_argument("--seconds", type=int, default=5, help="mic record length")
    args = ap.parse_args()

    if args.loop:
        loop(use_kws=args.kws)
        return

    path = args.audio
    if not path:
        print(f"Recording {args.seconds}s — speak now!", file=sys.stderr)
        path = record_mic(args.seconds)
        print("Done recording.", file=sys.stderr)
    run_once(path, text_only=args.text_only, use_kws=args.kws)


if __name__ == "__main__":
    main()
