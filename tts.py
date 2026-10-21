#!/usr/bin/env python3
"""Offline TTS via sherpa-onnx (Piper en_US-lessac-medium).

Usage:
    python tts.py "Hello world"                 # writes out.wav
    python tts.py "Hello world" -o hello.wav    # custom output
    python tts.py --play "Hello world"          # speak immediately (afplay)
"""
import argparse
import os
import re
import subprocess

import sherpa_onnx

VOICE_DIR = os.path.join(os.path.dirname(__file__), "voices", "vits-piper-en_US-lessac-medium")

# piper-lessac handles sentences of ~<200 chars well; longer than that the
# prosody gets rambling. Keep chunks in the sweet spot.
MAX_SENTENCE_CHARS = 180


def build_tts():
    return sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=os.path.join(VOICE_DIR, "en_US-lessac-medium.onnx"),
                    lexicon=os.path.join(VOICE_DIR, "lexicon.txt"),
                    tokens=os.path.join(VOICE_DIR, "tokens.txt"),
                    data_dir=os.path.join(VOICE_DIR, "espeak-ng-data"),
                )
            )
        )
    )


def split_sentences(text, max_chars=MAX_SENTENCE_CHARS):
    """Split prose into TTS-ready sentence chunks.

    Splits on sentence-final punctuation, then greedily merges short
    sentences (and hard-wraps overlong ones) so every chunk is speakable
    in one natural breath. Abbreviations like "Mr." are not handled —
    the voice assistant's replies are short and conversational.
    """
    parts = [p.strip() for p in re.split(r"(?<=[.!?;])\s+", text.strip()) if p.strip()]
    chunks = []
    for p in parts:
        # hard-wrap anything still too long at clause boundaries if possible
        while len(p) > max_chars:
            cut = p.rfind(", ", 0, max_chars)
            if cut < max_chars // 2:
                cut = p.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            chunks.append(p[:cut].strip().rstrip(","))
            p = p[cut:].strip().lstrip(",")
        if not p:
            continue
        if chunks and len(chunks[-1]) + len(p) + 1 <= max_chars:
            chunks[-1] = chunks[-1] + " " + p
        else:
            chunks.append(p)
    return chunks or [text.strip()]


def synthesize_pcm(tts, text, speed=1.0):
    """Generate audio in-memory. Returns (int16 numpy array, sample_rate)."""
    audio = tts.generate(text, sid=0, speed=speed)
    if audio.samples is None or len(audio.samples) == 0:
        raise RuntimeError("TTS produced no audio")
    import numpy as np
    pcm = (np.array(audio.samples) * 32767).clip(-32768, 32767).astype(np.int16)
    return pcm, audio.sample_rate


def pcm_to_wav_bytes(pcm, sample_rate):
    """Wrap raw int16 mono PCM in a WAV container, return bytes."""
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())
    return buf.getvalue()


def synthesize(text, out_path="out.wav", speed=1.0):
    tts = build_tts()
    pcm, sample_rate = synthesize_pcm(tts, text, speed)
    wav = pcm_to_wav_bytes(pcm, sample_rate)
    with open(out_path, "wb") as f:
        f.write(wav)
    return out_path, len(pcm) / sample_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("-o", "--output", default="out.wav")
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    path, dur = synthesize(args.text, args.output, args.speed)
    print(f"wrote {path} ({dur:.1f}s)")
    if args.play:
        subprocess.run(["afplay", path])


if __name__ == "__main__":
    main()
