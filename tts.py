#!/usr/bin/env python3
"""Offline TTS via sherpa-onnx (Piper voices).

Voices (auto-detected from voices/ — add a piper model dir and it appears):
    lessac   en_US-lessac-medium      single-speaker default
    libritts en_US-libritts_r-medium  904 speakers — pick with speaker_id

Usage:
    python tts.py "Hello world"                    # writes out.wav (lessac)
    python tts.py "Hello world" --voice libritts --speaker 42
    python tts.py "Hello world" --speed 1.3 -o fast.wav
    python tts.py --play "Hello world"             # speak immediately (afplay)
"""
import argparse
import os
import re
import subprocess

import sherpa_onnx

VOICES_DIR = os.path.join(os.path.dirname(__file__), "voices")

# piper-lessac handles sentences of ~<200 chars well; longer than that the
# prosody gets rambling. Keep chunks in the sweet spot.
MAX_SENTENCE_CHARS = 180

# How the libritts multi-speaker model's speakers are surfaced in the UI.
LIBRITTS_PRESETS = {
    "nova": 19,    # warm female
    "forge": 121,  # deep male
    "pixel": 307,  # bright, quick
    "onyx": 522,   # gravelly male
    "wisp": 655,   # soft female
}


def _find_piper_dir(prefix):
    if not os.path.isdir(VOICES_DIR):
        return None
    for entry in sorted(os.listdir(VOICES_DIR)):
        path = os.path.join(VOICES_DIR, entry)
        if entry.startswith(prefix) and os.path.isdir(path):
            return path
    return None


def list_voices():
    """Voice configs available right now (model dirs actually on disk)."""
    out = {}
    lessac = _find_piper_dir("vits-piper-en_US-lessac")
    if lessac:
        out["lessac"] = {
            "dir": lessac,
            "multi_speaker": False,
            "num_speakers": 1,
            "presets": {},
        }
    lib = _find_piper_dir("vits-piper-en_US-libritts")
    if lib:
        out["libritts"] = {
            "dir": lib,
            "multi_speaker": True,
            "num_speakers": 904,
            "presets": dict(LIBRITTS_PRESETS),
        }
    return out


def build_tts(voice="lessac"):
    voices = list_voices()
    cfg = voices.get(voice) or voices.get("lessac")
    if cfg is None:
        raise RuntimeError(
            "no piper voice found — see README for model download links"
        )
    d = cfg["dir"]
    onnx_name = next(
        (f for f in os.listdir(d) if f.endswith(".onnx") and not f.endswith(".json")),
        None,
    )
    if not onnx_name:
        raise RuntimeError(f"no .onnx model in {d}")
    lexicon = os.path.join(d, "lexicon.txt")
    return sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=os.path.join(d, onnx_name),
                    lexicon=lexicon if os.path.exists(lexicon) else "",
                    tokens=os.path.join(d, "tokens.txt"),
                    data_dir=os.path.join(d, "espeak-ng-data"),
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


def synthesize_pcm(tts, text, speed=1.0, speaker_id=0):
    """Generate audio in-memory. Returns (int16 numpy array, sample_rate)."""
    audio = tts.generate(text, sid=speaker_id, speed=speed)
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


def synthesize_with(text, out_path, tts_engine, speed=1.0, speaker_id=0):
    """Synthesize with an already-built engine (server caches engines per voice)."""
    pcm, sample_rate = synthesize_pcm(tts_engine, text, speed, speaker_id)
    wav = pcm_to_wav_bytes(pcm, sample_rate)
    with open(out_path, "wb") as f:
        f.write(wav)
    return out_path, len(pcm) / sample_rate


def synthesize(text, out_path="out.wav", speed=1.0, voice="lessac", speaker_id=0):
    tts = build_tts(voice)
    return synthesize_with(text, out_path, tts, speed, speaker_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("-o", "--output", default="out.wav")
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--voice", default="lessac", choices=sorted(list_voices()) or ["lessac"])
    ap.add_argument("--speaker", type=int, default=0, help="speaker id (multi-speaker voices)")
    args = ap.parse_args()

    path, dur = synthesize(args.text, args.output, args.speed, args.voice, args.speaker)
    print(f"wrote {path} ({dur:.1f}s, voice={args.voice}, speaker={args.speaker}, speed={args.speed})")
    if args.play:
        subprocess.run(["afplay", path])


if __name__ == "__main__":
    main()
