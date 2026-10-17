#!/usr/bin/env python3
"""Offline TTS via sherpa-onnx (Piper en_US-lessac-medium).

Usage:
    python tts.py "Hello world"                 # writes out.wav
    python tts.py "Hello world" -o hello.wav    # custom output
    python tts.py --play "Hello world"          # speak immediately (afplay)
"""
import argparse
import os
import subprocess

import sherpa_onnx

VOICE_DIR = os.path.join(os.path.dirname(__file__), "voices", "vits-piper-en_US-lessac-medium")


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


def synthesize(text, out_path="out.wav", speed=1.0):
    tts = build_tts()
    audio = tts.generate(text, sid=0, speed=speed)
    if audio.samples is None or len(audio.samples) == 0:
        raise RuntimeError("TTS produced no audio")
    # sherpa returns float32 samples at the model sample rate
    import wave
    import numpy as np
    pcm = (np.array(audio.samples) * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(out_path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(audio.sample_rate)
        f.writeframes(pcm.tobytes())
    return out_path, len(audio.samples) / audio.sample_rate


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
