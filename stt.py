#!/usr/bin/env python3
"""Offline STT via sherpa-onnx (Whisper tiny.en, int8).

Usage:
    python stt.py path/to/audio.wav
    python stt.py --mic        # record 5s from default mic
"""
import sys
import argparse
import wave
import subprocess
import tempfile
import os

import sherpa_onnx
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "sherpa-onnx-whisper-tiny.en")


def build_recognizer():
    return sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=os.path.join(MODEL_DIR, "tiny.en-encoder.int8.onnx"),
        decoder=os.path.join(MODEL_DIR, "tiny.en-decoder.int8.onnx"),
        tokens=os.path.join(MODEL_DIR, "tiny.en-tokens.txt"),
        num_threads=4,
        decoding_method="greedy_search",
    )


def read_wave(path):
    with wave.open(path, "rb") as f:
        assert f.getnchannels() == 1, "need mono"
        assert f.getsampwidth() == 2, "need 16-bit PCM"
        sample_rate = f.getframerate()
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    assert sample_rate == 16000, f"need 16kHz, got {sample_rate}"
    return samples.astype(np.float32) / 32768.0


def record_mic(seconds=5):
    """Record from the default microphone to a 16kHz mono wav via ffmpeg."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-f", "avfoundation", "-i", ":0",
         "-t", str(seconds), "-ar", "16000", "-ac", "1", tmp.name],
        check=True, capture_output=True,
    )
    return tmp.name


def transcribe(path):
    samples = read_wave(path)
    recognizer = build_recognizer()
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, samples)
    recognizer.decode_stream(stream)
    return stream.result.text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?", help="16kHz mono wav file")
    ap.add_argument("--mic", action="store_true", help="record 5s from mic")
    ap.add_argument("--seconds", type=int, default=5)
    args = ap.parse_args()

    path = args.audio
    if args.mic or not path:
        if args.mic:
            print(f"Recording {args.seconds}s from mic…", file=sys.stderr)
            path = record_mic(args.seconds)
        else:
            ap.error("provide a wav file or --mic")

    text = transcribe(path)
    print(text)


if __name__ == "__main__":
    main()
