"""One-off fixture generator: wake-word test wavs at 16kHz mono.

Run: .venv/bin/python test/gen_wake_fixtures.py
"""
import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tts import synthesize


def to16k(src, dst):
    with wave.open(src, "rb") as f:
        rate = f.getframerate()
        n = f.getnframes()
        pcm = np.frombuffer(f.readframes(n), dtype=np.int16).astype(np.float32)
    ratio = 16000 / rate
    out_len = int(len(pcm) * ratio)
    idx = np.arange(out_len) / ratio
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, len(pcm) - 1)
    frac = idx - i0
    out = (pcm[i0] + (pcm[i1] - pcm[i0]) * frac).astype(np.int16)
    with wave.open(dst, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(out.tobytes())
    print(f"{dst}: {out_len / 16000:.1f}s @16k mono")


if __name__ == "__main__":
    synthesize("what time is it right now", "/tmp/nowake.wav")
    synthesize("hey hermes, what time is it right now", "/tmp/wake.wav")
    to16k("/tmp/nowake.wav", os.path.join(HERE, "nowake.wav"))
    to16k("/tmp/wake.wav", os.path.join(HERE, "wake.wav"))
