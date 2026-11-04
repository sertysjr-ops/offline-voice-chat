"""Always-on wake-word detection via sherpa-onnx keyword spotting (KWS).

Unlike the transcript gate in wakeword.py (STT first, string-match second),
this listens on raw audio with a tiny 3.3M-param zipformer — cheap enough to
run continuously, and it catches the wake phrase even when STT would garble
it (whisper tiny loves "hey hermes" → "hey hermes"/"hey miss"/"a hermes"...).

Model (gitignored, ~14MB download — see README):
  models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/

Keywords live in keywords.txt (repo-tracked), one BPE-tokenized phrase per
line. Tokenize new phrases with: .venv/bin/python bin/gen_keywords.py "OK HERMES"
(sentencepiece + the model's bpe.model — build-time only).
"""
import os
import wave

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models", "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01")
KEYWORDS_FILE = os.path.join(ROOT, "keywords.txt")

_ENCODER = "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
_DECODER = "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
_JOINER = "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"

_spotter = None


def available():
    """True when the KWS model + keywords file are present."""
    return all(
        os.path.exists(os.path.join(MODEL_DIR, f))
        for f in ("tokens.txt", _ENCODER, _DECODER, _JOINER)
    ) and os.path.exists(KEYWORDS_FILE)


def get_spotter():
    global _spotter
    if _spotter is None:
        import sherpa_onnx

        if not available():
            raise RuntimeError(
                "KWS model missing — download it (see README, models/ section)"
            )
        _spotter = sherpa_onnx.KeywordSpotter(
            tokens=os.path.join(MODEL_DIR, "tokens.txt"),
            encoder=os.path.join(MODEL_DIR, _ENCODER),
            decoder=os.path.join(MODEL_DIR, _DECODER),
            joiner=os.path.join(MODEL_DIR, _JOINER),
            keywords_file=KEYWORDS_FILE,
            num_threads=2,
            provider="cpu",
        )
    return _spotter


def resample_16k(samples, sr):
    """Linear-resample float32 mono to 16kHz (numpy-only, good enough for KWS)."""
    if sr == 16000:
        return samples
    new_n = int(len(samples) / sr * 16000)
    if new_n == 0:
        return np.zeros(0, dtype=np.float32)
    return np.interp(
        np.linspace(0, len(samples) - 1, new_n), np.arange(len(samples)), samples
    ).astype(np.float32)


def read_wav(path):
    """wav → (float32 mono samples, sample_rate)."""
    with wave.open(path, "rb") as f:
        sr = f.getframerate()
        ch = f.getnchannels()
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32)
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    return pcm / 32768.0, sr


def detect(samples_f32_16k):
    """One utterance (16kHz float32 mono) → detected keyword string or None.

    Feeds the spotter in 100ms chunks exactly like the always-on loop would,
    so latency/behaviour here matches production streaming use.
    """
    spotter = get_spotter()
    stream = spotter.create_stream()
    chunk = 1600  # 100ms @ 16k
    for i in range(0, len(samples_f32_16k), chunk):
        stream.accept_waveform(16000, samples_f32_16k[i : i + chunk])
        while spotter.is_ready(stream):
            spotter.decode_stream(stream)
        result = spotter.get_result(stream)
        if result:
            return result
    return None


def detect_wav(path):
    """wav file (any rate/channels) → detected keyword string or None."""
    samples, sr = read_wav(path)
    return detect(resample_16k(samples, sr))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: .venv/bin/python kws.py <wav> [wav...]")
        sys.exit(1)
    for p in sys.argv[1:]:
        print(f"{p}: {detect_wav(p)!r}")
