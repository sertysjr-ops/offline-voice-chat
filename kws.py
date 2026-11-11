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


# --- runtime keyword management -----------------------------------------
# keywords.txt is BPE-tokenized; adding/removing phrases needs the model's
# bpe.model + sentencepiece (venv-only dep, same as bin/gen_keywords.py).
# After any edit the cached spotter is dropped and rebuilt so the change
# is live without a server restart — and a broken file fails the request
# that caused it, not the next /api/wake call minutes later.

def _bpe():
    import sentencepiece as spm

    bpe_model = os.path.join(MODEL_DIR, "bpe.model")
    if not os.path.exists(bpe_model):
        raise RuntimeError("bpe.model missing from the KWS model dir")
    return spm.SentencePieceProcessor(model_file=bpe_model)


def _read_lines():
    if not os.path.exists(KEYWORDS_FILE):
        return []
    with open(KEYWORDS_FILE) as f:
        return [ln.strip() for ln in f if ln.strip()]


def reset_spotter():
    """Drop the cached spotter; next get_spotter() rebuilds from disk."""
    global _spotter
    _spotter = None


def list_keywords():
    """Human-readable wake phrases, detokenized via the model's bpe."""
    sp = _bpe()
    return [sp.decode(line.split()) for line in _read_lines()]


def add_keyword(phrase):
    """Tokenize + append a wake phrase. Returns the normalized phrase.

    Raises ValueError on empty/untokenizable input, KeyError on duplicate.
    """
    phrase = " ".join((phrase or "").upper().split())
    if not phrase:
        raise ValueError("empty phrase")
    sp = _bpe()
    tokenized = " ".join(sp.encode(phrase, out_type=str))
    if not tokenized:
        raise ValueError(f"could not tokenize {phrase!r}")
    lines = _read_lines()
    if tokenized in lines:
        raise KeyError(phrase)
    with open(KEYWORDS_FILE, "a") as f:
        f.write(tokenized + "\n")
    reset_spotter()
    get_spotter()  # rebuild NOW so a bad edit fails this request, not the next
    return phrase


def remove_keyword(phrase):
    """Remove a wake phrase by its human-readable form. Returns it.

    Raises KeyError when not present, ValueError when it is the last one
    (an empty keywords file would make every /api/wake a permanent reject).
    """
    target = " ".join((phrase or "").upper().split())
    if not target:
        raise ValueError("empty phrase")
    sp = _bpe()
    lines = _read_lines()
    decoded = [sp.decode(ln.split()) for ln in lines]
    if target not in decoded:
        raise KeyError(target)
    if len(lines) == 1:
        raise ValueError("cannot remove the last wake phrase")
    kept = [ln for ln, dec in zip(lines, decoded) if dec != target]
    with open(KEYWORDS_FILE, "w") as f:
        f.write("\n".join(kept) + "\n")
    reset_spotter()
    get_spotter()
    return target


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: .venv/bin/python kws.py <wav> [wav...]")
        sys.exit(1)
    for p in sys.argv[1:]:
        print(f"{p}: {detect_wav(p)!r}")
