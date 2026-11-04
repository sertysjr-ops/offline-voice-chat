#!/usr/bin/env python
"""Tokenize wake phrases into keywords.txt entries.

The KWS model expects BPE-tokenized keywords (its own bpe.model). This is a
build-time tool — the shipped keywords.txt is repo-tracked, so sentencepiece
is only needed when ADDING phrases, never at runtime.

Usage:
  .venv/bin/python bin/gen_keywords.py "OK HERMES" ["COMPUTER" ...]

Appends tokenized lines to keywords.txt (dedupes, uppercases input).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kws import KEYWORDS_FILE, MODEL_DIR  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=os.path.join(MODEL_DIR, "bpe.model"))

    existing = set()
    if os.path.exists(KEYWORDS_FILE):
        with open(KEYWORDS_FILE) as f:
            existing = {line.strip() for line in f if line.strip()}

    added = []
    with open(KEYWORDS_FILE, "a") as f:
        for phrase in sys.argv[1:]:
            tokenized = " ".join(sp.encode(phrase.upper(), out_type=str))
            if tokenized and tokenized not in existing:
                f.write(tokenized + "\n")
                existing.add(tokenized)
                added.append((phrase.upper(), tokenized))

    for phrase, tokenized in added:
        print(f"+ {phrase}  →  {tokenized}")
    if not added:
        print("nothing new to add")
    print(f"keywords.txt now has {len(existing)} phrase(s)")


if __name__ == "__main__":
    main()
