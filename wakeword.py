"""Wake-word matching — pure stdlib, no model imports.

Shared by pipeline.py (CLI loop) and server.py (hands-free web mode).
"""

WAKE_WORDS = ("hey hermes", "hermes", "ok hermes")


def match_wake_word(transcript, wake_words=WAKE_WORDS):
    """Returns (matched, stripped_transcript).

    Matches a wake word at the start of the utterance (most natural:
    "hey hermes, what's the weather"). Also tolerates a trailing comma
    or period. The stripped transcript is what the LLM should hear.
    """
    lower = transcript.lower().strip()
    for w in wake_words:
        if lower.startswith(w):
            rest = transcript.strip()[len(w):].strip()
            if rest.startswith((",", ".")):
                rest = rest[1:].strip()
            return True, rest
    return False, transcript.strip()
