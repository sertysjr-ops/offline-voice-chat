#!/usr/bin/env python3
"""Local LLM chat via Ollama (works offline once model is pulled).

Usage:
    from brain import ask
    print(ask("What is the capital of France?"))
"""
import os
import json
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("VOICE_LLM_MODEL", "qwen3.5:4b")

SYSTEM_PROMPT = (
    "You are a concise voice assistant. Answer in 1-3 short sentences, "
    "conversationally, no markdown, no bullet points, nothing that would "
    "sound weird when read aloud."
)


def _messages(user_text, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages


def ask(user_text, history=None):
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps({
            "model": MODEL,
            "think": False,
            "messages": _messages(user_text, history),
            "stream": False,
            "options": {"temperature": 0.6, "num_predict": 220},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as res:
        data = json.loads(res.read())
    return data["message"]["content"].strip()


def ask_stream(user_text, history=None):
    """Yield reply text chunks as the model generates them.

    Same request as ask() but with stream:true — Ollama answers NDJSON,
    one object per token group. Used by the server to start TTS on the
    first complete sentence while the rest of the reply still generates.
    """
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps({
            "model": MODEL,
            "think": False,
            "messages": _messages(user_text, history),
            "stream": True,
            "options": {"temperature": 0.6, "num_predict": 220},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as res:
        for line in res:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            chunk = (data.get("message") or {}).get("content") or ""
            if chunk:
                yield chunk
            if data.get("done"):
                break


def main():
    import sys
    q = " ".join(sys.argv[1:]) or "Say hello like a pirate."
    print(ask(q))


if __name__ == "__main__":
    main()
