# Offline Voice Chat

Fully local voice interface for Hermes. No cloud APIs, no accounts, no telemetry.

```
mic → Whisper STT (sherpa-onnx) → Ollama LLM (qwen3.5) → Piper TTS (sherpa-onnx) → speaker
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python sherpa-onnx sounddevice numpy
# models (already downloaded, see models/ and voices/)
```

Requires [Ollama](https://ollama.com) running with `qwen3.5:4b` pulled (`ollama pull qwen3.5:4b`).

## Usage

```bash
.venv/bin/python pipeline.py                 # record 5s from mic, get spoken reply
.venv/bin/python pipeline.py --loop          # continuous conversation mode
.venv/bin/python pipeline.py input.wav --text-only   # from file, no speech out
.venv/bin/python stt.py --mic --seconds 8    # STT only
.venv/bin/python tts.py "Hello" --play       # TTS only
.venv/bin/python brain.py "your question"    # LLM only
```

## Architecture

| File | Role | Model |
|------|------|-------|
| `stt.py` | Speech → text | Whisper tiny.en (int8 ONNX) |
| `brain.py` | Text → text | qwen3.5:4b via Ollama (`think: false`) |
| `tts.py` | Text → speech | Piper en_US-lessac-medium |
| `pipeline.py` | Wires it all together | — |

All models run on-device via ONNX Runtime / Ollama. First run loads models into RAM (whisper ~75MB, piper ~60MB, qwen ~4GB).

## Roadmap

- [x] End-to-end offline pipeline (file-based)
- [x] Mic input via ffmpeg
- [ ] Voice Activity Detection (silence-based recording instead of fixed 5s)
- [ ] Wake word ("Hey Hermes") via openwakeword
- [ ] Streaming TTS (speak sentence-by-sentence while LLM generates)
- [ ] Web UI (FastAPI + Web Audio API)
- [ ] Conversation memory across sessions
