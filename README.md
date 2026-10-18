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

.venv/bin/python server.py                   # web UI + JSON API → http://localhost:8787
.venv/bin/python test/test_server.py         # integration test: boots server, hits every endpoint
```

## Web UI & API

`server.py` (stdlib `http.server`, zero extra deps) serves:

- `GET  /`            → single-page chat UI (`index.html`) with push-to-talk
- `POST /api/talk`    → `{audio: base64 16kHz-mono wav, history}` → `{transcript, reply, audio}`
- `POST /api/ask`     → `{text, history}` → `{reply, audio}` (text-only path, skips STT)
- `GET  /api/health`  → `{ok, models}`

The UI records via Web Audio API, resamples to 16kHz mono, encodes WAV in JS,
and auto-stops on silence (browser-side VAD: RMS < threshold for ~1.4s, 15s max).
Replies come back as base64 WAV and autoplay in the chat. Plain-text fallback
input included. 6-turn conversation memory is kept client-side.

## Architecture

| File | Role | Model |
|------|------|-------|
| `stt.py` | Speech → text | Whisper tiny.en (int8 ONNX) |
| `brain.py` | Text → text | qwen3.5:4b via Ollama (`think: false`) |
| `tts.py` | Text → speech | Piper en_US-lessac-medium |
| `pipeline.py` | Wires it all together | — |
| `server.py` | Stdlib HTTP server: web UI + `/api/talk` + `/api/ask` | — |
| `index.html` | Chat UI: Web Audio recording, browser VAD, WAV playback | — |

All models run on-device via ONNX Runtime / Ollama. First run loads models into RAM (whisper ~75MB, piper ~60MB, qwen ~4GB).

## Roadmap

- [x] End-to-end offline pipeline (file-based)
- [x] Mic input via ffmpeg
- [x] Voice Activity Detection — browser-side (RMS silence auto-stop); CLI still fixed-length
- [x] Web UI (stdlib HTTP server + Web Audio API, no FastAPI needed)
- [x] `/api/ask` text-only endpoint for the UI's text fallback
- [ ] Wake word ("Hey Hermes") via openwakeword (constants in `pipeline.py` WAKE_WORDS)
- [ ] Streaming TTS (speak sentence-by-sentence while LLM generates)
- [ ] Conversation memory across sessions
