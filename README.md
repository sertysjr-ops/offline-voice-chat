# Offline Voice Chat

Fully local voice interface for Hermes. No cloud APIs, no accounts, no telemetry.

```
mic → Whisper STT (sherpa-onnx) → Ollama LLM (qwen3.5) → Piper TTS (sherpa-onnx) → speaker
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python sherpa-onnx sounddevice numpy
# models (already downloaded, see models/ and voices/):
#   whisper tiny.en  → models/sherpa-onnx-whisper-tiny.en/
#   piper lessac     → voices/vits-piper-en_US-lessac-medium/
#   piper libritts   → voices/vits-piper-en_US-libritts_r-medium/  (904 speakers)
#   kws gigaspeech   → models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/
curl -sL -o /tmp/libritts.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2 \
  && tar -xjf /tmp/libritts.tar.bz2 -C voices/ && rm /tmp/libritts.tar.bz2
# always-on wake-word detector (KWS, ~14MB):
curl -sL -o /tmp/kws.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2 \
  && tar -xjf /tmp/kws.tar.bz2 -C models/ && rm /tmp/kws.tar.bz2
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
- `POST /api/talk`    → `{audio, session?, handsFree?, speed?, voice?, speaker?}` → `{transcript, reply, audio, voice}`
- `POST /api/ask`     → `{text, session?, speed?, voice?, speaker?}` → `{reply, audio, voice}` (text-only path, skips STT)
- `GET  /api/voices`  → installed voice models + named speaker presets (`libritts:nova`, `:forge`, `:pixel`, `:onyx`, `:wisp`)
- `POST /api/wake`    → `{audio}` → `{awake, keyword}` — audio-native wake-word detection via sherpa KWS (3.3M params, ~0.5s, no STT). The always-on gate: cheap enough to run on every mic frame. `503` if the KWS model isn't downloaded.
- `GET  /api/health`  → `{ok, models}`

### Wake words: two gates

- **`wakeword.py`** (transcript gate) — STT first, then string-match "hey hermes".
  Flexible (custom `wakeWords` per request) but pays full STT latency on every
  frame and can miss when whisper garbles the phrase.
- **`kws.py`** (audio gate) — keyword spotting directly on raw audio. Fast
  (~0.5s), robust to STT mishearings, this is what an always-on listener
  should run. Keywords live in `keywords.txt` (BPE-tokenized; add phrases
  with `.venv/bin/python bin/gen_keywords.py "OK HERMES"` — needs
  `sentencepiece` in the venv, build-time only).
- Try it: `.venv/bin/python kws.py test/wake.wav test/nowake.wav`

Voice prefs: per-request `speed` (0.5–2.5×) and `voice`/`speaker` win; otherwise the
session's saved profile applies (persisted server-side in `data/sessions.json`).
Unknown voice names fall back to `lessac` rather than erroring.

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
