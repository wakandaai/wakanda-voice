# Wakanda Voice Pipeline

Configurable cascaded voice pipeline for African languages. Supports two modes:

- **S2ST** — Cascaded speech-to-speech translation (STT → MT → TTS)
- **Multilingual Voice Bot** — Voice assistant with tool calling (STT → MT → LLM → MT → TTS)

## Architecture

```
Browser/Mic → [LiveKit Server] → Orchestrator → Model Servers → [LiveKit Server] → Browser/Speaker
                                      │
                                      ├── STT Server  (ws://localhost:8001)
                                      ├── MT Server   (ws://localhost:8002)
                                      ├── TTS Server  (ws://localhost:8003)
                                      └── LLM Server  (http://localhost:8000)  [vLLM]
```

## Baseline Models

| Role | Model | Languages |
|------|-------|-----------|
| ASR  | facebook/mms-1b-all | 1,100+ |
| MT   | facebook/nllb-200-3.3B | 200 |
| TTS  | facebook/mms-tts-{lang} | 1,100+ |
| LLM  | almanach/AfriqueLLM-Qwen2.5-3B | African + EN |

All models are swappable via config.

## Quick Start

### 1. Install dependencies

```bash
pip install -e ".[dev]"
```

### 2. Start model servers (in separate terminals)

```bash
# STT
python servers/stt_server.py --model facebook/mms-1b-all --language yor

# MT
python servers/mt_server.py --model facebook/nllb-200-3.3B

# TTS
python servers/tts_server.py --model facebook/mms-tts-eng
```

### 3. Test the pipeline

```bash
# With microphone (requires sounddevice)
python scripts/test_pipeline.py --config configs/default.yaml

# With audio file
python scripts/test_pipeline.py --config configs/default.yaml --file test.wav
```

### 4. (Later) Run with LiveKit

```bash
# Start LiveKit Server
livekit-server --dev

# Start orchestrator
python orchestrator/main.py --config configs/default.yaml
```

## Configuration

Edit `configs/default.yaml` to swap models, change languages, or switch pipeline mode.

```yaml
mode: s2st  # or "multilingual_voice_bot"
default_lang: yor

stt:
  url: ws://localhost:8001
  model: facebook/mms-1b-all    # swap to openai/whisper-large-v3, etc.
  streaming: false

mt:
  url: ws://localhost:8002
  model: facebook/nllb-200-3.3B
  streaming: false
```

## Project Structure

```
wakanda-voice/
├── servers/           # Model server wrappers (WebSocket)
│   ├── base.py        # BaseModelServer template
│   ├── stt_server.py  # STT (MMS-1B / Whisper)
│   ├── mt_server.py   # MT (NLLB)
│   └── tts_server.py  # TTS (MMS-TTS)
├── orchestrator/      # Pipeline orchestration
│   ├── orchestrator.py
│   ├── vad.py         # Silero VAD
│   ├── config.py      # YAML config loader
│   ├── session_logger.py
│   ├── clients/       # Model server clients
│   │   ├── ws_client.py
│   │   └── llm_client.py
│   └── pipelines/     # Pipeline implementations
│       ├── s2st.py
│       └── multilingual_bot.py
├── configs/           # YAML configurations
├── scripts/           # Test and utility scripts
├── frontend/          # Browser UI (Phase 5)
└── logs/              # Session logs (JSONL)
```
