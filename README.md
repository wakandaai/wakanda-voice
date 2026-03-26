# Wakanda Voice Pipeline

A configurable cascaded speech pipeline for African languages. Supports two modes:

- **S2ST** — Speech-to-speech translation (STT → MT → TTS)
- **Multilingual Voice Bot** — Voice assistant with tool calling (STT → MT → LLM → MT → TTS)

Users select a task and language pair in the browser. Models load on demand.

## Architecture

```
Browser (mic/speaker)
    │
    │  WebRTC (LiveKit)
    ▼
Orchestrator (Python, CPU)
    │
    ├── STT Server  (ws://localhost:8001)  ← MMS-1B / Whisper / custom
    ├── MT Server   (ws://localhost:8002)  ← NLLB / custom
    ├── TTS Server  (ws://localhost:8003)  ← MMS-TTS / custom
    └── LLM Server  (http://localhost:8000) ← vLLM / Ollama / OpenAI API
```

All model servers communicate over WebSocket with a unified protocol.
Models are swappable via config — change the model name, restart, done.

## Baseline Models

| Role | Model | Languages |
|------|-------|-----------|
| ASR  | `facebook/mms-1b-all` | 1,100+ |
| MT   | `facebook/nllb-200-1.3B` | 200 |
| TTS  | `facebook/mms-tts-{lang}` | 1,100+ |
| LLM  | `almanach/AfriqueLLM-Qwen2.5-3B` | African + EN |
| VAD  | Silero VAD v5 | Language-agnostic |

## Supported Languages

| Language | Code | STT | MT | TTS |
|----------|------|-----|----|-----|
| Yoruba | `yor` | ✓ | ✓ | ✓ |
| Hausa | `hau` | ✓ | ✓ | ✓ |
| Igbo | `ibo` | ✓ | ✓ | ✓ |
| Bemba | `bem` | ✓ | ✓ | ✓ |
| Swahili | `swa` | ✓ | ✓ | ✓ |
| Kinyarwanda | `kin` | ✓ | ✓ | ✓ |
| English | `eng` | ✓ | ✓ | ✓ |
| French | `fra` | ✓ | ✓ | ✓ |

More languages can be added by extending the language maps in the server code and frontend.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (16GB+ VRAM recommended)
- Linux (tested on Ubuntu 22.04)

## Installation

```bash
git clone <repo-url>
cd wakanda-voice

# Create a conda environment (recommended)
conda create -n wakanda python=3.10
conda activate wakanda

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install the project and all dependencies
pip install -e .

# Install LiveKit Server
curl -sSL https://get.livekit.io | bash
```

## Quick Start

### Option A: Single launcher (recommended)

```bash
python launch.py --config configs/default.yaml --livekit
```

This starts all model servers (without preloading models), LiveKit Server, the orchestrator, and the frontend. Open `http://localhost:3000` in your browser.

### Option B: Separate terminals (for debugging)

```bash
# Terminal 1: Model servers + frontend
python launch.py --config configs/default.yaml

# Terminal 2: LiveKit Server
livekit-server --dev --bind 0.0.0.0

# Terminal 3: Orchestrator (watch this one for pipeline logs)
python orchestrator/main.py --config configs/default.yaml
```

Then open `http://localhost:3000`.

### Option C: Local testing without LiveKit

Test the pipeline with a local audio file — no browser, no LiveKit:

```bash
# Start model servers manually
python servers/stt_server.py --model facebook/mms-1b-all --language eng --port 8001
python servers/mt_server.py --model facebook/nllb-200-1.3B --port 8002
python servers/tts_server.py --model facebook/mms-tts-fra --port 8003

# Run pipeline on an audio file
python scripts/test_pipeline.py --config configs/default.yaml --file test_audio.wav
```

### Remote access (SSH)

If the GPU server is remote, use VS Code's SSH extension and forward ports 3000, 7880, and 8080 in the Ports tab. Then open `http://localhost:3000` in your local browser.

Alternatively, use SSH port forwarding:

```bash
ssh -i ~/.ssh/mykey -L 3000:localhost:3000 -L 7880:localhost:7880 -L 8080:localhost:8080 user@your-server-ip
```

## Usage

1. Open the browser UI
2. **Select task**: S2ST Translation or Voice Bot
3. **Select languages**: Source → Target
4. **Click Connect**: Models load on demand (~10-15s first time)
5. **Speak**: Audio flows through the pipeline and you hear the result

When you disconnect and reconnect with different languages, the model servers reconfigure automatically.

## Configuration

Edit `configs/default.yaml` to change models, ports, or behavior:

```yaml
mode: s2st
default_lang: eng

stt:
  url: ws://localhost:8001
  model: facebook/mms-1b-all       # swap to openai/whisper-large-v3
  streaming: false

mt:
  url: ws://localhost:8002
  model: facebook/nllb-200-1.3B    # swap to facebook/nllb-200-3.3B
  streaming: false

tts:
  url: ws://localhost:8003
  model: facebook/mms-tts-eng      # auto-swapped based on target language
  streaming: false
  sample_rate: 16000

llm:
  url: http://localhost:8000/v1
  model: almanach/AfriqueLLM-Qwen2.5-3B
  provider: openai_compat
```

### Swapping models

Change the model name in the config. The server framework supports any HuggingFace model:

| Swap | Config change | Effect |
|------|--------------|--------|
| MMS-1B → Whisper | `model: openai/whisper-large-v3` | Better accuracy, auto language detection |
| NLLB 1.3B → 3.3B | `model: facebook/nllb-200-3.3B` | Better translation quality (needs more VRAM) |
| MMS-TTS → custom | `model: your-org/your-tts-model` | Your lab's TTS model |
| AfriqueLLM → GPT-4.1 | `url: https://api.openai.com/v1`, `model: gpt-4.1` | Cloud LLM |

## Project Structure

```
wakanda-voice/
├── launch.py                      # Single-command launcher
├── configs/
│   └── default.yaml               # Pipeline configuration
├── servers/
│   ├── base.py                    # BaseModelServer (shared WebSocket protocol)
│   ├── stt_server.py              # STT server (MMS-1B / Whisper)
│   ├── mt_server.py               # MT server (NLLB)
│   └── tts_server.py              # TTS server (MMS-TTS)
├── orchestrator/
│   ├── main.py                    # LiveKit bridge + HTTP config endpoint
│   ├── orchestrator.py            # Core pipeline orchestrator
│   ├── vad.py                     # Silero VAD segmenter
│   ├── config.py                  # YAML config loader
│   ├── session_logger.py          # Bilingual JSONL conversation logger
│   ├── clients/
│   │   ├── ws_client.py           # WebSocket client for model servers
│   │   └── llm_client.py          # OpenAI-compatible LLM client
│   ├── pipelines/
│   │   ├── s2st.py                # S2ST pipeline logic
│   │   └── multilingual_bot.py    # Voice bot pipeline logic
│   └── tools/
│       └── (future: web_search.py, etc.)
├── frontend/
│   ├── index.html                 # Browser UI
│   └── livekit-client.umd.js      # LiveKit JS SDK (download separately)
├── scripts/
│   ├── test_pipeline.py           # Local testing without LiveKit
│   └── token_server.py            # Frontend file server + JWT token generator
├── logs/                          # Session logs (JSONL)
└── pyproject.toml
```

## How It Works

### Data flow (S2ST mode)

```
Browser mic → WebRTC → LiveKit Server → Orchestrator
    → Silero VAD (detects speech segments)
    → STT Server (MMS-1B: audio → source text)
    → MT Server (NLLB: source text → target text)
    → TTS Server (MMS-TTS: target text → audio)
    → LiveKit Server → WebRTC → Browser speaker
```

### Data flow (Voice Bot mode)

```
Browser mic → ... → Orchestrator
    → VAD → STT (source language)
    → MT inbound (source → English)
    → LLM reasoning + tool calls (English)
    → MT outbound (English → source)
    → TTS (source language)
    → ... → Browser speaker
```

### On-demand model loading

Models load only when a user connects with a specific language pair:

1. User selects Swahili → English in the browser
2. Browser sends `POST /configure` to orchestrator (port 8080)
3. Orchestrator tells STT server to load Swahili adapter
4. Orchestrator tells MT server to load NLLB
5. Orchestrator tells TTS server to load English TTS model
6. Once all models are ready, orchestrator responds OK
7. Browser connects to LiveKit for audio streaming

### Conversation logging

Every session is logged to `logs/session_{timestamp}.jsonl` with:
- Original text (user's language)
- Translated text (English)
- LLM response and tool calls
- Per-stage latency (STT, MT, TTS, total)

## Ports

| Port | Service | Purpose |
|------|---------|---------|
| 3000 | Token Server + Frontend | Browser UI and JWT token generation |
| 7880 | LiveKit Server | WebRTC signaling |
| 7881 | LiveKit TURN | WebRTC relay (used automatically) |
| 7882 | LiveKit UDP | WebRTC media |
| 8001 | STT Server | Speech-to-text model |
| 8002 | MT Server | Machine translation model |
| 8003 | TTS Server | Text-to-speech model |
| 8080 | Orchestrator HTTP | Model configuration endpoint |

## Troubleshooting

### CUDA out of memory
Use smaller models (`facebook/nllb-200-1.3B` instead of `3.3B`) or run MT on CPU:
```bash
python servers/mt_server.py --model facebook/nllb-200-1.3B --device cpu
```

### Browser can't access microphone
Browsers require HTTPS or `localhost` for mic access. Use VS Code port forwarding or SSH tunnels so the browser sees `localhost`.

### Models take long to load on first connect
First load downloads model weights from HuggingFace. Set `HF_TOKEN` for faster downloads:
```bash
export HF_TOKEN=hf_your_token_here
```
Subsequent loads use the cached weights and are much faster.

### LiveKit JS SDK not loading
Download it locally to `frontend/livekit-client.umd.js` (see Installation).

## License

MIT
