#!/usr/bin/env python3
"""
Test the pipeline locally without LiveKit.

Uses your system microphone and speaker directly.
Run model servers first, then run this script.

Usage:
    # Terminal 1: Start STT server
    python servers/stt_server.py --model facebook/mms-1b-all --language yor

    # Terminal 2: Start MT server
    python servers/mt_server.py --model facebook/nllb-200-3.3B

    # Terminal 3: Start TTS server
    python servers/tts_server.py --model facebook/mms-tts-eng

    # Terminal 4: Run this test script
    python scripts/test_pipeline.py --config configs/default.yaml

    Speak into your mic. The pipeline will transcribe → translate → speak.
    Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.config import load_config
from orchestrator.orchestrator import Orchestrator


logger = logging.getLogger(__name__)


async def main(config_path: str):
    config = load_config(config_path)

    # Create orchestrator
    orch = Orchestrator(config)

    # Set up audio output callback
    playback_queue: asyncio.Queue[tuple[bytes, int]] = asyncio.Queue()

    async def on_audio(pcm_bytes: bytes, sample_rate: int):
        await playback_queue.put((pcm_bytes, sample_rate))

    async def on_subtitle(text: str, lang: str):
        role_icon = "🗣" if lang == config.default_lang else "🔊"
        print(f"  {role_icon} [{lang}] {text}")

    async def on_state(stage: str, detail: str | None):
        icons = {
            "transcribing": "📝",
            "translating_inbound": "🔄",
            "thinking": "🧠",
            "tool_call": "🔧",
            "translating_outbound": "🔄",
            "speaking": "🔊",
            "idle": "⏸",
        }
        icon = icons.get(stage, "▶")
        suffix = f" ({detail})" if detail else ""
        print(f"  {icon} {stage}{suffix}")

    orch.on_audio = on_audio
    orch.on_subtitle = on_subtitle
    orch.on_state = on_state

    # Connect to model servers
    print(f"\n🔌 Connecting to model servers...")
    print(f"   STT: {config.stt.url}")
    print(f"   MT:  {config.mt.url}")
    print(f"   TTS: {config.tts.url}")
    if config.mode == "multilingual_voice_bot":
        print(f"   LLM: {config.llm.url}")
    print()

    await orch.connect()
    print(f"✅ Connected! Mode: {config.mode}, Language: {config.default_lang}\n")

    # Audio playback task
    async def playback_loop():
        try:
            import sounddevice as sd
            while True:
                pcm_bytes, sample_rate = await playback_queue.get()
                audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                # Play audio (blocking, but in executor)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, sd.play, audio_np, sample_rate)
                await loop.run_in_executor(None, sd.wait)
        except asyncio.CancelledError:
            pass

    playback_task = asyncio.create_task(playback_loop())

    # Audio capture loop
    print("🎤 Listening... Speak into your microphone. Ctrl+C to stop.\n")

    try:
        import sounddevice as sd

        sample_rate = 16000
        frame_duration_ms = 30
        frame_size = int(sample_rate * frame_duration_ms / 1000)

        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio status: {status}")
            # Convert float32 to int16 PCM
            pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            audio_queue.put_nowait(pcm)

        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frame_size,
            callback=audio_callback,
        )

        with stream:
            while True:
                pcm_bytes = await audio_queue.get()
                await orch.process_audio(pcm_bytes)

    except KeyboardInterrupt:
        print("\n\n⏹ Stopping...")
    except ImportError:
        print("❌ sounddevice not installed. Install with: pip install sounddevice")
        print("   Alternatively, test with audio files using test_with_file() below.")
    finally:
        playback_task.cancel()
        await orch.disconnect()
        print("👋 Done.")


async def test_with_file(config_path: str, audio_path: str):
    """
    Test pipeline with an audio file instead of microphone.
    Useful for automated testing and CI.
    """
    import soundfile as sf

    config = load_config(config_path)
    orch = Orchestrator(config)

    results = []

    async def on_audio(pcm_bytes: bytes, sample_rate: int):
        results.append(("audio", len(pcm_bytes), sample_rate))
        # Optionally save to file
        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        sf.write("test_output.wav", audio_np, sample_rate)
        print(f"  💾 Saved output to test_output.wav")

    async def on_subtitle(text: str, lang: str):
        results.append(("subtitle", text, lang))
        print(f"  📝 [{lang}] {text}")

    async def on_state(stage: str, detail: str | None):
        print(f"  ▶ {stage}" + (f" ({detail})" if detail else ""))

    orch.on_audio = on_audio
    orch.on_subtitle = on_subtitle
    orch.on_state = on_state

    await orch.connect()

    # Read audio file
    audio_np, sr = sf.read(audio_path, dtype="int16")
    if sr != 16000:
        # Basic resampling (for proper resampling, use librosa)
        print(f"⚠ Audio is {sr}Hz, expected 16000Hz. Results may be degraded.")

    pcm_bytes = audio_np.tobytes()

    # Process as one utterance (skip VAD for file-based testing)
    await orch._handle_utterance(pcm_bytes)
    await orch.disconnect()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Wakanda Voice Pipeline locally")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--file", default=None,
                        help="Audio file to process (instead of mic)")
    parser.add_argument("--lang", default=None,
                        help="Override user language")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.file:
        asyncio.run(test_with_file(args.config, args.file))
    else:
        asyncio.run(main(args.config))
