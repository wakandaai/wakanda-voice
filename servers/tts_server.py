"""
TTS (Text-to-Speech) model server.

Baseline: facebook/mms-tts-{lang} (VITS-based, per-language models)
Swappable to: Kyutai TTS, Coqui XTTS, or any HuggingFace TTS model.

Batch mode: receives full text, returns full audio.
"""

from __future__ import annotations

import asyncio
import argparse
import logging
from typing import Any, AsyncIterator

import numpy as np
import torch

from base import BaseModelServer

logger = logging.getLogger(__name__)


# MMS-TTS uses per-language model repos
MMS_TTS_MODELS = {
    "yor": "facebook/mms-tts-yor",
    "hau": "facebook/mms-tts-hau",
    "ibo": "facebook/mms-tts-ibo",
    "bem": "facebook/mms-tts-bem",
    "swa": "facebook/mms-tts-swh",
    "kin": "facebook/mms-tts-kin",
    "eng": "facebook/mms-tts-eng",
    "fra": "facebook/mms-tts-fra",
}


class TTSServer(BaseModelServer):
    """
    Text-to-speech server.
    Supports MMS-TTS (VITS-based) and generic HuggingFace TTS models.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = None
        self.sample_rate: int = 16000

    async def load_model(self, model_name: str, **kwargs) -> None:
        """Load TTS model."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync, model_name)

    def _load_model_sync(self, model_name: str) -> None:
        from transformers import VitsModel, AutoTokenizer

        logger.info(f"Loading TTS model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = VitsModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.sample_rate = self.model.config.sampling_rate
        logger.info(f"TTS model loaded (sample_rate={self.sample_rate})")

    async def process(self, data: Any, config: dict) -> AsyncIterator:
        """
        Synthesize speech from text.

        data: dict with "text" and optionally "language"
        Yields: binary PCM audio bytes, then {"type": "done"}
        """
        if not isinstance(data, dict):
            yield {"type": "error", "message": "Expected JSON with text"}
            return

        if not self.model:
            yield {"type": "error", "message": "No model loaded"}
            return

        text = data.get("text", "")
        language = data.get("language") or config.get("language")

        # MMS-TTS uses per-language models — check if we need to swap
        if language and language in MMS_TTS_MODELS:
            needed_model = MMS_TTS_MODELS[language]
            if needed_model != self.current_model_name:
                logger.info(f"Switching TTS model for language: {language}")
                await self.load_model(needed_model)
                self.current_model_name = needed_model

        if not text.strip():
            yield {"type": "done", "sample_rate": self.sample_rate}
            return

        loop = asyncio.get_event_loop()
        pcm_bytes = await loop.run_in_executor(None, self._synthesize_sync, text)

        # Send audio as binary
        yield pcm_bytes

        # Signal completion with metadata
        yield {
            "type": "done",
            "sample_rate": self.sample_rate,
        }

    def _synthesize_sync(self, text: str) -> bytes:
        """Synchronous speech synthesis."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model(**inputs)

        waveform = output.waveform[0].cpu().numpy()

        # Normalize and convert to 16-bit PCM
        waveform = np.clip(waveform, -1.0, 1.0)
        pcm_int16 = (waveform * 32767).astype(np.int16)
        return pcm_int16.tobytes()


def main():
    parser = argparse.ArgumentParser(description="Wakanda Voice — TTS Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="facebook/mms-tts-yor")
    parser.add_argument("--no-preload", action="store_true",
                        help="Start without loading a model (wait for config)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [TTS] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    server = TTSServer(host=args.host, port=args.port, device=args.device)

    async def start():
        if not args.no_preload:
            await server.load_model(args.model)
            server.current_model_name = args.model
            server.metrics.model_name = args.model
            server.metrics.model_loaded = True
        else:
            logger.info("Started without preloading — waiting for config message")
        await server.serve()

    asyncio.run(start())


if __name__ == "__main__":
    main()
