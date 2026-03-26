"""
STT (Speech-to-Text) model server.

Baseline: facebook/mms-1b-all (1,100+ languages, CTC-based)
Swappable to: openai/whisper-large-v3, or any HuggingFace ASR model.

Batch mode: receives complete audio segment, returns full transcript.
Streaming mode: (future) receives audio chunks, returns partial transcripts.
"""

from __future__ import annotations

import asyncio
import argparse
import logging
import sys
from typing import Any, AsyncIterator

import numpy as np
import torch

from base import BaseModelServer

logger = logging.getLogger(__name__)


# ── Language code mappings ──
# MMS uses ISO 639-3 codes, NLLB uses xx_Script codes.
# This maps common short codes to MMS adapter names.
LANG_TO_MMS = {
    "yor": "yor",
    "hau": "hau",
    "ibo": "ibo",  # Igbo
    "igb": "ibo",  # alias
    "bem": "bem",
    "swa": "swh",
    "kin": "kin",
    "eng": "eng",
    "lug": "lug",
    "fra": "fra",
    # Add more as needed
}


class STTServer(BaseModelServer):
    """
    Speech-to-text server.

    Supports two model families:
    - MMS-1B (CTC-based, facebook/mms-1b-all)
    - Whisper (seq2seq, openai/whisper-*)

    The model family is auto-detected from the model name.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.processor = None
        self.model_family: str = "unknown"  # "mms" or "whisper"
        self._current_lang: str = "eng"

    async def load_model(self, model_name: str, **kwargs) -> None:
        """Load STT model onto GPU."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync, model_name, kwargs)

    def _load_model_sync(self, model_name: str, kwargs: dict) -> None:
        """Synchronous model loading (runs in thread pool)."""
        # Detect model family
        if "whisper" in model_name.lower():
            self._load_whisper(model_name, kwargs)
        elif "mms" in model_name.lower():
            self._load_mms(model_name, kwargs)
        else:
            # Try as a generic CTC model
            self._load_mms(model_name, kwargs)

    def _load_mms(self, model_name: str, kwargs: dict) -> None:
        """Load MMS-1B or similar CTC model."""
        from transformers import AutoModelForCTC, AutoProcessor

        logger.info(f"Loading MMS model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.model_family = "mms"
        logger.info("MMS model loaded successfully")

    def _load_whisper(self, model_name: str, kwargs: dict) -> None:
        """Load Whisper model."""
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        logger.info(f"Loading Whisper model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        self.model_family = "whisper"
        logger.info("Whisper model loaded successfully")

    def _set_language(self, lang: str) -> None:
        """Set the target language for transcription."""
        if lang == self._current_lang:
            return

        mms_lang = LANG_TO_MMS.get(lang, lang)

        if self.model_family == "mms":
            try:
                self.processor.tokenizer.set_target_lang(mms_lang)
                self.model.load_adapter(mms_lang)
                self._current_lang = lang
                logger.info(f"MMS language set to: {mms_lang}")
            except Exception as e:
                logger.error(f"Failed to set MMS language '{mms_lang}': {e}")
                raise

        elif self.model_family == "whisper":
            # Whisper handles language via generate() kwargs
            self._current_lang = lang

    async def process(self, data: Any, config: dict) -> AsyncIterator:
        """
        Process audio data and yield transcription.

        data: bytes (raw PCM audio, 16-bit signed int, 16kHz mono)
        config: dict with "language" key
        """
        if not isinstance(data, bytes):
            yield {"type": "error", "message": "Expected binary audio data"}
            return

        if not self.model:
            yield {"type": "error", "message": "No model loaded"}
            return

        # Set language
        lang = config.get("language", "eng")
        self._set_language(lang)

        # Run inference in thread pool (blocking GPU ops)
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, self._transcribe_sync, data)

        yield {
            "type": "transcript",
            "text": text,
            "language": lang,
            "final": True,
        }

    def _transcribe_sync(self, audio_bytes: bytes) -> str:
        """Synchronous transcription. Runs in thread pool."""
        # Decode PCM bytes to numpy float32
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio_np) == 0:
            return ""

        if self.model_family == "mms":
            return self._transcribe_mms(audio_np)
        elif self.model_family == "whisper":
            return self._transcribe_whisper(audio_np)
        else:
            raise ValueError(f"Unknown model family: {self.model_family}")

    def _transcribe_mms(self, audio_np: np.ndarray) -> str:
        """Transcribe with MMS-1B."""
        inputs = self.processor(
            audio_np,
            sampling_rate=16000,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(predicted_ids)[0]
        return text.strip()

    def _transcribe_whisper(self, audio_np: np.ndarray) -> str:
        """Transcribe with Whisper."""
        inputs = self.processor(
            audio_np,
            sampling_rate=16000,
            return_tensors="pt",
        ).to(self.device)

        generate_kwargs = {}
        if self._current_lang and self._current_lang != "auto":
            generate_kwargs["language"] = self._current_lang

        with torch.no_grad():
            predicted_ids = self.model.generate(
                inputs.input_features,
                max_new_tokens=448,
                **generate_kwargs,
            )

        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Wakanda Voice — STT Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="facebook/mms-1b-all",
                        help="Model to preload on startup")
    parser.add_argument("--language", default="eng",
                        help="Default language adapter to load")
    parser.add_argument("--no-preload", action="store_true",
                        help="Start without loading a model (wait for config)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [STT] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    server = STTServer(host=args.host, port=args.port, device=args.device)

    async def start():
        if not args.no_preload:
            await server.load_model(args.model)
            server.current_model_name = args.model
            server.metrics.model_name = args.model
            server.metrics.model_loaded = True
            # Set default language
            server._set_language(args.language)
        else:
            logger.info("Started without preloading — waiting for config message")

        # Start serving
        await server.serve()

    asyncio.run(start())


if __name__ == "__main__":
    main()
