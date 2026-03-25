"""
MT (Machine Translation) model server.

Baseline: facebook/nllb-200-3.3B (200 languages)
Swappable to: any HuggingFace seq2seq translation model, or LLM-based MT.

Batch mode: receives full sentence, returns full translation.
"""

from __future__ import annotations

import asyncio
import argparse
import logging
from typing import Any, AsyncIterator

import torch

from base import BaseModelServer

logger = logging.getLogger(__name__)


# NLLB language codes for target African languages
LANG_CODE_MAP = {
    # Short code → NLLB code
    "yor": "yor_Latn",
    "hau": "hau_Latn",
    "ibo": "ibo_Latn",
    "igb": "ibo_Latn",
    "bem": "bem_Latn",
    "swa": "swh_Latn",
    "kin": "kin_Latn",
    "eng": "eng_Latn",
    "fra": "fra_Latn",
    # Pass through codes that already have the _Script suffix
}


def resolve_lang_code(code: str) -> str:
    """Resolve a short language code to NLLB format."""
    if "_" in code:
        return code  # already in xx_Script format
    return LANG_CODE_MAP.get(code, f"{code}_Latn")


class MTServer(BaseModelServer):
    """
    Machine Translation server using NLLB-200 or compatible seq2seq models.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = None

    async def load_model(self, model_name: str, **kwargs) -> None:
        """Load MT model."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync, model_name)

    def _load_model_sync(self, model_name: str) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info(f"Loading MT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info("MT model loaded successfully")

    async def process(self, data: Any, config: dict) -> AsyncIterator:
        """
        Translate text.

        data: dict with "text", "src", "tgt" keys
        """
        if not isinstance(data, dict):
            yield {"type": "error", "message": "Expected JSON with text/src/tgt"}
            return

        if not self.model:
            yield {"type": "error", "message": "No model loaded"}
            return

        text = data.get("text", "")
        src_lang = resolve_lang_code(data.get("src", "eng"))
        tgt_lang = resolve_lang_code(data.get("tgt", "eng"))

        if not text.strip():
            yield {"type": "translation", "text": "", "final": True}
            return

        loop = asyncio.get_event_loop()
        translation = await loop.run_in_executor(
            None, self._translate_sync, text, src_lang, tgt_lang
        )

        yield {
            "type": "translation",
            "text": translation,
            "src": src_lang,
            "tgt": tgt_lang,
            "final": True,
        }

    def _translate_sync(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Synchronous translation."""
        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        tgt_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                forced_bos_token_id=tgt_token_id,
                max_new_tokens=256,
            )

        translation = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return translation.strip()


def main():
    parser = argparse.ArgumentParser(description="Wakanda Voice — MT Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="facebook/nllb-200-3.3B")
    parser.add_argument("--no-preload", action="store_true",
                        help="Start without loading a model (wait for config)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [MT] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    server = MTServer(host=args.host, port=args.port, device=args.device)

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
