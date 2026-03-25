"""
Cascaded S2ST pipeline: STT → MT → TTS

The simplest pipeline. User speaks in language A, hears translated speech in language B.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Awaitable

from orchestrator.clients.ws_client import ModelClient

logger = logging.getLogger(__name__)


async def run_s2st_pipeline(
    transcript: str,
    source_lang: str,
    target_lang: str,
    mt_client: ModelClient,
    tts_client: ModelClient,
    on_subtitle: Callable[[str, str], Awaitable[None]],
    on_audio: Callable[[bytes, int], Awaitable[None]],
) -> dict[str, float]:
    """
    Run the S2ST pipeline: translate transcript, synthesize speech.

    Args:
        transcript: Source language text from STT.
        source_lang: Source language code (e.g. "yor_Latn").
        target_lang: Target language code (e.g. "eng_Latn").
        mt_client: Connected MT model client.
        tts_client: Connected TTS model client.
        on_subtitle: Callback to send subtitle to client. (text, lang) → None
        on_audio: Callback to send audio to client. (pcm_bytes, sample_rate) → None

    Returns:
        Dict of per-stage latency in ms.
    """
    latency = {}

    # ── Translate ──
    t0 = time.monotonic()
    translation = await mt_client.translate(
        text=transcript,
        src=source_lang,
        tgt=target_lang,
    )
    latency["mt"] = (time.monotonic() - t0) * 1000

    logger.info(f"MT: '{transcript}' → '{translation}' ({latency['mt']:.0f}ms)")

    # Send target-language subtitle
    await on_subtitle(translation, target_lang)

    # ── Synthesize ──
    t0 = time.monotonic()

    # Extract short lang code for TTS (e.g. "eng_Latn" → "eng")
    tts_lang = target_lang.split("_")[0] if "_" in target_lang else target_lang

    audio_bytes, sample_rate = await tts_client.synthesize(
        text=translation,
        language=tts_lang,
    )
    latency["tts"] = (time.monotonic() - t0) * 1000

    logger.info(f"TTS: {len(audio_bytes)} bytes, {sample_rate}Hz ({latency['tts']:.0f}ms)")

    # Send audio to client
    await on_audio(audio_bytes, sample_rate)

    return latency, translation
