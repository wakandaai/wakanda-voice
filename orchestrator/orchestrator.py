"""
Wakanda Voice Pipeline — Orchestrator

The central coordinator. Connects to model servers, manages VAD,
dispatches to the appropriate pipeline (S2ST or multilingual bot),
and handles conversation state.

Can run in two modes:
1. Local test mode (test_pipeline.py): mic → VAD → pipeline → speaker
2. LiveKit mode (main.py): LiveKit room → VAD → pipeline → LiveKit room
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Awaitable, Optional

from orchestrator.config import PipelineConfig
from orchestrator.vad import VADSegmenter
from orchestrator.clients.ws_client import ModelClient
from orchestrator.clients.llm_client import LLMClient
from orchestrator.session_logger import SessionLogger
from orchestrator.pipelines.s2st import run_s2st_pipeline
from orchestrator.pipelines.multilingual_bot import run_multilingual_bot_pipeline

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main pipeline orchestrator.

    Receives audio, runs VAD, dispatches to the configured pipeline,
    and returns synthesized audio.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.mode = config.mode

        # VAD
        self.vad = VADSegmenter(
            threshold=config.vad.threshold,
            min_silence_ms=config.vad.min_silence_ms,
            min_speech_ms=config.vad.min_speech_ms,
        )

        # Model clients (not yet connected)
        self.stt_client = ModelClient(config.stt.url)
        self.mt_client = ModelClient(config.mt.url)
        self.tts_client = ModelClient(config.tts.url)
        self.llm_client = LLMClient(
            url=config.llm.url,
            model=config.llm.model,
            api_key=config.llm.api_key,
        )

        # Session state
        self.user_lang = config.default_lang
        self.conversation_history: list[dict] = []
        self.session_logger = SessionLogger(
            log_dir=config.logging.dir,
        )

        # Callbacks (set by the transport layer — LiveKit or local test)
        self.on_subtitle: Callable[[str, str], Awaitable[None]] = self._noop_subtitle
        self.on_audio: Callable[[bytes, int], Awaitable[None]] = self._noop_audio
        self.on_state: Callable[[str, Optional[str]], Awaitable[None]] = self._noop_state

        # Tool execution (placeholder, extended by tool registry)
        self.tools: Optional[list[dict]] = None
        self.execute_tool: Optional[Callable] = None

    async def connect(self) -> None:
        """Connect to all model servers."""
        logger.info("Connecting to model servers...")

        await self.stt_client.connect()
        await self.stt_client.configure(
            model=self.config.stt.model,
            language=self.user_lang,
        )

        await self.mt_client.connect()
        await self.mt_client.configure(model=self.config.mt.model)

        await self.tts_client.connect()
        await self.tts_client.configure(model=self.config.tts.model)

        logger.info("All model servers connected")

    async def disconnect(self) -> None:
        """Disconnect from all servers and flush logs."""
        self.session_logger.close()
        await self.stt_client.disconnect()
        await self.mt_client.disconnect()
        await self.tts_client.disconnect()
        await self.llm_client.close()
        logger.info("Disconnected from all model servers")

    def set_language(self, lang: str) -> None:
        """Set the user's language (called when user selects in UI)."""
        self.user_lang = lang
        logger.info(f"User language set to: {lang}")
        self.session_logger.log_event("language_changed", language=lang)

    async def process_audio(self, pcm_bytes: bytes) -> None:
        """
        Feed audio into the pipeline.
        Called continuously with audio frames from the transport layer.
        """
        async for utterance in self.vad.feed(pcm_bytes):
            await self._handle_utterance(utterance)

    async def _handle_utterance(self, audio_bytes: bytes) -> None:
        """Process one complete utterance through the pipeline."""
        t_start = time.monotonic()
        latency = {}

        # ── STT ──
        t0 = time.monotonic()
        await self.on_state("transcribing", None)

        transcript = await self.stt_client.transcribe(
            audio_bytes=audio_bytes,
            lang=self.user_lang,
        )
        latency["stt"] = (time.monotonic() - t0) * 1000

        if not transcript.strip():
            logger.debug("Empty transcript, skipping")
            return

        logger.info(f"STT [{self.user_lang}]: '{transcript}' ({latency['stt']:.0f}ms)")

        # Show source-language subtitle
        await self.on_subtitle(transcript, self.user_lang)

        # ── Dispatch to pipeline ──
        if self.mode == "s2st":
            pipeline_latency, translation = await run_s2st_pipeline(
                transcript=transcript,
                source_lang=self.config.s2st.source_lang,
                target_lang=self.config.s2st.target_lang,
                mt_client=self.mt_client,
                tts_client=self.tts_client,
                on_subtitle=self.on_subtitle,
                on_audio=self.on_audio,
            )
            latency.update(pipeline_latency)
            latency["total"] = (time.monotonic() - t_start) * 1000

            # Log user turn (source language)
            self.session_logger.log_user_turn(
                original_text=transcript,
                english_text=translation,
                user_lang=self.user_lang,
                latency={"stt": latency["stt"]},
            )

            # Log translated output
            self.session_logger.log_assistant_turn(
                english_text=translation,
                translated_text=translation,
                user_lang=self.config.s2st.target_lang,
                latency={
                    "mt": latency.get("mt", 0),
                    "tts": latency.get("tts", 0),
                    "total": latency.get("total", 0),
                },
            )

        elif self.mode == "multilingual_voice_bot":
            pipeline_latency, english_response, translated_response = \
                await run_multilingual_bot_pipeline(
                    transcript=transcript,
                    user_lang=self.user_lang,
                    mt_client=self.mt_client,
                    llm_client=self.llm_client,
                    tts_client=self.tts_client,
                    conversation_history=self.conversation_history,
                    tools=self.tools,
                    execute_tool=self.execute_tool,
                    on_subtitle=self.on_subtitle,
                    on_audio=self.on_audio,
                    on_state=self.on_state,
                )
            latency.update(pipeline_latency)

            # Compute english_input for logging
            english_input = transcript
            if self.user_lang not in ("eng", "eng_Latn", "en"):
                # The MT inbound already happened inside the pipeline
                # We can infer it from the conversation history
                if self.conversation_history:
                    for msg in reversed(self.conversation_history):
                        if msg["role"] == "user":
                            english_input = msg["content"]
                            break

            # Log user turn
            self.session_logger.log_user_turn(
                original_text=transcript,
                english_text=english_input,
                user_lang=self.user_lang,
                latency={"stt": latency["stt"], "mt_inbound": latency.get("mt_inbound", 0)},
            )

            # Log assistant turn
            self.session_logger.log_assistant_turn(
                english_text=english_response,
                translated_text=translated_response,
                user_lang=self.user_lang,
                latency={
                    "llm": latency.get("llm", 0),
                    "mt_outbound": latency.get("mt_outbound", 0),
                    "tts": latency.get("tts", 0),
                    "total": latency.get("total", 0),
                },
            )

        logger.info(f"Pipeline complete: {latency}")
        await self.on_state("idle", None)

    # ── Default no-op callbacks ──

    @staticmethod
    async def _noop_subtitle(text: str, lang: str) -> None:
        pass

    @staticmethod
    async def _noop_audio(pcm_bytes: bytes, sample_rate: int) -> None:
        pass

    @staticmethod
    async def _noop_state(stage: str, detail: Optional[str]) -> None:
        pass
