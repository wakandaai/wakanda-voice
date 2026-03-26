"""
Wakanda Voice Pipeline — LiveKit Entry Point

Usage:
    python orchestrator/main.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from aiohttp import web
from livekit import rtc, api

from orchestrator.config import load_config
from orchestrator.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class LiveKitBridge:

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.room = rtc.Room()
        self.audio_source: rtc.AudioSource | None = None
        self.orchestrator: Orchestrator | None = None
        self.tts_sample_rate = self.config.tts.sample_rate or 16000
        self.models_ready = False

    async def start(self, room_name: str = "wakanda-room"):
        # Create orchestrator (don't connect to model servers yet)
        self.orchestrator = Orchestrator(self.config)
        self.orchestrator.on_audio = self._on_pipeline_audio
        self.orchestrator.on_subtitle = self._on_pipeline_subtitle
        self.orchestrator.on_state = self._on_pipeline_state

        logger.info("Orchestrator created — waiting for user language selection")

        # ── Start HTTP config endpoint ──
        app = web.Application(middlewares=[self._cors_middleware])
        app.router.add_post("/configure", self._handle_configure)
        app.router.add_options("/configure", self._handle_options)
        app.router.add_get("/status", self._handle_status)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", 8080)
        await site.start()
        logger.info("Config endpoint at http://0.0.0.0:8080/configure")

        # ── Connect to LiveKit ──
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("participant_connected", self._on_participant_connected)
        self.room.on("participant_disconnected", self._on_participant_disconnected)

        token = (
            api.AccessToken(
                api_key=self.config.livekit_api_key,
                api_secret=self.config.livekit_api_secret,
            )
            .with_identity("wakanda-bot")
            .with_name("Wakanda Voice Bot")
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
            )
            .to_jwt()
        )

        logger.info(f"Connecting to LiveKit room '{room_name}' at {self.config.livekit_url}")
        await self.room.connect(
            self.config.livekit_url,
            token,
            options=rtc.RoomOptions(auto_subscribe=True),
        )
        logger.info(f"Connected to room: {self.room.name}")

        # Publish audio track for TTS output
        self.audio_source = rtc.AudioSource(
            sample_rate=self.tts_sample_rate,
            num_channels=1,
        )
        track = rtc.LocalAudioTrack.create_audio_track("agent-voice", self.audio_source)
        await self.room.local_participant.publish_track(
            track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        logger.info("Published audio track: agent-voice")
        logger.info("LiveKit bridge ready. Waiting for participants...")

        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    async def shutdown(self):
        logger.info("Shutting down...")
        if self.orchestrator:
            await self.orchestrator.disconnect()
        await self.room.disconnect()
        logger.info("Shutdown complete")

    # ── HTTP config endpoint ──

    @web.middleware
    async def _cors_middleware(self, request, handler):
        response = await handler(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    async def _handle_options(self, request):
        return web.Response(status=200)

    async def _handle_status(self, request):
        return web.json_response({
            "models_ready": self.models_ready,
            "mode": self.orchestrator.mode if self.orchestrator else None,
            "language": self.orchestrator.user_lang if self.orchestrator else None,
        })

    async def _handle_configure(self, request):
        data = await request.json()
        logger.info(f"Config received: {data}")

        # Require all fields
        required = ["source", "target", "source_nllb", "target_nllb", "mode"]
        missing = [f for f in required if f not in data]
        if missing:
            return web.json_response(
                {"status": "error", "message": f"Missing fields: {missing}"},
                status=400,
            )

        src = data["source"]
        tgt = data["target"]
        src_nllb = data["source_nllb"]
        tgt_nllb = data["target_nllb"]
        mode = data["mode"]

        # Update orchestrator config
        self.orchestrator.set_language(src)
        self.orchestrator.mode = mode
        self.orchestrator.config.s2st.source_lang = src_nllb
        self.orchestrator.config.s2st.target_lang = tgt_nllb

        # Load models
        try:
            await self._load_models(src, tgt, src_nllb, tgt_nllb)
            return web.json_response({"status": "ok", "models_ready": True})
        except Exception as e:
            logger.error(f"Failed to load models: {e}", exc_info=True)
            return web.json_response(
                {"status": "error", "message": str(e)},
                status=500,
            )

    async def _load_models(self, src: str, tgt: str, src_nllb: str, tgt_nllb: str):
        logger.info(f"Loading models for {src} → {tgt}...")

        # Connect to model servers if not already connected
        if not self.models_ready:
            await self.orchestrator.connect()

        # Configure STT for source language
        await self.orchestrator.stt_client.configure(
            model=self.config.stt.model,
            language=src,
        )
        logger.info(f"STT configured for {src}")

        # Configure MT
        await self.orchestrator.mt_client.configure(
            model=self.config.mt.model,
        )
        logger.info("MT configured")

        # Configure TTS for target language
        # MMS-TTS uses per-language model repos
        tts_lang_map = {
            "swa": "swh", "ibo": "ibo", "yor": "yor", "hau": "hau",
            "bem": "bem", "kin": "kin", "eng": "eng", "fra": "fra",
            "lug": "lug",
        }
        tts_code = tts_lang_map.get(tgt, tgt)
        tts_model = f"facebook/mms-tts-{tts_code}"
        await self.orchestrator.tts_client.configure(
            model=tts_model,
            language=tgt,
        )
        logger.info(f"TTS configured for {tgt} ({tts_model})")

        self.models_ready = True
        logger.info("All models loaded and ready")

    # ── LiveKit event handlers ──

    def _on_track_subscribed(self, track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Subscribed to audio from: {participant.identity}")
            audio_stream = rtc.AudioStream(track, sample_rate=16000, num_channels=1)
            asyncio.create_task(self._process_audio_stream(audio_stream, participant))

    def _on_participant_connected(self, participant):
        logger.info(f"Participant connected: {participant.identity}")

    def _on_participant_disconnected(self, participant):
        logger.info(f"Participant disconnected: {participant.identity}")

    # ── Audio processing ──

    async def _process_audio_stream(self, audio_stream, participant):
        logger.info(f"Processing audio from {participant.identity}")
        frame_count = 0
        try:
            async for frame_event in audio_stream:
                frame = frame_event.frame
                pcm_bytes = frame.data.tobytes()
                frame_count += 1
                if frame_count % 500 == 1:
                    logger.info(f"Audio frame #{frame_count}: {len(pcm_bytes)} bytes")
                if self.models_ready:
                    await self.orchestrator.process_audio(pcm_bytes)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Audio processing error: {e}", exc_info=True)

    # ── Pipeline callbacks → LiveKit ──

    async def _on_pipeline_audio(self, pcm_bytes: bytes, sample_rate: int):
        if not self.audio_source:
            return

        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16)

        # Resample if needed
        if sample_rate != self.tts_sample_rate:
            duration = len(audio_np) / sample_rate
            target_samples = int(duration * self.tts_sample_rate)
            indices = np.linspace(0, len(audio_np) - 1, target_samples)
            audio_np = np.interp(indices, np.arange(len(audio_np)), audio_np.astype(np.float64))
            audio_np = audio_np.astype(np.int16)

        # Push in 20ms frames
        samples_per_frame = self.tts_sample_rate // 50
        offset = 0
        while offset < len(audio_np):
            chunk = audio_np[offset:offset + samples_per_frame]
            if len(chunk) < samples_per_frame:
                chunk = np.pad(chunk, (0, samples_per_frame - len(chunk)))
            frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=self.tts_sample_rate,
                num_channels=1,
                samples_per_channel=len(chunk),
            )
            await self.audio_source.capture_frame(frame)
            offset += samples_per_frame

    async def _on_pipeline_subtitle(self, text: str, lang: str):
        try:
            await self.room.local_participant.publish_data(
                json.dumps({"type": "subtitle", "text": text, "language": lang}).encode(),
                reliable=True,
            )
        except Exception as e:
            logger.error(f"Failed to send subtitle: {e}")

    async def _on_pipeline_state(self, stage: str, detail: str | None):
        try:
            await self.room.local_participant.publish_data(
                json.dumps({"type": "state", "stage": stage, "detail": detail}).encode(),
                reliable=True,
            )
        except Exception as e:
            logger.error(f"Failed to send state: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Wakanda Voice — LiveKit Bridge")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--room", default="wakanda-room")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [BRIDGE] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    bridge = LiveKitBridge(args.config)
    await bridge.start(room_name=args.room)


if __name__ == "__main__":
    asyncio.run(main())
