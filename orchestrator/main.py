"""
Wakanda Voice Pipeline — LiveKit Entry Point

Joins a LiveKit room as a server-side participant, subscribes to user audio,
runs it through the pipeline, and publishes synthesized audio back.

Usage:
    # 1. Install LiveKit Server:  curl -sSL https://get.livekit.io | bash
    # 2. Start LiveKit:           livekit-server --dev --bind 0.0.0.0
    # 3. Start model servers (same as before)
    # 4. Start this:              python orchestrator/main.py --config configs/default.yaml
    # 5. Start token server:      python scripts/token_server.py
    # 6. Open browser:            http://localhost:7881
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
from livekit import rtc, api

from orchestrator.config import load_config
from orchestrator.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class LiveKitBridge:
    """
    Bridges LiveKit rooms to the Wakanda Voice pipeline orchestrator.

    - Joins a LiveKit room as "wakanda-bot"
    - Subscribes to audio from human participants
    - Feeds audio through orchestrator (VAD → STT → MT → TTS)
    - Publishes TTS audio back to the room
    - Sends subtitles and state via data channel
    """

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.room = rtc.Room()
        self.audio_source: rtc.AudioSource | None = None
        self.orchestrator: Orchestrator | None = None
        self.tts_sample_rate = self.config.tts.sample_rate or 16000

    async def start(self, room_name: str = "wakanda-room"):
        # Create orchestrator
        self.orchestrator = Orchestrator(self.config)

        # Wire orchestrator callbacks to LiveKit
        self.orchestrator.on_audio = self._on_pipeline_audio
        self.orchestrator.on_subtitle = self._on_pipeline_subtitle
        self.orchestrator.on_state = self._on_pipeline_state

        # Connect to model servers
        logger.info("Connecting to model servers...")
        await self.orchestrator.connect()
        logger.info("Model servers connected")

        # Set up LiveKit event handlers
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("participant_connected", self._on_participant_connected)
        self.room.on("participant_disconnected", self._on_participant_disconnected)
        self.room.on("data_received", self._on_data_received)

        # Generate access token for the bot
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

        # Connect to LiveKit room
        logger.info(f"Connecting to LiveKit room '{room_name}' at {self.config.livekit_url}")
        await self.room.connect(
            self.config.livekit_url,
            token,
            options=rtc.RoomOptions(auto_subscribe=True),
        )
        logger.info(f"Connected to room: {self.room.name}")

        # Create audio source for publishing TTS output
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

        # Send initial state
        await self._send_data({
            "type": "config",
            "mode": self.config.mode,
            "language": self.config.default_lang,
        })

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

    def _on_data_received(self, data):
        try:
            payload = json.loads(data.data.decode())
            msg_type = payload.get("type")

            if msg_type == "set_language":
                lang = payload.get("language", "eng")
                logger.info(f"Language set by user: {lang}")
                if self.orchestrator:
                    self.orchestrator.set_language(lang)

            elif msg_type == "set_mode":
                mode = payload.get("mode", "s2st")
                logger.info(f"Mode set by user: {mode}")
                if self.orchestrator:
                    self.orchestrator.mode = mode

            elif msg_type == "set_languages":
                src = payload.get("source", "eng")
                tgt = payload.get("target", "eng")
                src_nllb = payload.get("source_nllb", "eng_Latn")
                tgt_nllb = payload.get("target_nllb", "eng_Latn")
                logger.info(f"Languages set: {src} → {tgt} ({src_nllb} → {tgt_nllb})")
                if self.orchestrator:
                    self.orchestrator.set_language(src)
                    self.orchestrator.config.s2st.source_lang = src_nllb
                    self.orchestrator.config.s2st.target_lang = tgt_nllb

        except Exception as e:
            logger.error(f"Error parsing data message: {e}")

    # ── Audio processing ──

    async def _process_audio_stream(self, audio_stream, participant):
        logger.info(f"Processing audio from {participant.identity}")
        frame_count = 0
        try:
            async for frame_event in audio_stream:
                frame = frame_event.frame
                pcm_bytes = frame.data.tobytes()
                frame_count += 1
                if frame_count % 100 == 1:
                    logger.info(f"Audio frame #{frame_count}: {len(pcm_bytes)} bytes, sr={frame.sample_rate}, ch={frame.num_channels}, samples={frame.samples_per_channel}")
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
        await self._send_data({"type": "subtitle", "text": text, "language": lang})

    async def _on_pipeline_state(self, stage: str, detail: str | None):
        await self._send_data({"type": "state", "stage": stage, "detail": detail})

    async def _send_data(self, payload: dict):
        try:
            await self.room.local_participant.publish_data(
                json.dumps(payload).encode(),
                reliable=True,
            )
        except Exception as e:
            logger.error(f"Failed to send data: {e}")


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
