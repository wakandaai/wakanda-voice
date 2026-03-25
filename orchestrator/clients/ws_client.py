"""
Unified WebSocket client for model servers.

Speaks the standardized protocol defined in servers/base.py.
Used by the orchestrator to talk to STT, MT, and TTS servers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

import websockets

logger = logging.getLogger(__name__)


class ModelClient:
    """
    Async WebSocket client for any model server in the pipeline.

    Usage:
        client = ModelClient("ws://localhost:8001")
        await client.connect()
        await client.configure(model="facebook/mms-1b-all", language="yor")
        result = await client.transcribe(audio_bytes)
        translation = await client.translate("hello", src="eng", tgt="yor")
        audio = await client.synthesize("hello", language="eng")
    """

    def __init__(self, url: str, reconnect: bool = True):
        self.url = url
        self.reconnect = reconnect
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = asyncio.Event()
        self._response_queue: asyncio.Queue = asyncio.Queue()
        self._listen_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Establish WebSocket connection to model server."""
        try:
            self.ws = await websockets.connect(
                self.url,
                max_size=50 * 1024 * 1024,
                ping_interval=30,
                ping_timeout=10,
            )
            self._listen_task = asyncio.create_task(self._listen_loop())

            # Wait for "ready" signal from server
            msg = await asyncio.wait_for(self._response_queue.get(), timeout=30)
            if isinstance(msg, dict) and msg.get("type") == "ready":
                self._connected.set()
                logger.info(f"Connected to model server: {self.url}")
            else:
                logger.warning(f"Unexpected first message from server: {msg}")
                self._connected.set()  # proceed anyway

        except Exception as e:
            logger.error(f"Failed to connect to {self.url}: {e}")
            raise

    async def disconnect(self) -> None:
        """Close connection."""
        if self._listen_task:
            self._listen_task.cancel()
        if self.ws:
            await self.ws.close()
        self._connected.clear()

    async def configure(self, model: str, language: Optional[str] = None,
                        **options) -> dict:
        """Send config message to server. Returns config_ack."""
        msg = {"type": "config", "model": model, "options": options}
        if language:
            msg["language"] = language

        await self._send_json(msg)

        # Wait for config_ack
        response = await self._get_response()
        return response

    # ── High-level methods ──

    async def transcribe(self, audio_bytes: bytes, lang: str = "eng") -> str:
        """Send audio to STT server, return transcript text."""
        # Ensure language is set in config
        await self._send_json({
            "type": "config",
            "model": None,  # keep current model
            "language": lang,
        })
        ack = await self._get_response()  # config_ack

        # Send audio
        await self._send_bytes(audio_bytes)

        # Collect transcript
        full_text = ""
        while True:
            response = await self._get_response()
            if isinstance(response, dict):
                if response.get("type") == "transcript":
                    full_text += response.get("text", "")
                    if response.get("final", False):
                        break
                elif response.get("type") == "error":
                    raise RuntimeError(f"STT error: {response.get('message')}")
                else:
                    break

        return full_text

    async def translate(self, text: str, src: str, tgt: str) -> str:
        """Send text to MT server, return translation."""
        await self._send_json({
            "type": "translate",
            "text": text,
            "src": src,
            "tgt": tgt,
        })

        full_text = ""
        while True:
            response = await self._get_response()
            if isinstance(response, dict):
                if response.get("type") == "translation":
                    full_text += response.get("text", "")
                    if response.get("final", False):
                        break
                elif response.get("type") == "error":
                    raise RuntimeError(f"MT error: {response.get('message')}")
                else:
                    break

        return full_text

    async def synthesize(self, text: str, language: str = "eng") -> tuple[bytes, int]:
        """Send text to TTS server, return (pcm_bytes, sample_rate)."""
        await self._send_json({
            "type": "synthesize",
            "text": text,
            "language": language,
        })

        audio_chunks = bytearray()
        sample_rate = 16000

        while True:
            response = await self._get_response()
            if isinstance(response, bytes):
                audio_chunks.extend(response)
            elif isinstance(response, dict):
                if response.get("type") == "done":
                    sample_rate = response.get("sample_rate", 16000)
                    break
                elif response.get("type") == "error":
                    raise RuntimeError(f"TTS error: {response.get('message')}")

        return bytes(audio_chunks), sample_rate

    # ── Internal ──

    async def _send_json(self, data: dict) -> None:
        """Send JSON message."""
        await self._connected.wait()
        # Filter out None values
        clean = {k: v for k, v in data.items() if v is not None}
        await self.ws.send(json.dumps(clean))

    async def _send_bytes(self, data: bytes) -> None:
        """Send binary message."""
        await self._connected.wait()
        await self.ws.send(data)

    async def _get_response(self, timeout: float = 60.0) -> Any:
        """Wait for next response from server."""
        return await asyncio.wait_for(self._response_queue.get(), timeout=timeout)

    async def _listen_loop(self) -> None:
        """Background task: read messages from server, put on queue."""
        try:
            async for message in self.ws:
                if isinstance(message, str):
                    data = json.loads(message)
                    await self._response_queue.put(data)
                elif isinstance(message, bytes):
                    await self._response_queue.put(message)
        except websockets.ConnectionClosed:
            logger.warning(f"Connection closed: {self.url}")
            self._connected.clear()
            if self.reconnect:
                logger.info(f"Attempting reconnect to {self.url}...")
                await asyncio.sleep(2)
                try:
                    await self.connect()
                except Exception:
                    logger.error(f"Reconnect failed: {self.url}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Listen loop error: {e}", exc_info=True)
