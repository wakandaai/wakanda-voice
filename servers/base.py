"""
Base model server framework.

Every model server (STT, MT, TTS) subclasses BaseModelServer and implements:
  - load_model(): load the model onto GPU
  - process(): handle one request, yield results

The base class handles WebSocket protocol, connection management, and the
unified message format so that the orchestrator speaks one protocol to all services.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import websockets
from websockets.server import serve

logger = logging.getLogger(__name__)


@dataclass
class ServerMetrics:
    """Track per-request metrics."""
    requests_total: int = 0
    errors_total: int = 0
    last_latency_ms: float = 0
    model_name: str = ""
    model_loaded: bool = False


class BaseModelServer(ABC):
    """
    Base class for all model servers in the Wakanda Voice Pipeline.

    Subclass this and implement load_model() and process().
    The server exposes a WebSocket endpoint with a standardized protocol.

    Protocol (client → server):
        {"type": "config", "model": "...", "language": "...", "options": {}}
        {"type": "translate", "text": "...", "src": "...", "tgt": "..."}   (MT)
        {"type": "synthesize", "text": "...", "language": "..."}           (TTS)
        binary PCM bytes                                                    (STT)
        {"type": "flush"}
        {"type": "interrupt"}

    Protocol (server → client):
        {"type": "transcript", "text": "...", "final": true/false}         (STT)
        {"type": "translation", "text": "...", "final": true/false}        (MT)
        binary PCM bytes                                                    (TTS)
        {"type": "done"}
        {"type": "error", "message": "..."}
        {"type": "ready"}
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8001, device: str = "cuda"):
        self.host = host
        self.port = port
        self.device = device
        self.current_model_name: Optional[str] = None
        self.model: Any = None
        self.metrics = ServerMetrics()
        self._lock = asyncio.Lock()  # serialize model loads

    @abstractmethod
    async def load_model(self, model_name: str, **kwargs) -> None:
        """Load a model into memory. Called when config message specifies a new model."""
        ...

    @abstractmethod
    async def process(self, data: Any, config: dict) -> AsyncIterator:
        """
        Process one request. Yield results.

        For STT: data is bytes (PCM audio), yield {"type": "transcript", "text": ..., "final": ...}
        For MT:  data is dict with "text", "src", "tgt", yield {"type": "translation", ...}
        For TTS: data is dict with "text", "language", yield bytes (PCM) then {"type": "done"}
        """
        ...

    async def handle_connection(self, websocket: websockets.WebSocketServerProtocol) -> None:
        """Handle one WebSocket connection from the orchestrator."""
        config: dict = {}
        peer = websocket.remote_address
        logger.info(f"Client connected: {peer}")

        try:
            # Send ready signal
            await websocket.send(json.dumps({"type": "ready"}))

            async for message in websocket:
                try:
                    if isinstance(message, str):
                        data = json.loads(message)
                        msg_type = data.get("type")

                        if msg_type == "config":
                            await self._handle_config(data, websocket)
                            config = data

                        elif msg_type == "interrupt":
                            logger.debug("Interrupt received")
                            # Subclasses can override to cancel work
                            pass

                        elif msg_type == "flush":
                            logger.debug("Flush received")
                            pass

                        elif msg_type in ("translate", "synthesize"):
                            await self._handle_process(data, config, websocket)

                        elif msg_type == "ping":
                            await websocket.send(json.dumps({
                                "type": "pong",
                                "metrics": {
                                    "model": self.metrics.model_name,
                                    "loaded": self.metrics.model_loaded,
                                    "requests": self.metrics.requests_total,
                                    "last_latency_ms": self.metrics.last_latency_ms,
                                },
                            }))

                        else:
                            logger.warning(f"Unknown message type: {msg_type}")

                    elif isinstance(message, bytes):
                        # Binary data — audio for STT
                        await self._handle_process(message, config, websocket)

                except Exception as e:
                    self.metrics.errors_total += 1
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    try:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": str(e),
                        }))
                    except Exception:
                        pass

        except websockets.ConnectionClosed:
            logger.info(f"Client disconnected: {peer}")
        except Exception as e:
            logger.error(f"Connection error: {e}", exc_info=True)

    async def _handle_config(self, data: dict, websocket) -> None:
        """Handle a config message — potentially load a new model."""
        requested_model = data.get("model")
        if requested_model and requested_model != self.current_model_name:
            async with self._lock:
                if requested_model != self.current_model_name:
                    logger.info(f"Loading model: {requested_model}")
                    t0 = time.monotonic()
                    await self.load_model(
                        requested_model,
                        **data.get("options", {}),
                    )
                    self.current_model_name = requested_model
                    self.metrics.model_name = requested_model
                    self.metrics.model_loaded = True
                    elapsed = (time.monotonic() - t0) * 1000
                    logger.info(f"Model loaded in {elapsed:.0f}ms: {requested_model}")

        await websocket.send(json.dumps({
            "type": "config_ack",
            "model": self.current_model_name,
        }))

    async def _handle_process(self, data: Any, config: dict, websocket) -> None:
        """Run the process method and send results back."""
        self.metrics.requests_total += 1
        t0 = time.monotonic()

        async for result in self.process(data, config):
            if isinstance(result, bytes):
                await websocket.send(result)
            elif isinstance(result, dict):
                await websocket.send(json.dumps(result))
            else:
                logger.warning(f"Unexpected result type: {type(result)}")

        self.metrics.last_latency_ms = (time.monotonic() - t0) * 1000

    async def serve(self) -> None:
        """Start the WebSocket server."""
        logger.info(f"Starting {self.__class__.__name__} on {self.host}:{self.port}")

        async with serve(
            self.handle_connection,
            self.host,
            self.port,
            max_size=50 * 1024 * 1024,  # 50MB max message (for large audio)
            ping_interval=30,
            ping_timeout=10,
        ):
            logger.info(f"{self.__class__.__name__} listening on ws://{self.host}:{self.port}")
            await asyncio.Future()  # run forever

    def run(self) -> None:
        """Convenience method to run the server."""
        asyncio.run(self.serve())
