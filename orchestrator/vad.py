"""
Voice Activity Detection using Silero VAD.

Segments continuous audio from the microphone into discrete utterances.
Each utterance is a complete speech segment bounded by silence.
The orchestrator feeds continuous audio frames in and gets complete
utterance byte arrays out when the speaker stops talking.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import AsyncIterator, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VADSegmenter:
    """
    Buffers audio frames and uses Silero VAD to detect speech boundaries.

    Feed audio frames via feed(). When VAD detects end-of-speech (silence
    exceeding min_silence_ms after speech), yields the complete utterance
    audio as bytes.

    Args:
        threshold: VAD speech probability threshold (0-1). Higher = stricter.
        min_silence_ms: Minimum silence after speech to trigger end-of-utterance.
        min_speech_ms: Minimum speech duration to consider valid (filters noise bursts).
        sample_rate: Audio sample rate (must be 16000 for Silero).
        frame_ms: Duration of each audio frame in milliseconds.
        pre_speech_ms: How much audio before speech onset to include (captures word starts).
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_ms: int = 700,
        min_speech_ms: int = 250,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        pre_speech_ms: int = 300,
    ):
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms

        # Load Silero VAD
        self.vad_model, _ = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            trust_repo=True,
        )
        self.vad_model.eval()

        # State
        self.is_speaking = False
        self.speech_buffer = bytearray()
        self.silence_frames = 0
        self.speech_frames = 0

        # Pre-speech ring buffer: stores recent frames before speech starts
        # so we capture the beginning of words
        pre_speech_frames = max(1, pre_speech_ms // frame_ms)
        self.pre_speech_buffer: deque[bytes] = deque(maxlen=pre_speech_frames)

        # Silero expects specific chunk sizes
        self.samples_per_frame = int(sample_rate * frame_ms / 1000)

        logger.info(
            f"VAD initialized: threshold={threshold}, "
            f"min_silence={min_silence_ms}ms, min_speech={min_speech_ms}ms"
        )

    def reset(self) -> None:
        """Reset VAD state for a new session."""
        self.is_speaking = False
        self.speech_buffer.clear()
        self.silence_frames = 0
        self.speech_frames = 0
        self.pre_speech_buffer.clear()
        self.vad_model.reset_states()

    async def feed(self, pcm_bytes: bytes) -> AsyncIterator[bytes]:
        """
        Feed a chunk of PCM audio (16-bit, 16kHz, mono).
        Yields complete utterance audio when end-of-speech is detected.
        """
        # Buffer incoming bytes
        self._raw_buffer = getattr(self, '_raw_buffer', bytearray())
        self._raw_buffer.extend(pcm_bytes)

        # Process in 512-sample chunks (1024 bytes at 16-bit)
        chunk_bytes = 512 * 2  # 512 samples * 2 bytes per sample
        while len(self._raw_buffer) >= chunk_bytes:
            chunk = bytes(self._raw_buffer[:chunk_bytes])
            del self._raw_buffer[:chunk_bytes]
            async for utterance in self._process_frame(chunk):
                yield utterance

    async def _process_frame(self, frame_bytes: bytes) -> AsyncIterator[bytes]:
        """Process one audio frame through VAD."""
        # Convert to float tensor for Silero
        frame_np = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        frame_tensor = torch.from_numpy(frame_np)

        # Get speech probability
        speech_prob = self.vad_model(frame_tensor, self.sample_rate).item()

        if speech_prob >= self.threshold:
            # Speech detected
            if not self.is_speaking:
                # Speech onset — include pre-speech buffer
                self.is_speaking = True
                self.speech_frames = 0
                self.silence_frames = 0

                # Prepend buffered pre-speech audio
                for pre_frame in self.pre_speech_buffer:
                    self.speech_buffer.extend(pre_frame)

                logger.debug(f"Speech started (prob={speech_prob:.2f})")

            self.speech_buffer.extend(frame_bytes)
            self.speech_frames += 1
            self.silence_frames = 0

        else:
            # Silence
            if self.is_speaking:
                # Still in speech segment, counting silence
                self.speech_buffer.extend(frame_bytes)
                self.silence_frames += 1

                silence_ms = self.silence_frames * self.frame_ms
                speech_ms = self.speech_frames * self.frame_ms

                if silence_ms >= self.min_silence_ms:
                    # End of speech detected
                    if speech_ms >= self.min_speech_ms:
                        # Valid utterance — yield it
                        utterance = bytes(self.speech_buffer)
                        logger.debug(
                            f"Speech ended: {speech_ms}ms speech, "
                            f"{silence_ms}ms silence, "
                            f"{len(utterance)} bytes"
                        )
                        yield utterance
                    else:
                        logger.debug(
                            f"Speech too short ({speech_ms}ms), discarding"
                        )

                    # Reset for next utterance
                    self.speech_buffer.clear()
                    self.is_speaking = False
                    self.speech_frames = 0
                    self.silence_frames = 0
            else:
                # Not speaking — keep in pre-speech ring buffer
                self.pre_speech_buffer.append(frame_bytes)
