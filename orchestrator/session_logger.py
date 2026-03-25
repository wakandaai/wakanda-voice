"""
Session logger — captures bilingual conversation history to JSONL files.

Each session gets its own log file. Every turn records:
- Original text (user's language)
- English translation (what LLM saw)
- LLM response (English)
- Back-translation (user's language)
- Tool calls if any
- Per-stage latency
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SessionLogger:
    """Logs conversation turns to a JSONL file."""

    def __init__(self, log_dir: str = "./logs", session_id: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"session_{self.session_id}.jsonl"
        self.turn_count = 0

        # Write session header
        self._write({
            "type": "session_start",
            "session_id": self.session_id,
            "timestamp": self._now(),
        })

        logger.info(f"Session log: {self.log_path}")

    def log_user_turn(
        self,
        original_text: str,
        english_text: str,
        user_lang: str,
        latency: dict[str, float],
    ) -> None:
        """Log a user's speech turn."""
        self.turn_count += 1
        self._write({
            "type": "turn",
            "turn": self.turn_count,
            "role": "user",
            "timestamp": self._now(),
            "user_lang": user_lang,
            "original_text": original_text,
            "english_text": english_text,
            "latency_ms": latency,
        })

    def log_assistant_turn(
        self,
        english_text: str,
        translated_text: str,
        user_lang: str,
        tool_calls: Optional[list[dict]] = None,
        latency: dict[str, float] = None,
    ) -> None:
        """Log the assistant's response turn."""
        self._write({
            "type": "turn",
            "turn": self.turn_count,
            "role": "assistant",
            "timestamp": self._now(),
            "user_lang": user_lang,
            "english_text": english_text,
            "translated_text": translated_text,
            "tool_calls": tool_calls or [],
            "latency_ms": latency or {},
        })

    def log_event(self, event_type: str, **data) -> None:
        """Log a generic event (pipeline state, errors, etc.)."""
        self._write({
            "type": event_type,
            "timestamp": self._now(),
            **data,
        })

    def close(self) -> None:
        """Write session end marker."""
        self._write({
            "type": "session_end",
            "session_id": self.session_id,
            "timestamp": self._now(),
            "total_turns": self.turn_count,
        })
        logger.info(f"Session ended: {self.turn_count} turns logged to {self.log_path}")

    def _write(self, entry: dict) -> None:
        """Append one JSON line to the log file."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
