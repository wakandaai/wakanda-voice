"""
LLM client — speaks the OpenAI-compatible chat completions API.

Works with: vLLM, Ollama, OpenAI, Anthropic (via proxy), TGI, etc.
Supports streaming and tool calling.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    has_tool_calls: bool = False
    finish_reason: Optional[str] = None


class LLMClient:
    """
    Async client for OpenAI-compatible LLM APIs.
    Supports streaming chat completions and tool calling.
    """

    def __init__(self, url: str, model: str, api_key: Optional[str] = None):
        self.base_url = url.rstrip("/")
        self.model = model
        self.api_key = api_key or "not-needed"
        self._client = httpx.AsyncClient(timeout=120)

    async def complete(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Non-streaming completion. Returns full response."""
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            body["tools"] = tools

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        response = await self._client.post(
            f"{self.base_url}/chat/completions",
            json=body,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        result = LLMResponse(finish_reason=choice.get("finish_reason"))

        msg = choice.get("message", {})
        result.text = msg.get("content", "") or ""

        if msg.get("tool_calls"):
            result.has_tool_calls = True
            for tc in msg["tool_calls"]:
                result.tool_calls.append(ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ))

        return result

    async def complete_stream(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """
        Streaming completion. Yields text tokens as they arrive.

        Note: if the model decides to call tools, this yields nothing
        and you should use complete() instead to get tool_calls.
        For the voice pipeline, we first try streaming. If we detect
        tool calls, we fall back to non-streaming.
        """
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            body["tools"] = tools

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        async with self._client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=body,
            headers=headers,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                payload = line[6:]  # strip "data: "
                if payload.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    async def close(self):
        await self._client.aclose()
