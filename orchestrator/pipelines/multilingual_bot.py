"""
Multilingual Voice Bot pipeline: STT → MT(in) → LLM+tools → MT(out) → TTS

User speaks any language → translated to English → LLM reasons + calls tools →
response translated back → spoken in user's language.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Awaitable, Optional

from orchestrator.clients.ws_client import ModelClient
from orchestrator.clients.llm_client import LLMClient

logger = logging.getLogger(__name__)


def is_sentence_end(text: str) -> bool:
    """Heuristic check for sentence boundary."""
    text = text.rstrip()
    return text.endswith(('.', '!', '?', '。', '！', '？'))


async def run_multilingual_bot_pipeline(
    transcript: str,
    user_lang: str,
    mt_client: ModelClient,
    llm_client: LLMClient,
    tts_client: ModelClient,
    conversation_history: list[dict],
    tools: Optional[list[dict]],
    execute_tool: Optional[Callable],
    on_subtitle: Callable[[str, str], Awaitable[None]],
    on_audio: Callable[[bytes, int], Awaitable[None]],
    on_state: Callable[[str, Optional[str]], Awaitable[None]],
) -> tuple[dict[str, float], str, str]:
    """
    Run the multilingual voice bot pipeline.

    Returns:
        (latency_dict, english_response, translated_response)
    """
    latency = {}

    # ── INBOUND MT: user language → English ──
    if user_lang not in ("eng", "eng_Latn", "en"):
        t0 = time.monotonic()
        await on_state("translating_inbound", None)

        src_code = user_lang if "_" in user_lang else f"{user_lang}_Latn"
        english_input = await mt_client.translate(
            text=transcript,
            src=src_code,
            tgt="eng_Latn",
        )
        latency["mt_inbound"] = (time.monotonic() - t0) * 1000
        logger.info(f"MT(in): '{transcript}' → '{english_input}' ({latency['mt_inbound']:.0f}ms)")
    else:
        english_input = transcript
        latency["mt_inbound"] = 0

    # ── LLM REASONING + TOOLS ──
    t0 = time.monotonic()
    await on_state("thinking", None)

    conversation_history.append({
        "role": "user",
        "content": english_input,
    })

    # Run LLM with tool loop
    english_response = await _run_llm_with_tools(
        llm_client=llm_client,
        messages=conversation_history,
        tools=tools,
        execute_tool=execute_tool,
        on_state=on_state,
    )
    latency["llm"] = (time.monotonic() - t0) * 1000
    logger.info(f"LLM: '{english_response[:80]}...' ({latency['llm']:.0f}ms)")

    # Store English response in history
    conversation_history.append({
        "role": "assistant",
        "content": english_response,
    })

    # ── OUTBOUND MT: English → user language ──
    if user_lang not in ("eng", "eng_Latn", "en"):
        t0 = time.monotonic()
        await on_state("translating_outbound", None)

        tgt_code = user_lang if "_" in user_lang else f"{user_lang}_Latn"
        translated_response = await mt_client.translate(
            text=english_response,
            src="eng_Latn",
            tgt=tgt_code,
        )
        latency["mt_outbound"] = (time.monotonic() - t0) * 1000
        logger.info(f"MT(out): '{english_response[:40]}' → '{translated_response[:40]}' ({latency['mt_outbound']:.0f}ms)")
    else:
        translated_response = english_response
        latency["mt_outbound"] = 0

    # Send translated subtitle
    await on_subtitle(translated_response, user_lang)

    # ── TTS ──
    t0 = time.monotonic()
    await on_state("speaking", None)

    tts_lang = user_lang.split("_")[0] if "_" in user_lang else user_lang
    audio_bytes, sample_rate = await tts_client.synthesize(
        text=translated_response,
        language=tts_lang,
    )
    latency["tts"] = (time.monotonic() - t0) * 1000
    logger.info(f"TTS: {len(audio_bytes)} bytes ({latency['tts']:.0f}ms)")

    # Send audio
    await on_audio(audio_bytes, sample_rate)

    # Total
    latency["total"] = sum(latency.values())

    return latency, english_response, translated_response


async def _run_llm_with_tools(
    llm_client: LLMClient,
    messages: list[dict],
    tools: Optional[list[dict]],
    execute_tool: Optional[Callable],
    on_state: Callable,
    max_tool_rounds: int = 5,
) -> str:
    """
    Run LLM with tool-calling loop.
    Returns the final text response.
    """
    for round_num in range(max_tool_rounds):
        response = await llm_client.complete(
            messages=messages,
            tools=tools if tools else None,
        )

        if response.has_tool_calls and execute_tool:
            for tc in response.tool_calls:
                logger.info(f"Tool call [{round_num+1}]: {tc.name}({tc.arguments})")
                await on_state("tool_call", tc.name)

                try:
                    result = await execute_tool(tc.name, tc.arguments)
                except Exception as e:
                    result = f"Error executing tool: {e}"

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })
        else:
            # Final response (no more tool calls)
            return response.text

    logger.warning(f"Hit max tool rounds ({max_tool_rounds})")
    # One more try without tools to force a text response
    response = await llm_client.complete(messages=messages, tools=None)
    return response.text
