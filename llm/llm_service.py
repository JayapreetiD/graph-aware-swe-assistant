

"""
llm/llm_service.py

Responsibility: call the Gemini API with a fully-assembled prompt
(from prompt_builder.py) and return the raw answer text.

This file does NOT build prompts, retrieve chunks, or verify
citations - those are separate concerns (prompt_builder.py upstream,
a citation verifier downstream). Keeping this file to "send prompt,
get answer" makes it trivial to swap models or providers later
without touching retrieval or verification logic.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

logger = logging.getLogger(__name__)

load_dotenv()

# Pinned, non-preview, non-lite flash model. Note: gemini-3.7-flash
# (the newest at time of writing) returned repeated 503 UNAVAILABLE
# errors during development, likely due to being very recently
# released with limited serving capacity. gemini-2.5-flash is
# deprecated for new accounts (404). gemini-3.6-flash is the current
# working choice: stable, non-preview, non-lite, and not hitting
# capacity issues in testing. Revisit gemini-3.7-flash later once
# it's had time to stabilize - this is a one-line change either way.
MODEL_NAME = "gemini-3.6-flash"

# Caps the OUTPUT, not the input context (that's prompt_builder's
# CONTEXT_TOKEN_BUDGET). Raised from 1024 -> 2048 -> 3072 during
# development after repeatedly hitting FinishReason.MAX_TOKENS on
# multi-step technical answers with several citations. Gemini 3.x
# models appear to spend some of the output budget on internal
# reasoning before producing visible text, which eats into this cap
# faster than the visible answer length alone would suggest. Even at
# 3072, truncation is possible and is now surfaced via
# LLMResult.truncated rather than silently returned as a complete
# answer - see generate_answer().
MAX_OUTPUT_TOKENS = 8192


@dataclass
class LLMResult:
    answer: str
    model: str
    success: bool
    error: str | None = None
    # True if Gemini stopped generating because it hit MAX_OUTPUT_TOKENS
    # rather than finishing naturally. A truncated answer may end
    # mid-sentence or mid-citation-list. Downstream code (Phase 5 eval
    # in particular) should check this and flag/exclude truncated runs
    # rather than scoring an incomplete answer as if it were complete.
    truncated: bool = False


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found in environment. Check that .env "
            "exists in the project root and contains GEMINI_API_KEY=..."
        )
    return genai.Client(api_key=api_key)


def generate_answer(prompt: str) -> LLMResult:
    """
    Send a fully-built prompt (already under the context token budget
    from prompt_builder.py) to Gemini and return the answer.

    Failures (bad key, rate limit, network issue) are caught and
    returned as a non-raising LLMResult with success=False, rather
    than propagating an exception - this matters for Phase 5, where
    a single failed call out of 90-150 shouldn't crash the whole
    evaluation run. The caller decides how to handle failures (skip,
    retry, log).
    """
    client = _get_client()

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={"max_output_tokens": MAX_OUTPUT_TOKENS},
        )
    except genai_errors.APIError as e:
        logger.error("Gemini API error: %s", e)
        return LLMResult(answer="", model=MODEL_NAME, success=False, error=str(e))
    except Exception as e:  # noqa: BLE001 - deliberately broad: any
        # failure here must not crash an eval loop; log and surface it.
        logger.error("Unexpected error calling Gemini: %s", e)
        return LLMResult(answer="", model=MODEL_NAME, success=False, error=str(e))

    # Determine whether generation was cut off by the token cap rather
    # than finishing naturally. Checked before the empty-response
    # check below, since a truncated-but-non-empty response should
    # still surface truncated=True, not just success=True silently.
    finish_reason = (
        getattr(response.candidates[0], "finish_reason", None)
        if response.candidates else None
    )
    truncated = str(finish_reason) == "FinishReason.MAX_TOKENS"
    if truncated:
        logger.warning(
            "Response truncated by MAX_TOKENS despite %d token budget.",
            MAX_OUTPUT_TOKENS,
        )

    if not response.text:
        logger.warning("Gemini returned an empty response for this prompt.")
        return LLMResult(
            answer="", model=MODEL_NAME, success=False,
            error="Empty response from model",
        )

    return LLMResult(
        answer=response.text,
        model=MODEL_NAME,
        success=True,
        truncated=truncated,
    )


if __name__ == "__main__":
    test_prompt = (
        "You are a software engineering assistant. In one sentence, "
        "explain what a Python decorator is."
    )
    result = generate_answer(test_prompt)

    print(f"Success: {result.success}")
    print(f"Model: {result.model}")
    print(f"Truncated: {result.truncated}")
    if result.success:
        print(f"Answer: {result.answer}")
    else:
        print(f"Error: {result.error}")