

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

MODEL_NAME = "gemini-3.6-flash"
MAX_OUTPUT_TOKENS = 2048


@dataclass
class LLMResult:
    answer: str
    model: str
    success: bool
    error: str | None = None


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found in environment. Check that .env "
            "exists in the project root and contains GEMINI_API_KEY=..."
        )
    return genai.Client(api_key=api_key)


def generate_answer(prompt: str) -> LLMResult:
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
    except Exception as e:  # noqa: BLE001
        logger.error("Unexpected error calling Gemini: %s", e)
        return LLMResult(answer="", model=MODEL_NAME, success=False, error=str(e))
    print(
        "finish_reason:",
        getattr(response.candidates[0], "finish_reason", "unknown") if response.candidates else "no candidates",)
    if not response.text:
        logger.warning("Gemini returned an empty response for this prompt.")
        return LLMResult(
            answer="", model=MODEL_NAME, success=False,
            error="Empty response from model",
        )

    return LLMResult(answer=response.text, model=MODEL_NAME, success=True)


if __name__ == "__main__":
    test_prompt = (
        "You are a software engineering assistant. In one sentence, "
        "explain what a Python decorator is."
    )
    result = generate_answer(test_prompt)

    print(f"Success: {result.success}")
    print(f"Model: {result.model}")
    if result.success:
        print(f"Answer: {result.answer}")
    else:
        print(f"Error: {result.error}")