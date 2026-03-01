"""
BaseAgent — shared LLM API wrapper with retry logic.
Supports both OpenAI (gpt-4o) and Anthropic (claude-*) models.
"""
from __future__ import annotations

import json
import os
import time
from typing import Union

from lift.schemas import LLMConfig


class LLMCallError(Exception):
    """Raised after exhausting all retries."""


class BaseAgent:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = None
        self._backend = self._detect_backend(config.model)

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    @staticmethod
    def _detect_backend(model: str) -> str:
        if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
            return "openai"
        if model.startswith("claude"):
            return "anthropic"
        # OpenRouter passes any provider/model slug (e.g. "qwen/qwen3.5-35b-a3b")
        if "/" in model or os.environ.get("OPENROUTER_API_KEY"):
            return "openrouter"
        raise ValueError(
            f"Cannot determine LLM backend for model '{model}'. "
            "Set OPENROUTER_API_KEY for third-party models, or use a model name "
            "starting with 'gpt' (OpenAI) or 'claude' (Anthropic)."
        )

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self._backend == "openai":
            import openai
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY not set in environment.")
            self._client = openai.OpenAI(api_key=api_key)
        elif self._backend == "openrouter":
            import openai
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENROUTER_API_KEY not set in environment.")
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=self.OPENROUTER_BASE_URL,
            )
        else:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY not set in environment.")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    # ------------------------------------------------------------------
    # Core LLM call
    # ------------------------------------------------------------------

    def call_llm(
        self,
        prompt: str,
        system_prompt: str,
        expect_json: bool = True,
    ) -> Union[dict, str]:
        """
        Calls the configured LLM with retry logic.
        If expect_json=True, parses and returns a dict.
        Raises LLMCallError after exhausting retries.
        """
        last_exc: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                raw_text = self._call_once(prompt, system_prompt)
                if expect_json:
                    return self._parse_json(raw_text)
                return raw_text
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.config.max_retries:
                    wait = 60 if "429" in str(exc) else 2 ** attempt
                    time.sleep(wait)
        raise LLMCallError(
            f"LLM call failed after {self.config.max_retries} attempts. "
            f"Last error: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Backend-specific call
    # ------------------------------------------------------------------

    def _call_once(self, prompt: str, system_prompt: str) -> str:
        client = self._get_client()
        if self._backend in ("openai", "openrouter"):
            response = client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or ""
        else:
            response = client.messages.create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text or ""

    # ------------------------------------------------------------------
    # JSON parsing helper
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop opening fence (and optional language tag) and closing fence
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        return json.loads(text)
