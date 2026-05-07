"""
BaseAgent — shared LLM API wrapper using the OpenAI API.
Model: configured via config.yaml  (default: gpt-5.2)
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Union

from lift.schemas import LLMConfig

logger = logging.getLogger(__name__)


class LLMCallError(Exception):
    """Raised after exhausting all retries."""


class BaseAgent:

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = None

    # ------------------------------------------------------------------
    # Client — OpenAI API
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is not None:
            return self._client
        import openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Add OPENAI_API_KEY=<your-key> to your .env file."
            )
        self._client = openai.OpenAI(api_key=api_key)
        logger.info("LLM client → OpenAI / model: %s", self.config.model)
        return self._client

    # ------------------------------------------------------------------
    # Core LLM call with retry
    # ------------------------------------------------------------------

    def call_llm(
        self,
        prompt: str,
        system_prompt: str,
        expect_json: bool = True,
    ) -> Union[dict, str]:
        """
        Calls the configured LLM via the OpenAI API with retry logic.
        Returns a parsed dict when expect_json=True, raw str otherwise.
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

    def _call_once(self, prompt: str, system_prompt: str) -> str:
        client = self._get_client()
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

    # ------------------------------------------------------------------
    # JSON parsing helper
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        return json.loads(text)
