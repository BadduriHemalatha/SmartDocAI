"""Gemini API client with safe error handling."""

from __future__ import annotations

from google import genai

from smartdoc.exceptions import LLMError, MissingAPIKeyError
from smartdoc.logging_setup import get_logger

logger = get_logger()


class GeminiClient:
    def __init__(self, api_key: str, model_name: str) -> None:
        self.api_key = (api_key or "").strip()
        self.model_name = model_name
        self._client = None

    def require_key(self) -> None:
        if not self.api_key:
            raise MissingAPIKeyError(
                "GEMINI_API_KEY is missing",
                user_message=(
                    "Gemini API key is not configured. "
                    "Set GEMINI_API_KEY in the .env file, then restart the app."
                ),
            )

    def _load(self):
        self.require_key()

        if self._client is not None:
            return self._client

        try:
            self._client = genai.Client(api_key=self.api_key)
        except Exception as exc:
            logger.exception("Failed to initialize Gemini")
            raise LLMError(
                f"Gemini init failed: {exc}",
                user_message=(
                    "Could not initialize the Gemini API client. "
                    "Check the API key and network connection."
                ),
            ) from exc

        return self._client

    def generate(self, prompt: str) -> str:
        client = self._load()

        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
        except Exception as exc:
            logger.exception("Gemini generation failed")

            message = str(exc).lower()

            if (
                "api key" in message
                or "permission" in message
                or "401" in message
                or "403" in message
            ):
                user = "Gemini rejected the API key. Verify GEMINI_API_KEY."
            elif (
                "quota" in message
                or "429" in message
                or "resource exhausted" in message
            ):
                user = (
                    "Gemini rate limit or quota was exceeded. "
                    "Wait a moment and try again."
                )
            elif "404" in message or "not found" in message:
                user = (
                    f"The Gemini model '{self.model_name}' is unavailable. "
                    "Check the configured model name."
                )
            else:
                user = (
                    "The Gemini API request failed. "
                    "Check your network connection and try again."
                )

            raise LLMError(
                f"Gemini generate failed: {exc}",
                user_message=user,
            ) from exc

        text = _response_text(response)

        if not text:
            raise LLMError(
                "Empty Gemini response",
                user_message=(
                    "Gemini returned an empty response. "
                    "Try again with a shorter question."
                ),
            )

        return text.strip()


def _response_text(response) -> str:
    text = getattr(response, "text", None)

    if text:
        return str(text)

    return ""