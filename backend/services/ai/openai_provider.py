import os
import logging
import time
import inspect

from dotenv import load_dotenv
from openai import OpenAI

from services.ai.provider import AIProvider


load_dotenv()

logger = logging.getLogger(__name__)


class OpenAIProvider(AIProvider):
    """
    AI provider.

    Uses Groq when GROQ_API_KEY is available.
    Falls back to OpenRouter when it is not.

    Supports:
    - Normal text generation
    - Vision/image analysis
    """

    def __init__(self):
        self.groq_api_key = os.getenv(
            "GROQ_API_KEY"
        )

        self.openrouter_api_key = os.getenv(
            "OPENROUTER_API_KEY"
        )

        if self.groq_api_key:

            self.client = OpenAI(
                api_key=self.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            )

            self.provider_name = "Groq"

            self.model = os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-20b",
            )

        elif self.openrouter_api_key:

            self.client = OpenAI(
                api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
            )

            self.provider_name = "OpenRouter"

            self.model = os.getenv(
                "OPENROUTER_MODEL",
                "openrouter/free",
            )

        else:

            raise RuntimeError(
                "Neither GROQ_API_KEY nor "
                "OPENROUTER_API_KEY is configured."
            )

    def generate(
        self,
        prompt,
        conversation_history=None,
        temperature=0.7,
        max_tokens=1000,
    ):
        """
        Generate an AI response.
        """

        messages = [
            {
                "role": "system",
                "content": str(prompt),
            }
        ]

        if conversation_history:

            for message in conversation_history:

                if not isinstance(
                    message,
                    dict
                ):
                    continue

                role = message.get(
                    "role"
                )

                content = message.get(
                    "content"
                )

                if role not in (
                    "user",
                    "assistant"
                ):
                    continue

                if not content:
                    continue

                messages.append(
                    {
                        "role": role,
                        "content": str(content),
                    }
                )

        try:

            ai_start = time.perf_counter()

            request_kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if (
                self.provider_name == "Groq"
                and self.model.startswith(
                    "openai/gpt-oss"
                )
            ):
                request_kwargs[
                    "reasoning_effort"
                ] = "low"

            response = None
            content = ""

            for attempt in range(2):

                try:

                    response = (
                        self.client
                        .chat
                        .completions
                        .create(
                            **request_kwargs
                        )
                    )

                except Exception:

                    if attempt == 1:
                        raise

                    continue

                if not response.choices:

                    if attempt == 0:

                        logger.warning(
                            "%s returned no choices; retrying once.",
                            self.provider_name,
                        )

                        continue

                    raise RuntimeError(
                        "The AI provider returned "
                        "no choices."
                    )

                choice = response.choices[0]

                content = (
                    choice.message.content
                )

                if content:

                    content = str(
                        content
                    ).strip()

                    if content:
                        break

                if attempt == 0:

                    logger.warning(
                        "%s returned an empty response; "
                        "retrying once.",
                        self.provider_name,
                    )

                    continue

                raise RuntimeError(
                    "The AI provider returned "
                    "an empty response."
                )

            elapsed = (
                time.perf_counter()
                - ai_start
            )

            logger.warning(
                "[PERF] %s %.2fs model=%s",
                self.provider_name,
                elapsed,
                self.model,
            )

            if not content:

                raise RuntimeError(
                    "The AI provider returned "
                    "empty text."
                )

            return content

        except Exception as exc:

            logger.exception(
                "%s request failed",
                self.provider_name,
            )

            raise RuntimeError(
                f"{self.provider_name} "
                f"request failed: {exc}"
            ) from exc

    def analyze_image(
        self,
        image_data_url,
        prompt,
        max_tokens=220,
    ):
        """
        Analyze an image using a Groq vision-capable model.

        Uses non-thinking mode so that internal reasoning is
        not returned as visible text.
        """

        if not image_data_url:
            raise ValueError(
                "Image data URL is required."
            )

        if self.provider_name != "Groq":
            raise RuntimeError(
                "Vision image analysis currently "
                "requires the Groq provider."
            )

        vision_model = os.getenv(
            "GROQ_VISION_MODEL",
            "qwen/qwen3.8-27b",
        )

        logger.warning(
            "[VISION] provider=%s model=%s image_bytes=%s",
            self.provider_name,
            vision_model,
            len(image_data_url),
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": str(prompt),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                        },
                    },
                ],
            }
        ]

        try:

            ai_start = time.perf_counter()

            response = (
                self.client
                .chat
                .completions
                .create(
                    model=vision_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    reasoning_effort="none",
                )
            )

            logger.warning(
                "[VISION] request completed in %.2fs",
                time.perf_counter() - ai_start,
            )

            if not response.choices:

                raise RuntimeError(
                    "The vision model returned "
                    "no choices."
                )

            content = (
                response.choices[0]
                .message
                .content
                or ""
            ).strip()

            if not content:

                raise RuntimeError(
                    "The vision model returned "
                    "empty text."
                )

            # Defensive cleanup in case a provider/model
            # still returns reasoning tags.
            if "<think>" in content:

                content = (
                    content.split(
                        "<think>",
                        1
                    )[0]
                    .strip()
                )

            if not content:

                raise RuntimeError(
                    "The vision model returned "
                    "no usable final text."
                )

            elapsed = (
                time.perf_counter()
                - ai_start
            )

            logger.warning(
                "[PERF] %s vision %.2fs model=%s",
                self.provider_name,
                elapsed,
                vision_model,
            )

            return content

        except Exception as exc:

            logger.exception(
                "%s vision request failed",
                self.provider_name,
            )

            raise RuntimeError(
                f"{self.provider_name} "
                f"vision request failed: {exc}"
            ) from exc

