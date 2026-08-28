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
    Fast AI provider.

    Uses Groq when GROQ_API_KEY is available.
    Falls back to OpenRouter when it is not.
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
                and self.model.startswith("openai/gpt-oss")
            ):
                request_kwargs["reasoning_effort"] = "low"

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