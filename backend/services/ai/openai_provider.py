import os
import logging

from dotenv import load_dotenv
from openai import OpenAI

from services.ai.provider import AIProvider


load_dotenv()

logger = logging.getLogger(__name__)


class OpenAIProvider(AIProvider):
    """
    OpenRouter-backed AI provider.

    OpenRouter exposes an OpenAI-compatible API, so the
    OpenAI Python SDK can be used with OpenRouter's base URL.
    """

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing. "
                "Add your OpenRouter API key to the .env file."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        self.model = os.getenv(
            "OPENROUTER_MODEL",
            "openrouter/free",
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

        conversation_history should contain messages like:

        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        """

        messages = [
            {
                "role": "system",
                "content": str(prompt),
            }
        ]

        # ------------------------------------------------------
        # CONVERSATION HISTORY
        # ------------------------------------------------------

        if conversation_history:
            for message in conversation_history:

                if not isinstance(message, dict):
                    continue

                role = message.get("role")
                content = message.get("content")

                if role not in ("user", "assistant"):
                    continue

                if not content:
                    continue

                messages.append(
                    {
                        "role": role,
                        "content": str(content),
                    }
                )

        # ------------------------------------------------------
        # API REQUEST
        # ------------------------------------------------------

        try:

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # --------------------------------------------------
            # CHECK RESPONSE
            # --------------------------------------------------

            if not response.choices:
                raise RuntimeError(
                    "The AI provider returned no choices."
                )

            choice = response.choices[0]

            content = choice.message.content

            # --------------------------------------------------
            # EMPTY RESPONSE
            # --------------------------------------------------

            if not content:

                finish_reason = getattr(
                    choice,
                    "finish_reason",
                    None,
                )

                logger.warning(
                    "OpenRouter returned empty content. "
                    "model=%s finish_reason=%s",
                    self.model,
                    finish_reason,
                )

                # ------------------------------------------------
                # RETRY
                # ------------------------------------------------

                logger.info(
                    "Retrying OpenRouter request with "
                    "larger max_tokens."
                )

                retry_response = (
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max(
                            max_tokens,
                            100,
                        ),
                    )
                )

                if not retry_response.choices:
                    raise RuntimeError(
                        "The AI provider returned no choices "
                        "on retry."
                    )

                retry_choice = retry_response.choices[0]

                content = retry_choice.message.content

                if not content:

                    retry_finish_reason = getattr(
                        retry_choice,
                        "finish_reason",
                        None,
                    )

                    logger.error(
                        "OpenRouter returned empty content "
                        "after retry. "
                        "model=%s finish_reason=%s",
                        self.model,
                        retry_finish_reason,
                    )

                    raise RuntimeError(
                        "The AI provider returned an empty "
                        "response after retry."
                    )

            # --------------------------------------------------
            # FINAL RESPONSE
            # --------------------------------------------------

            content = str(content).strip()

            if not content:
                raise RuntimeError(
                    "The AI provider returned empty text."
                )

            return content

        except Exception as exc:

            logger.exception(
                "AI provider request failed"
            )

            raise RuntimeError(
                f"AI provider request failed: {exc}"
            ) from exc