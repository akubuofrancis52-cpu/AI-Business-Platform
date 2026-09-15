import logging


logger = logging.getLogger(__name__)


class AIProvider:
    """
    Base interface for all Botify AI providers.
    """

    def generate(
        self,
        prompt,
        conversation_history=None,
        temperature=0.7,
        max_tokens=1000,
    ):
        raise NotImplementedError(
            "AI providers must implement generate()."
        )


def get_provider():
    """
    Return the configured Botify AI provider.

    OpenAIProvider internally selects:
    1. Groq when GROQ_API_KEY is configured.
    2. OpenRouter when GROQ_API_KEY is unavailable.

    The import is intentionally lazy to avoid a circular import:
    openai_provider.py imports AIProvider from this module.
    """
    from .openai_provider import OpenAIProvider

    return OpenAIProvider()
