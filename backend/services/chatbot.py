"""
Botify Chatbot Compatibility Layer

IMPORTANT:
This is NOT a rule-based chatbot.

All intelligence is handled by AIService + the configured
LLM provider.

This file exists so older parts of the application can still
use a simple chatbot-style interface.
"""

from services.ai.ai_service import (
    AIService,
    generate_ai_response,
)


class Chatbot:
    """
    Thin wrapper around the real Botify AI service.

    No hard-coded:
        "if hi -> hello"

    No fixed responses.

    The language model handles natural-language understanding.
    """

    def __init__(self, provider=None):
        self.ai = AIService(
            provider=provider
        )

    def respond(
        self,
        business_id,
        message,
        language="English",
        customer_name=None,
        conversation_history=None,
    ):
        """
        Generate a natural AI response.
        """

        return self.ai.generate_response(
            business_id=business_id,
            customer_message=message,
            language=language,
            customer_name=customer_name,
            conversation_history=conversation_history,
        )


def chat(
    business_id,
    message,
    language="English",
    customer_name=None,
    conversation_history=None,
):
    """
    Simple function interface.

    Example:

        response = chat(
            business_id=2,
            message="Hi"
        )
    """

    return generate_ai_response(
        business_id=business_id,
        customer_message=message,
        language=language,
        customer_name=customer_name,
        conversation_history=conversation_history,
    )