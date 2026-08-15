"""
Botify AI Service

This is the main intelligence/orchestration layer.

It does NOT contain hard-coded chatbot replies.

The LLM remains responsible for understanding the customer's
natural language. This service gives the LLM the right context:
- restaurant
- menu
- customer
- language
- conversation history
"""

import logging

from services.ai.openai_provider import OpenAIProvider
from services.ai.prompt_builder import build_restaurant_prompt


logger = logging.getLogger(__name__)


class AIService:
    """
    Main AI service used by Botify.

    This class connects:
        Business data
            +
        Menu intelligence
            +
        Conversation memory
            +
        LLM provider
    """

    def __init__(self, provider=None):
        self.provider = provider or OpenAIProvider()

    def generate_response(
        self,
        business_id,
        customer_message,
        language="English",
        customer_name=None,
        conversation_history=None,
    ):
        """
        Generate an intelligent restaurant response.

        Parameters
        ----------
        business_id:
            ID of the restaurant/business.

        customer_message:
            The customer's latest message.

        language:
            Language the AI should respond in.

        customer_name:
            Optional customer name.

        conversation_history:
            Previous messages in this conversation.

            Example:
            [
                {
                    "role": "user",
                    "content": "Hi"
                },
                {
                    "role": "assistant",
                    "content": "Hello! How can I help?"
                }
            ]

        Returns
        -------
        str
            Natural-language AI response.
        """

        if not customer_message:
            return (
                "I'm here to help. "
                "What would you like to know?"
            )

        customer_message = str(
            customer_message
        ).strip()

        if not customer_message:
            return (
                "I'm here to help. "
                "What would you like to know?"
            )

        try:
            # --------------------------------------------------
            # BUILD RESTAURANT-SPECIFIC AI INSTRUCTIONS
            # --------------------------------------------------

            prompt = build_restaurant_prompt(
                business_id=business_id,
                customer_message=customer_message,
                language=language,
                customer_name=customer_name,
            )

            # --------------------------------------------------
            # CALL THE ACTUAL AI MODEL
            # --------------------------------------------------

            response = self.provider.generate(
                prompt=prompt,
                conversation_history=conversation_history or [],
                temperature=0.7,
                max_tokens=1000,
            )

            if not response:
                raise RuntimeError(
                    "AI provider returned an empty response."
                )

            return response.strip()

        except Exception:
            logger.exception(
                "AIService failed while generating response"
            )

            # Re-raise the real error.
            #
            # The Flask route can decide how to display it
            # to the customer while the terminal keeps the
            # actual technical error for debugging.
            raise


# --------------------------------------------------------------
# SIMPLE FUNCTION API
# --------------------------------------------------------------
#
# This makes it easy for existing Flask code to call:
#
#     generate_ai_response(...)
#
# without having to manually create AIService().
# --------------------------------------------------------------

_default_service = None


def get_ai_service():
    global _default_service

    if _default_service is None:
        _default_service = AIService()

    return _default_service


def generate_ai_response(
    business_id,
    customer_message,
    language="English",
    customer_name=None,
    conversation_history=None,
):
    """
    Convenience function for Flask routes.
    """

    service = get_ai_service()

    return service.generate_response(
        business_id=business_id,
        customer_message=customer_message,
        language=language,
        customer_name=customer_name,
        conversation_history=conversation_history,
    )