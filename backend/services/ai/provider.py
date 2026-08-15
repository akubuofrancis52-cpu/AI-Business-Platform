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