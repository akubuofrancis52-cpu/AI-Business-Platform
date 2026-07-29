class AIProvider:

    def generate(self, prompt):

        raise NotImplementedError(
            "AI providers must implement generate()."
        )