import os

from dotenv import load_dotenv
from openai import OpenAI

from services.ai.provider import AIProvider

# Load variables from .env
load_dotenv()

print("Dotenv loaded.")
print("OPENROUTER_API_KEY:", os.getenv("OPENROUTER_API_KEY"))


class OpenAIProvider(AIProvider):

    def __init__(self):

        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )

    def generate(self, prompt):

        response = self.client.chat.completions.create(
            model="inclusionai/ling-3.0-flash:free",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful restaurant assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content