import os

import pytest
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def test_openrouter_connection():
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        pytest.skip("OPENROUTER_API_KEY is not configured")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    response = client.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL", "openrouter/free"),
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: Hello, this is a test.",
            }
        ],
        max_tokens=32,
    )

    assert response.choices
    assert response.choices[0].message.content is not None
