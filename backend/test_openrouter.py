import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

response = client.chat.completions.create(
    model=os.getenv("OPENROUTER_MODEL", "openrouter/free"),
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: Hello, this is a test."
        }
    ]
)

print("TYPE:", type(response))
print("\nFULL RESPONSE:")
print(response)

print("\nMESSAGE:")
print(response.choices[0].message)

print("\nCONTENT:")
print(repr(response.choices[0].message.content))

print("\nFINISH REASON:")
print(response.choices[0].finish_reason)