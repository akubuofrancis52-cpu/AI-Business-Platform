import json

from services.ai.openai_provider import OpenAIProvider
from services.ai.prompt_builder import build_restaurant_prompt


def extract_order(business_id, customer_message):

    ai = OpenAIProvider()

    prompt = build_restaurant_prompt(
        business_id,
        customer_message
    )

    prompt += """

Your job is to extract the customer's order.

Return ONLY valid JSON.

Example:

{
  "items": [
    {
      "name": "Burger",
      "quantity": 2
    },
    {
      "name": "Coke",
      "quantity": 1
    }
  ]
}

Do not explain anything.
Return only JSON.
"""

    response = ai.generate(prompt)

    return json.loads(response)