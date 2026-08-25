import json
import re

from services.ai.openai_provider import OpenAIProvider


def _extract_json(text):
    """
    Safely extract a JSON object from an AI response.
    Handles normal JSON and ```json code blocks.
    """

    if not text:
        return None

    text = text.strip()

    # Direct JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # JSON code block
    match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        re.DOTALL
    )

    if match:

        try:
            return json.loads(
                match.group(1)
            )
        except json.JSONDecodeError:
            return None

    # Find first JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        try:
            return json.loads(
                text[start:end + 1]
            )
        except json.JSONDecodeError:
            return None

    return None


def interpret_order_request(
    message,
    order
):

    order_items = []

    for item in order.items:

        order_items.append({
            "name": item.name,
            "quantity": item.quantity,
            "price": float(item.price)
        })

    prompt = f"""
You are an order-management intent parser for a restaurant.

The customer wants to modify or cancel an existing order.

Customer message:
{message}

Current order:
{json.dumps(order_items, ensure_ascii=False)}

Return ONLY valid JSON.

Allowed actions:

cancel
Remove the entire order.

remove_item
Remove a specific item from the order.

set_quantity
Change the quantity of a specific item.

add_item
Add another menu item.

replace_item
Replace one existing item with another item.

unknown
The request is unclear.

Return exactly this structure:

{{
    "action": "cancel|remove_item|set_quantity|add_item|replace_item|unknown",
    "item_name": null,
    "quantity": null,
    "new_item_name": null
}}

Rules:

- For add_item:
  - item_name must be the menu item the customer wants to add.
  - Extract the complete item name from the customer's message.
  - Phrases such as "add another", "add one more", "I want another",
    "give me another", or "add 2 more" refer to adding that item.
  - Example:
    "Add another Signature Lebanese Shawarma"
    must produce:
    "action": "add_item",
    "item_name": "Signature Lebanese Shawarma",
    "quantity": 1

- item_name must refer to an item already in the order when using
  remove_item, set_quantity, or replace_item.
- quantity must be an integer when relevant.
- new_item_name is only used for replace_item.
- Do not invent items.
- If the request is unclear, use unknown.
- Do not invent items.
- If the request is unclear, use unknown.
"""

    provider = OpenAIProvider()

    response = provider.generate(
        prompt
    )

    result = _extract_json(
        response
    )

    if not isinstance(result, dict):
        return {
            "action": "unknown",
            "item_name": None,
            "quantity": None,
            "new_item_name": None
        }

    return {
        "action": result.get(
            "action",
            "unknown"
        ),
        "item_name": result.get(
            "item_name"
        ),
        "quantity": result.get(
            "quantity"
        ),
        "new_item_name": result.get(
            "new_item_name"
        )
    }