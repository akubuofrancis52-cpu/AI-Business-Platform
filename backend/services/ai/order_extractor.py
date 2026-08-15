import json
import logging

from services.ai.openai_provider import OpenAIProvider
from services.ai.menu_intelligence import resolve_menu_object

from models.menu import Menu


logger = logging.getLogger(__name__)


# ============================================================
# JSON EXTRACTION
# ============================================================

def _extract_json(response):

    if not response:
        return None

    response = str(
        response
    ).strip()

    # Remove markdown fences
    response = response.replace(
        "```json",
        ""
    )

    response = response.replace(
        "```JSON",
        ""
    )

    response = response.replace(
        "```",
        ""
    )

    response = response.strip()

    # Direct JSON
    try:

        data = json.loads(
            response
        )

        if isinstance(data, dict):
            return data

    except json.JSONDecodeError:
        pass

    # Embedded JSON
    start = response.find("{")
    end = response.rfind("}")

    if (
        start != -1
        and end != -1
        and end > start
    ):

        try:

            data = json.loads(
                response[
                    start:end + 1
                ]
            )

            if isinstance(data, dict):
                return data

        except json.JSONDecodeError:
            pass

    return None


# ============================================================
# EXTRACT ORDER
# ============================================================

def extract_order(
    business_id,
    customer_message
):

    customer_message = str(
        customer_message or ""
    ).strip()

    if not customer_message:

        return {
            "items": [],
            "total": 0.0,
            "currency": "FCFA",
            "status": "pending",
            "unmatched": []
        }

    # ========================================================
    # LOAD REAL MENU
    # ========================================================

    menu_items = (
        Menu.query
        .filter_by(
            business_id=business_id,
            available=True
        )
        .order_by(
            Menu.category.asc(),
            Menu.name.asc()
        )
        .all()
    )

    if not menu_items:

        return {
            "items": [],
            "total": 0.0,
            "currency": "FCFA",
            "status": "pending",
            "unmatched": []
        }

    # ========================================================
    # MENU NAMES
    # ========================================================

    menu_names = [
        item.name
        for item in menu_items
    ]

    menu_text = "\n".join(
        f"- {name}"
        for name in menu_names
    )

    # ========================================================
    # DEDICATED EXTRACTION PROMPT
    # ========================================================

    prompt = f"""
You are a restaurant order extraction engine.

Your ONLY job is to identify the food or drink items
that the customer explicitly wants to order.

You are NOT the customer-facing assistant.

Do not greet the customer.

Do not explain anything.

Do not recommend anything.

Do not calculate prices.

Do not calculate totals.

Return ONLY valid JSON.

============================================================
AVAILABLE MENU ITEMS
============================================================

{menu_text}

============================================================
RULES
============================================================

1. Only use menu items from the available menu list.

2. Never invent a menu item.

3. If the customer mentions a product that cannot be
   matched to a real menu item, put it in "unmatched".

4. Quantity must be an integer.

5. If quantity is not specified, use 1.

6. "a", "an", and "one" mean quantity 1.

7. "two", "2", etc. mean quantity 2.

8. Handle normal spelling mistakes when the intended
   menu item is clear.

9. Use the exact menu item name in the output.

10. If the customer is NOT actually requesting a food
    or drink item, return an empty items list.

11. Do not create products from descriptions.

12. Do not create products that are not on the menu.

13. Do not include explanations.

14. Do not use markdown.

15. Do not write anything outside the JSON object.

============================================================
REQUIRED FORMAT
============================================================

{{
    "items": [
        {{
            "name": "Exact Menu Item Name",
            "quantity": 1
        }}
    ],
    "unmatched": []
}}

============================================================
CUSTOMER MESSAGE
============================================================

{customer_message}
"""

    # ========================================================
    # AI CALL
    # ========================================================

    ai = OpenAIProvider()

    response = ai.generate(
        prompt,
        temperature=0,
        max_tokens=300,
    )

    logger.info(
        "Order extraction response: %s",
        response
    )

    data = _extract_json(
        response
    )

    if not isinstance(data, dict):

        logger.error(
            "Invalid order extraction response: %r",
            response
        )

        raise ValueError(
            "AI returned invalid order JSON."
        )

    ai_items = data.get(
        "items",
        []
    )

    ai_unmatched = data.get(
        "unmatched",
        []
    )

    if not isinstance(
        ai_items,
        list
    ):

        ai_items = []

    if not isinstance(
        ai_unmatched,
        list
    ):

        ai_unmatched = []

    validated_items = []
    unmatched = []

    # ========================================================
    # VALIDATE EACH ITEM
    # ========================================================

    for item in ai_items:

        if not isinstance(
            item,
            dict
        ):
            continue

        name = str(
            item.get(
                "name",
                ""
            )
        ).strip()

        if not name:
            continue

        # ====================================================
        # QUANTITY
        # ====================================================

        try:

            quantity = int(
                item.get(
                    "quantity",
                    1
                )
            )

        except (
            TypeError,
            ValueError
        ):

            quantity = 1

        if quantity < 1:
            quantity = 1

        # ====================================================
        # REAL MENU RESOLUTION
        # ====================================================

        menu = resolve_menu_object(
            business_id,
            name,
            menu_items=menu_items
        )

        if not menu:

            unmatched.append(
                name
            )

            continue

        # ====================================================
        # REAL DATABASE PRICE
        # ====================================================

        price = float(
            menu.price or 0
        )

        subtotal = (
            price * quantity
        )

        validated_items.append({
            "name": menu.name,
            "quantity": quantity,
            "price": price,
            "subtotal": float(
                subtotal
            )
        })

    # ========================================================
    # AI UNMATCHED
    # ========================================================

    for item in ai_unmatched:

        item = str(
            item
        ).strip()

        if (
            item
            and item not in unmatched
        ):

            unmatched.append(
                item
            )

    # ========================================================
    # REAL TOTAL
    # ========================================================

    total = sum(
        item["subtotal"]
        for item in validated_items
    )

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "items": validated_items,
        "total": float(total),
        "currency": "FCFA",
        "status": "pending",
        "unmatched": unmatched
    }