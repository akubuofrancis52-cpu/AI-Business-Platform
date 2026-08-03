import json

from services.ai.openai_provider import OpenAIProvider
from services.ai.prompt_builder import build_restaurant_prompt
from services.ai.menu_intelligence import resolve_menu_object

from models.menu import Menu


# ==========================
# JSON EXTRACTION
# ==========================

def _extract_json(response):
    """
    Safely extract JSON from the AI response.

    Handles:
    - normal JSON
    - markdown JSON blocks
    - JSON embedded in additional text
    """

    if not response:
        return None

    response = response.strip()

    # Remove markdown fences
    response = response.replace(
        "```json",
        ""
    )

    response = response.replace(
        "```",
        ""
    )

    response = response.strip()

    # Direct JSON
    try:

        return json.loads(
            response
        )

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

            return json.loads(
                response[
                    start:end + 1
                ]
            )

        except json.JSONDecodeError:
            pass

    return None


# ==========================
# EXTRACT ORDER
# ==========================

def extract_order(
    business_id,
    customer_message
):
    """
    AI identifies what the customer wants.

    Python/database handles:
    - menu validation
    - smart menu resolution
    - real prices
    - subtotals
    - total calculation
    """

    # ==========================
    # LOAD MENU ONCE
    # ==========================

    menu_items = Menu.query.filter_by(
        business_id=business_id,
        available=True
    ).order_by(
        Menu.category.asc(),
        Menu.name.asc()
    ).all()

    if not menu_items:

        return {
            "items": [],
            "total": 0.0,
            "currency": "FCFA",
            "status": "pending",
            "unmatched": []
        }

    # ==========================
    # AI PROMPT
    # ==========================

    ai = OpenAIProvider()

    prompt = build_restaurant_prompt(
        business_id,
        customer_message
    )

    menu_names = [
        item.name
        for item in menu_items
    ]

    prompt += f"""

ORDER EXTRACTION MODE

Your ONLY job is to identify the products
the customer explicitly wants to order.

AVAILABLE MENU ITEM NAMES:

{json.dumps(
    menu_names,
    ensure_ascii=False
)}

RULES:

1. Extract only products the customer requested.
2. Never invent a product.
3. Quantity must be an integer.
4. "one" means 1.
5. If quantity is not specified, use 1.
6. Use the closest real menu item name.
7. Never create a new menu item.
8. Do not calculate prices.
9. Do not calculate totals.
10. Return ONLY valid JSON.
11. Do not include explanations.
12. Do not include markdown.

FORMAT:

{{
    "items": [
        {{
            "name": "Chicken Burger",
            "quantity": 2
        }}
    ]
}}

CUSTOMER MESSAGE:

{customer_message}
"""

    # ==========================
    # CALL AI
    # ==========================

    response = ai.generate(
        prompt
    )

    data = _extract_json(
        response
    )

    if not isinstance(
        data,
        dict
    ):

        raise ValueError(
            "AI returned invalid order JSON."
        )

    ai_items = data.get(
        "items",
        []
    )

    if not isinstance(
        ai_items,
        list
    ):

        raise ValueError(
            "Invalid items format."
        )

    validated_items = []
    unmatched = []

    total = 0.0

    # ==========================
    # VALIDATE EACH ITEM
    # ==========================

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

        # --------------------------
        # QUANTITY
        # --------------------------

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

        # --------------------------
        # SMART MENU RESOLUTION
        # --------------------------

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

        # --------------------------
        # REAL DATABASE PRICE
        # --------------------------

        price = float(
            menu.price
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

        total += subtotal

    # ==========================
    # RETURN
    # ==========================

    return {

        "items": validated_items,

        "total": float(
            total
        ),

        "currency": "FCFA",

        "status": "pending",

        "unmatched": unmatched

    }