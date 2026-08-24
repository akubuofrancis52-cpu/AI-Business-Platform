import json
import logging
import re 

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

def _try_fast_order_extraction(
    customer_message,
    menu_items,
):
    """
    Resolve simple single-item orders without an LLM.

    Only activates when exactly one real menu item is
    explicitly present in the customer message and the
    message contains a clear ordering phrase.
    """

    text = str(
        customer_message or ""
    ).strip().lower()

    if not text or not menu_items:
        return None

    # --------------------------------------------------------
    # CLEAR ORDERING LANGUAGE
    # --------------------------------------------------------

    order_phrases = (
        "i want ",
        "i'd like ",
        "id like ",
        "i would like ",
        "give me ",
        "get me ",
        "bring me ",
        "i'll have ",
        "ill have ",
        "can i get ",
        "can i have ",
        "order ",
    )

    if not any(
        phrase in text
        for phrase in order_phrases
    ):
        return None

    # Avoid treating obvious availability questions as orders.
    question_phrases = (
        "do you have",
        "what do you have",
        "what is on",
        "what's on",
        "is there",
        "are there",
    )

    if any(
        phrase in text
        for phrase in question_phrases
    ):
        return None

    # --------------------------------------------------------
    # FIND EXACT MENU ITEMS
    # --------------------------------------------------------

    matches = []

    for menu_item in menu_items:

        menu_name = str(
            menu_item.name or ""
        ).strip()

        if not menu_name:
            continue

        normalized_name = re.sub(
            r"\s+",
            " ",
            re.sub(
                r"[^\w\s]",
                " ",
                menu_name.lower(),
            ),
        ).strip()

        if (
            normalized_name
            and normalized_name in text
        ):
            matches.append(
                menu_item
            )

    # Only bypass the LLM when exactly one item matches.
    if len(matches) != 1:
        return None

    menu_item = matches[0]

    # --------------------------------------------------------
    # QUANTITY
    # --------------------------------------------------------

    quantity = 1

    number_match = re.search(
        r"\b(\d+)\b",
        text,
    )

    if number_match:

        try:
            quantity = max(
                1,
                int(
                    number_match.group(1)
                ),
            )
        except ValueError:
            quantity = 1

    else:

        word_quantities = {
            "one": 1,
            "a": 1,
            "an": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
        }

        for word, value in word_quantities.items():

            if re.search(
                rf"\b{word}\b",
                text,
            ):
                quantity = value
                break

    price = float(
        menu_item.price or 0
    )

    subtotal = (
        price * quantity
    )

    return {
        "items": [
            {
                "name": menu_item.name,
                "quantity": quantity,
                "price": price,
                "subtotal": float(
                    subtotal
                ),
            }
        ],
        "total": float(
            subtotal
        ),
        "currency": "FCFA",
        "status": "pending",
        "unmatched": [],
    }

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
    # FAST EXACT-MENU ORDER PATH
    # ========================================================

    fast_result = _try_fast_order_extraction(
        customer_message,
        menu_items,
    )

    if fast_result is not None:

        logger.info(
            "Fast order extraction used for: %s",
            customer_message,
        )

        return fast_result

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
You extract restaurant orders into JSON.

Use ONLY exact items from the AVAILABLE MENU.
Never invent products, prices, or descriptions.

Rules:
- Return ONLY valid JSON.
- "items" contains foods/drinks the customer explicitly wants.
- Use the exact menu item name.
- Quantity must be an integer; default to 1.
- "a", "an", "one" = 1.
- Handle obvious spelling mistakes when the intended menu item is clear.
- Put requested items that cannot be matched into "unmatched".
- If the customer is not requesting food/drinks, return empty "items".
- Do not recommend, explain, greet, calculate prices, or calculate totals.
- No markdown or text outside the JSON.

AVAILABLE MENU:
{menu_text}

REQUIRED JSON:
{{
  "items": [
    {{
      "name": "Exact Menu Item Name",
      "quantity": 1
    }}
  ],
  "unmatched": []
}}

CUSTOMER:
{customer_message}
"""

    # ========================================================
    # AI CALL
    # ========================================================

    ai = OpenAIProvider()

    response = ai.generate(
        prompt,
        temperature=0,
        max_tokens=120,
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