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
    Resolve obvious menu-item orders locally without an LLM.

    Handles:
    - Exact menu names
    - Common shortened names
    - Multiple items
    - Basic numeric quantities

    Returns None when the request is ambiguous so the LLM
    can handle it.
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
        # English
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

        # French ordering
        "je veux ",
        "je voudrais ",
        "donnez-moi ",
        "donne-moi ",
        "je prendrai ",
        "je souhaite ",
        "je peux avoir ",
        "je vais prendre ",

        # French pending-order additions
        "ajoute ",
        "ajouter ",
        "ajoute-moi ",
        "ajoute moi ",
        "ajouter-moi ",
        "ajouter moi ",
        "je veux ajouter ",
        "je voudrais ajouter ",
        "je souhaite ajouter ",
    )

    if not any(
        phrase in text
        for phrase in order_phrases
    ):
        return None

    # --------------------------------------------------------
    # AVOID AVAILABILITY QUESTIONS
    # --------------------------------------------------------

    question_phrases = (
        "do you have",
        "what do you have",
        "what is on",
        "what's on",
        "is there",
        "are there",
        "do you serve",
        "what can i get",
    )

    if any(
        phrase in text
        for phrase in question_phrases
    ):
        return None

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    def normalize(value):
        value = str(
            value or ""
        ).lower()

        value = re.sub(
            r"[^\w\s]",
            " ",
            value,
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value.strip()

    normalized_text = normalize(text)

    # --------------------------------------------------------
    # COMMON MENU ALIASES
    # --------------------------------------------------------

    aliases = {
        "shawarma": (
            "shawarma",
            "lebanese shawarma",
            "shawarma libanais",
        ),

        "lemonade": (
            "lemonade",
            "limonade",
            "limonades",
        ),

        "club sandwich": (
            "club sandwich",
            "sandwich",
            "club sandwich varié",
        ),

        "french fries": (
            "french fries",
            "fries",
            "frites",
            "frites croquantes",
        ),

        "margherita pizza": (
            "margherita pizza",
            "margherita pizzas",
            "pizza margherita",
            "pizzas margherita",
            "pizza margherita classique",
            "pizzas margherita classiques",
),

        "chicken cheese pizza": (
            "chicken cheese pizza",
            "loaded chicken pizza",
            "pizza poulet fromage",
            "pizza poulet et fromage",
            "pizza poulet & fromage garnie",
        ),

        "caramel sundae": (
            "caramel sundae",
            "sundae caramel",
            "sundae caramel croquant",
        ),

        "chocolate ice cream": (
            "chocolate ice cream",
            "chocolate scoop",
            "chocolate cone",
            "chocolat",
            "chocolat surchargé",
            "corne de chocolat",
        ),

        "vanilla cornetto": (
            "vanilla cornetto",
            "cornetto",
            "cornetto vanille",
            "cornetto classique à la vanille",
        ),

        "fruit ice cream": (
            "fruit ice cream",
            "glace aux fruits",
            "glace aux fruits mélangés",
            "tasse de glace aux fruits mélangés",
        ),

        "strawberry gelato": (
            "strawberry gelato",
            "gelato aux fraises",
        ),
    }

    # --------------------------------------------------------
    # BUILD MENU MATCHES
    # --------------------------------------------------------

    candidates = []

    for menu_item in menu_items:

        menu_name = str(
            menu_item.name or ""
        ).strip()

        if not menu_name:
            continue

        normalized_name = normalize(
            menu_name
        )

        candidates.append(
            {
                "item": menu_item,
                "name": menu_name,
                "normalized": normalized_name,
            }
        )

    # --------------------------------------------------------
    # FIND MATCHES
    # --------------------------------------------------------

    matched = {}

    for candidate in candidates:

        menu_item = candidate["item"]
        menu_name = candidate["name"]
        normalized_name = candidate["normalized"]

        terms = {
            normalized_name
        }

        # Add safe aliases based on the actual menu item.
        for alias, variants in aliases.items():

            if (
                alias in normalized_name
                or normalized_name in variants
            ):
                terms.update(
                    variants
                )

        matched_terms = []

        for term in terms:

            normalized_term = normalize(
                term
            )

            if not normalized_term:
                continue

            if (
                normalized_term
                in normalized_text
            ):
                matched_terms.append(
                    normalized_term
                )

        if not matched_terms:
            continue

        # ----------------------------------------------------
        # AMBIGUOUS GENERIC TERMS
        # ----------------------------------------------------

        generic_terms = {
            "pizza",
            "sandwich",
            "ice cream",
            "dessert",
        }

        if any(
            term in generic_terms
            for term in matched_terms
        ):
            continue

        matched[
            menu_item.id
        ] = {
            "item": menu_item,
            "terms": matched_terms,
        }

    # --------------------------------------------------------
    # NO LOCAL MATCH
    # --------------------------------------------------------

    if not matched:
        return None

    # --------------------------------------------------------
    # QUANTITY
    # --------------------------------------------------------

    quantity_words = {
        "one": 1,
        "a": 1,
        "an": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,

        # French
        "un": 1,
        "une": 1,
        "deux": 2,
        "trois": 3,
        "quatre": 4,
        "cinq": 5,
    }

    # --------------------------------------------------------
    # BUILD RESULT
    # --------------------------------------------------------

    extracted_items = []

    for match in matched.values():

        menu_item = match["item"]

        quantity = 1

        # Look for a number immediately before the
        # matched menu term.
        for term in match["terms"]:

            # Numeric quantity.
            number_match = re.search(
                rf"(\d+)\s+{re.escape(term)}",
                normalized_text,
            )

            if number_match:

                quantity = max(
                    1,
                    int(
                        number_match.group(1)
                    ),
                )

                break

            # Word quantity.
            for word, value in quantity_words.items():

                word_pattern = (
                    rf"{re.escape(word)}\s+"
                    rf"{re.escape(term)}"
                )

                if re.search(
                    word_pattern,
                    normalized_text,
                ):
                    quantity = value
                    break

        price = float(
            menu_item.price or 0
        )

        subtotal = (
            price * quantity
        )

        extracted_items.append(
            {
                "name": menu_item.name,
                "quantity": quantity,
                "price": price,
                "subtotal": float(
                    subtotal
                ),
            }
        )

    if not extracted_items:
        return None

    total = sum(
        item["subtotal"]
        for item in extracted_items
    )

    logger.info(
        "Fast multi-item order extraction used for: %s",
        customer_message,
    )

    return {
        "items": extracted_items,
        "total": float(total),
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