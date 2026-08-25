import json
import logging
import re
import time
from functools import lru_cache
from database.db import db
from contextvars import ContextVar

from services.ai.openai_provider import OpenAIProvider
from services.ai.menu_intelligence import (
    search_menu,
    recommend_menu,
)

from services.ai.agent_tools import (
    tool_search_menu,
    get_customer,
    get_active_order,
    get_pending_order,
    create_order_preview,
    confirm_pending_order,
    discard_pending_order,
    modify_active_order,
    cancel_active_order,
)

CUSTOMER_LANGUAGE = ContextVar(
    "customer_language",
    default="English",
)

logger = logging.getLogger(__name__)


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(value):
    """
    Convert a value to clean text.
    """

    if value is None:
        return ""

    return str(value).strip()


def normalize_text(value):
    """
    Normalize text for simple intent and confirmation checks.
    """

    text = clean_text(value).lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()

# ============================================================
# EMOJI HELPER
# ============================================================

def add_natural_emoji(message, customer_message=""):
    """
    Add an occasional natural emoji to customer-facing
    responses.
    """

    message = clean_text(message)

    if not message:
        return message

    if any(
        ord(char) > 0x1F000
        for char in message
    ):
        return message

    customer_text = normalize_text(
        customer_message
    )

    if any(
        word in customer_text
        for word in (
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        )
    ):
        emoji = "👋"

    elif any(
        word in customer_text
        for word in (
            "order",
            "food",
            "eat",
            "pizza",
            "shawarma",
            "burger",
            "drink",
        )
    ):
        emoji = "🍽️"

    elif any(
        word in customer_text
        for word in (
            "menu",
            "available",
            "have",
            "recommend",
            "recommendation",
            "suggest",
        )
    ):
        emoji = "😋"

    elif any(
        word in customer_text
        for word in (
            "thank",
            "thanks",
            "appreciate",
        )
    ):
        emoji = "😊"

    elif any(
        word in customer_text
        for word in (
            "confirm",
            "yes",
            "okay",
            "correct",
            "place",
        )
    ):
        emoji = "✅"

    else:
        return message

    selector = (
        len(message)
        + len(customer_text)
    ) % 4

    if selector not in (0, 1):
        return message

    return f"{message} {emoji}"


def customer_response(
    message,
    customer_message="",
):
    """
    Standardize a customer-facing response and optionally
    add
    a natural emoji.

    The current customer language is taken from the
    request-scoped CUSTOMER_LANGUAGE context.
    """

    message = clean_text(
        message
    )

    if not message:
        return ""

    language = CUSTOMER_LANGUAGE.get()

    fixed_translations = {
        "French": {
            "Sure. What would you like to order?":
                "Bien sûr. Que souhaitez-vous commander ?",

            "Sure! Let me know how I can help with the restaurant.":
                "Bien sûr ! Dites-moi comment je peux vous aider avec le restaurant.",

            "I'm here to help. What would you like to know?":
                "Je suis là pour vous aider. Que souhaitez-vous savoir ?",

            "I'm the restaurant's AI assistant, so I can only help with the menu, orders, delivery, and other restaurant-related questions.":
                "Je suis l'assistant IA du restaurant. Je peux vous aider avec le menu, les commandes, la livraison et les autres questions liées au restaurant.",

            "I couldn't retrieve that restaurant information right now.":
                "Je n'arrive pas à récupérer ces informations du restaurant pour le moment.",

            "I couldn't create the order preview right now. Please try again.":
                "Je n'arrive pas à créer l'aperçu de la commande pour le moment. Veuillez réessayer.",

            "I couldn't modify the order right now.":
                "Je n'arrive pas à modifier la commande pour le moment.",

            "Your order has been updated.":
                "Votre commande a été mise à jour.",

            "I couldn't cancel the order right now. Please try again.":
                "Je n'arrive pas à annuler la commande pour le moment. Veuillez réessayer.",

            "I couldn't confirm the order right now. Please try again.":
                "Je n'arrive pas à confirmer la commande pour le moment. Veuillez réessayer.",

            "I couldn't find any menu items right now. Please try again later.":
                "Je ne trouve aucun article du menu pour le moment. Veuillez réessayer plus tard.",

            "I'm unable to access the menu right now. Please try again.":
                "Je n'arrive pas à accéder au menu pour le moment. Veuillez réessayer.",

            "Here are some good options:":
                "Voici quelques bonnes options :",

            "I couldn't get recommendations right now. Please try the menu instead.":
                "Je n'arrive pas à obtenir de recommandations pour le moment. Veuillez consulter le menu à la place.",
        }
    }

    translated = (
        fixed_translations
        .get(language, {})
        .get(message)
    )

    if translated:
        message = translated

    return add_natural_emoji(
        message,
        customer_message,
    )


def pending_order_confirmation_is_valid(
    history,
    pending,
):
    """
    Kept for compatibility with existing callers.
    Pending-order confirmation is now primarily controlled
    by the actual pending-order state.
    """

    if not pending:
        return False

    return True


CONFIRMATION_PHRASES = {
    "yes",
    "yes please",
    "yes pls",
    "confirm",
    "confirmed",
    "confirm it",
    "go ahead",
    "go for it",
    "place it",
    "place the order",
    "do it",
    "proceed",
    "proceed with the order",
    "proceed with my order",
    "i would like to proceed",
    "i would like to proceed with the order",
    "i want to proceed",
    "i want to proceed with the order",
    "thats correct",
    "that is correct",
    "correct",
    "okay",
    "ok",
    "sure",
    "sure thing",
    "sounds good",
    "looks good",
    "thats good",
    "that is good",

    # French
    "oui",
    "oui merci",
    "oui s'il vous plaît",
    "oui s il vous plait",
    "je confirme",
    "confirmer",
    "confirme",
    "d'accord",
    "d accord",
}

REJECTION_PHRASES = {
    "no",
    "no thanks",
    "no thank you",
    "cancel",
    "never mind",
    "nevermind",
    "forget it",
    "dont place it",
    "do not place it",
    "cancel it",

    # French
    "non",
    "non merci",
    "annuler",
    "annule",
    "je ne veux pas",
    "je ne veux plus",
    "laissez tomber",
    "laisse tomber",
}

def is_confirmation(message):
    text = normalize_text(
        message
    )

    return text in CONFIRMATION_PHRASES


def is_rejection(message):
    text = normalize_text(
        message
    )

    return text in REJECTION_PHRASES

# ============================================================
# PENDING ORDER HELPERS
# ============================================================

def _extract_pending_items(pending_data):
    """
    Extract pending items regardless of whether the result
    uses items, order.items, or preview.items.
    """

    if not isinstance(
        pending_data,
        dict,
    ):
        return []

    if "items" in pending_data:
        return (
            pending_data.get("items")
            or []
        )

    order = pending_data.get(
        "order"
    )

    if isinstance(
        order,
        dict,
    ):
        return (
            order.get("items")
            or []
        )

    preview = pending_data.get(
        "preview"
    )

    if isinstance(
        preview,
        dict,
    ):
        return (
            preview.get("items")
            or []
        )

    return []


# ============================================================
# TOOL EXECUTION
# ============================================================

def execute_tool(
    business_id,
    phone,
    message,
    tool,
    arguments=None,
):
    arguments = arguments or {}

    if not isinstance(
        arguments,
        dict,
    ):
        arguments = {}

    if tool == "search_menu":

        query = (
            arguments.get("query")
            or arguments.get("message")
            or message
        )

        return tool_search_menu(
            business_id,
            query,
        )

    if tool == "get_customer":

        return get_customer(
            business_id,
            phone,
        )

    if tool == "get_active_order":

        return get_active_order(
            business_id,
            phone,
        )

    if tool == "get_pending_order":

        return get_pending_order(
            business_id,
            phone,
        )

    if tool == "create_order_preview":

        order_message = (
            arguments.get("message")
            or arguments.get("query")
            or message
        )

        return create_order_preview(
            business_id,
            phone,
            order_message,
        )

    if tool == "confirm_order":

        return confirm_pending_order(
            business_id,
            phone,
        )

    if tool == "discard_order_preview":

        return discard_pending_order(
            business_id,
            phone,
        )

    if tool == "modify_order":

        modification_message = (
            arguments.get("message")
            or arguments.get("query")
            or message
        )

        return modify_active_order(
            business_id,
            phone,
            modification_message,
        )

    if tool == "cancel_order":

        return cancel_active_order(
            business_id,
            phone,
        )

    return {
        "success": False,
        "message": "That tool is not available.",
    }


# ============================================================
# CONVERSATION MEMORY
# ============================================================

def format_history(history):

    if not history:
        return "No previous conversation."

    parts = []

    for item in history:

        if not isinstance(
            item,
            dict,
        ):
            continue

        customer_message = clean_text(
            item.get("message")
        )

        assistant_response = clean_text(
            item.get("response")
        )

        if not customer_message:
            continue

        parts.append(
            "Customer:\n"
            f"{customer_message}\n\n"
            "Assistant:\n"
            f"{assistant_response}"
        )

    if not parts:
        return "No previous conversation."

    return "\n\n--------------------\n\n".join(parts)


# ============================================================
# RESTAURANT INFORMATION
# ============================================================

def get_restaurant_context(
    business_id,
):
    """
    Load only the restaurant and menu fields required by
    build_agent_prompt().
    """

    try:

        from models.business import Business
        from models.menu import Menu

        # ----------------------------------------------------
        # BUSINESS
        # ----------------------------------------------------

        business_row = (
            Business.query
            .with_entities(
                Business.name,
                Business.business_type,
                Business.address,
                Business.phone,
                Business.opening_hours,
                Business.delivery_policy,
            )
            .filter(
                Business.id == business_id
            )
            .first()
        )

        if not business_row:
            return {
                "business": {},
                "menu": [],
            }

        business_data = {
            "name": clean_text(
                business_row.name
            ),
            "business_type": clean_text(
                business_row.business_type
            ),
            "address": clean_text(
                business_row.address
            ),
            "phone": clean_text(
                business_row.phone
            ),
            "opening_hours": clean_text(
                business_row.opening_hours
            ),
            "delivery_policy": clean_text(
                business_row.delivery_policy
            ),
        }

        # ----------------------------------------------------
        # AVAILABLE MENU
        # ----------------------------------------------------

        menu_rows = (
            Menu.query
            .with_entities(
                Menu.name,
                Menu.description,
                Menu.category,
                Menu.price,
            )
            .filter(
                Menu.business_id == business_id,
                Menu.available.is_(True),
            )
            .order_by(
                Menu.id.asc()
            )
            .all()
        )

        menu = [
            {
                "name": clean_text(
                    row.name
                ),
                "description": clean_text(
                    row.description
                ),
                "category": clean_text(
                    row.category
                ),
                "price": float(
                    row.price or 0
                ),
            }
            for row in menu_rows
        ]

        return {
            "business": business_data,
            "menu": menu,
        }

    except Exception:

        logger.exception(
            "Could not load restaurant context."
        )

        return {
            "business": {},
            "menu": [],
        }


# ============================================================
# RESTAURANT INFORMATION TOOL
# ============================================================

def get_restaurant_info(
    business_id,
    message,
):
    """
    Retrieve restaurant information using a minimal database
    query and a single normalized customer message.
    """

    try:

        from models.business import Business
        from models.menu import Menu

        # ----------------------------------------------------
        # LOAD ONLY REQUIRED FIELDS
        # ----------------------------------------------------

        business = (
            Business.query
            .with_entities(
                Business.name,
                Business.business_type,
                Business.address,
                Business.phone,
                Business.opening_hours,
                Business.delivery_policy,
            )
            .filter(
                Business.id == business_id
            )
            .first()
        )

        if not business:
            return {
                "success": False,
                "message": (
                    "Restaurant information is unavailable."
                ),
            }

        text = normalize_text(
            message
        )

        # ----------------------------------------------------
        # OPENING HOURS
        # ----------------------------------------------------

        if any(
            phrase in text
            for phrase in (
                "opening hours",
                "opening time",
                "when do you open",
                "when are you open",
                "what time do you open",
                "what time do you close",
                "when do you close",
                "closing time",
                "are you open",
                "are you closed",
                "open today",
            )
        ):

            hours = clean_text(
                business.opening_hours
            )

            if not hours:
                return {
                    "success": False,
                    "info_type": "opening_hours",
                    "message": (
                        "Opening hours have not been "
                        "provided by the restaurant yet."
                    ),
                }

            return {
                "success": True,
                "info_type": "opening_hours",
                "opening_hours": hours,
            }

        # ----------------------------------------------------
        # DELIVERY
        # ----------------------------------------------------

        if any(
            phrase in text
            for phrase in (
                "do you deliver",
                "delivery information",
                "information about delivery",
                "info about delivery",
                "delivery options",
                "delivery available",
                "do you offer delivery",
            )
        ):

            delivery = clean_text(
                business.delivery_policy
            )

            if not delivery:
                return {
                    "success": False,
                    "info_type": "delivery",
                    "message": (
                        "Delivery information has not been "
                        "provided by the restaurant yet."
                    ),
                }

            return {
                "success": True,
                "info_type": "delivery",
                "delivery_policy": delivery,
            }

        # ----------------------------------------------------
        # ADDRESS
        # ----------------------------------------------------

        if any(
            phrase in text
            for phrase in (
                "where are you located",
                "where are you",
                "what is your address",
                "your address",
                "restaurant address",
                "location",
                "where is the restaurant",
            )
        ):

            address = clean_text(
                business.address
            )

            if not address:
                return {
                    "success": False,
                    "info_type": "address",
                    "message": (
                        "The restaurant address has not "
                        "been provided yet."
                    ),
                }

            return {
                "success": True,
                "info_type": "address",
                "address": address,
            }

        # ----------------------------------------------------
        # PHONE
        # ----------------------------------------------------

        if any(
            phrase in text
            for phrase in (
                "phone number",
                "your phone number",
                "contact number",
                "contact",
                "call you",
            )
        ):

            phone = clean_text(
                business.phone
            )

            if not phone:
                return {
                    "success": False,
                    "info_type": "phone",
                    "message": (
                        "The restaurant phone number "
                        "has not been provided yet."
                    ),
                }

            return {
                "success": True,
                "info_type": "phone",
                "phone": phone,
            }

        # ----------------------------------------------------
        # ABOUT THE RESTAURANT
        # ----------------------------------------------------

        if any(
            phrase in text
            for phrase in (
                "tell me about your restaurant",
                "tell me about this restaurant",
                "tell me about your place",
                "tell me about this place",
                "what can you tell me about your restaurant",
                "what can you tell me about this restaurant",
                "what is your restaurant like",
                "what is this restaurant like",
                "about your restaurant",
                "about this restaurant",
            )
        ):

            menu_items = (
                Menu.query
                .with_entities(
                    Menu.name,
                    Menu.price,
                    Menu.category,
                )
                .filter(
                    Menu.business_id == business_id,
                    Menu.available.is_(True),
                )
                .order_by(
                    Menu.category.asc(),
                    Menu.name.asc(),
                )
                .limit(20)
                .all()
            )

            name = clean_text(
                business.name
            )

            business_type = clean_text(
                business.business_type
            )

            category_counts = {}

            for item in menu_items:

                category = clean_text(
                    item.category
                ) or "Menu"

                category_counts[
                    category
                ] = (
                    category_counts.get(
                        category,
                        0
                    ) + 1
                )

            categories = list(
                category_counts.keys()
            )

            return {
                "success": True,
                "info_type": "about",
                "name": name,
                "business_type": business_type,
                "menu_count": len(menu_items),
                "menu_categories": categories,
                "menu_items": [
                    {
                        "name": clean_text(
                            item.name
                        ),
                        "price": float(
                            item.price or 0
                        ),
                        "category": (
                            clean_text(
                                item.category
                            )
                            or "Menu"
                        ),
                    }
                    for item in menu_items
                ],
            }

        # ----------------------------------------------------
        # GENERAL RESTAURANT INFORMATION
        # ----------------------------------------------------

        return {
            "success": True,
            "info_type": "general",
            "name": clean_text(
                business.name
            ),
            "business_type": clean_text(
                business.business_type
            ),
            "address": clean_text(
                business.address
            ),
            "phone": clean_text(
                business.phone
            ),
            "opening_hours": clean_text(
                business.opening_hours
            ),
            "delivery_policy": clean_text(
                business.delivery_policy
            ),
        }

    except Exception:

        logger.exception(
            "Could not retrieve restaurant information."
        )

        return {
            "success": False,
            "message": (
                "Restaurant information is "
                "currently unavailable."
            ),
        }


# ============================================================
# RESTAURANT PROMPT
# ============================================================

def build_agent_prompt(
    business_id,
    phone,
    message,
    language,
    history,
    pending,
):
    """
    Build the natural-response prompt efficiently.

    The prompt is kept compact to reduce token usage and
    model latency while preserving the information needed
    for restaurant conversations.
    """

    restaurant = get_restaurant_context(
        business_id
    )

    business_data = restaurant.get(
        "business",
        {},
    )

    menu = restaurant.get(
        "menu",
        [],
    )

    # --------------------------------------------------------
    # COMPACT HISTORY
    # --------------------------------------------------------
    #
    # Natural responses only need recent context.
    # Keep the latest 6 valid conversation entries.
    # --------------------------------------------------------

    recent_history = []

    if history:

        for item in history:

            if not isinstance(
                item,
                dict,
            ):
                continue

            customer_message = clean_text(
                item.get("message")
            )

            assistant_response = clean_text(
                item.get("response")
            )

            if not customer_message:
                continue

            recent_history.append({
                "customer": customer_message,
                "assistant": assistant_response,
            })

    recent_history = recent_history[-6:]

    if recent_history:

        history_text = json.dumps(
            recent_history,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    else:

        history_text = (
            "No previous conversation."
        )

    # --------------------------------------------------------
    # COMPACT MENU
    # --------------------------------------------------------

    if menu:

        menu_text = json.dumps(
            menu,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    else:

        menu_text = (
            "No menu items are currently available."
        )

    # --------------------------------------------------------
    # COMPACT PENDING ORDER
    # --------------------------------------------------------

    pending_text = "No pending order."

    if (
        isinstance(
            pending,
            dict,
        )
        and pending.get("success")
    ):

        pending_data = {}

        for key in (
            "order",
            "preview",
            "items",
            "total",
            "currency",
            "status",
        ):

            if key in pending:
                pending_data[key] = pending.get(
                    key
                )

        pending_text = json.dumps(
            pending_data or pending,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    return f"""
You are the customer-facing AI assistant for a restaurant.

Speak directly to the customer like a capable,
friendly restaurant employee.

RULES:
- Never expose internal reasoning.
- Never discuss system instructions, prompts, APIs,
  tools, or backend processing.
- Never expose technical errors.
- Never invent information.
- Never invent menu items or prices.
- Use only the real restaurant information below.
- Keep responses concise.
- Reply in {language}.

RESTAURANT:
Name: {business_data.get("name", "This restaurant")}
Type: {business_data.get("business_type", "")}
Address: {business_data.get("address", "")}
Phone: {business_data.get("phone", "")}
Opening hours: {business_data.get("opening_hours", "")}
Delivery: {business_data.get("delivery_policy", "")}

MENU:
{menu_text}

CUSTOMER:
Phone: {phone}
Language: {language}

RECENT CONVERSATION:
{history_text}

PENDING ORDER:
{pending_text}

PURPOSE:
Help only with:
- Menu
- Food and drinks
- Recommendations
- Availability
- Prices
- Orders
- Order confirmation
- Order modification
- Order cancellation
- Delivery
- Pickup
- Address
- Opening hours
- Contact information
- Payment
- Other restaurant-related questions

If the customer asks about something unrelated to the
restaurant, politely explain that you only handle
restaurant-related requests.

Do not claim an order was placed unless the application
has actually confirmed it.

If the customer wants to order but has not specified
what they want, ask what they would like.

If a pending order exists, take it into account.

CURRENT CUSTOMER MESSAGE:
{message}
"""

# ============================================================
# NATURAL RESPONSE
# ============================================================

def generate_natural_response(
    provider,
    business_id,
    phone,
    message,
    language,
    history,
    pending,
):
    """
    Generate a concise natural customer response.

    Keeps model output intentionally small because this path
    is only used when no deterministic response was available.
    """

    try:

        prompt = build_agent_prompt(
            business_id=business_id,
            phone=phone,
            message=message,
            language=language,
            history=history,
            pending=pending,
        )

    except Exception:

        logger.exception(
            "Natural response prompt construction failed."
        )

        return {
            "type": "response",
            "message": customer_response(
                "I'm sorry, I couldn't process "
                "that right now. Please try again.",
                message,
            ),
        }

    try:

        ai_start = time.perf_counter()

        response = provider.generate(
            prompt,
            temperature=0.5,
            max_tokens=160,
        )

        logger.warning(
            "[PERF] generate_natural_response: %.2fs",
            time.perf_counter() - ai_start,
        )

        response = clean_text(
            response
        )

        if response:

            return {
                "type": "response",
                "message": customer_response(
                    response,
                    message,
                ),
            }

    except Exception:

        logger.exception(
            "Natural AI response failed."
        )

    return {
        "type": "response",
        "message": customer_response(
            "I'm sorry, I couldn't process "
            "that right now. Please try again.",
            message,
        ),
    }


# ============================================================
# TOOL RESPONSE
# ============================================================

@lru_cache(maxsize=256)
def _translate_order_text_cached(
    order_text,
    language_name,
):
    """
    Translate identical order-response text only once.
    """

    provider = OpenAIProvider()

    prompt = f"""
Translate the following restaurant message into {language_name}.

STRICT RULES:
- Translate only the existing text.
- Do NOT add a heading.
- Do NOT add a title.
- Do NOT add a confirmation section.
- Do NOT add a greeting.
- Do NOT add extra sentences.
- Do NOT add or remove lines.
- Return EXACTLY the same number of lines as the input.
- Keep the same line order.
- Keep quantities exactly unchanged.
- Keep prices exactly unchanged.
- Keep "FCFA" exactly unchanged.
- Keep order numbers exactly unchanged.
- Keep URLs exactly unchanged.
- Keep emojis exactly unchanged.
- Keep WhatsApp Markdown such as *bold* exactly intact.
- Do NOT convert FCFA into another currency.
- Return ONLY the translated text.

INPUT:
{order_text}
"""

    try:

        translated = provider.generate(
            prompt,
            temperature=0,
            max_tokens=120,
        )

        translated = clean_text(
            translated
        )

        if translated:
            return translated

    except Exception:

        logger.exception(
            "Failed to translate order preview to %s",
            language_name,
        )

    return ""


def translate_order_preview_for_customer(
    lines,
    language,
    provider=None,
):
    """
    Translate order-response lines only when necessary.
    """

    if not language or not lines:
        return lines

    language_name = str(
        language
    ).strip()

    if (
        not language_name
        or language_name.lower()
        in (
            "english",
            "en",
            "en-us",
            "en-gb",
        )
    ):
        return lines

    order_text = "\n".join(
        lines
    )

    # Reuse the existing provider when available.
    if provider is not None:

        prompt = f"""
Translate the following restaurant message into {language_name}.

STRICT RULES:
- Translate only the existing text.
- Do NOT add a heading.
- Do NOT add a title.
- Do NOT add a confirmation section.
- Do NOT add a greeting.
- Do NOT add extra sentences.
- Do NOT add or remove lines.
- Return EXACTLY the same number of lines as the input.
- Keep the same line order.
- Keep quantities exactly unchanged.
- Keep prices exactly unchanged.
- Keep "FCFA" exactly unchanged.
- Keep order numbers exactly unchanged.
- Keep URLs exactly unchanged.
- Keep emojis exactly unchanged.
- Keep WhatsApp Markdown such as *bold* exactly intact.
- Do NOT convert FCFA into another currency.
- Return ONLY the translated text.

INPUT:
{order_text}
"""

        try:

            translated = provider.generate(
                prompt,
                temperature=0,
                max_tokens=300,
            )

            translated = clean_text(
                translated
            )

            if translated:
                return translated.splitlines()

        except Exception:

            logger.exception(
                "Failed to translate order preview to %s",
                language_name,
            )

        return lines

    # Cached fallback.
    translated = _translate_order_text_cached(
        order_text,
        language_name,
    )

    if translated:
        return translated.splitlines()

    return lines

def build_tool_response(
    provider,
    business_id,
    phone,
    message,
    language,
    history,
    tool_result,
):
    """
    Convert a tool result into a customer-facing response.

    Fast paths are used for known restaurant operations so that
    common operations do not require another LLM call.
    """

    # --------------------------------------------------------
    # NORMALIZE TOOL RESULT
    # --------------------------------------------------------

    if not isinstance(
        tool_result,
        dict,
    ):
        tool_result = {}

    success = bool(
        tool_result.get("success")
    )

    unmatched = (
        tool_result.get("unmatched")
        or []
    )

    matched_items = (
        tool_result.get("matched_items")
        or []
    )

    preview = tool_result.get(
        "preview"
    )

    order = tool_result.get(
        "order"
    )

    payment = tool_result.get(
        "payment"
    )

    action = tool_result.get(
        "action"
    )

    order_id = tool_result.get(
        "order_id"
    )

    total = tool_result.get(
        "total"
    )

    tool_message = tool_result.get(
        "message"
    )

    # --------------------------------------------------------
    # SMALL RESPONSE HELPER
    # --------------------------------------------------------

    def make_response(text):
        return {
            "type": "response",
            "message": customer_response(
                text,
                message,
            ),
        }

    # --------------------------------------------------------
    # RESTAURANT INFORMATION FAST PATH
    # --------------------------------------------------------

    if (
        success
        and tool_result.get("info_type")
    ):

        info_type = str(
            tool_result.get("info_type")
        ).strip()

        if info_type == "opening_hours":

            hours = clean_text(
                tool_result.get("opening_hours")
            )

            if hours:
                return make_response(
                    f"Our opening hours are: {hours}"
                )

        elif info_type == "delivery":

            delivery = clean_text(
                tool_result.get("delivery_policy")
            )

            if delivery:
                return make_response(
                    delivery
                )

        elif info_type == "address":

            address = clean_text(
                tool_result.get("address")
            )

            if address:
                return make_response(
                    f"Our address is: {address}"
                )

        elif info_type == "phone":

            restaurant_phone = clean_text(
                tool_result.get("phone")
            )

            if restaurant_phone:
                return make_response(
                    f"You can reach us at: "
                    f"{restaurant_phone}"
                )

        elif info_type == "about":

            name = clean_text(
                tool_result.get("name")
            )

            business_type = clean_text(
                tool_result.get("business_type")
            )

            categories = [
                clean_text(category)
                for category in (
                    tool_result.get(
                        "menu_categories",
                        []
                    )
                )
                if clean_text(category)
            ]

            menu_items = tool_result.get(
                "menu_items",
                []
            )

            lines = []

            if name:
                lines.append(
                    f"Welcome to {name}!"
                )

            if business_type:
                lines.append(
                    f"{name} is a {business_type.lower()} "
                    "bringing together satisfying meals, "
                    "refreshing drinks, and sweet treats "
                    "for different tastes and occasions."
                )

            else:
                lines.append(
                    f"{name} offers a variety of "
                    "meals, drinks, and desserts "
                    "for different tastes and occasions."
                )

            if categories:

                if len(categories) > 1:
                    category_text = (
                        ", ".join(
                            categories[:-1]
                        )
                        + f", and {categories[-1]}"
                    )
                else:
                    category_text = categories[0]

                lines.append(
                    "Our menu brings together "
                    f"{category_text}."
                )

            category_items = {}

            for item in menu_items:

                item_name = clean_text(
                    item.get("name")
                )

                category = clean_text(
                    item.get("category")
                ) or "Menu"

                if not item_name:
                    continue

                category_items.setdefault(
                    category,
                    []
                ).append(item_name)

            if category_items:

                highlight_lines = [
                    "Here are some of the highlights:"
                ]

                for category, items in (
                    category_items.items()
                ):

                    selected_items = items[:4]

                    highlight_lines.append(
                        f"• {category}: "
                        + ", ".join(
                            selected_items
                        )
                        + "."
                    )

                lines.append(
                    "\n".join(
                        highlight_lines
                    )
                )

            lines.append(
                "Whether you're looking for something "
                "quick and savory, a refreshing drink, "
                "or a dessert to finish things off, "
                "there's plenty to explore."
            )

            lines.append(
                "Tell me what you're craving and "
                "I'll help you choose something "
                "from the menu."
            )

            return make_response(
                "\n\n".join(lines)
            )

        elif info_type == "general":

            name = clean_text(
                tool_result.get("name")
            )

            business_type = clean_text(
                tool_result.get("business_type")
            )

            menu_count = tool_result.get(
                "menu_count",
                0
            )

            categories = [
                clean_text(category)
                for category in (
                    tool_result.get(
                        "menu_categories",
                        []
                    )
                )
                if clean_text(category)
            ]

            menu_items = tool_result.get(
                "menu_items",
                []
            )

            lines = []

            if name:
                lines.append(
                    f"Welcome to {name}!"
                )

            if business_type:
                lines.append(
                    f"We're a {business_type.lower()} "
                    "focused on giving customers a "
                    "variety of tasty options in one place."
                )

            if menu_count:
                lines.append(
                    f"Our menu currently features "
                    f"{menu_count} available items"
                    + (
                        " across "
                        + ", ".join(categories)
                        if categories
                        else ""
                    )
                    + "."
                )

            if categories:
                lines.append(
                    "You can choose from "
                    + ", ".join(categories)
                    + ", with options ranging from "
                    "savory meals and snacks to drinks "
                    "and desserts."
                )

            # Mention a few real menu highlights.
            highlights = []

            for item in menu_items[:6]:

                item_name = clean_text(
                    item.get("name")
                )

                if item_name:
                    highlights.append(
                        item_name
                    )

            if highlights:

                lines.append(
                    "Some of the menu highlights "
                    "include "
                    + ", ".join(
                        highlights[:-1]
                    )
                    + (
                        f", and {highlights[-1]}."
                        if len(highlights) > 1
                        else "."
                    )
                )

            lines.append(
                "Whether you're looking for "
                "something quick and filling, "
                "a refreshing drink, or a sweet "
                "dessert, there are several options "
                "to choose from."
            )

            lines.append(
                "Tell me what you're in the mood for "
                "and I can help you choose something "
                "from the menu."
            )

            return make_response(
                "\n\n".join(
                    lines
                )
            )

        elif info_type == "general":

            name = clean_text(
                tool_result.get("name")
            )

            address = clean_text(
                tool_result.get("address")
            )

            phone_number = clean_text(
                tool_result.get("phone")
            )

            opening_hours = clean_text(
                tool_result.get("opening_hours")
            )

            delivery_policy = clean_text(
                tool_result.get("delivery_policy")
            )

            lines = []

            if name:
                lines.append(
                    f"Restaurant: {name}"
                )

            if address:
                lines.append(
                    f"Address: {address}"
                )

            if phone_number:
                lines.append(
                    f"Phone: {phone_number}"
                )

            if opening_hours:
                lines.append(
                    f"Opening hours: {opening_hours}"
                )

            if delivery_policy:
                lines.append(
                    f"Delivery: {delivery_policy}"
                )

            if lines:
                return make_response(
                    "\n".join(lines)
                )

    # --------------------------------------------------------
    # UNMATCHED MENU ITEMS
    # --------------------------------------------------------

    if not success and unmatched:

        unmatched_text = ", ".join(
            str(item)
            for item in unmatched
        )

        if matched_items:

            matched_text = ", ".join(
                f"{item.get('quantity', 1)}x "
                f"{item.get('name', 'item')}"
                for item in matched_items
                if isinstance(item, dict)
            )

            response = (
                f"I couldn't find "
                f"{unmatched_text} on the menu. "
                f"I found {matched_text}. "
                f"Would you like to replace "
                f"{unmatched_text} with something else?"
            )

        else:

            response = (
                f"I couldn't find "
                f"{unmatched_text} on the menu. "
                f"Please choose another item."
            )

        return make_response(
            response
        )

    # --------------------------------------------------------
    # ORDER PREVIEW
    # --------------------------------------------------------

    if (
        success
        and isinstance(
            preview,
            dict,
        )
    ):

        preview_items = (
            preview.get("items")
            or []
        )

        preview_total = preview.get(
            "total",
            0,
        )

        preview_currency = (
            preview.get("currency")
            or "FCFA"
        )

        lines = [
            "Your order:",
            "",
        ]

        for item in preview_items:

            if not isinstance(
                item,
                dict,
            ):
                continue

            item_name = clean_text(
                item.get("name")
                or "Item"
            )

            quantity = item.get(
                "quantity",
                1,
            )

            lines.append(
                f"• *{item_name}* × {quantity}"
            )

        lines.append("")

        try:

            total_text = (
                f"{float(preview_total):,.0f} "
                f"{preview_currency}"
            )

        except (
            TypeError,
            ValueError,
        ):

            total_text = (
                f"{preview_total} "
                f"{preview_currency}"
            )

        lines.extend([
            f"Total: {total_text}",
            "",
            "Please confirm your order.",
        ])

        lines = translate_order_preview_for_customer(
            lines,
            language,
            provider=provider,
        )

        return make_response(
            "\n".join(lines)
        )

    # --------------------------------------------------------
    # CONFIRMED ORDER + PAYMENT
    # --------------------------------------------------------

    if (
        success
        and isinstance(
            order,
            dict,
        )
        and isinstance(
            payment,
            dict,
        )
    ):

        confirmed_order_id = order.get(
            "id"
        )

        confirmed_total = order.get(
            "total",
            0,
        )

        currency = (
            order.get("currency")
            or "FCFA"
        )

        checkout_url = payment.get(
            "checkout_url"
        )

        try:

            confirmed_total_text = (
                f"{float(confirmed_total):,.0f} "
                f"{currency}"
            )

        except (
            TypeError,
            ValueError,
        ):

            confirmed_total_text = (
                f"{confirmed_total} "
                f"{currency}"
            )

        lines = [
            (
                f"Order #{confirmed_order_id} has been "
                "created successfully."
            ),
            "",
            f"Total: {confirmed_total_text}",
            "",
            "Payment is required to complete your order.",
        ]

        if checkout_url:

            lines.extend([
                "",
                f"Pay here: {checkout_url}",
            ])

        lines = translate_order_preview_for_customer(
            lines,
            language,
            provider=provider,
        )

        return make_response(
            "\n".join(lines)
        )

    # --------------------------------------------------------
    # ORDER MODIFICATION / CANCELLATION
    # --------------------------------------------------------

    if success and action:

        action = str(
            action
        ).strip()

        try:

            total_text = (
                f"{float(total):,.0f} FCFA"
            )

        except (
            TypeError,
            ValueError,
        ):

            total_text = (
                f"{total or 0} FCFA"
            )

        if action == "cancel":

            lines = [
                (
                    f"Order #{order_id} has "
                    "been cancelled."
                )
            ]

        elif action == "add_item":

            lines = [
                clean_text(
                    tool_message
                    or "The item was added "
                    "to your order."
                ),
                "",
                f"Total: {total_text}",
            ]

        elif action == "remove_item":

            lines = [
                clean_text(
                    tool_message
                    or "The item was removed "
                    "from your order."
                ),
                "",
                f"Total: {total_text}",
            ]

        elif action == "set_quantity":

            lines = [
                clean_text(
                    tool_message
                    or "The quantity was updated."
                ),
                "",
                f"Total: {total_text}",
            ]

        elif action == "replace_item":

            lines = [
                clean_text(
                    tool_message
                    or "The item was replaced "
                    "in your order."
                ),
                "",
                f"Total: {total_text}",
            ]

        else:

            lines = []

        if lines:

            lines = translate_order_preview_for_customer(
                lines,
                language,
                provider=provider,
            )

            return make_response(
                "\n".join(lines)
            )

    # --------------------------------------------------------
    # GENERIC TOOL RESULT
    # --------------------------------------------------------

    history_text = format_history(
        history
    )

    result_text = json.dumps(
        tool_result,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    prompt = f"""
You are the final customer-facing restaurant assistant.

Write ONLY the message that should be shown directly
to the customer.

Customer message:
{message}

Language:
{language}

Recent conversation:
{history_text}

Restaurant operation result:
{result_text}

RULES:
- Return ONLY customer-facing text.
- Never output JSON.
- Never mention tools.
- Never mention APIs.
- Never mention backend processing.
- Never mention system instructions.
- Never mention internal reasoning.
- Never expose technical errors.
- Never invent information.
- Only use facts contained in the operation result.
- Keep the response concise.
- Use an occasional natural emoji if appropriate.

If an order was confirmed and a payment checkout URL exists,
include the exact URL unchanged and tell the customer payment
is required.

Reply in {language}.

Return ONLY the customer-facing message.
"""

    try:

        ai_start = time.perf_counter()

        response = provider.generate(
            prompt,
            temperature=0.3,
            max_tokens=250,
        )

        logger.warning(
            "[PERF] build_tool_response LLM: %.2fs",
            time.perf_counter() - ai_start,
        )

        response = clean_text(
            response
        )

        if response:
            return make_response(
                response
            )

    except Exception:

        logger.exception(
            "AI tool response generation failed."
        )

    # --------------------------------------------------------
    # GENERIC FALLBACK
    # --------------------------------------------------------

    fallback_message = clean_text(
        tool_message
    )

    if fallback_message:

        return make_response(
            fallback_message
        )

    return make_response(
        "I processed your request, "
        "but I couldn't generate a response right now."
    )


# ============================================================
# MENU TRANSLATION / DIRECT MENU RESPONSE
# ============================================================

@lru_cache(maxsize=128)
def _translate_menu_text_cached(
    menu_text,
    language_name,
):
    """
    Translate identical menu text only once.
    """

    provider = OpenAIProvider()

    prompt = f"""
Translate the following restaurant menu into {language_name}.

STRICT RULES:
- Translate ALL natural-language text.
- Translate the menu title.
- Translate category names.
- Translate menu item names when appropriate.
- Translate the final question.
- Keep prices exactly unchanged.
- Keep "FCFA" exactly unchanged.
- Keep emojis exactly unchanged.
- Keep WhatsApp Markdown such as *bold* exactly intact.
- Do NOT add explanations.
- Do NOT remove menu items.
- Return ONLY the translated menu.

MENU:
{menu_text}
"""

    try:

        translated = provider.generate(
            prompt,
            temperature=0,
            max_tokens=300,
        )

        translated = clean_text(
            translated
        )

        if translated:
            return translated

    except Exception:

        logger.exception(
            "Failed to translate menu to %s",
            language_name,
        )

    return ""


def translate_menu_for_customer(
    lines,
    language,
    provider=None,
):
    """
    Translate menu lines only when necessary.

    Reuses the existing provider when available and falls
    back to cached translation for repeated identical menus.
    """

    if not language or not lines:
        return lines

    language_name = str(
        language
    ).strip()

    if (
        not language_name
        or language_name.lower()
        in (
            "english",
            "en",
            "en-us",
            "en-gb",
        )
    ):
        return lines

    menu_text = "\n".join(
        lines
    )

    # --------------------------------------------------------
    # REUSE EXISTING PROVIDER
    # --------------------------------------------------------

    if provider is not None:

        prompt = f"""
Translate the following restaurant menu into {language_name}.

STRICT RULES:
- Translate ALL natural-language text.
- Translate the menu title.
- Translate category names.
- Translate menu item names when appropriate.
- Translate the final question.
- Keep prices exactly unchanged.
- Keep "FCFA" exactly unchanged.
- Keep emojis exactly unchanged.
- Keep WhatsApp Markdown such as *bold* exactly intact.
- Do NOT add explanations.
- Do NOT remove menu items.
- Return ONLY the translated menu.

MENU:
{menu_text}
"""

        try:

            translated = provider.generate(
                prompt,
                temperature=0,
                max_tokens=300,
            )

            translated = clean_text(
                translated
            )

            if translated:
                return translated.splitlines()

        except Exception:

            logger.exception(
                "Failed to translate menu to %s",
                language_name,
            )

        return lines

    # --------------------------------------------------------
    # CACHED FALLBACK
    # --------------------------------------------------------

    translated = _translate_menu_text_cached(
        menu_text,
        language_name,
    )

    if translated:
        return translated.splitlines()

    return lines


def build_menu_response(
    tool_result,
    language="English",
    customer_message="",
    provider=None,
):

    if not isinstance(
        tool_result,
        dict,
    ):

        return {
            "type": "response",
            "message": customer_response(
                "I'm unable to access the menu right now. "
                "Please try again.",
                customer_message,
            ),
        }

    if not tool_result.get(
        "success"
    ):

        return {
            "type": "response",
            "message": customer_response(
                tool_result.get("message")
                or
                "I'm unable to access the menu right now. "
                "Please try again.",
                customer_message,
            ),
        }

    results = (
        tool_result.get("results")
        or []
    )

    full_menu = bool(
        tool_result.get("full_menu")
    )

    if not results and not full_menu:

        return {
            "type": "response",
            "message": customer_response(
                "I couldn't find any menu items "
                "right now. Please try again later.",
                customer_message,
            ),
        }

    category_emojis = {
        "burger": "🍔",
        "burgers": "🍔",
        "pizza": "🍕",
        "pizzas": "🍕",
        "chicken": "🍗",
        "chickens": "🍗",
        "drink": "🥤",
        "drinks": "🥤",
        "beverage": "🥤",
        "beverages": "🥤",
        "dessert": "🍰",
        "desserts": "🍰",
        "breakfast": "🍳",
        "rice": "🍚",
        "sandwich": "🥪",
        "sandwiches": "🥪",
        "salad": "🥗",
        "salads": "🥗",
        "snack": "🍿",
        "snacks": "🍿",
        "pasta": "🍝",
        "fish": "🐟",
        "seafood": "🦐",
        "meat": "🥩",
        "fries": "🍟",
        "sides": "🍟",
    }

    categories = {}

    for item in results:

        if not isinstance(
            item,
            dict,
        ):
            continue

        name = clean_text(
            item.get("name")
            or item.get("title")
            or "Menu item"
        )

        if not name:
            continue

        category = clean_text(
            item.get("category")
            or "Other"
        )

        category_key = category.lower()

        if category_key not in categories:

            categories[
                category_key
            ] = {
                "name": category,
                "items": [],
            }

        categories[
            category_key
        ]["items"].append(item)

    if not categories:

        return {
            "type": "response",
            "message": customer_response(
                "I couldn't find any menu items "
                "right now. Please try again later.",
                customer_message,
            ),
        }

    menu_header = {
        "French": "🍽️ *Notre Menu*",
        "Spanish": "🍽️ *Nuestro Menú*",
        "Portuguese": "🍽️ *Nosso Menu*",
        "Italian": "🍽️ *Il Nostro Menu*",
        "German": "🍽️ *Unsere Speisekarte*",
    }.get(
        language,
        "🍽️ *Our Menu*"
    )

    lines = [
        menu_header,
        "",
    ]

    for category_data in categories.values():

        category_name = category_data[
            "name"
        ]

        emoji = category_emojis.get(
            category_name.lower(),
            "🍽️",
        )

        lines.append(
            f"{emoji} *{category_name}*"
        )

        for item in category_data[
            "items"
        ]:

            name = clean_text(
                item.get("name")
                or item.get("title")
                or "Menu item"
            )

            price = item.get(
                "price"
            )

            price_text = ""

            if price is not None:

                try:

                    price_text = (
                        f"{float(price):,.0f} FCFA"
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    price_text = clean_text(
                        price
                    )

            line = f"• *{name}*"

            if price_text:
                line += (
                    f" — {price_text}"
                )

            lines.append(
                line
            )

    lines.append("")

    order_prompt = {
        "French": "Que souhaitez-vous commander ?",
        "Spanish": "¿Qué le gustaría pedir?",
        "Portuguese": "O que gostaria de pedir?",
        "Italian": "Cosa desidera ordinare?",
        "German": "Was möchten Sie bestellen?",
    }.get(
        language,
        "What would you like to order?"
    )

    lines.append(
        order_prompt
    )

    lines = translate_menu_for_customer(
        lines,
        language,
        provider=provider,
    )

    response = "\n".join(
        lines
    )

    return {
        "type": "response",
        "message": response,
    }


# ============================================================
# FAST INTENT CLASSIFICATION CONSTANTS
# ============================================================

FAST_RECOMMENDATION_PHRASES = (
    "hungry",
    "starving",
    "something to eat",
    "something good",
)

FAST_CANCEL_PHRASES = (
    "cancel my order",
    "cancel the order",
    "cancel order",
    "cancel it",
    "cancel my food",
    "i want to cancel",
    "i need to cancel",
    "remove my order",
)

FAST_MODIFY_PHRASES = (
    "change my order",
    "change the order",
    "modify my order",
    "modify the order",
    "edit my order",
    "edit the order",
    "remove from my order",
    "take off my order",
    "take it off my order",
    "remove it from my order",
    "add to my order",
    "add something to my order",
    "change the quantity",
    "change the amount",
    "make it two",
    "make it three",
    "make it one",
)

FAST_MENU_PHRASES = (
    "menu",
    "show menu",
    "show me menu",
    "show me the menu",
    "show the menu",
    "see the menu",
    "view the menu",
    "can i look at the menu",
    "can i see the menu",
    "i would like to see your menu",
    "i'd like to see your menu",
    "i would like to see the menu",
    "i'd like to see the menu",
    "look at the menu",
    "what is on the menu",
    "what's on the menu",
    "what do you have",
    "what do you guys have",
    "what food do you have",
    "what foods do you have",
    "what drinks do you have",
    "what pizzas do you have",
    "what meals do you have",
    "menu please",
    "send me the menu",
)

FAST_RESTAURANT_INFO_PHRASES = (
    "opening hours",
    "opening time",
    "what time do you open",
    "what time do you close",
    "when do you open",
    "when are you open",
    "when do you close",
    "when are you closed",
    "closing time",
    "what are your hours",
    "what time are you open",
    "what time are you closed",
    "are you open",
    "are you closed",
    "open today",
    "open now",
    "when can i visit",
    "do you deliver",
    "do you offer delivery",
    "is delivery available",
    "delivery available",
    "delivery information",
    "information about delivery",
    "info about delivery",
    "delivery options",
    "how does delivery work",
    "how much is delivery",
    "delivery fee",
    "delivery charge",
    "where are you located",
    "where are you",
    "where is the restaurant",
    "what is your address",
    "what's your address",
    "whats your address",
    "your address",
    "restaurant address",
    "location",
    "restaurant location",
    "where can i find you",
    "phone number",
    "your phone number",
    "contact number",
    "contact information",
    "contact info",
    "how can i contact you",
    "how do i contact you",
    "can i call you",
    "what number can i call",
    "tell me about your restaurant",
    "tell me about this restaurant",
    "tell me about your place",
    "tell me about this place",
    "what can you tell me about your restaurant",
    "what can you tell me about this restaurant",
    "what is your restaurant like",
    "what is this restaurant like",
    "about your restaurant",
    "about this restaurant",
)

FAST_ORDER_PHRASES = (
    "i want ",
    "i'd like ",
    "id like ",
    "give me ",
    "get me ",
    "bring me ",
    "i'll have ",
    "ill have ",
    "can i get ",
    "can i have ",
    "i would like ",
    "order ",
)

FAST_GENERIC_ORDER_PHRASES = (
    "i want to place an order",
    "i want to order",
    "i'd like to place an order",
    "id like to place an order",
    "i would like to place an order",
    "can i place an order",
    "i want an order",
)

FAST_FOOD_PATTERNS = (
    "pizza",
    "shawarma",
    "burger",
    "fries",
    "sandwich",
    "chicken",
    "ice cream",
    "gelato",
    "sundae",
    "lemonade",
    "cone",
    "drink",
    "food",
    "meal",
    "coffee",
    "juice",
    "water",
    "cake",
    "rice",
    "pasta",
)

FAST_FALLBACK_FOOD_PATTERNS = (
    "pizza",
    "shawarma",
    "burger",
    "fries",
    "sandwich",
    "chicken",
    "ice cream",
    "gelato",
    "sundae",
    "lemonade",
    "cone",
    "drink",
    "food",
)

FAST_ORDER_VERBS = (
    "want",
    "need",
    "get",
    "give",
    "have",
    "order",
    "take",
    "bring",
    "like",
)

FAST_RECOMMENDATION_PHRASES_FULL = (
    "what is good here",
    "what's good here",
    "what is good",
    "what's good",
    "what do you recommend",
    "what would you recommend",
    "what should i get",
    "what should i order",
    "recommend something",
    "recommend me something",
    "any recommendations",
    "your recommendation",
    "best thing here",
    "best item here",
    "most popular",
    "popular item",
)

FAST_OFF_TOPIC_PHRASES = (
    "write me a poem",
    "write a poem",
    "tell me a joke",
    "who is the president",
    "what is bitcoin",
    "solve my homework",
    "give me legal advice",
    "give me medical advice",
    "pretend to be",
    "roleplay",
)

FAST_CHAT_PHRASES = (
    # English
    "hello",
    "hi",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "thank",
    "okay",
    "ok",
    "alright",

    # French
    "bonjour",
    "bonsoir",
    "salut",
    "merci",

    # German
    "hallo",
    "guten morgen",
    "guten abend",
    "danke",

    # Spanish
    "hola",
    "buenos dias",
    "gracias",

    # Portuguese
    "ola",
    "bom dia",
    "obrigado",
    "obrigada",

    # Italian
    "ciao",
    "buongiorno",
    "grazie",
)


# ============================================================
# FAST INTENT CLASSIFICATION
# ============================================================

def classify_message_fast(message):

    text = normalize_text(
        message
    )

    if not text:
        return "chat"

    def contains_any(
        phrases,
    ):
        return any(
            phrase in text
            for phrase in phrases
        )

    # --------------------------------------------------------
    # OFF-TOPIC
    # --------------------------------------------------------

    if contains_any(
        FAST_OFF_TOPIC_PHRASES
    ):
        return "off_topic"

    # --------------------------------------------------------
    # HUNGER / RECOMMENDATION
    # --------------------------------------------------------

    if contains_any(
        FAST_RECOMMENDATION_PHRASES
    ):
        return "recommendation"

    # --------------------------------------------------------
    # CANCEL ORDER
    # --------------------------------------------------------

    if contains_any(
        FAST_CANCEL_PHRASES
    ):
        return "cancel_order"

    # --------------------------------------------------------
    # MODIFY ORDER
    # --------------------------------------------------------

    if contains_any(
        FAST_MODIFY_PHRASES
    ):
        return "modify_order"

    if (
        "remove " in text
        and " from my order" in text
    ):
        return "modify_order"

    # --------------------------------------------------------
    # MENU SEARCH
    # --------------------------------------------------------

    if contains_any(
        FAST_MENU_PHRASES
    ):
        return "menu_search"

    # --------------------------------------------------------
    # RESTAURANT INFORMATION
    # --------------------------------------------------------

    if contains_any(
        FAST_RESTAURANT_INFO_PHRASES
    ):
        return "restaurant_info"

    # --------------------------------------------------------
    # ORDER INTENT
    # --------------------------------------------------------

    if (
        contains_any(FAST_GENERIC_ORDER_PHRASES)
        and not contains_any(FAST_FOOD_PATTERNS)
    ):
        return "chat"

    has_order_phrase = contains_any(
        FAST_ORDER_PHRASES
    )

    has_food = contains_any(
        FAST_FOOD_PATTERNS
    )

    if has_order_phrase and has_food:
        return "order"

    # --------------------------------------------------------
    # RECOMMENDATION INTENT
    # --------------------------------------------------------

    if contains_any(
        FAST_RECOMMENDATION_PHRASES_FULL
    ):
        return "recommendation"

    # --------------------------------------------------------
    # CHAT
    # --------------------------------------------------------

    if (
        text in FAST_CHAT_PHRASES
        or any(
            text.startswith(
                phrase + " "
            )
            for phrase in FAST_CHAT_PHRASES
        )
    ):
        return "chat"

    # --------------------------------------------------------
    # FALLBACK ORDER DETECTION
    # --------------------------------------------------------

    has_fallback_food = contains_any(
        FAST_FALLBACK_FOOD_PATTERNS
    )

    has_order_verb = contains_any(
        FAST_ORDER_VERBS
    )

    if has_fallback_food and has_order_verb:
        return "order"

    return None


# ============================================================
# AI INTENT CLASSIFICATION FALLBACK
# ============================================================

def classify_message(
    provider,
    message,
):
    """
    LLM fallback classifier used only when the fast
    local classifier cannot determine the intent.
    """

    prompt = f"""
Classify this restaurant customer message.

CUSTOMER MESSAGE:
{message}

Return exactly ONE category:

chat
off_topic
order
modify_order
cancel_order
menu_search
restaurant_info
recommendation

CATEGORY RULES:

chat:
Greetings, thanks, acknowledgements, or natural
restaurant-related conversation.

off_topic:
Anything unrelated to the restaurant, menu, food,
drinks, ordering, delivery, pickup, payment,
location, hours, or restaurant customer service.

order:
The customer clearly wants specific food or drinks.

modify_order:
The customer wants to change an existing order.

cancel_order:
The customer wants to cancel an existing order.

menu_search:
The customer wants to see, browse, or search the menu.

restaurant_info:
The customer asks about hours, delivery,
address, phone number, or other restaurant information.

recommendation:
The customer asks what they should eat, what is good,
or asks for recommendations.

Return ONLY the category name.
"""

    try:

        result = provider.generate(
            prompt,
            temperature=0,
            max_tokens=50,
        )

        result = normalize_text(
            result
        ).replace(
            "`",
            "",
        ).strip()

        # Handle occasional punctuation from the model.
        result = result.rstrip(
            ".,:;!?"
        ).strip()

        valid_categories = {
            "chat",
            "off_topic",
            "order",
            "modify_order",
            "cancel_order",
            "menu_search",
            "restaurant_info",
            "recommendation",
        }

        if result in valid_categories:
            return result

        logger.warning(
            "Unexpected AI classification result: %r",
            result,
        )

        return "chat"

    except Exception:

        logger.exception(
            "AI intent classification failed."
        )

        return "chat"

# ============================================================
# PENDING ORDER QUANTITY FOLLOW-UP
# ============================================================

def update_pending_order_quantity(
    business_id,
    phone,
    message,
):
    """
    Handle quantity-only follow-ups for an existing
    pending order preview.
    """

    from models.customer import Customer
    from models.pending_order import PendingOrder

    text = normalize_text(
        message
    )

    quantity_match = re.fullmatch(
        r"(?:"
        r"i will have|"
        r"i ll have|"
        r"ill have|"
        r"i want|"
        r"i would like|"
        r"make it|"
        r"make that|"
        r"give me|"
        r"two please|"
        r"three please|"
        r"four please|"
        r"five please|"
        r"one please"
        r")?"
        r"\s*"
        r"(?:"
        r"\d+|"
        r"one|"
        r"two|"
        r"three|"
        r"four|"
        r"five"
        r")"
        r"\s*(?:please)?",
        text,
    )

    if not quantity_match:
        return None

    quantity_text = (
        quantity_match.group(0)
        .strip()
        .split()
        [-1]
    )

    word_quantities = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
    }

    try:

        if quantity_text.isdigit():

            quantity = max(
                1,
                int(
                    quantity_text
                ),
            )

        else:

            quantity = (
                word_quantities
                .get(quantity_text)
            )

    except (
        TypeError,
        ValueError,
    ):

        return None

    if quantity is None:
        return None

    customer = (
        Customer.query
        .filter_by(
            business_id=business_id,
            phone=phone,
        )
        .first()
    )

    if not customer:
        return None

    preview = (
        PendingOrder.query
        .filter_by(
            business_id=business_id,
            customer_id=customer.id,
            status="pending",
        )
        .order_by(
            PendingOrder.id.desc()
        )
        .first()
    )

    if not preview:
        return None

    try:

        items = json.loads(
            preview.items_json
        )

    except (
        json.JSONDecodeError,
        TypeError,
    ):

        return None

    if not isinstance(
        items,
        list,
    ) or not items:

        return None

    target = items[-1]

    if not isinstance(
        target,
        dict,
    ):

        return None

    target_name = clean_text(
        target.get("name")
        or "Item"
    )

    try:

        unit_price = float(
            target.get(
                "price",
                0,
            )
            or 0
        )

    except (
        TypeError,
        ValueError,
    ):

        return None

    target["quantity"] = quantity

    target["subtotal"] = (
        unit_price * quantity
    )

    total = 0.0

    for item in items:

        if not isinstance(
            item,
            dict,
        ):
            continue

        try:

            item_quantity = max(
                1,
                int(
                    item.get(
                        "quantity",
                        1,
                    )
                    or 1
                ),
            )

            item_price = float(
                item.get(
                    "price",
                    0,
                )
                or 0
            )

        except (
            TypeError,
            ValueError,
        ):

            continue

        item["quantity"] = (
            item_quantity
        )

        item["subtotal"] = (
            item_price
            * item_quantity
        )

        total += (
            item["subtotal"]
        )

    preview.items_json = json.dumps(
        items,
        ensure_ascii=False,
    )

    preview.total_price = (
        float(total)
    )

    db.session.commit()

    lines = [
        "Your updated order:",
        "",
    ]

    for item in items:

        if not isinstance(
            item,
            dict,
        ):
            continue

        name = clean_text(
            item.get("name")
            or "Item"
        )

        item_quantity = item.get(
            "quantity",
            1,
        )

        lines.append(
            f"• *{name}* × "
            f"{item_quantity}"
        )

    lines.extend([
        "",
        f"Total: {total:,.0f} FCFA",
        "",
        "Please confirm your order.",
    ])

    return {
        "type": "response",
        "message": customer_response(
            "\n".join(lines),
            message,
        ),
    }


# ============================================================
# MAIN AGENT
# ============================================================

def run_agent(
    business_id,
    phone,
    message,
    language="English",
    history=None,
):

    message = clean_text(
        message
    )

    if not message:

        return {
            "type": "response",
            "message": (
                "I'm here to help. "
                "What would you like to know?"
            ),
        }

    history = history or []

    # ========================================================
    # LAZY PROVIDER
    # ========================================================

    provider = None

    def get_provider():

        nonlocal provider

        if provider is None:
            provider = OpenAIProvider()

        return provider

    # ========================================================
    # FAST GREETING
    # ========================================================

    normalized_message = normalize_text(
        message
    )

    greeting_responses = {
        "English": {
            "hi": "Hi! How can I help you today?",
            "hello": "Hello! How can I help you today?",
            "hey": "Hey! How can I help you today?",
            "good morning": "Good morning! How can I help you today?",
            "good afternoon": "Good afternoon! How can I help you today?",
            "good evening": "Good evening! How can I help you today?",
        },
        "French": {
            "bonjour": "Bonjour ! Comment puis-je vous aider ?",
            "bonsoir": "Bonsoir ! Comment puis-je vous aider ?",
            "salut": "Salut ! Comment puis-je vous aider ?",
        },
        "Spanish": {
            "hola": "¡Hola! ¿Cómo puedo ayudarte?",
        },
        "Portuguese": {
            "olá": "Olá! Como posso ajudá-lo?",
        },
        "Italian": {
            "ciao": "Ciao! Come posso aiutarti?",
        },
        "German": {
            "hallo": "Hallo! Wie kann ich Ihnen helfen?",
        },
    }

    language_greetings = greeting_responses.get(
        language,
        greeting_responses["English"]
    )

    greeting_response = language_greetings.get(
        normalized_message
    )

    if greeting_response:
        return {
            "type": "response",
            "message": customer_response(
                greeting_response,
                message,
            ),
        }

    pending = None

    # ========================================================
    # FAST PENDING-ORDER CONFIRMATION / REJECTION
    # ========================================================

    confirmation = is_confirmation(
        message
    )

    rejection = is_rejection(
        message
    )

    if confirmation or rejection:

        pending = get_pending_order(
            business_id,
            phone,
        )

        if (
            isinstance(
                pending,
                dict,
            )
            and pending.get("success")
        ):

            if confirmation:

                try:

                    result = confirm_pending_order(
                        business_id,
                        phone,
                    )

                    return build_tool_response(
                        provider=get_provider(),
                        business_id=business_id,
                        phone=phone,
                        message=message,
                        language=language,
                        history=history,
                        tool_result=result,
                    )

                except Exception:

                    logger.exception(
                        "Pending order confirmation failed."
                    )

                    return {
                        "type": "response",
                        "message": customer_response(
                            "I couldn't confirm the order "
                            "right now. Please try again.",
                            message,
                        ),
                    }

            if rejection:

                try:

                    result = discard_pending_order(
                        business_id,
                        phone,
                    )

                    return build_tool_response(
                        provider=get_provider(),
                        business_id=business_id,
                        phone=phone,
                        message=message,
                        language=language,
                        history=history,
                        tool_result=result,
                    )

                except Exception:

                    logger.exception(
                        "Pending order rejection failed."
                    )

                    return {
                        "type": "response",
                        "message": customer_response(
                            "I couldn't cancel the order "
                            "preview right now. Please try again.",
                            message,
                        ),
                    }

    # ========================================================
    # AMBIGUOUS BROAD ORDER REQUEST
    # ========================================================

    normalized_order_message = normalize_text(
        message
    )

    if normalized_order_message in (
        "i want everything",
        "i want one of everything",
    ):

        return {
            "type": "response",
            "message": customer_response(
                "I'd be happy to help. "
                "Which items would you like to order?",
                message,
            ),
        }

    # ========================================================
    # PENDING ORDER QUANTITY FOLLOW-UP
    # ========================================================

    quantity_update = (
        update_pending_order_quantity(
            business_id,
            phone,
            message,
        )
    )

    if quantity_update:

        return quantity_update

    # ========================================================
    # FAST CLASSIFICATION
    # ========================================================

    classification = classify_message_fast(
        message
    )

    logger.info(
        "Restaurant AI fast classification: %s",
        classification,
    )

    # ========================================================
    # FAST CLASSIFICATION
    # ========================================================

    classification = classify_message_fast(
        message
    )

    logger.info(
        "Restaurant AI fast classification: %s",
        classification,
    )

    # ========================================================
    # AI CLASSIFICATION FALLBACK
    # ========================================================

    if classification is None:

        classification = classify_message(
            get_provider(),
            message,
        )

    logger.info(
        "Restaurant AI classification: %s",
        classification,
    )

    # ========================================================
    # FAST CHAT RESPONSE
    # ========================================================

    if classification == "chat":

        chat_responses = {
            # ------------------------------------------------
            # GREETINGS
            # ------------------------------------------------

            "hi": "Hi! How can I help you today?",
            "hello": "Hello! How can I help you today?",
            "hey": "Hey! How can I help you today?",
            "good morning": "Good morning! How can I help you today?",
            "good afternoon": "Good afternoon! How can I help you today?",
            "good evening": "Good evening! How can I help you today?",

            # ------------------------------------------------
            # ENGLISH ACKNOWLEDGEMENTS
            # ------------------------------------------------

            "thanks": "You're welcome! Let me know if you'd like anything from the menu.",
            "thank you": "You're welcome! Let me know if you'd like anything from the menu.",
            "thank": "You're welcome! Let me know if you'd like anything from the menu.",
            "okay": "Alright! Let me know if you need anything.",
            "ok": "Alright! Let me know if you need anything.",
            "alright": "Alright! Let me know if you need anything.",

            # ------------------------------------------------
            # FRENCH
            # ------------------------------------------------

            "bonjour": "Bonjour ! Comment puis-je vous aider ?",
            "bonsoir": "Bonsoir ! Comment puis-je vous aider ?",
            "salut": "Salut ! Comment puis-je vous aider ?",
            "merci": "Avec plaisir ! N'hésitez pas si vous avez besoin de quoi que ce soit.",

            # ------------------------------------------------
            # GERMAN
            # ------------------------------------------------

            "hallo": "Hallo! Wie kann ich Ihnen helfen?",
            "guten morgen": "Guten Morgen! Wie kann ich Ihnen helfen?",
            "guten abend": "Guten Abend! Wie kann ich Ihnen helfen?",
            "danke": "Gerne! Lassen Sie mich wissen, wenn Sie etwas brauchen.",

            # ------------------------------------------------
            # SPANISH
            # ------------------------------------------------

            "hola": "¡Hola! ¿Cómo puedo ayudarte?",
            "buenos dias": "¡Buenos días! ¿Cómo puedo ayudarte?",
            "gracias": "¡De nada! Avísame si necesitas algo más.",

            # ------------------------------------------------
            # PORTUGUESE
            # ------------------------------------------------

            "ola": "Olá! Como posso ajudar?",
            "bom dia": "Bom dia! Como posso ajudar?",
            "obrigado": "De nada! Avise-me se precisar de mais alguma coisa.",
            "obrigada": "De nada! Avise-me se precisar de mais alguma coisa.",

            # ------------------------------------------------
            # ITALIAN
            # ------------------------------------------------

            "ciao": "Ciao! Come posso aiutarti?",
            "buongiorno": "Buongiorno! Come posso aiutarti?",
            "grazie": "Prego! Fammi sapere se hai bisogno di altro.",
        }

        # ----------------------------------------------------
        # ACKNOWLEDGEMENT PHRASES
        # ----------------------------------------------------

        acknowledgement_phrases = (
            "ok thank you",
            "okay thank you",
            "alright thank you",
            "thanks a lot",
            "thank you very much",
            "thanks a lot",
            "merci beaucoup",
            "grazie mille",
            "muchas gracias",
            "obrigado muito",
            "obrigada muito",
            "danke schön",
            "danke schon",
        )

        if any(
            phrase in normalized_message
            for phrase in acknowledgement_phrases
        ):

            return {
                "type": "response",
                "message": customer_response(
                    "You're welcome! Let me know if you'd like anything else from the menu.",
                    message,
                ),
            }

        direct_chat_response = chat_responses.get(
            normalized_message
        )

        if direct_chat_response:

            return {
                "type": "response",
                "message": customer_response(
                    direct_chat_response,
                    message,
                ),
            }

        return {
            "type": "response",
            "message": customer_response(
                "Sure! Let me know how I can help with the restaurant.",
                message,
            ),
        }

    # ========================================================
    # OFF TOPIC
    # ========================================================

    if classification == "off_topic":

        return {
            "type": "response",
            "message": customer_response(
                "I'm the restaurant's AI assistant, so I can "
                "only help with the menu, orders, delivery, "
                "and other restaurant-related questions.",
                message,
            ),
        }

    # ========================================================
    # RESTAURANT INFO
    # ========================================================

    if classification == "restaurant_info":

        try:

            result = get_restaurant_info(
                business_id,
                message,
            )

            if not result.get(
                "success"
            ):

                return {
                    "type": "response",
                    "message": customer_response(
                        result.get(
                            "message",
                            "That information is not available right now.",
                        ),
                        message,
                    ),
                }

            return build_tool_response(
                provider=get_provider(),
                business_id=business_id,
                phone=phone,
                message=message,
                language=language,
                history=history,
                tool_result=result,
            )

        except Exception:

            logger.exception(
                "Restaurant information lookup failed."
            )

            return {
                "type": "response",
                "message": customer_response(
                    "I couldn't retrieve that restaurant information right now.",
                    message,
                ),
            }

    # ========================================================
    # ORDER
    # ========================================================

    if classification == "order":

        # --------------------------------------------------------
        # AMBIGUOUS BROAD ORDER REQUEST
        # --------------------------------------------------------

        normalized_order_message = normalize_text(
            message
        )

        if normalized_order_message in (
            "i want everything",
            "i want one of everything",
        ):

            return {
                "type": "response",
                "message": customer_response(
                    "I'd be happy to help. "
                    "Which items would you like to order?",
                    message,
                ),
            }

        try:

            result = create_order_preview(
                business_id,
                phone,
                message,
            )

            if not result.get(
                "success"
            ):

                unmatched = result.get(
                    "unmatched",
                    [],
                )

                if unmatched:

                    return {
                        "type": "response",
                        "message": customer_response(
                            "I couldn't find "
                            + ", ".join(
                                unmatched
                            )
                            + " on the menu. "
                            "What would you like to order?",
                            message,
                        ),
                    }

            return build_tool_response(
                provider=get_provider(),
                business_id=business_id,
                phone=phone,
                message=message,
                language=language,
                history=history,
                tool_result=result,
            )

        except Exception:

            logger.exception(
                "Order preview creation failed."
            )

            return {
                "type": "response",
                "message": customer_response(
                    "I couldn't create the order "
                    "preview right now. Please try again.",
                    message,
                ),
            }

    # ========================================================
    # MODIFY ORDER
    # ========================================================

    if classification == "modify_order":

        try:

            result = modify_active_order(
                business_id,
                phone,
                message,
            )

            if not result.get("success"):

                return {
                    "type": "response",
                    "message": customer_response(
                        result.get("message")
                        or "I couldn't modify the order right now.",
                        message,
                    ),
                }

            if result.get("action") in (
                "add_item",
                "remove_item",
                "set_quantity",
                "replace_item",
            ):

                action_message = (
                    result.get("message")
                    or "Your order has been updated."
                )

                total = result.get(
                    "total"
                )

                try:

                    total_text = (
                        f"{float(total):,.0f} FCFA"
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    total_text = (
                        f"{total or 0} FCFA"
                    )

                response_lines = [
                    action_message,
                    "",
                    f"Total: {total_text}",
                ]

                response_lines = (
                    translate_order_preview_for_customer(
                        response_lines,
                        language,
                        provider=get_provider(),
                    )
                )

                return {
                    "type": "response",
                    "message": customer_response(
                        "\n".join(
                            response_lines
                        ),
                        message,
                    ),
                }

            return build_tool_response(
                provider=get_provider(),
                business_id=business_id,
                phone=phone,
                message=message,
                language=language,
                history=history,
                tool_result=result,
            )

        except Exception:

            logger.exception(
                "Order modification failed."
            )

            return {
                "type": "response",
                "message": customer_response(
                    "I couldn't modify the order "
                    "right now. Please try again.",
                    message,
                ),
            }

    # ========================================================
    # CANCEL ORDER
    # ========================================================

    if classification == "cancel_order":

        try:

            result = cancel_active_order(
                business_id,
                phone,
            )

            if not result.get("success"):

                return {
                    "type": "response",
                    "message": customer_response(
                        result.get("message")
                        or "No active order found.",
                        message,
                    ),
                }

            if (
                result.get("success")
                and str(
                    result.get("status", "")
                ).lower()
                == "cancelled"
            ):

                order_id = result.get(
                    "order_id"
                )

                return {
                    "type": "response",
                    "message": customer_response(
                        (
                            f"Your order #{order_id} "
                            "has been cancelled. ✅"
                        ),
                        message,
                    ),
                }

            return build_tool_response(
                provider=get_provider(),
                business_id=business_id,
                phone=phone,
                message=message,
                language=language,
                history=history,
                tool_result=result,
            )

        except Exception:

            logger.exception(
                "Order cancellation failed."
            )

            return {
                "type": "response",
                "message": customer_response(
                    "I couldn't cancel the order "
                    "right now. Please try again.",
                    message,
                ),
            }

    # ========================================================
    # MENU SEARCH
    # ========================================================

    if classification == "menu_search":

        try:

            result = tool_search_menu(
                business_id,
                message,
            )

            return build_menu_response(
                result,
                language=language,
                customer_message=message,
                provider=provider,
            )

        except Exception:

            logger.exception(
                "Menu search failed."
            )

            return {
                "type": "response",
                "message": customer_response(
                    "I'm unable to access the menu right now. "
                    "Please try again.",
                    message,
                ),
            }

    # ========================================================
    # RECOMMENDATION
    # ========================================================

    if classification == "recommendation":

        try:

            recommendations = recommend_menu(
                business_id,
                limit=3,
            )

            if not recommendations:

                return {
                    "type": "response",
                    "message": customer_response(
                        "I couldn't find any menu items to recommend right now.",
                        message,
                    ),
                }

            lines = [
                "Here are some good options:"
            ]

            for item in recommendations:

                name = clean_text(
                    item.get("name")
                    or "Item"
                )

                description = clean_text(
                    item.get("description")
                )

                price = item.get(
                    "price",
                    0,
                )

                try:

                    price_text = (
                        f"{float(price):,.0f}"
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    price_text = str(
                        price
                    )

                lines.append(
                    f"*{name}* — {price_text} FCFA"
                )

                if description:

                    lines.append(
                        description
                    )

            lines.append(
                "\nWould you like any of these?"
            )

            return {
                "type": "response",
                "message": customer_response(
                    "\n".join(
                        lines
                    ),
                    message,
                ),
            }

        except Exception:

            logger.exception(
                "Recommendation lookup failed."
            )

            return {
                "type": "response",
                "message": customer_response(
                    "I couldn't get recommendations right now. "
                    "Please try the menu instead.",
                    message,
                ),
            }

    # ========================================================
    # FINAL NATURAL RESPONSE FALLBACK
    # ========================================================

    if pending is None:

        pending = get_pending_order(
            business_id,
            phone,
        )

    return generate_natural_response(
        provider=get_provider(),
        business_id=business_id,
        phone=phone,
        message=message,
        language=language,
        history=history,
        pending=pending,
    )