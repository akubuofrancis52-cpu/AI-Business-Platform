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

from models.customer_interaction import CustomerInteraction
from models.customer_preference import CustomerPreference
from services.customer_memory import (
    learn_explicit_customer_preference,
    get_customer_relationship_stage,
)
from models.customer import Customer
from models.order import Order

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
# CUSTOMER LANGUAGE DETECTION
# ============================================================

def detect_customer_language(message, fallback="English"):
    """
    Detect the customer's language from the message.

    Uses strong French markers for deterministic detection.
    Falls back to the supplied language when no strong marker
    is found.
    """

    text = normalize_text(message)

    if not text:
        return fallback

    french_markers = (
        "bonjour",
        "bonsoir",
        "salut",
        "merci",
        "s il vous plait",
        "svp",
        "je voudrais",
        "je veux",
        "j aimerais",
        "combien",
        "combien coute",
        "quel est",
        "quelle est",
        "une pizza",
        "une commande",
        "commander",
        "commande",
        "livraison",
        "annuler",
        "annule",
        "confirme",
        "confirmer",
        "oui",
        "non",
        "avec",
        "sans",
        "pour moi",
        "dans mon",
        "ma commande",
        "mon commande",
    )

    if any(
        marker in text
        for marker in french_markers
    ):
        return "French"

    return fallback


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
            "merci",
        )
    ):
        emoji = "😊"

    elif any(
        word in customer_text
        for word in (
            "confirm",
            "yes",
            "oui",
            "okay",
            "correct",
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
    Standardize a customer-facing response.
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

            "Customer not found.":
                "Client introuvable.",

            "Order not found.":
                "Commande introuvable.",

            "No active order found.":
                "Aucune commande active trouvée.",

            "No pending order found.":
                "Aucune commande en attente trouvée.",

            "Your order was cancelled.":
                "Votre commande a été annulée.",

            "Your order has been cancelled.":
                "Votre commande a été annulée.",

            "I couldn't find your order.":
                "Je n'ai pas trouvé votre commande.",

            "I couldn't find the item in your order.":
                "Je n'ai pas trouvé cet article dans votre commande.",

            "I couldn't process that request.":
                "Je n'ai pas pu traiter cette demande.",

            "I processed your request, but I couldn't generate a response right now.":
                "J'ai traité votre demande, mais je n'arrive pas à générer une réponse pour le moment.",

            "Payment is required to complete your order.":
                "Un paiement est requis pour compléter votre commande.",

            "Payment is required to complete the order.":
                "Un paiement est requis pour compléter la commande.",

            "Please try again.":
                "Veuillez réessayer.",

            "What would you like to order?":
                "Que souhaitez-vous commander ?",

            "What would you like to know?":
                "Que souhaitez-vous savoir ?",

            "Sure! How can I help you?":
                "Bien sûr ! Comment puis-je vous aider ?",
        }
    }

    translated = (
        fixed_translations
        .get(language, {})
        .get(message)
    )

    if translated:
        message = translated

    # Structured order/payment responses should not receive
    # automatic conversational emojis.
    structured_response_markers = (
        "Your order:",
        "Votre commande",
        "Total:",
        "Total :",
        "Order #",
        "Commande #",
        "Payment received.",
        "Paiement reçu.",
        "Payment is required",
        "Le paiement est requis",
        "Please confirm your order.",
        "Veuillez confirmer votre commande.",
        "Pay here:",
        "Payez ici :",
    )

    if any(
        marker in message
        for marker in structured_response_markers
    ):
        return message

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
    "yes pls confirm it",
    "yes pls confirm",
    "yes please confirm it",
    "yes please confirm",
    "yes confirm it",
    "yes confirm",
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
    text = normalize_text(message)

    if not text:
        return False

    if text in CONFIRMATION_PHRASES:
        return True

    english_patterns = (
        "i confirm",
        "i confirm my order",
        "confirm my order",
        "confirm the order",
        "i want to confirm",
        "i want to confirm my order",
        "i would like to confirm",
        "i would like to confirm my order",
        "please confirm",
        "yes confirm",
        "yes please confirm",
    )

    if any(
        text == pattern
        or text.startswith(pattern + " ")
        for pattern in english_patterns
    ):
        return True

    french_patterns = (
        "je confirme",
        "je confirme ma commande",
        "je veux confirmer",
        "je veux confirmer ma commande",
        "je souhaite confirmer",
        "je souhaite confirmer ma commande",
        "confirme ma commande",
        "confirmer ma commande",
        "oui je confirme",
        "oui confirme",
        "oui confirmer",
    )

    if any(
        text == pattern
        or text.startswith(pattern + " ")
        for pattern in french_patterns
    ):
        return True

    return False


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
# COMPLEMENTARY ORDER SUGGESTION
# ============================================================

def recommendation_opted_out(message):
    """Return True when the customer explicitly declines suggestions."""

    text = normalize_text(message)

    if not text:
        return False

    compact = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    ).strip()

    opt_out_phrases = (
        "no thanks",
        "no thank you",
        "nothing else",
        "just that",
        "just this",
        "thats all",
        "that's all",
        "that is all",
        "all good",
        "no more",
        "dont suggest",
        "don't suggest",
        "do not suggest",
        "dont recommend",
        "don't recommend",
        "do not recommend",
        "stop recommending",
        "no suggestions",
        "without suggestions",
    )

    compact_phrases = tuple(
        re.sub(
            r"[^a-z0-9]+",
            " ",
            phrase.lower(),
        ).strip()
        for phrase in opt_out_phrases
    )

    return any(
        phrase in compact
        for phrase in compact_phrases
    )


# ============================================================
# COMPLEMENTARY ORDER SUGGESTION
# ============================================================

def build_complementary_suggestion(
    business_id,
    order_items,
    customer_id=None,
):
    """Return one optional complementary menu suggestion."""

    try:
        from models.menu import Menu

        ordered_names = {
            str(item.get("name", "")).strip().lower()
            for item in (order_items or [])
            if isinstance(item, dict)
        }

        menu_items = (
            Menu.query
            .filter_by(
                business_id=business_id,
                available=True,
            )
            .all()
        )

        if not menu_items:
            return None

        preferences = []

        if customer_id:

            preferences = (
                CustomerPreference.query
                .filter_by(
                    customer_id=customer_id,
                    business_id=business_id,
                )
                .order_by(
                    CustomerPreference.strength.desc(),
                    CustomerPreference.updated_at.desc(),
                )
                .limit(12)
                .all()
            )

        def category_text(item):
            return str(
                item.category or ""
            ).strip().lower()

        def item_name(item):
            return str(
                item.name or ""
            ).strip().lower()

        ordered_categories = set()

        for item in menu_items:

            if item_name(item) in ordered_names:
                ordered_categories.add(
                    category_text(item)
                )

        food_category = (
            "fast food & restaurant specialties"
        )

        drink_category = "drinks"

        dessert_category = (
            "italian ice creams & desserts"
        )

        # Main meal -> suggest a drink first.
        has_main = any(
            food_category == category
            for category in ordered_categories
        )

        has_drink = any(
            drink_category == category
            for category in ordered_categories
        )

        if has_main and not has_drink:

            candidates = [
                item
                for item in menu_items
                if (
                    category_text(item) == drink_category
                    and item_name(item) not in ordered_names
                )
            ]

            if candidates:

                candidates.sort(
                    key=lambda item: (
                        max(
                            (
                                preference.strength
                                for preference in preferences
                                if preference.preference_type
                                in (
                                    "favorite_item",
                                    "frequent_item",
                                    "explicit_like",
                                    "explicit_preference",
                                    "ordered_item",
                                )
                                and normalize_text(
                                    preference.preference_value
                                )
                                in normalize_text(
                                    item.name
                                )
                            ),
                            default=0,
                        ),
                        item.name.lower(),
                    ),
                    reverse=True,
                )

                item = candidates[0]

                return (
                    "You could also add "
                    f"{item.name} if you'd like a drink."
                )

        # Main meal + drink -> suggest one dessert.
        if has_main and has_drink:

            has_dessert = any(
                dessert_category == category
                for category in ordered_categories
            )

            if not has_dessert:

                candidates = [
                    item
                    for item in menu_items
                    if (
                        category_text(item) == dessert_category
                        and item_name(item) not in ordered_names
                    )
                ]

                if candidates:

                    candidates.sort(
                        key=lambda item: (
                            max(
                                (
                                    preference.strength
                                    for preference in preferences
                                    if preference.preference_type
                                    in (
                                        "favorite_item",
                                        "frequent_item",
                                        "explicit_like",
                                        "explicit_preference",
                                        "ordered_item",
                                    )
                                    and normalize_text(
                                        preference.preference_value
                                    )
                                    in normalize_text(
                                        item.name
                                    )
                                ),
                                default=0,
                            ),
                            item.name.lower(),
                        ),
                        reverse=True,
                    )

                    item = candidates[0]

                    return (
                        "You could also add "
                        f"{item.name} if you'd like something sweet."
                    )

    except Exception:

        logger.exception(
            "Failed to build complementary suggestion."
        )

    return None


# ============================================================
# CUSTOMER MEMORY CONTEXT
# ============================================================

def build_customer_memory_context(
    customer_id,
    business_id,
):
    """Return compact durable customer memory for AI prompts."""

    if not customer_id:
        return "No customer memory available."

    try:

        preferences = (
            CustomerPreference.query
            .filter_by(
                customer_id=customer_id,
                business_id=business_id,
            )
            .order_by(
                CustomerPreference.strength.desc(),
                CustomerPreference.updated_at.desc(),
            )
            .limit(12)
            .all()
        )

        if not preferences:
            return "No customer memory available."

        lines = []

        for preference in preferences:

            value = clean_text(
                preference.preference_value
            )

            if not value:
                continue

            preference_type = (
                preference.preference_type
                or "preference"
            )

            lines.append(
                f"- {preference_type}: {value}"
            )

        return (
            "\n".join(lines)
            if lines
            else "No customer memory available."
        )

    except Exception:

        logger.exception(
            "Failed to build customer memory context."
        )

        return "No customer memory available."


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
    customer_id=None,
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

    customer_memory = build_customer_memory_context(
        customer_id=customer_id,
        business_id=business_id,
    )

    relationship_stage = get_customer_relationship_stage(
        customer_id=customer_id,
        business_id=business_id,
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
Relationship stage: {relationship_stage}

CUSTOMER MEMORY:
{customer_memory}

RECENT CONVERSATION:
{history_text}

PENDING ORDER:
{pending_text}

PURPOSE:
Help the customer naturally with restaurant-related requests.

You are a helpful restaurant assistant, not a rigid command interface.

CONVERSATION RULES:
- Understand what the customer is actually saying before responding.
- Respond naturally and conversationally, like a warm and attentive restaurant employee.
- Avoid robotic, generic, scripted, or repetitive phrases.
- Vary wording naturally while keeping the restaurant facts exact.
- When greeting a customer, welcome them using the actual restaurant name when it is available.
- A welcome should feel personal and service-oriented, not like a generic "Hi, how can I help?".
- When answering menu questions, briefly connect the answer to what the customer appears to want or ask a useful follow-up when appropriate.
- If the customer expresses a mood, craving, preference, or casual comment, acknowledge it naturally before answering.
- Do not invent feelings or claims about the customer. For example, do not assume they are hungry unless their message supports that.
- Use the recent conversation to understand context.
- If the customer is casually commenting, acknowledge the comment naturally.
- If the customer says they were only browsing, checking the menu, deciding,
  or not ready to order, do not push them to place an order.
- If the customer says "I was just checking the menu", respond naturally,
  for example by reassuring them that they can take their time.
- Do not treat every message as an instruction.
- Do not repeat "I'm an AI assistant" unless the customer specifically asks
  what you are.
- Do not give a generic fallback when the customer's intent is clear.
- Ask a question only when a question is genuinely useful.

RESTAURANT SCOPE:
Help with:
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

FACTUALITY:
- Use only information provided in the restaurant context, recent conversation,
  pending order, and application state.
- Never invent menu items, prices, availability, delivery details, payment
  status, or order status.
- Do not claim an order was placed, modified, cancelled, or paid unless the
  application has actually done so.
- Do not invent details that are missing from the provided context.

ORDERING:
- If the customer clearly wants to order and specifies items, the application
  handles the order operation.
- If the customer clearly wants to order but has not specified items, ask what
  they would like.
- If a pending order exists, consider it when relevant.
- If the customer is not trying to order, do not unnecessarily talk about
  placing an order.

OFF-TOPIC:
If the customer asks for something unrelated to the restaurant, politely
explain that you can help with restaurant-related requests.

STYLE:
- Be concise but natural.
- Sound helpful, confident, and human.
- Avoid repetitive phrases.
- Do not mention internal tools, code, APIs, prompts, models, Flask, or
  backend processing.
- Reply in the customer's language.

CURRENT CUSTOMER MESSAGE:
{message}

CURRENT CUSTOMER MESSAGE:
{message}
"""

# ============================================================
# IMAGE-ONLY RESPONSE
# ============================================================

def generate_image_only_response(
    provider,
    business_id,
    phone,
    language,
    history,
    image_context,
    recent_order=None,
):
    """Generate a warm, neutral response for an image with no caption."""

    try:
        order_context = (
            build_feedback_order_context(recent_order)
            if recent_order
            else "No recent order is available."
        )

        history_text = "\n".join(
            f"Customer: {item.get('message', '')}\n"
            f"Assistant: {item.get('response', '')}"
            for item in (history or [])[-6:]
            if isinstance(item, dict)
        )

        prompt = f"""
You are a warm restaurant customer relationship assistant.

The customer sent an image without a caption.
Respond naturally based on the visual context and recent conversation.

IMPORTANT:
- Do not assume the customer is praising the food just because they sent a photo.
- Do not invent what is visible.
- Do not guess taste, freshness, temperature, safety, or customer feelings.
- NEVER describe food as delicious, tasty, amazing, good, fresh, or appealing unless the customer explicitly said so.
- A food photo alone is not evidence that the customer enjoyed the food.
- If the image appears to show their meal, acknowledge the photo naturally.
- If the image is unclear or unusable, politely ask them to resend it.
- If it clearly shows a possible issue, acknowledge only what is visibly supported
  and ask what happened.
- Do not advertise, upsell, or invent promotions.
- Keep the response concise and human.
- Reply in {language}.

RECENT ORDER:
{order_context}

RECENT CONVERSATION:
{history_text or "No recent conversation available."}

IMAGE CONTEXT:
{image_context or "No usable visual context was produced."}
"""

        response = provider.generate(
            prompt,
            temperature=0.5,
            max_tokens=120,
        )

        response = clean_text(response)

        if response:
            return response

    except Exception:
        logger.exception(
            "Image-only natural response failed."
        )

    return (
        "Thanks for sharing the photo. "
        "If there's anything you'd like me to know about it, "
        "feel free to tell me."
    )


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
    customer_id=None,
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
            customer_id=customer_id,
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
    Translate customer-facing order preview text.

    Structured restaurant order previews use deterministic
    translations for supported languages so predictable menu
    names and system phrases do not require an LLM call.
    """

    if not language or not lines:
        return lines

    language_name = str(
        language
    ).strip()

    if not language_name:
        return lines

    # --------------------------------------------------------
    # ENGLISH
    # --------------------------------------------------------

    if language_name.lower() in (
        "english",
        "en",
        "en-us",
        "en-gb",
    ):
        return lines

    # --------------------------------------------------------
    # FRENCH
    # --------------------------------------------------------

    if language_name.lower() in (
        "french",
        "fr",
        "fr-fr",
    ):

        translations = {
            "Your order:": "Votre commande :",
            "Your updated order:": "Votre commande mise à jour :",
            "Please confirm your order.": (
                "Veuillez confirmer votre commande."
            ),
            "Total:": "Total :",

            "Lemonade": "Limonade",

            "Classic Margherita Pizza": (
                "Pizza Margherita Classique"
            ),

            "Loaded Chicken & Cheese Pizza": (
                "Pizza Poulet & Fromage Garnie"
            ),

            "Crispy French Fries": (
                "Frites Croquantes"
            ),

            "Assorted Club Sandwich": (
                "Club Sandwich Varié"
            ),

            "Classic Vanilla Cornetto Cone": (
                "Cornetto Classique à la Vanille"
            ),

            "Chocolate Overload Scoop / Cone": (
                "Tasse / Corne de Chocolat Surchargé"
            ),

            "Strawberry Fruit Gelato": (
                "Gelato aux Fraises"
            ),

            "Caramel Crunch Sundae": (
                "Sundae Caramel Croquant"
            ),

            "Mixed Fruit-Flavored Ice Cream Cup": (
                "Tasse de Glace aux Fruits Mélangés"
            ),

            "Signature Lebanese Shawarma": (
                "Shawarma Libanais Signature"
            ),
        }

        translated_lines = []

        for line in lines:

            translated_line = str(line)

            # ------------------------------------------------
            # COMPLETE SYSTEM PHRASES
            # ------------------------------------------------

            if translated_line in translations:

                translated_line = translations[
                    translated_line
                ]

            else:

                # ------------------------------------------------
                # MENU ITEM NAMES
                # ------------------------------------------------

                for english_name, french_name in translations.items():

                    translated_line = translated_line.replace(
                        f"*{english_name}*",
                        f"*{french_name}*",
                    )

                    translated_line = translated_line.replace(
                        english_name,
                        french_name,
                    )

                # ------------------------------------------------
                # TOTAL LINE
                # ------------------------------------------------

                if translated_line.startswith("Total:"):

                    translated_line = translated_line.replace(
                        "Total:",
                        "Total :",
                        1,
                    )

                # ------------------------------------------------
                # DYNAMIC COMPLEMENTARY SUGGESTIONS
                # ------------------------------------------------

                lower_line = translated_line.lower()

                drink_prefix = "you could also add "
                drink_suffixes = (
                    " if you'd like a drink.",
                    " if you would like a drink.",
                    " if you'd like something to drink.",
                    " if you would like something to drink.",
                )

                if lower_line.startswith(drink_prefix):

                    for suffix in drink_suffixes:

                        if lower_line.endswith(suffix):

                            item_name = translated_line[
                                len(drink_prefix):
                                len(translated_line) - len(suffix)
                            ].strip()

                            translated_line = (
                                "Vous pouvez également ajouter "
                                f"{item_name} si vous souhaitez une boisson."
                            )

                            break

                dessert_prefix = "you could also add "
                dessert_suffixes = (
                    " if you'd like something sweet.",
                    " if you would like something sweet.",
                )

                lower_line = translated_line.lower()

                if lower_line.startswith(dessert_prefix):

                    for suffix in dessert_suffixes:

                        if lower_line.endswith(suffix):

                            item_name = translated_line[
                                len(dessert_prefix):
                                len(translated_line) - len(suffix)
                            ].strip()

                            translated_line = (
                                "Vous pouvez également ajouter "
                                f"{item_name} si vous souhaitez quelque chose de sucré."
                            )

                            break

                # ------------------------------------------------
                # ADD-TO-ORDER QUESTION
                # ------------------------------------------------

                lower_line = translated_line.lower()

                question_prefix = "would you like to add "
                question_suffix = " to your order?"

                if (
                    lower_line.startswith(question_prefix)
                    and lower_line.endswith(question_suffix)
                ):

                    item_name = translated_line[
                        len(question_prefix):
                        len(translated_line) - len(question_suffix)
                    ].strip()

                    translated_line = (
                        "Souhaitez-vous ajouter "
                        f"{item_name} à votre commande ?"
                    )

            translated_lines.append(
                translated_line
            )

        return translated_lines

    # --------------------------------------------------------
    # OTHER LANGUAGES
    # --------------------------------------------------------
    #
    # Do not introduce an LLM call here for structured order
    # responses. Returning the canonical text is safer than
    # potentially changing quantities, prices, or line structure.

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

        # --------------------------------------------------------
        # BUILD CONFIRMATION RESPONSE
        # --------------------------------------------------------

        if language == "French":

            lines = [
                (
                    f"Commande #{confirmed_order_id} "
                    "a été créée avec succès."
                ),
                "",
                f"Total : {confirmed_total_text}",
            ]

            if payment.get("demo") or payment.get("status") == "Paid":
                lines.extend([
                    "",
                    "Paiement reçu. Votre commande est confirmée."
                ])
            else:
                lines.extend([
                    "",
                    "Le paiement est requis pour finaliser "
                    "votre commande."
                ])

            if checkout_url:
                lines.extend([
                    "",
                    f"Payez ici : {checkout_url}",
                ])

        else:

            lines = [
                (
                    f"Order #{confirmed_order_id} has been "
                    "created successfully."
                ),
                "",
                f"Total: {confirmed_total_text}",
            ]

            if payment.get("demo") or payment.get("status") == "Paid":
                lines.extend([
                    "",
                    "Payment received. Your order is confirmed."
                ])
            else:
                lines.extend([
                    "",
                    "Payment is required to complete your order."
                ])

            if checkout_url:
                lines.extend([
                    "",
                    f"Pay here: {checkout_url}",
                ])

        return make_response(
            "\n".join(lines)
        )

        return make_response(
            "\n".join(lines)
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
def _translate_menu_text_cached(menu_text, language_name):
    """
    Deterministically translate known restaurant menu text.

    This avoids an unnecessary LLM call for the common supported languages
    while preserving prices, emojis, Markdown, and menu structure.
    """
    if language_name == "French":
        translations = {
            "Our Menu": "Notre Menu",
            "Fast Food & Restaurant Specialties": "Fast Food & Spécialités de Restaurant",
            "Italian Gelato & Desserts": "Glaces italiennes & Desserts",
            "Italian Ice Creams & Desserts": "Glaces italiennes & Desserts",
            "Drinks": "Boissons",
            "Beverages": "Boissons",
            "Drink": "Boisson",
            "Desserts": "Desserts",
            "Dessert": "Dessert",
            "Burgers": "Burgers",
            "Burger": "Burger",
            "Pizzas": "Pizzas",
            "Pizza": "Pizza",
            "Chicken": "Poulet",
            "Sandwiches": "Sandwichs",
            "Sandwich": "Sandwich",
            "Fries": "Frites",
            "Sides": "Accompagnements",
            "Rice": "Riz",
            "Salads": "Salades",
            "Salad": "Salade",
            "Snacks": "Snacks",
            "Snack": "Snack",
            "Pasta": "Pâtes",
            "Fish": "Poisson",
            "Seafood": "Fruits de mer",
            "Meat": "Viande",
            "Breakfast": "Petit-déjeuner",
            "Other": "Autres",
            "Lemonade": "Limonade",
            "Classic Margherita Pizza": "Pizza Margherita Classique",
            "Loaded Chicken & Cheese Pizza": "Pizza Poulet & Fromage Garnie",
            "Crispy French Fries": "Frites Croquantes",
            "Assorted Club Sandwich": "Club Sandwich Varié",
            "Classic Vanilla Cornetto Cone": "Cornetto Classique à la Vanille",
            "Chocolate Overload Scoop / Cone": "Glace au Chocolat Intense / Cornet",
            "Strawberry Fruit Gelato": "Gelato aux Fraises",
            "Caramel Crunch Sundae": "Sundae Caramel Croquant",
            "Mixed Fruit-Flavored Ice Cream Cup": "Coupe de Glace aux Fruits Mélangés",
            "Signature Lebanese Shawarma": "Shawarma Libanais Signature",
            "What would you like to order?": "Que souhaitez-vous commander ?",
        }

        translated = menu_text

        for source, target in sorted(
            translations.items(),
            key=lambda pair: len(pair[0]),
            reverse=True,
        ):
            translated = translated.replace(source, target)

        return translated

    return menu_text


def translate_menu_for_customer(lines, language, provider=None):
    """
    Translate a structured menu while preserving its exact prices,
    emojis, Markdown, and item count.

    French is handled deterministically because the restaurant's known
    menu vocabulary has stable translations. Other languages continue
    using the existing provider-based translation path.
    """
    if not lines:
        return lines

    if language == "English":
        return lines

    menu_text = "\n".join(lines)

    if language == "French":
        translated = _translate_menu_text_cached(
            menu_text,
            "French",
        )
        return translated.splitlines()

    language_name = clean_text(language) or "English"

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
            translated = provider.generate(prompt)

            if translated:
                return translated.splitlines()

        except Exception:
            pass

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
    # English
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

    # French
    "ajoute ",
    "ajouter ",
    "ajoute à ma commande",
    "ajouter à ma commande",
    "ajoute a ma commande",
    "ajouter a ma commande",
    "je veux ajouter",
    "je voudrais ajouter",
    "je souhaite ajouter",
    "enleve ",
    "enlever ",
    "enlève ",
    "retire ",
    "retirer ",
    "supprime ",
    "supprimer ",
    "enlève de ma commande",
    "retire de ma commande",
    "supprime de ma commande",
    "change ma commande",
    "modifier ma commande",
    "modifie ma commande",
    "change la quantité",
    "change la quantite",
    "mets ",
    "met ",
    "mets-en ",
    "mets en ",
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
    # English
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

    # French
    "je veux ",
    "je voudrais ",
    "j'aimerais ",
    "j aimerais ",
    "jaimerais ",
    "je souhaite ",
    "je vais prendre ",
    "je prends ",
    "je peux avoir ",
    "je peux commander ",
    "je voudrais commander ",
    "je veux commander ",
    "je souhaite commander ",
    "donne moi ",
    "donnez moi ",
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
    # English
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

    # French
    "limonade",
    "limonades",
    "frites",
    "sandwich",
    "poulet",
    "glace",
    "boisson",
    "boissons",
    "café",
    "cafe",
    "jus",
    "eau",
    "gâteau",
    "gateau",
    "riz",
    "pâtes",
    "pates",
)

FAST_FALLBACK_FOOD_PATTERNS = (
    # English
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

    # French
    "limonade",
    "limonades",
    "frites",
    "sandwich",
    "poulet",
    "glace",
    "boisson",
    "boissons",
    "pizza",
    "shawarma",
    "burger",
    "jus",
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
# CUSTOMER FEEDBACK / RELATIONSHIP INTELLIGENCE
# ============================================================

FAST_FEEDBACK_PHRASES = (
    "was delicious",
    "was amazing",
    "was great",
    "was really good",
    "was so good",
    "was nice",
    "was really nice",
    "was lovely",
    "was really lovely",
    "was excellent",
    "was really excellent",
    "was fantastic",
    "was really fantastic",
    "was incredible",
    "was really incredible",
    "was wonderful",
    "was really wonderful",
    "was tasty",
    "loved it",
    "love it",
    "really enjoyed",
    "really liked",
    "i enjoyed",
    "i liked it",
    "i loved",
    "very good",
    "so delicious",
    "really delicious",
    "thank you for the food",
    "thanks for the food",
    "food was good",
    "food was amazing",
    "food was delicious",
    "meal was good",
    "meal was amazing",
    "meal was delicious",
    "that was delicious",
    "that was amazing",
    "that was really good",
    "that was really nice",
    "that was lovely",
    "that hit the spot",
    "you guys nailed it",
    "you nailed it",
    "absolutely loved it",
    "i absolutely loved it",
    "the food was incredible",
    "the food was excellent",
    "the food was fantastic",
    "the meal was incredible",
    "the meal was excellent",
    "the meal was fantastic",
    "i really enjoyed this",
    "i really enjoyed that",
)

FAST_COMPLAINT_PHRASES = (
    "was cold",
    "food was cold",
    "meal was cold",
    "was not good",
    "wasn't good",
    "wasnt good",
    "not good",
    "not happy",
    "unhappy",
    "disappointed",
    "disappointing",
    "terrible",
    "bad food",
    "food was bad",
    "meal was bad",
    "wrong order",
    "missing item",
    "missing items",
    "something was missing",
    "late delivery",
    "arrived late",
    "arrived cold",
    "too cold",
)


def classify_customer_feedback(message):

    text = normalize_text(message)

    if not text:
        return None

    if any(
        phrase in text
        for phrase in FAST_COMPLAINT_PHRASES
    ):
        return "negative"

    if any(
        phrase in text
        for phrase in FAST_FEEDBACK_PHRASES
    ):
        return "positive"

    return None


def classify_customer_feedback_ai(
    provider,
    message,
):
    """
    Classify feedback sentiment only when the fast local
    detector cannot determine it.

    Returns:
        positive / negative / neutral
    """

    prompt = f"""
Classify the sentiment of this restaurant customer feedback.

CUSTOMER MESSAGE:
{message}

Return exactly ONE category:

positive
negative
neutral

positive:
The customer is clearly praising, enjoying, loving, or expressing
strong satisfaction with the food, restaurant, service, or experience.

negative:
The customer is clearly unhappy, disappointed, complaining, or
reporting a problem with the food, restaurant, service, delivery,
or experience.

neutral:
The message is feedback but the sentiment is unclear or mixed.

Return ONLY the category name.
"""

    try:

        result = provider.generate(
            prompt,
            temperature=0,
            max_tokens=20,
        )

        result = normalize_text(
            result
        ).strip(
            " .,!?:;"
        )

        if result in (
            "positive",
            "negative",
            "neutral",
        ):
            return result

    except Exception:

        logger.exception(
            "AI feedback sentiment classification failed."
        )

    return "neutral"


def get_recent_relevant_order(
    customer_id,
    business_id,
):

    """
    Return the most relevant recent customer order.

    Priority:
    1. Completed / Delivered / Paid orders
    2. Most recent non-cancelled order
    """

    completed_statuses = (
        "Completed",
        "Delivered",
        "Paid",
    )

    completed_order = (
        Order.query
        .filter(
            Order.customer_id == customer_id,
            Order.business_id == business_id,
            (
                Order.status.in_(completed_statuses)
                | (
                    db.func.lower(
                        db.func.coalesce(
                            Order.payment_status,
                            ""
                        )
                    ) == "paid"
                )
            ),
        )
        .order_by(
            Order.id.desc()
        )
        .first()
    )

    if completed_order:
        return completed_order

    cancelled_status = db.func.lower(
        db.func.coalesce(
            Order.status,
            ""
        )
    ).in_(
        (
            "cancelled",
            "canceled",
        )
    )

    return (
        Order.query
        .filter(
            Order.customer_id == customer_id,
            Order.business_id == business_id,
            ~cancelled_status,
        )
        .order_by(
            Order.id.desc()
        )
        .first()
    )



def save_customer_interaction(
    customer,
    business_id,
    message,
    interaction_type,
    sentiment=None,
    order=None,
    metadata=None,
):

    interaction = CustomerInteraction(
        customer_id=customer.id,
        business_id=business_id,
        order_id=(
            order.id
            if order
            else None
        ),
        interaction_type=interaction_type,
        sentiment=sentiment,
        message=message,
        metadata_json=(
            json.dumps(
                metadata,
                ensure_ascii=False
            )
            if metadata
            else None
        ),
    )

    db.session.add(
        interaction
    )

    return interaction


def build_feedback_order_context(order):

    if not order:
        return (
            "No recent completed order was found."
        )

    items = getattr(
        order,
        "items",
        []
    )

    lines = []

    for item in items:
        lines.append(
            f"{item.name} × {item.quantity}"
        )

    return (
        "\n".join(lines)
        or "Order found, but item details are unavailable."
    )


def generate_feedback_response(
    provider,
    business_id,
    phone,
    message,
    language,
    history,
    customer,
    sentiment,
    order=None,
    image_context=None,
):

    order_context = build_feedback_order_context(
        order
    )

    history_text = format_history(
        history
    )

    prompt = f"""
You are the relationship assistant for a restaurant.

The customer is sharing feedback about their restaurant experience.

Your job is to make the customer feel genuinely appreciated and increase
the likelihood that they will want to return, without sounding like an ad.

CUSTOMER MESSAGE:
{message}

CUSTOMER LANGUAGE:
{language}

FEEDBACK SENTIMENT:
{sentiment}

RECENT ORDER:
{order_context}

RECENT CONVERSATION:
{history_text}

IMAGE CONTEXT:
{image_context or "No image was provided."}

POSITIVE FEEDBACK BEHAVIOR:
- Thank the customer naturally.
- If the recent order contains a specific item and it is relevant, you may
  mention that item naturally.
- Do not list every item unless the customer is clearly discussing the full order.
- Reinforce the positive experience without exaggerating.
- A warm invitation to return is encouraged when natural.
- Do not immediately push another sale.
- Do not invent promotions, discounts, coupons, or offers.
- Do not sound like a marketing campaign.

NEGATIVE FEEDBACK BEHAVIOR:
- Acknowledge the customer's experience.
- Apologize when appropriate.
- Focus on understanding and resolving the problem.
- Do not upsell.
- Do not invent compensation, refunds, or staff actions.

FACTUALITY:
- Only use information contained in the recent order, recent conversation,
  image context, and restaurant context supplied by the application.
- Never invent ingredients, prices, availability, order status, or image details.
- Do not claim the restaurant did something unless the application has confirmed it.

STYLE:
- Warm, natural, human, and concise.
- Do not repeat the same phrase every time.
- Do not sound robotic.
- Do not mention AI, tools, APIs, prompts, models, Flask, or backend systems.
- Reply only with the customer-facing message.
- Reply in the customer's language.

Write the response now.
"""

    response = provider.generate(
        prompt,
        temperature=0.65,
        max_tokens=180,
    )

    return (
        response.strip()
        if response
        else customer_response(
            "Thank you for your feedback. "
            "We really appreciate you taking the time to let us know.",
            message,
        )
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
    # CANCEL ORDER
    # --------------------------------------------------------

    if contains_any(
        FAST_CANCEL_PHRASES
    ):
        return "cancel_order"

    # --------------------------------------------------------
    # CONTEXTUAL CONVERSATION OVERRIDES
    # --------------------------------------------------------

    conversational_phrases = (
        "i was just checking the menu",
        "i was just checking",
        "just checking the menu",
        "just checking",
        "i'm just checking the menu",
        "im just checking the menu",
        "i am just checking the menu",
        "i'm just browsing",
        "im just browsing",
        "i am just browsing",
        "just browsing",
        "i was just browsing",
        "i'm only browsing",
        "im only browsing",
        "i am only browsing",
        "just looking",
        "i was just looking",
        "i'm still deciding",
        "im still deciding",
        "i am still deciding",
        "still deciding what to get",
        "still deciding",
        "i haven't decided",
        "i havent decided",
        "not sure what to get",
        "i'm not sure what to get",
        "im not sure what to get",
    )

    if any(
        phrase in text
        for phrase in conversational_phrases
    ):
        return "chat"

    # --------------------------------------------------------
    # CUSTOMER FEEDBACK / COMPLAINT
    # --------------------------------------------------------

    feedback_sentiment = classify_customer_feedback(
        message
    )

    if feedback_sentiment:
        return "feedback"

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

    # Common natural-language additions/removals should stay
    # on the fast local path instead of falling through to the LLM.
    modify_prefixes = (
        "add ",
        "add a ",
        "add an ",
        "add some ",
        "add to my order ",
        "i want to add ",
        "i would like to add ",
        "i'd like to add ",
        "please add ",
        "actually add ",
        "remove ",
        "remove a ",
        "remove an ",
        "take off ",
        "take out ",
        "delete ",
        "actually remove ",
        "actually take off ",
        "please remove ",
        "enlève ",
        "enleve ",
        "retire ",
        "supprime ",
        "ajoute ",
        "ajouter ",
        "retirer ",
        "supprimer ",
    )

    if text.startswith(modify_prefixes):
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
            "feedback",
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

    Supports:
    - "make it 2"
    - "make it two"
    - "make it 2 lemonades"
    - "make it two lemonades"
    - "change it to 3 pizzas"
    - "set 4 lemonades"
    - French equivalents such as "mets 3 limonades"

    Item-qualified requests modify only the specified item.
    """

    from models.customer import Customer
    from models.pending_order import PendingOrder

    text = normalize_text(
        message
    )

    if not text:
        return None

    # --------------------------------------------------------
    # QUANTITY / ITEM FOLLOW-UP PREFIXES
    # --------------------------------------------------------

    quantity_prefixes = (
        "make it ",
        "make that ",
        "change it to ",
        "change it ",
        "set ",
        "mets ",
        "met ",
        "mets en ",
        "mets-en ",
        "change la quantité de ",
        "change la quantite de ",
    )

    # --------------------------------------------------------
    # DETERMINE WHETHER THIS IS A QUANTITY FOLLOW-UP
    # --------------------------------------------------------

    matched_prefix = None

    for prefix in quantity_prefixes:

        if text.startswith(prefix):
            matched_prefix = prefix
            break

    # Keep support for the original quantity-only forms.
    if matched_prefix is None:

        quantity_only_match = re.fullmatch(
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

        if not quantity_only_match:
            return None

        quantity_text = (
            quantity_only_match.group(0)
            .strip()
            .split()[-1]
        )

        item_text = ""

    else:

        remainder = (
            text[len(matched_prefix):]
            .strip()
        )

        if not remainder:
            return None

        item_text = remainder

        quantity_match = re.match(
            r"^(?P<quantity>\d+|one|two|three|four|five)"
            r"(?:\s+please)?"
            r"(?:\s+)?"
            r"(?P<item>.*)$",
            remainder,
        )

        if not quantity_match:
            return None

        quantity_text = (
            quantity_match.group(
                "quantity"
            )
        )

        item_text = (
            quantity_match.group(
                "item"
            )
            .strip()
        )

    # --------------------------------------------------------
    # QUANTITY WORDS
    # --------------------------------------------------------

    word_quantities = {
        "one": 1,
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

    try:

        if quantity_text.isdigit():

            quantity = max(
                1,
                int(quantity_text),
            )

        else:

            quantity = word_quantities.get(
                quantity_text
            )

    except (
        TypeError,
        ValueError,
    ):

        return None

    if quantity is None:
        return None

    # --------------------------------------------------------
    # FIND CUSTOMER
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # FIND PENDING PREVIEW
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # LOAD ITEMS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # FIND TARGET
    # --------------------------------------------------------

    target = None

    if item_text:

        # Remove polite words that can appear after the item.
        item_text = re.sub(
            r"\s+please$",
            "",
            item_text,
        ).strip()

        # Resolve the requested item through the existing
        # fast menu extractor.
        from services.ai.order_extractor import extract_order

        extraction_message = (
            f"je voudrais {quantity_text} {item_text}"
        )

        extracted = extract_order(
            business_id,
            extraction_message,
        )

        requested_items = extracted.get(
            "items",
            [],
        )

        if not requested_items:
            return None

        requested = requested_items[0]

        requested_name = clean_text(
            requested.get("name")
            or ""
        )

        if not requested_name:
            return None

        for item in items:

            if not isinstance(
                item,
                dict,
            ):
                continue

            item_name = clean_text(
                item.get("name")
                or ""
            )

            if (
                item_name.lower()
                == requested_name.lower()
            ):
                target = item
                break

    else:

        # Original behavior:
        # a quantity-only follow-up applies to the
        # most recently added item.
        target = items[-1]

    if not isinstance(
        target,
        dict,
    ):

        return None

    # --------------------------------------------------------
    # UPDATE QUANTITY
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RECALCULATE TOTAL
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    preview.items_json = json.dumps(
        items,
        ensure_ascii=False,
    )

    preview.total_price = float(
        total
    )

    db.session.commit()

    # --------------------------------------------------------
    # BUILD RESPONSE
    # --------------------------------------------------------

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

    lines = translate_order_preview_for_customer(
        lines,
        CUSTOMER_LANGUAGE.get(),
        provider=None,
    )

    return {
        "type": "response",
        "message": customer_response(
            "\n".join(lines),
            message,
        ),
    }

def modify_pending_order(
    business_id,
    phone,
    message,
    language,
):
    """
    Modify an existing pending order preview locally.

    Supports straightforward add-item and remove-item requests
    without creating a real Order or calling the LLM.
    """

    from models.customer import Customer
    from models.pending_order import PendingOrder
    from database.db import db
    from services.ai.order_extractor import extract_order

    text = normalize_text(
        message
    )

    # --------------------------------------------------------
    # DETECT MODIFICATION TYPE
    # --------------------------------------------------------

    add_markers = (
        "ajoute ",
        "ajouter ",
        "ajoute-moi ",
        "ajoute moi ",
        "ajouter-moi ",
        "ajouter moi ",
        "je veux ajouter ",
        "je voudrais ajouter ",
        "je souhaite ajouter ",
        "add ",
        "add to my order ",
        "i want to add ",
        "i would like to add ",
    )

    remove_markers = (
        "enlève ",
        "enleve ",
        "enlever ",
        "retire ",
        "retirer ",
        "supprime ",
        "supprimer ",
        "enlève de ma commande ",
        "enleve de ma commande ",
        "retire de ma commande ",
        "supprime de ma commande ",
        "remove ",
        "actually remove ",
        "please remove ",
        "remove from my order ",
        "actually remove from my order ",
        "take off ",
        "actually take off ",
        "take it off ",
        "delete ",
    )

    set_quantity_markers = (
        "mets ",
        "met ",
        "mets en ",
        "mets-en ",
        "change ",
        "change la quantité de ",
        "change la quantite de ",
        "set ",
        "make ",
    )

    is_add = any(
        text.startswith(marker)
        for marker in add_markers
    )

    is_remove = any(
        text.startswith(marker)
        for marker in remove_markers
    )

    is_set_quantity = any(
        text.startswith(marker)
        for marker in set_quantity_markers
    )

    if not is_add and not is_remove and not is_set_quantity:
        return None

    # --------------------------------------------------------
    # FIND CUSTOMER
    # --------------------------------------------------------

    customer = Customer.query.filter_by(
        business_id=business_id,
        phone=phone,
    ).first()

    if not customer:
        return None

    # --------------------------------------------------------
    # FIND PENDING PREVIEW
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # LOAD EXISTING PREVIEW
    # --------------------------------------------------------

    try:

        items = json.loads(
            preview.items_json
        )

    except (
        json.JSONDecodeError,
        TypeError,
    ):

        return {
            "type": "response",
            "message": customer_response(
                "I couldn't read your pending order.",
                message,
            ),
        }

    if not isinstance(items, list):
        items = []

    # ========================================================
    # ADD ITEM
    # ========================================================

    if is_add:

        extracted = extract_order(
            business_id,
            message,
        )

        new_items = extracted.get(
            "items",
            [],
        )

        if not new_items:
            return {
                "type": "response",
                "message": customer_response(
                    "I couldn't identify the item you want to add.",
                    message,
                ),
            }

        for new_item in new_items:

            if not isinstance(
                new_item,
                dict,
            ):
                continue

            new_name = clean_text(
                new_item.get("name")
                or ""
            )

            if not new_name:
                continue

            try:

                new_quantity = max(
                    1,
                    int(
                        new_item.get(
                            "quantity",
                            1,
                        )
                        or 1
                    ),
                )

                new_price = float(
                    new_item.get(
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

            existing = None

            for item in items:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                existing_name = clean_text(
                    item.get("name")
                    or ""
                )

                if (
                    existing_name.lower()
                    == new_name.lower()
                ):

                    existing = item
                    break

            if existing:

                try:

                    current_quantity = max(
                        1,
                        int(
                            existing.get(
                                "quantity",
                                1,
                            )
                            or 1
                        ),
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    current_quantity = 1

                existing["quantity"] = (
                    current_quantity
                    + new_quantity
                )

                existing["price"] = new_price

                existing["subtotal"] = (
                    new_price
                    * existing["quantity"]
                )

            else:

                items.append({
                    "name": new_name,
                    "quantity": new_quantity,
                    "price": new_price,
                    "subtotal": (
                        new_price
                        * new_quantity
                    ),
                })

    # ========================================================
    # REMOVE ITEM
    # ========================================================

    elif is_remove:

        # Convert a removal request into a normal order-style
        # phrase so the existing fast extractor can identify
        # the menu item and quantity locally.
        remove_prefixes = (
            "enlève ",
            "enleve ",
            "enlever ",
            "retire ",
            "retirer ",
            "supprime ",
            "supprimer ",
            "enlève de ma commande ",
            "enleve de ma commande ",
            "retire de ma commande ",
            "supprime de ma commande ",
            "remove ",
            "remove from my order ",
            "take off ",
            "take it off ",
            "delete ",
        )

        extraction_message = text

        for prefix in remove_prefixes:

            if extraction_message.startswith(prefix):
                extraction_message = (
                    extraction_message[len(prefix):].strip()
                )
                break

        if not extraction_message:
            return {
                "type": "response",
                "message": customer_response(
                    "I couldn't identify the item you want to remove.",
                    message,
                ),
            }

        extraction_message = (
            f"je voudrais {extraction_message}"
        )

        extracted = extract_order(
            business_id,
            extraction_message,
        )

        requested_items = extracted.get(
            "items",
            [],
        )

        if not requested_items:

            return {
                "type": "response",
                "message": customer_response(
                    "I couldn't identify the item you want to remove.",
                    message,
                ),
            }

        for requested in requested_items:

            if not isinstance(
                requested,
                dict,
            ):
                continue

            requested_name = clean_text(
                requested.get("name")
                or ""
            )

            if not requested_name:
                continue

            try:

                remove_quantity = max(
                    1,
                    int(
                        requested.get(
                            "quantity",
                            1,
                        )
                        or 1
                    ),
                )

            except (
                TypeError,
                ValueError,
            ):

                remove_quantity = 1

            target = None

            for item in items:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                item_name = clean_text(
                    item.get("name")
                    or ""
                )

                if (
                    item_name.lower()
                    == requested_name.lower()
                ):

                    target = item
                    break

            if not target:

                return {
                    "type": "response",
                    "message": customer_response(
                        (
                            f"I couldn't find "
                            f"'{requested_name}' "
                            "in your pending order."
                        ),
                        message,
                    ),
                }

            try:

                current_quantity = max(
                    1,
                    int(
                        target.get(
                            "quantity",
                            1,
                        )
                        or 1
                    ),
                )

            except (
                TypeError,
                ValueError,
            ):

                current_quantity = 1

            remaining_quantity = (
                current_quantity
                - remove_quantity
            )

            if remaining_quantity <= 0:

                items.remove(
                    target
                )

            else:

                target["quantity"] = (
                    remaining_quantity
                )

                try:

                    price = float(
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

                    price = 0.0

                target["subtotal"] = (
                    price
                    * remaining_quantity
                )

    # ========================================================
    # SET QUANTITY
    # ========================================================

    elif is_set_quantity:

        extraction_message = text

        for prefix in set_quantity_markers:

            if extraction_message.startswith(prefix):
                extraction_message = (
                    extraction_message[len(prefix):].strip()
                )
                break

        if not extraction_message:
            return {
                "type": "response",
                "message": customer_response(
                    "I couldn't identify the item and quantity.",
                    message,
                ),
            }

        extraction_message = (
            f"je voudrais {extraction_message}"
        )

        extracted = extract_order(
            business_id,
            extraction_message,
        )

        requested_items = extracted.get(
            "items",
            [],
        )

        if not requested_items:
            return {
                "type": "response",
                "message": customer_response(
                    "I couldn't identify the item and quantity.",
                    message,
                ),
            }

        for requested in requested_items:

            if not isinstance(
                requested,
                dict,
            ):
                continue

            requested_name = clean_text(
                requested.get("name")
                or ""
            )

            if not requested_name:
                continue

            try:

                requested_quantity = max(
                    1,
                    int(
                        requested.get(
                            "quantity",
                            1,
                        )
                        or 1
                    ),
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            target = None

            for item in items:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                item_name = clean_text(
                    item.get("name")
                    or ""
                )

                if (
                    item_name.lower()
                    == requested_name.lower()
                ):
                    target = item
                    break

            if not target:

                return {
                    "type": "response",
                    "message": customer_response(
                        (
                            f"I couldn't find "
                            f"'{requested_name}' "
                            "in your pending order."
                        ),
                        message,
                    ),
                }

            try:

                price = float(
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

                price = 0.0

            target["quantity"] = requested_quantity

            target["price"] = price

            target["subtotal"] = (
                price
                * requested_quantity
            )

    # --------------------------------------------------------
    # EMPTY ORDER CHECK
    # --------------------------------------------------------

    if not items:

        return {
            "type": "response",
            "message": customer_response(
                "Your pending order is now empty.",
                message,
            ),
        }

    # --------------------------------------------------------
    # RECALCULATE TOTAL
    # --------------------------------------------------------

    total = 0.0

    for item in items:

        if not isinstance(
            item,
            dict,
        ):
            continue

        try:

            quantity = max(
                1,
                int(
                    item.get(
                        "quantity",
                        1,
                    )
                    or 1
                ),
            )

            price = float(
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

        item["quantity"] = quantity
        item["price"] = price
        item["subtotal"] = (
            price * quantity
        )

        total += item["subtotal"]

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    preview.items_json = json.dumps(
        items,
        ensure_ascii=False,
    )

    preview.total_price = float(
        total
    )

    db.session.commit()

    # --------------------------------------------------------
    # BUILD RESPONSE
    # --------------------------------------------------------

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

        quantity = item.get(
            "quantity",
            1,
        )

        lines.append(
            f"• *{name}* × {quantity}"
        )

    lines.extend([
        "",
        f"Total: {total:,.0f} FCFA",
        "",
        "Please confirm your order.",
    ])

    lines = translate_order_preview_for_customer(
        lines,
        language,
        provider=None,
    )

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
    image_context=None,
):

    message = clean_text(
        message
    )

    # ========================================================
    # CUSTOMER LANGUAGE
    # ========================================================

    detected_language = detect_customer_language(
        message,
        fallback=language,
    )

    language = detected_language

    CUSTOMER_LANGUAGE.set(
        language
    )

    if not message:

        return {
            "type": "response",
            "message": customer_response(
                "I'm here to help. "
                "What would you like to know?",
                message,
            ),
        }

    history = history or []

    # ========================================================
    # EXPLICIT CUSTOMER PREFERENCE MEMORY
    # ========================================================

    try:

        customer = Customer.query.filter_by(
            phone=phone,
            business_id=business_id,
        ).first()

        if customer:

            learn_explicit_customer_preference(
                customer=customer,
                business_id=business_id,
                message=message,
            )

            db.session.commit()

    except Exception:

        db.session.rollback()

        logger.exception(
            "Explicit customer preference learning failed."
        )

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
    # IMAGE-ONLY RELATIONSHIP RESPONSE
    # ========================================================

    normalized_message = normalize_text(
        message
    )

    if image_context and normalized_message == "customer sent a photo":

        customer = Customer.query.filter_by(
            phone=phone,
            business_id=business_id,
        ).first()

        recent_order = None

        if customer:
            recent_order = get_recent_relevant_order(
                customer.id,
                business_id,
            )

            try:
                save_customer_interaction(
                    customer=customer,
                    business_id=business_id,
                    message=message,
                    interaction_type="image",
                    sentiment=None,
                    order=recent_order,
                )

                db.session.commit()

            except Exception:
                db.session.rollback()
                logger.exception(
                    "Failed to save customer image interaction."
                )

        return {
            "type": "response",
            "message": customer_response(
                generate_image_only_response(
                    provider=get_provider(),
                    business_id=business_id,
                    phone=phone,
                    language=language,
                    history=history,
                    image_context=image_context,
                    recent_order=recent_order,
                ),
                message,
            ),
        }

    # ========================================================
    # FAST GREETING
    # ========================================================

    normalized_message = normalize_text(
        message
    )

    greeting_phrases = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "bonjour",
        "bonsoir",
        "salut",
        "hola",
        "olá",
        "ciao",
        "hallo",
    }

    if normalized_message in greeting_phrases:
        # Use the same restaurant context already used by the AI response layer.
        # This keeps greetings tenant-specific without hardcoding a restaurant name.
        greeting_context = get_restaurant_context(business_id)
        business_data = greeting_context.get("business", {})
        restaurant_name = (
            business_data.get("name")
            or "our restaurant"
        )

        if language == "French":
            greeting_variants = [
                f"Bienvenue chez {restaurant_name} ! Nous sommes ravis de vous accueillir. Que puis-je vous servir aujourd’hui ?",
                f"Bienvenue chez {restaurant_name} ! C’est un plaisir de vous recevoir. Qu’est-ce qui vous ferait plaisir aujourd’hui ?",
                f"Bonjour et bienvenue chez {restaurant_name} ! Nous sommes prêts à vous servir. Que puis-je vous proposer ?",
            ]
        elif language == "Spanish":
            greeting_variants = [
                f"¡Bienvenido a {restaurant_name}! Nos alegra mucho recibirte. ¿Qué te gustaría disfrutar hoy?",
                f"¡Hola y bienvenido a {restaurant_name}! Estamos listos para atenderte. ¿Qué te apetece hoy?",
            ]
        elif language == "Portuguese":
            greeting_variants = [
                f"Bem-vindo ao {restaurant_name}! É um prazer receber você. O que gostaria de pedir hoje?",
                f"Olá e bem-vindo ao {restaurant_name}! Estamos prontos para atendê-lo. O que gostaria de experimentar?",
            ]
        elif language == "Italian":
            greeting_variants = [
                f"Benvenuto da {restaurant_name}! Siamo felici di averti qui. Cosa posso servirti oggi?",
                f"Ciao e benvenuto da {restaurant_name}! Siamo pronti a servirti. Cosa ti andrebbe oggi?",
            ]
        elif language == "German":
            greeting_variants = [
                f"Willkommen bei {restaurant_name}! Schön, dass Sie da sind. Was darf ich Ihnen heute anbieten?",
                f"Herzlich willkommen bei {restaurant_name}! Wir freuen uns, Sie zu bedienen. Was möchten Sie heute genießen?",
            ]
        else:
            greeting_variants = [
                f"Welcome to {restaurant_name}! We're delighted to have you here and ready to serve you. What can I get for you today?",
                f"Welcome to {restaurant_name}! It's lovely to have you here. What are you in the mood for today?",
                f"Welcome to {restaurant_name}! We're happy to have you with us. What can I help you find today?",
            ]

        # Rotate naturally instead of returning the same canned sentence every time.
        greeting_response = greeting_variants[
            sum(ord(char) for char in normalized_message) % len(greeting_variants)
        ]

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
    # CUSTOMER FEEDBACK / RELATIONSHIP RESPONSE
    # ========================================================

    if classification == "feedback":

        customer = Customer.query.filter_by(
            phone=phone,
            business_id=business_id,
        ).first()

        if not customer:

            return {
                "type": "response",
                "message": customer_response(
                    "Thank you for your feedback. "
                    "We really appreciate you taking "
                    "the time to let us know.",
                    message,
                ),
            }

        feedback_sentiment = classify_customer_feedback(
            message
        )

        if not feedback_sentiment:

            feedback_sentiment = (
                classify_customer_feedback_ai(
                    get_provider(),
                    message,
                )
            )

        recent_order = (
            get_recent_relevant_order(
                customer.id,
                business_id,
            )
        )

        try:

            save_customer_interaction(
                customer=customer,
                business_id=business_id,
                message=message,
                interaction_type="feedback",
                sentiment=feedback_sentiment,
                order=recent_order,
            )

            db.session.commit()

        except Exception:

            db.session.rollback()

            logger.exception(
                "Failed to save customer feedback interaction."
            )

        return {
            "type": "response",
            "message": generate_feedback_response(
                provider=get_provider(),
                business_id=business_id,
                phone=phone,
                message=message,
                language=language,
                history=history,
                customer=customer,
                sentiment=feedback_sentiment,
                order=recent_order,
                image_context=image_context,
            ),
        }

    # ========================================================
    # FAST CHAT RESPONSE
    # ========================================================

    if classification == "chat":

        # ----------------------------------------------------
        # KEEP VERY SIMPLE CHAT FAST AND LOCAL
        # ----------------------------------------------------

        chat_responses = {
            "hi": "Hi! How can I help you today?",
            "hello": "Hello! How can I help you today?",
            "hey": "Hey! How can I help you today?",
            "good morning": "Good morning! How can I help you today?",
            "good afternoon": "Good afternoon! How can I help you today?",
            "good evening": "Good evening! How can I help you today?",

            "thanks": (
                "You're welcome! "
                "Let me know if you need anything else."
            ),
            "thank you": (
                "You're welcome! "
                "Let me know if you need anything else."
            ),
            "thank": (
                "You're welcome! "
                "Let me know if you need anything else."
            ),

            "okay": "Alright! Let me know if you need anything.",
            "ok": "Alright! Let me know if you need anything.",
            "alright": "Alright! Let me know if you need anything.",

            "bonjour": "Bonjour ! Comment puis-je vous aider ?",
            "bonsoir": "Bonsoir ! Comment puis-je vous aider ?",
            "salut": "Salut ! Comment puis-je vous aider ?",
            "merci": "Avec plaisir ! N'hésitez pas si vous avez besoin de quoi que ce soit.",

            "hallo": "Hallo! Wie kann ich Ihnen helfen?",
            "guten morgen": "Guten Morgen! Wie kann ich Ihnen helfen?",
            "guten abend": "Guten Abend! Wie kann ich Ihnen helfen?",
            "danke": "Gerne! Lassen Sie mich wissen, wenn Sie etwas brauchen.",

            "hola": "¡Hola! ¿Cómo puedo ayudarte?",
            "buenos dias": "¡Buenos días! ¿Cómo puedo ayudarte?",
            "gracias": "¡De nada! Avísame si necesitas algo más.",

            "ola": "Olá! Como posso ajudar?",
            "bom dia": "Bom dia! Como posso ajudar?",
            "obrigado": "De nada! Avise-me se precisar de mais alguma coisa.",
            "obrigada": "De nada! Avise-me se precisar de mais alguma coisa.",

            "ciao": "Ciao! Come posso aiutarti?",
            "buongiorno": "Buongiorno! Come posso aiutarti?",
            "grazie": "Prego! Fammi sapere se hai bisogno di altro.",
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

        # ----------------------------------------------------
        # CONTEXTUAL CONVERSATION
        # ----------------------------------------------------

        if pending is None:

            pending = get_pending_order(
                business_id,
                phone,
            )

        customer = Customer.query.filter_by(
            phone=phone,
            business_id=business_id,
        ).first()

        return generate_natural_response(
            provider=get_provider(),
            business_id=business_id,
            phone=phone,
            message=message,
            language=language,
            history=history,
            pending=pending,
            customer_id=(
                customer.id
                if customer
                else None
            ),
        )

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

        # --------------------------------------------------------
        # CONTEXTUAL ORDER REFERENCE RESOLUTION
        # --------------------------------------------------------

        normalized_message = normalize_text(
            message
        )

        contextual_reference_phrases = (
            "the one",
            "that one",
            "this one",
            "the chicken one",
            "the pizza one",
            "that pizza",
            "this pizza",
            "i'll take that",
            "ill take that",
            "i will take that",
            "i'll take the",
            "ill take the",
            "i will take the",
            "give me that",
            "give me the",
            "i want that one",
            "i want the",
            "i'll have that",
            "ill have that",
            "i will have that",
        )

        is_contextual_reference = any(
            phrase in normalized_message
            for phrase in contextual_reference_phrases
        )

        if is_contextual_reference:

            try:

                from models.menu import Menu

                menu_items = (
                    Menu.query
                    .filter_by(
                        business_id=business_id,
                        available=True,
                    )
                    .order_by(
                        Menu.category.asc(),
                        Menu.name.asc(),
                    )
                    .all()
                )

                menu_context = "\n".join(
                    f"- {clean_text(item.name)}"
                    for item in menu_items
                    if item.name
                )

                history_text = format_history(
                    history
                )

                prompt = f"""
You resolve a customer's contextual restaurant order reference.

CUSTOMER MESSAGE:
{message}

RECENT CONVERSATION:
{history_text}

AVAILABLE MENU:
{menu_context}

TASK:
Determine whether the customer is referring to one specific menu item
based on the current message and recent conversation.

Return ONLY valid JSON:

{{
  "item": "Exact Menu Item Name",
  "quantity": 1
}}

RULES:
- Use ONLY an exact item from the available menu.
- Use the conversation to resolve references such as "the chicken one",
  "that pizza", "the one you mentioned", or "I'll take that one".
- Do not guess when the reference is genuinely ambiguous.
- If the reference cannot be resolved confidently, return:
  {{"item": null, "quantity": 0}}
- Quantity must be an integer.
"""

                response = get_provider().generate(
                    prompt,
                    temperature=0,
                    max_tokens=120,
                )

                response = clean_text(
                    response
                )

                resolved = None

                try:
                    resolved = json.loads(
                        response
                    )

                except (
                    json.JSONDecodeError,
                    TypeError,
                ):

                    start = response.find("{")
                    end = response.rfind("}")

                    if (
                        start != -1
                        and end != -1
                        and end > start
                    ):
                        try:
                            resolved = json.loads(
                                response[
                                    start:end + 1
                                ]
                            )
                        except (
                            json.JSONDecodeError,
                            TypeError,
                        ):
                            resolved = None

                if isinstance(
                    resolved,
                    dict,
                ):

                    resolved_name = clean_text(
                        resolved.get("item")
                        or ""
                    )

                    try:

                        resolved_quantity = max(
                            1,
                            int(
                                resolved.get(
                                    "quantity",
                                    1,
                                )
                                or 1
                            ),
                        )

                    except (
                        TypeError,
                        ValueError,
                    ):

                        resolved_quantity = 1

                    if resolved_name:

                        matched_menu_item = None

                        for menu_item in menu_items:

                            menu_name = clean_text(
                                menu_item.name
                                or ""
                            )

                            if (
                                menu_name.lower()
                                == resolved_name.lower()
                            ):

                                matched_menu_item = (
                                    menu_name
                                )
                                break

                        if matched_menu_item:

                            resolved_order_message = (
                                f"order "
                                f"{resolved_quantity} "
                                f"{matched_menu_item}"
                            )

                            result = create_order_preview(
                                business_id,
                                phone,
                                resolved_order_message,
                            )

                            if result.get("success"):

                                preview = result.get(
                                    "preview"
                                )

                                if preview:

                                    items = preview.get(
                                        "items",
                                        [],
                                    )

                                    total = preview.get(
                                        "total",
                                        0,
                                    )

                                    lines = [
                                        "Your order:",
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

                                        quantity = item.get(
                                            "quantity",
                                            1,
                                        )

                                        lines.append(
                                            f"• *{name}* × {quantity}"
                                        )

                                    suggestion = None

                                    if not recommendation_opted_out(message):
                                        suggestion = build_complementary_suggestion(
                                            business_id,
                                            items,
                                            customer_id=(
                                                customer.id
                                                if customer
                                                else None
                                            ),
                                        )

                                    if suggestion:
                                        logger.info(
                                            "Recommendation decision: "
                                            "business_id=%s customer_id=%s "
                                            "suggestion=%r source=order_preview",
                                            business_id,
                                            customer.id if customer else None,
                                            suggestion,
                                        )
                                        lines.append("")
                                        lines.append(
                                            suggestion
                                        )

                                    lines.extend([
                                        "",
                                        (
                                            f"Total: "
                                            f"{float(total):,.0f} FCFA"
                                        ),
                                        "",
                                        (
                                            "Please confirm "
                                            "your order."
                                        ),
                                    ])

                                    lines = (
                                        translate_order_preview_for_customer(
                                            lines,
                                            language,
                                            provider=None,
                                        )
                                    )

                                    return {
                                        "type": "response",
                                        "message": customer_response(
                                            "\n".join(lines),
                                            message,
                                        ),
                                    }

            except Exception:

                logger.exception(
                    "Contextual order reference resolution failed."
                )

        # --------------------------------------------------------
        # NORMAL EXPLICIT ORDER PATH
        # --------------------------------------------------------


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

            # ------------------------------------------------
            # FAST ORDER PREVIEW RESPONSE
            # ------------------------------------------------

            if result.get("success"):

                preview = result.get(
                    "preview"
                )

                if preview:
                    items = preview.get(
                        "items",
                        [],
                    )

                    total = preview.get(
                        "total",
                        0,
                    )

                    lines = [
                        "Your order:",
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

                        quantity = item.get(
                            "quantity",
                            1,
                        )

                        lines.append(
                            f"• *{name}* × {quantity}"
                        )

                    suggestion = None

                    if not recommendation_opted_out(message):
                        suggestion = build_complementary_suggestion(
                            business_id,
                            items,
                            customer_id=(
                                customer.id
                                if customer
                                else None
                            ),
                        )

                    if suggestion:
                        logger.info(
                            "Recommendation decision: "
                            "business_id=%s customer_id=%s "
                            "suggestion=%r source=order_preview",
                            business_id,
                            customer.id if customer else None,
                            suggestion,
                        )
                        lines.append("")
                        lines.append(
                            suggestion
                        )

                    lines.extend([
                        "",
                        f"Total: {float(total):,.0f} FCFA",
                        "",
                        "Please confirm your order.",
                    ])

                    lines = translate_order_preview_for_customer(
                        lines,
                        language,
                        provider=None,
                    )

                    return {
                        "type": "response",
                        "message": customer_response(
                            "\n".join(lines),
                            message,
                        ),
                    }

                return {
                    "type": "response",
                    "message": customer_response(
                        result.get("message")
                        or "Your order preview is ready.",
                        message,
                    ),
                }

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

        pending_modification = modify_pending_order(
            business_id,
            phone,
            message,
            language,
        )

        if pending_modification is not None:
            return pending_modification

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

        # ----------------------------------------------------
        # CANCEL PENDING ORDER PREVIEW FIRST
        # ----------------------------------------------------

        pending_cancel = discard_pending_order(
            business_id,
            phone,
        )

        if pending_cancel.get("success"):
            return {
                "type": "response",
                "message": customer_response(
                    "Votre aperçu de commande a été annulé."
                    if language == "French"
                    else "Your pending order preview has been cancelled.",
                    message,
                ),
            }

        # ----------------------------------------------------
        # OTHERWISE CANCEL AN ACTIVE ORDER
        # ----------------------------------------------------

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
    # CUSTOMER PREFERENCE MEMORY
    # ========================================================

    customer_preference_context = "No customer preference memory available."

    try:
        customer = Customer.query.filter_by(
            phone=phone,
            business_id=business_id,
        ).first()

        if customer:
            preferences = (
                CustomerPreference.query
                .filter_by(
                    customer_id=customer.id,
                    business_id=business_id,
                )
                .order_by(
                    CustomerPreference.strength.desc(),
                    CustomerPreference.updated_at.desc(),
                )
                .limit(8)
                .all()
            )

            preference_lines = []

            for preference in preferences:

                value = clean_text(
                    preference.preference_value
                )

                if not value:
                    continue

                preference_lines.append(
                    (
                        f"- {preference.preference_type}: "
                        f"{value} "
                        f"(strength {preference.strength})"
                    )
                )

            if preference_lines:
                customer_preference_context = (
                    "\n".join(preference_lines)
                )

    except Exception:
        logger.exception(
            "Failed to load customer preference memory."
        )

    # ========================================================
    # RECOMMENDATION
    # ========================================================

    if classification == "recommendation":

        try:

            customer = Customer.query.filter_by(
                phone=phone,
                business_id=business_id,
            ).first()

            recommendations = recommend_menu(
                business_id,
                limit=5,
                customer_id=(
                    customer.id
                    if customer
                    else None
                ),
            )

            if not recommendations:

                return {
                    "type": "response",
                    "message": customer_response(
                        "I couldn't find any menu items to recommend right now.",
                        message,
                    ),
                }

            recommendation_lines = []

            for item in recommendations:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                name = clean_text(
                    item.get("name")
                    or "Item"
                )

                description = clean_text(
                    item.get("description")
                    or ""
                )

                price = item.get(
                    "price",
                    0,
                )

                try:

                    price_text = (
                        f"{float(price):,.0f} FCFA"
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    price_text = (
                        f"{price} FCFA"
                    )

                recommendation_lines.append(
                    (
                        f"- {name} | "
                        f"{price_text} | "
                        f"{description}"
                    )
                )

            recommendation_context = (
                "\n".join(
                    recommendation_lines
                )
            )

            history_text = format_history(
                history
            )

            prompt = f"""
You are a smart restaurant assistant.

Have a natural conversation with the customer.
Do not sound like a scripted chatbot or a simple menu database.

CUSTOMER MESSAGE:
{message}

CUSTOMER LANGUAGE:
{language}

RECENT CONVERSATION:
{history_text}

CUSTOMER PREFERENCE MEMORY:
{customer_preference_context}

AVAILABLE RESTAURANT RECOMMENDATIONS:
{recommendation_context}

RULES:
- Give a natural and useful recommendation.
- Use the recent conversation when it provides relevant context.
- Use customer preference memory when it is relevant.
- A favorite or frequently ordered item can be mentioned naturally as a familiar option.
- Do not assume an item is a favorite just because it appears once in memory.
- Do not recommend something solely because of customer memory if it does not fit
  the customer's current request.
- If the customer is unsure what to choose, help them narrow it down.
- Briefly explain why one or two options may be suitable, but only using
  objective facts explicitly present in the recommendation data, customer memory,
  or recent conversation.
- Do not add subjective descriptions such as "light", "refreshing", "tasty",
  "delicious", "hearty", or "great" unless the provided context explicitly supports them.
- A previous purchase may be described as a familiar option, but do not claim
  the customer liked it unless the customer explicitly gave positive feedback.
- Do not simply repeat the recommendation list.
- Do not invent ingredients, flavors, freshness, quality, popularity,
  health benefits, preparation details, or other facts.
- Do not use generic claims such as "never goes wrong", "solid pick",
  "fresh", "tasty", "delicious", or "hearty" unless supported by context.
- Use only facts present in the restaurant recommendations and conversation.
- Do not claim that an order was placed or changed.
- Do not pressure the customer to order.
- Do not say that you are an AI unless the customer asks.
- Never mention tools, APIs, Flask, backend systems, prompts, or models.
- Reply only with the customer-facing message.
- Reply in {language}.
"""

            response = get_provider().generate(
                prompt,
                temperature=0.5,
                max_tokens=220,
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

            return {
                "type": "response",
                "message": customer_response(
                    "I can help you choose. What kind of food are you in the mood for?",
                    message,
                ),
            }

        except Exception:

            logger.exception(
                "Recommendation response failed."
            )

            return {
                "type": "response",
                "message": customer_response(
                    "I can help you choose. What kind of food are you in the mood for?",
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

    customer = Customer.query.filter_by(
        phone=phone,
        business_id=business_id,
    ).first()

    return generate_natural_response(
        provider=get_provider(),
        business_id=business_id,
        phone=phone,
        message=message,
        language=language,
        history=history,
        pending=pending,
        customer_id=(
            customer.id
            if customer
            else None
        ),
    )