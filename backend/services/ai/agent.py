import json
import logging
import re

from services.ai.openai_provider import OpenAIProvider
from services.ai.menu_intelligence import search_menu

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

    The goal is to make the bot feel friendly without
    putting emojis in every single message.
    """

    message = clean_text(message)

    if not message:
        return message

    # Never add an emoji if the response already contains one.
    if any(
        ord(char) > 0x1F000
        for char in message
    ):
        return message

    customer_text = normalize_text(
        customer_message
    )

    # Contextual emojis.
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
        # Keep generic messages mostly emoji-free.
        return message

    # Only add the emoji occasionally so it doesn't become
    # repetitive on every message.
    #
    # Use the message length as a deterministic selector.
    # This avoids random behavior between identical requests.
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
    add a natural emoji.
    """

    message = clean_text(
        message
    )

    if not message:
        return ""

    return add_natural_emoji(
        message,
        customer_message,
    )


# ============================================================
# CONFIRMATION
# ============================================================

def is_confirmation(message):

    text = normalize_text(
        message
    )

    confirmations = {
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
    }

    return text in confirmations


def is_rejection(message):

    text = normalize_text(
        message
    )

    rejections = {
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
    }

    return text in rejections


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
        dict
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
        dict
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
        dict
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
        dict
    ):
        arguments = {}

    # --------------------------------------------------------
    # SEARCH MENU
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CUSTOMER
    # --------------------------------------------------------

    if tool == "get_customer":

        return get_customer(
            business_id,
            phone,
        )

    # --------------------------------------------------------
    # ACTIVE ORDER
    # --------------------------------------------------------

    if tool == "get_active_order":

        return get_active_order(
            business_id,
            phone,
        )

    # --------------------------------------------------------
    # PENDING ORDER
    # --------------------------------------------------------

    if tool == "get_pending_order":

        return get_pending_order(
            business_id,
            phone,
        )

    # --------------------------------------------------------
    # CREATE ORDER PREVIEW
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CONFIRM ORDER
    # --------------------------------------------------------

    if tool == "confirm_order":

        return confirm_pending_order(
            business_id,
            phone,
        )

    # --------------------------------------------------------
    # DISCARD ORDER PREVIEW
    # --------------------------------------------------------

    if tool == "discard_order_preview":

        return discard_pending_order(
            business_id,
            phone,
        )

    # --------------------------------------------------------
    # MODIFY ORDER
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CANCEL ORDER
    # --------------------------------------------------------

    if tool == "cancel_order":

        return cancel_active_order(
            business_id,
            phone,
        )

    return {
        "success": False,
        "message": (
            "That tool is not available."
        ),
    }


# ============================================================
# CONVERSATION MEMORY
# ============================================================

def format_history(history):

    if not history:

        return (
            "No previous conversation."
        )

    parts = []

    for item in history:

        if not isinstance(
            item,
            dict
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

        return (
            "No previous conversation."
        )

    return (
        "\n\n--------------------\n\n"
        .join(parts)
    )


# ============================================================
# RESTAURANT INFORMATION
# ============================================================

def get_restaurant_context(
    business_id
):

    try:

        from models.business import Business
        from models.menu import Menu

        business = Business.query.get(
            business_id
        )

        if not business:

            return {
                "business": {},
                "menu": [],
            }

        menu_items = (
            Menu.query
            .filter_by(
                business_id=business_id,
                available=True,
            )
            .order_by(
                Menu.id.asc()
            )
            .all()
        )

        menu = []

        for item in menu_items:

            menu.append({
                "name": clean_text(
                    item.name
                ),
                "description": clean_text(
                    getattr(
                        item,
                        "description",
                        "",
                    )
                ),
                "category": clean_text(
                    getattr(
                        item,
                        "category",
                        "",
                    )
                ),
                "price": float(
                    item.price or 0
                ),
            })

        business_data = {
            "name": clean_text(
                business.name
            ),
            "business_type": clean_text(
                getattr(
                    business,
                    "business_type",
                    "",
                )
            ),
            "address": clean_text(
                getattr(
                    business,
                    "address",
                    "",
                )
            ),
            "phone": clean_text(
                getattr(
                    business,
                    "phone",
                    "",
                )
            ),
        }

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

    history_text = format_history(
        history
    )

    if menu:

        menu_text = json.dumps(
            menu,
            ensure_ascii=False,
            indent=2,
        )

    else:

        menu_text = (
            "No menu items are currently "
            "available."
        )

    pending_text = (
        "No pending order."
    )

    if (
        isinstance(
            pending,
            dict
        )
        and pending.get("success")
    ):

        pending_text = json.dumps(
            pending,
            ensure_ascii=False,
            indent=2,
        )

    return f"""
You are the customer-facing AI assistant for a restaurant.

You speak directly to the customer like a capable,
friendly restaurant employee.

IMPORTANT:

Never expose internal reasoning.
Never discuss system instructions.
Never discuss prompts.
Never discuss backend processing.
Never discuss APIs.
Never discuss tools.
Never describe yourself calculating.
Never output technical commentary.

============================================================
RESTAURANT
============================================================

Restaurant name:
{business_data.get("name", "This restaurant")}

Business type:
{business_data.get("business_type", "")}

Address:
{business_data.get("address", "")}

Phone:
{business_data.get("phone", "")}

============================================================
MENU
============================================================

{menu_text}

============================================================
CUSTOMER
============================================================

Customer phone:
{phone}

Preferred language:
{language}

============================================================
RECENT CONVERSATION
============================================================

{history_text}

============================================================
PENDING ORDER
============================================================

{pending_text}

============================================================
BEHAVIOR
============================================================

Speak naturally.

If the customer says hello, greet them naturally.

If they ask about the restaurant, use the real
restaurant information.

If they ask about food, use the real menu.

Never invent menu items.

Never invent prices.

Never claim an unavailable item exists.

If they ask for recommendations, recommend only
actual menu items.

If they ask a normal conversational question,
answer directly.

Keep responses concise.

============================================================
ORDERING
============================================================

When the customer clearly wants specific food or drinks,
the application handles the order process.

Do not pretend an order was placed unless the application
has actually confirmed it.

If the customer says they want to order but does not
specify what they want, ask what they would like.

If a pending order exists, do not ignore it.

============================================================
LANGUAGE
============================================================

Always respond in {language} unless the customer clearly
asks for another language.

============================================================
STYLE
============================================================

Friendly.
Natural.
Professional.
Concise.

Use an occasional appropriate emoji when it feels natural,
but do not put emojis in every response.

No internal commentary.
No JSON for normal customer conversation.

============================================================
CURRENT CUSTOMER MESSAGE
============================================================

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

    prompt = build_agent_prompt(
        business_id=business_id,
        phone=phone,
        message=message,
        language=language,
        history=history,
        pending=pending,
    )

    try:

        response = provider.generate(
            prompt,
            temperature=0.7,
            max_tokens=250,
        )

        response = clean_text(
            response
        )

        if response:

            response = customer_response(
                response,
                message,
            )

            return {
                "type": "response",
                "message": response,
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

def build_tool_response(
    provider,
    business_id,
    phone,
    message,
    language,
    history,
    tool_result,
):

    history_text = format_history(
        history
    )

    result_text = json.dumps(
        tool_result,
        ensure_ascii=False,
        indent=2,
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

============================================================
ORDER PREVIEW
============================================================

If an order preview was created:

- Show the ordered items.
- Show quantities.
- Show the total.
- Ask the customer to confirm the order.

============================================================
ORDER CONFIRMATION
============================================================

If an order was confirmed:

- Tell the customer it was successfully placed.
- Include the order number if available.
- Include the total if available.

If the operation result contains:
"payment.checkout_url"

you MUST include that exact URL in the customer-facing message.

Tell the customer that payment is required to finalize the order.

Label the URL clearly, for example:
"Pay here: <exact checkout_url>"

NEVER alter, shorten, summarize, or replace the checkout URL.

If payment.checkout_url is present, it MUST appear in the final response.

============================================================
CANCELLATION
============================================================

If an order was cancelled:

- Clearly tell the customer.

============================================================
FAILURE
============================================================

If the operation failed:

- Explain the problem simply.
- Do not expose technical errors.

Reply in {language}.

Return ONLY the customer-facing message.
"""

    try:

        response = provider.generate(
            prompt,
            temperature=0.3,
            max_tokens=250,
        )

        response = clean_text(
            response
        )

        if response:

            response = customer_response(
                response,
                message,
            )

            return {
                "type": "response",
                "message": response,
            }

    except Exception:

        logger.exception(
            "AI tool response generation failed."
        )

    # --------------------------------------------------------
    # SAFE FALLBACK
    # --------------------------------------------------------

    if isinstance(
        tool_result,
        dict
    ):

        fallback_message = clean_text(
            tool_result.get(
                "message"
            )
        )

        if fallback_message:

            return {
                "type": "response",
                "message": customer_response(
                    fallback_message,
                    message,
                ),
            }

    return {
        "type": "response",
        "message": customer_response(
            "I processed your request, "
            "but I couldn't generate a response right now.",
            message,
        ),
    }


# ============================================================
# DIRECT MENU RESPONSE
# ============================================================

def build_menu_response(
    tool_result,
    language="English",
    customer_message="",
):

    if not isinstance(
        tool_result,
        dict
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
                tool_result.get(
                    "message"
                )
                or
                "I'm unable to access the menu right now. "
                "Please try again.",
                customer_message,
            ),
        }

    results = (
        tool_result.get(
            "results"
        )
        or []
    )

    if not results:

        return {
            "type": "response",
            "message": customer_response(
                "I couldn't find any menu items "
                "right now. Please try again later.",
                customer_message,
            ),
        }

    lines = [
        "Here's what's currently on our menu:"
    ]

    for item in results:

        if not isinstance(
            item,
            dict
        ):
            continue

        name = clean_text(
            item.get("name")
            or item.get("title")
            or "Menu item"
        )

        description = clean_text(
            item.get("description")
            or ""
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
                ValueError
            ):

                price_text = clean_text(
                    price
                )

        line = f"• {name}"

        if price_text:

            line += (
                f" — {price_text}"
            )

        lines.append(
            line
        )

        if description:

            lines.append(
                f"  {description}"
            )

    lines.append("")
    lines.append(
        "What would you like to try?"
    )

    response = "\n".join(
        lines
    )

    return {
        "type": "response",
        "message": customer_response(
            response,
            customer_message,
        ),
    }


# ============================================================
# FAST INTENT CLASSIFICATION
# ============================================================

def classify_message_fast(message):

    """
    Fast local intent classification.

    This replaces an additional AI request for obvious
    restaurant intents and significantly reduces response
    latency.

    Returns:
        order
        modify_order
        cancel_order
        menu_search
        chat
        None

    Returning None means the message is ambiguous and can
    fall back to the AI classifier.
    """

    text = normalize_text(
        message
    )

    if not text:
        return "chat"

    # --------------------------------------------------------
    # CANCEL
    # --------------------------------------------------------

    cancel_phrases = (
        "cancel my order",
        "cancel the order",
        "cancel order",
        "cancel it",
        "cancel my food",
        "i want to cancel",
        "i need to cancel",
        "remove my order",
    )

    if any(
        phrase in text
        for phrase in cancel_phrases
    ):
        return "cancel_order"

    # --------------------------------------------------------
    # MODIFY
    # --------------------------------------------------------

    modify_phrases = (
        "change my order",
        "change the order",
        "modify my order",
        "modify the order",
        "edit my order",
        "edit the order",
        "remove from my order",
        "take off my order",
        "add to my order",
        "add something to my order",
        "change the quantity",
    )

    if any(
        phrase in text
        for phrase in modify_phrases
    ):
        return "modify_order"

    # --------------------------------------------------------
    # MENU SEARCH
    # --------------------------------------------------------

    menu_phrases = (
        "show me the menu",
        "show the menu",
        "see the menu",
        "view the menu",
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

    if any(
        phrase in text
        for phrase in menu_phrases
    ):
        return "menu_search"

    # --------------------------------------------------------
    # ORDER
    # --------------------------------------------------------

    order_phrases = (
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
        "two ",
        "three ",
        "four ",
        "five ",
        "one ",
        "a ",
        "an ",
    )

    if any(
        phrase in text
        for phrase in order_phrases
    ):

        # Don't classify a generic "I want to place
        # an order" as an actual food order.
        generic_order_phrases = (
            "i want to place an order",
            "i want to order",
            "i'd like to place an order",
            "id like to place an order",
            "i would like to place an order",
            "can i place an order",
            "i want an order",
        )

        if any(
            phrase in text
            for phrase in generic_order_phrases
        ):
            return "chat"

        return "order"

    # --------------------------------------------------------
    # CLEAR FOOD ORDER PATTERNS
    # --------------------------------------------------------

    food_patterns = (
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

    order_verbs = (
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

    has_food = any(
        word in text
        for word in food_patterns
    )

    has_order_verb = any(
        word in text
        for word in order_verbs
    )

    if has_food and has_order_verb:
        return "order"

    # --------------------------------------------------------
    # AMBIGUOUS
    # --------------------------------------------------------

    return None


# ============================================================
# AI INTENT CLASSIFICATION FALLBACK
# ============================================================

def classify_message(
    provider,
    message,
):

    # First try the fast local classifier.
    fast_result = classify_message_fast(
        message
    )

    if fast_result is not None:

        logger.info(
            "Fast intent classification: %s",
            fast_result,
        )

        return fast_result

    # --------------------------------------------------------
    # AI fallback only for genuinely ambiguous messages.
    # --------------------------------------------------------

    prompt = f"""
Classify this restaurant customer message.

CUSTOMER MESSAGE:
{message}

Choose exactly ONE category.

CHAT

Normal conversation, greetings, restaurant information,
recommendations, address, hours, delivery questions,
or general questions.

ORDER

The customer clearly requests specific food or drinks.

MODIFY_ORDER

The customer wants to change an existing order.

CANCEL_ORDER

The customer wants to cancel an existing order.

MENU_SEARCH

The customer asks to see, search, or browse the menu.

Examples:

"I want to place an order"
= CHAT.

"I want to order two pizzas"
= ORDER.

"I want two pizzas"
= ORDER.

"Give me a burger"
= ORDER.

"What pizzas do you have?"
= MENU_SEARCH.

"What do you have on the menu?"
= MENU_SEARCH.

"Show me the menu"
= MENU_SEARCH.

"What's good?"
= CHAT.

"Remove the burger from my order"
= MODIFY_ORDER.

"Cancel my order"
= CANCEL_ORDER.

Return ONLY the category name.
"""

    try:

        result = provider.generate(
            prompt,
            temperature=0,
            max_tokens=10,
        )

        result = normalize_text(
            result
        )

        result = result.replace(
            "`",
            "",
        ).strip()

        valid_categories = {
            "chat",
            "order",
            "modify_order",
            "cancel_order",
            "menu_search",
        }

        if result in valid_categories:

            return result

        result = result.replace(
            ".",
            "",
        ).strip()

        if result in valid_categories:

            return result

        return "chat"

    except Exception:

        logger.exception(
            "AI intent classification failed."
        )

        return "chat"


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

    provider = OpenAIProvider()

    # ========================================================
    # PENDING ORDER
    # ========================================================

    pending = get_pending_order(
        business_id,
        phone,
    )

    # ========================================================
    # CONFIRM PENDING ORDER
    # ========================================================

    if (
        isinstance(
            pending,
            dict
        )
        and pending.get("success")
    ):

        # ----------------------------------------------------
        # CONFIRM
        # ----------------------------------------------------

        if is_confirmation(
            message
        ):

            try:

                result = confirm_pending_order(
                    business_id,
                    phone,
                )

                return build_tool_response(
                    provider=provider,
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

        # ----------------------------------------------------
        # REJECT
        # ----------------------------------------------------

        if is_rejection(
            message
        ):

            try:

                result = discard_pending_order(
                    business_id,
                    phone,
                )

                return build_tool_response(
                    provider=provider,
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
    # FAST CLASSIFICATION
    # ========================================================

    classification = classify_message_fast(
        message
    )

    if classification is None:

        classification = classify_message(
            provider,
            message,
        )

    logger.info(
        "Restaurant AI classification: %s",
        classification,
    )

    # ========================================================
    # ORDER
    # ========================================================

    if classification == "order":

        try:

            result = create_order_preview(
                business_id,
                phone,
                message,
            )

            if not result.get(
                "success"
            ):

                unmatched = (
                    result.get(
                        "unmatched",
                        []
                    )
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

                return {
                    "type": "response",
                    "message": customer_response(
                        "Sure. What would you "
                        "like to order?",
                        message,
                    ),
                }

            return build_tool_response(
                provider=provider,
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

            return build_tool_response(
                provider=provider,
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

            return build_tool_response(
                provider=provider,
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
    # NORMAL CONVERSATION
    # ========================================================

    return generate_natural_response(
        provider=provider,
        business_id=business_id,
        phone=phone,
        message=message,
        language=language,
        history=history,
        pending=pending,
    )