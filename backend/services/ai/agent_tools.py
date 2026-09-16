import logging
import re

logger = logging.getLogger(__name__)
import json

from database.db import db

from models.customer import Customer
from models.order import Order
from models.order_item import OrderItem
from models.pending_order import PendingOrder

from services.ai.menu_intelligence import search_menu
from services.ai.order_extractor import extract_order
from services.ai.order_modifier import interpret_order_request
from services.ai.order_executor import execute_order_action

from services.inventory import (
    filter_inventory_available_menu_items,
    reserve_inventory_for_order,
    consume_inventory_for_order,
)

from services.payments.paydunya import (
    create_checkout_invoice,
)


# ============================================================
# CUSTOMER
# ============================================================

def get_customer(business_id, phone):

    customer = Customer.query.filter_by(
        business_id=business_id,
        phone=phone
    ).first()

    if not customer:
        return {
            "success": False,
            "message": "Customer not found."
        }

    return {
        "success": True,
        "customer": {
            "id": customer.id,
            "name": customer.name,
            "phone": customer.phone,
            "language": customer.language
        }
    }


# ============================================================
# MENU SEARCH
# ============================================================

def tool_search_menu(business_id, query):

    """
    Search the restaurant menu.

    Broad menu/browsing requests return the complete currently
    available menu from the database. Specific requests use the
    grounded menu search path.

    Inventory filtering is best-effort here: a broken inventory
    lookup must never make a valid restaurant menu disappear.
    """

    query_text = str(
        query or ""
    ).strip().lower()

    if not query_text:
        return {
            "success": True,
            "results": [],
            "full_menu": False,
        }

    # --------------------------------------------------------
    # NORMALIZE CUSTOMER WORDING
    # --------------------------------------------------------

    menu_query = (
        query_text
        .replace("?", "")
        .replace("!", "")
        .replace(".", "")
        .replace(",", "")
        .replace(";", "")
        .replace(":", "")
        .replace("’", "'")
        .replace("-", " ")
        .strip()
    )

    menu_query = " ".join(
        menu_query.split()
    )

    # --------------------------------------------------------
    # BROAD / FULL MENU REQUESTS
    # --------------------------------------------------------

    full_menu_phrases = (
        # English
        "menu",
        "show menu",
        "show me the menu",
        "show me your menu",
        "send me the menu",
        "send me your menu",
        "give me the menu",
        "give me your menu",
        "view menu",
        "see menu",
        "see the menu",
        "what is on the menu",
        "what's on the menu",
        "whats on the menu",
        "what do you have",
        "what do you guys have",
        "what food do you have",
        "what foods do you have",
        "what food do you guys have",
        "what foods do you guys have",
        "what do you serve",
        "what you got",
        "what've you got",
        "what have you got",
        "what yall got",
        "what y'all got",
        "what you guys got",
        "what yall got for food",
        "what y'all got for food",
        "what you got for food",
        "what you guys got for food",
        "whatcha got",
        "what are you guys serving",
        "what's good",
        "whats good",
        "what's good here",
        "whats good here",
        "list menu",
        "list the menu",
        "available menu",
        "full menu",

        # French
        "voir le menu",
        "voir votre menu",
        "voir ton menu",
        "je peux voir le menu",
        "je peux voir votre menu",
        "je peux voir ton menu",
        "puis je voir le menu",
        "puis-je voir le menu",
        "montrez moi le menu",
        "montrez-moi le menu",
        "montre moi le menu",
        "montre-moi le menu",
        "montrez votre menu",
        "montre votre menu",
        "donnez moi le menu",
        "donnez-moi le menu",
        "donnez votre menu",
        "quel est le menu",
        "qu est ce qu il y a au menu",
        "qu'est ce qu'il y a au menu",
        "qu'est-ce qu'il y a au menu",
        "que proposez vous",
        "que proposez-vous",
        "qu est ce que vous avez",
        "qu'est-ce que vous avez",
        "que servez vous",
        "que servez-vous",
        "je voudrais voir le menu",
        "je veux voir le menu",

        # Casual French / mixed
        "t as quoi",
        "t'as quoi",
        "tu as quoi",
        "t as quoi comme",
        "t'as quoi comme",
        "tu as quoi comme",
        "vous avez quoi",
        "vous avez quoi comme",
        "qu est ce que t as",
        "qu'est-ce que t'as",
        "qu est ce que tu as",
        "qu'est-ce que tu as",
        "c quoi comme",
        "c'est quoi comme",
        "c quoi comme food",
        "c'est quoi comme food",
        "quoi comme food",
        "quoi comme bouffe",
        "wesh t as quoi",
        "wesh t'as quoi",
        "wesh tu as quoi",
        "wesh vous avez quoi",
        "wech t as quoi",
        "wech t'as quoi",
        "wesh t as quoi comme food",
        "wesh t'as quoi comme food",
        "wesh t as quoi comme bouffe",
        "wesh t'as quoi comme bouffe",
        "t as quoi comme food",
        "t'as quoi comme food",
        "t as quoi comme bouffe",
        "t'as quoi comme bouffe",
        "vous avez quoi comme food",
        "vous avez quoi comme bouffe",
        "tu proposes quoi",
        "tu proposes quoi comme food",
        "vous proposez quoi",
        "vous proposez quoi comme food",

        # Spanish
        "ver el menu",
        "ver su menu",
        "puedo ver el menu",
        "puedo ver su menu",
        "muestreme el menu",
        "muéstrame el menú",
        "dame el menu",
        "dame el menú",
        "cual es el menu",
        "cuál es el menú",
        "que tienen",
        "qué tienen",

        # Portuguese
        "ver o menu",
        "ver seu menu",
        "posso ver o menu",
        "posso ver seu menu",
        "mostre o menu",
        "mostre-me o menu",
        "me mostre o menu",
        "me dê o menu",
        "me de o menu",
        "qual é o menu",
        "o que vocês têm",
        "o que voces tem",

        # German
        "zeig mir das menü",
        "zeige mir das menü",
        "kann ich das menü sehen",
        "kann ich eure speisekarte sehen",
        "was habt ihr",
        "was gibt es auf der speisekarte",

        # Italian
        "mostrami il menu",
        "fammi vedere il menu",
        "posso vedere il menu",
        "qual è il menu",
        "cosa avete",
        "cosa avete nel menu",

        # Swahili
        "onyesha menyu",
        "nionyeshe menyu",
        "naweza kuona menyu",
        "naweza kuona menu",
        "nipe menyu",
        "mna nini",
    )

    menu_words = (
        "menu",
        "menú",
        "menü",
        "menyu",
    )

    # Explicit broad-language signals that should work even when
    # the exact wording contains slang, an address term, or a
    # small mixed-language variation.
    broad_menu_patterns = (
        r"\bwhat\s+(?:do\s+you|you)\s+(?:guys\s+)?(?:have|got)\b",
        r"\bwhat\s+(?:food|foods)\s+(?:do\s+you|you)\s+(?:guys\s+)?(?:have|got)\b",
        r"\bwhat\s+(?:are\s+you\s+)?serv(?:e|ing)\b",
        r"\bwhatcha\s+got\b",
        r"\bwhat(?:'s|s)\s+good(?:\s+here)?\b",
        r"\bwesh\s+(?:bro\s+|fr[eé]rot\s+)?t[' ]?as\s+quoi\b",
        r"\bwech\s+(?:bro\s+|fr[eé]rot\s+)?t[' ]?as\s+quoi\b",
        r"\bt[' ]?as\s+quoi\s+comme\s+(?:food|bouffe)\b",
        r"\btu\s+as\s+quoi\s+comme\s+(?:food|bouffe)\b",
        r"\bvous\s+avez\s+quoi\s+comme\s+(?:food|bouffe)\b",
        r"\bc[' ]?est\s+quoi\s+comme\s+(?:food|bouffe)\b",
        r"\bc\s+quoi\s+comme\s+(?:food|bouffe)\b",
    )

    # A query can contain a broad-menu phrase while still asking for
    # one specific category. Specific category intent must win.
    #
    # Examples:
    #   "t'as quoi comme pizza?" -> specific: pizza
    #   "wesh t'as quoi comme food?" -> broad: full menu
    #
    # Do this before broad full-menu detection so "comme pizza",
    # "comme boissons", etc. are not swallowed by the generic
    # "t'as quoi comme" patterns.
    specific_category_query = bool(
        re.search(
            r"\b(?:t[' ]?as|tu\s+as|vous\s+avez)\s+quoi\s+"
            r"comme\s+(?!food\b|bouffe\b)(.+)$",
            menu_query,
            re.IGNORECASE,
        )
    )

    is_full_menu = (
        not specific_category_query
        and (
            menu_query in full_menu_phrases
            or any(
                phrase in menu_query
                for phrase in full_menu_phrases
            )
            or any(
                re.search(pattern, menu_query)
                for pattern in broad_menu_patterns
            )
            or (
                any(
                    word in menu_query
                    for word in menu_words
                )
                and any(
                    marker in menu_query
                    for marker in (
                        "show",
                        "see",
                        "view",
                        "send",
                        "give",
                        "list",
                        "voir",
                        "montre",
                        "montrez",
                        "donne",
                        "donnez",
                        "propose",
                        "proposez",
                        "what",
                        "quoi",
                    )
                )
            )
        )
    )

    # --------------------------------------------------------
    # LOAD THE REAL AVAILABLE MENU
    # --------------------------------------------------------

    from services.ai.menu_intelligence import (
        get_menu_items,
    )

    try:

        menu_items = get_menu_items(
            business_id
        )

    except Exception:

        logger.exception(
            "Menu database load failed for business_id=%s",
            business_id,
        )

        return {
            "success": False,
            "message": (
                "The menu is temporarily unavailable. "
                "Please try again."
            ),
        }

    if not menu_items:

        return {
            "success": True,
            "results": [],
            "full_menu": is_full_menu,
        }

    # --------------------------------------------------------
    # INVENTORY FILTER
    #
    # Inventory status is useful, but a failure in the inventory
    # subsystem must never make a healthy menu inaccessible.
    # --------------------------------------------------------

    try:

        filtered_items = (
            filter_inventory_available_menu_items(
                menu_items,
                quantity=1,
            )
        )

        if filtered_items is not None:
            menu_items = filtered_items

    except Exception:

        logger.exception(
            "Menu inventory filter failed for business_id=%s; "
            "falling back to Menu.available items.",
            business_id,
        )

    # --------------------------------------------------------
    # FULL MENU
    #
    # IMPORTANT: no search limit here. Every item returned by the
    # grounded menu source is passed to the response builder.
    # --------------------------------------------------------

    if is_full_menu:

        results = []

        for item in menu_items:

            results.append({
                "id": item.id,
                "name": item.name,
                "description": (
                    item.description
                    or ""
                ),
                "category": (
                    item.category
                    or "Other"
                ),
                "price": float(
                    item.price or 0
                ),
                "currency": "FCFA",
            })

        return {
            "success": True,
            "results": results,
            "full_menu": True,
        }

    # --------------------------------------------------------
    # NORMAL / SPECIFIC MENU SEARCH
    # --------------------------------------------------------

    # --------------------------------------------------------
    # NORMALIZE NATURAL-LANGUAGE SPECIFIC QUERIES
    #
    # Extract the useful menu subject from conversational
    # questions before handing the query to the grounded search.
    #
    # Examples:
    #   "what pizzas do you have?"       -> "pizzas"
    #   "what kind of pizza do you have?" -> "pizza"
    #   "show me your pizzas"             -> "pizzas"
    #   "which pizzas do you have?"       -> "pizzas"
    #   "wesh t'as quoi comme pizza?"     -> "pizza"
    # --------------------------------------------------------

    search_query = menu_query

    natural_query_patterns = (
        r"^what\s+(?:kind\s+of\s+)?(.+?)\s+do\s+(?:you|yall|you guys)\s+(?:have|got)$",
        r"^what\s+(?:kind\s+of\s+)?(.+?)\s+(?:do\s+)?(?:you|yall|you guys)\s+have$",
        r"^what\s+(?:kind\s+of\s+)?(.+?)\s+(?:do\s+)?(?:you|yall|you guys)\s+got$",
        r"^which\s+(.+?)\s+do\s+(?:you|yall|you guys)\s+have$",
        r"^which\s+(.+?)\s+do\s+(?:you|yall|you guys)\s+got$",
        r"^show\s+me\s+(?:your\s+)?(.+)$",
        r"^what\s+(.+?)\s+(?:are|is)\s+(?:available|on\s+the\s+menu)$",
        r"^what\s+(.+?)\s+(?:you|yall|you guys)\s+got$",
        r"^wesh\s+(?:bro\s+|fr[eé]rot\s+)?t[' ]?as\s+quoi\s+comme\s+(.+)$",
        r"^wech\s+(?:bro\s+|fr[eé]rot\s+)?t[' ]?as\s+quoi\s+comme\s+(.+)$",
        r"^t[' ]?as\s+quoi\s+comme\s+(.+)$",
        r"^tu\s+as\s+quoi\s+comme\s+(.+)$",
        r"^vous\s+avez\s+quoi\s+comme\s+(.+)$",
        r"^t[' ]?as\s+quelles?\s+(.+)$",
    )

    for pattern in natural_query_patterns:
        match = re.match(
            pattern,
            menu_query,
            re.IGNORECASE,
        )

        if match:
            candidate = match.group(1).strip()

            if candidate:
                search_query = candidate
                break

    # Strip conversational filler around the actual menu subject.
    search_query = re.sub(
        r"\b(?:please|pls|plz|stp|svp|bro|brother|fr[eé]rot|fr[eè]re|"
        r"fam|gang|guys|yall|you guys)\b",
        " ",
        search_query,
        flags=re.IGNORECASE,
    )

    search_query = re.sub(
        r"\s+",
        " ",
        search_query,
    ).strip()

    results = search_menu(
        business_id,
        search_query,
        limit=5,
        menu_items=menu_items,
    )

    return {
        "success": True,
        "results": results,
        "full_menu": False,
    }


# ============================================================
# ACTIVE ORDER
# ============================================================

def get_active_order(business_id, phone):

    customer = Customer.query.filter_by(
        business_id=business_id,
        phone=phone
    ).first()

    if not customer:
        return {
            "success": False,
            "message": "Customer not found."
        }

    order = Order.query.filter(
        Order.business_id == business_id,
        Order.customer_id == customer.id,
        Order.status.in_([
            "Pending",
            "Preparing"
        ])
    ).order_by(
        Order.id.desc()
    ).first()

    if not order:
        return {
            "success": False,
            "message": "No active order found."
        }

    return {
        "success": True,
        "order": {
            "id": order.id,
            "status": order.status,
            "payment_status": order.payment_status,
            "total": float(
                order.total_price or 0
            ),
            "items": [
                {
                    "name": item.name,
                    "quantity": item.quantity,
                    "price": float(
                        item.price or 0
                    ),
                    "subtotal": float(
                        item.subtotal or 0
                    )
                }
                for item in order.items
            ]
        }
    }


# ============================================================
# PENDING ORDER
# ============================================================

def get_pending_order(
    business_id,
    phone,
):
    """
    Retrieve the latest pending order using only the database
    columns required by the agent.
    """

    customer = (
        Customer.query
        .with_entities(
            Customer.id,
        )
        .filter(
            Customer.business_id == business_id,
            Customer.phone == phone,
        )
        .first()
    )

    if not customer:
        return {
            "success": False,
            "message": "Customer not found.",
        }

    preview = (
        PendingOrder.query
        .with_entities(
            PendingOrder.id,
            PendingOrder.items_json,
            PendingOrder.total_price,
            PendingOrder.status,
        )
        .filter(
            PendingOrder.business_id == business_id,
            PendingOrder.customer_id == customer.id,
            PendingOrder.status == "pending",
        )
        .order_by(
            PendingOrder.id.desc()
        )
        .first()
    )

    if not preview:
        return {
            "success": False,
            "message": "No pending order preview.",
        }

    try:
        items = json.loads(
            preview.items_json
        )
    except (
        json.JSONDecodeError,
        TypeError,
    ):
        return {
            "success": False,
            "message": (
                "The pending order preview "
                "could not be read."
            ),
        }

    return {
        "success": True,
        "preview": {
            "id": preview.id,
            "items": items,
            "total": float(
                preview.total_price or 0
            ),
            "status": preview.status,
        },
    }


# ============================================================
# CREATE ORDER PREVIEW
# ============================================================

def create_order_preview(
    business_id,
    phone,
    message
):

    # --------------------------------------------------------
    # RESOLVE COMMON CUSTOMER MENU SHORTHAND
    #
    # Keep the database menu as the source of truth while allowing
    # natural customer references such as "margaritas".
    # --------------------------------------------------------

    extraction_message = str(
        message or ""
    )

    import re

    menu_aliases = {
        "margarita": "Classic Margherita Pizza",
        "margaritas": "Classic Margherita Pizza",
        "margherita": "Classic Margherita Pizza",
        "margheritas": "Classic Margherita Pizza",
    }

    for alias, canonical_name in menu_aliases.items():

        if re.search(
            rf"\b{re.escape(alias)}\b",
            extraction_message,
            re.IGNORECASE,
        ):
            extraction_message = re.sub(
                rf"\b{re.escape(alias)}\b",
                canonical_name,
                extraction_message,
                flags=re.IGNORECASE,
            )
            break

    extracted = extract_order(
        business_id,
        extraction_message
    )

    items = extracted.get(
        "items",
        []
    )

    unmatched = extracted.get(
        "unmatched",
        []
    )

    # --------------------------------------------------------
    # UNMATCHED ITEMS
    # --------------------------------------------------------

    if unmatched:

        return {
            "success": False,
            "message": (
                "Some requested items could not "
                "be found on the menu."
            ),
            "unmatched": unmatched,
            "matched_items": items,
        }

    # --------------------------------------------------------
    # NO ITEMS
    # --------------------------------------------------------

    if not items:

        return {
            "success": False,
            "message": (
                "No food or drink items "
                "were specified."
            ),
            "unmatched": []
        }

    customer = Customer.query.filter_by(
        business_id=business_id,
        phone=phone
    ).first()

    if not customer:

        customer = Customer(
            name="New Customer",
            phone=phone,
            language="English",
            business_id=business_id
        )

        db.session.add(
            customer
        )

        db.session.flush()

    # --------------------------------------------------------
    # MERGE INTO EXISTING PENDING PREVIEW
    #
    # A pending preview represents the customer's current
    # working order. A new turn such as:
    #
    #   "jveux some pizza"
    #   "ajoute une lemonade"
    #
    # must preserve the pizza and add the lemonade.
    #
    # Never delete the previous pending preview merely because
    # the customer sent another order message.
    # --------------------------------------------------------

    existing_preview = (
        PendingOrder.query
        .filter_by(
            business_id=business_id,
            customer_id=customer.id,
            status="pending"
        )
        .order_by(
            PendingOrder.id.desc()
        )
        .first()
    )

    if existing_preview:
        try:
            existing_items = json.loads(
                existing_preview.items_json
            )
        except (
            json.JSONDecodeError,
            TypeError,
        ):
            existing_items = []

        if not isinstance(existing_items, list):
            existing_items = []

        # Merge newly requested items into the existing preview.
        for new_item in items:

            if not isinstance(new_item, dict):
                continue

            new_name = str(
                new_item.get("name") or ""
            ).strip()

            if not new_name:
                continue

            try:
                new_quantity = max(
                    1,
                    int(
                        new_item.get(
                            "quantity",
                            1
                        )
                        or 1
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                new_quantity = 1

            try:
                new_price = float(
                    new_item.get(
                        "price",
                        0
                    )
                    or 0
                )
            except (
                TypeError,
                ValueError,
            ):
                new_price = 0.0

            existing_item = None

            for current_item in existing_items:

                if not isinstance(current_item, dict):
                    continue

                current_name = str(
                    current_item.get("name") or ""
                ).strip()

                if (
                    current_name.lower()
                    == new_name.lower()
                ):
                    existing_item = current_item
                    break

            if existing_item:

                try:
                    current_quantity = max(
                        1,
                        int(
                            existing_item.get(
                                "quantity",
                                1
                            )
                            or 1
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    current_quantity = 1

                existing_item["quantity"] = (
                    current_quantity
                    + new_quantity
                )

                # Keep the current grounded price when available.
                try:
                    current_price = float(
                        existing_item.get(
                            "price",
                            new_price
                        )
                        or new_price
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    current_price = new_price

                existing_item["price"] = current_price
                existing_item["subtotal"] = (
                    current_price
                    * existing_item["quantity"]
                )

            else:

                existing_items.append({
                    "name": new_name,
                    "quantity": new_quantity,
                    "price": new_price,
                    "subtotal": (
                        new_price
                        * new_quantity
                    ),
                })

        # Recalculate the complete pending-order total.
        merged_total = 0.0

        for item in existing_items:

            if not isinstance(item, dict):
                continue

            try:
                quantity = max(
                    1,
                    int(
                        item.get(
                            "quantity",
                            1
                        )
                        or 1
                    )
                )

                price = float(
                    item.get(
                        "price",
                        0
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

            merged_total += item["subtotal"]

        existing_preview.items_json = json.dumps(
            existing_items,
            ensure_ascii=False
        )

        existing_preview.total_price = float(
            merged_total
        )

        db.session.commit()

        preview = existing_preview
        items = existing_items

    else:

        # No pending preview exists yet, so create the first one.
        preview = PendingOrder(
            business_id=business_id,
            customer_id=customer.id,
            items_json=json.dumps(
                items,
                ensure_ascii=False
            ),
            total_price=float(
                extracted.get(
                    "total",
                    0
                )
            ),
            status="pending"
        )

        db.session.add(
            preview
        )

        db.session.commit()

    return {
        "success": True,
        "preview": {
            "id": preview.id,
            "items": items,
            "total": float(
                preview.total_price or 0
            ),
            "currency": "FCFA",
            "status": "pending"
        },
        "message": (
            "Order preview created. "
            "Customer confirmation is required."
        )
    }


# ============================================================
# CONFIRM PENDING ORDER
# ============================================================

def confirm_pending_order(
    business_id,
    phone
):

    customer = Customer.query.filter_by(
        business_id=business_id,
        phone=phone
    ).first()

    if not customer:

        return {
            "success": False,
            "message": "Customer not found."
        }

    preview = PendingOrder.query.filter_by(
        business_id=business_id,
        customer_id=customer.id,
        status="pending"
    ).order_by(
        PendingOrder.id.desc()
    ).first()

    if not preview:

        return {
            "success": False,
            "message": (
                "There is no order waiting "
                "for confirmation."
            )
        }

    try:

        items = json.loads(
            preview.items_json
        )

    except (
        json.JSONDecodeError,
        TypeError
    ):

        return {
            "success": False,
            "message": (
                "The pending order is invalid."
            )
        }

    if not items:

        return {
            "success": False,
            "message": (
                "The pending order is empty."
            )
        }

    # ========================================================
    # REVALIDATE MENU
    # ========================================================

    from models.menu import Menu

    final_items = []
    final_total = 0.0

    for item in items:

        item_name = str(
            item.get(
                "name",
                ""
            )
        ).strip()

        if not item_name:
            continue

        menu = Menu.query.filter(
            Menu.business_id == business_id,
            Menu.name.ilike(item_name),
            Menu.available.is_(True)
        ).first()

        if not menu:

            return {
                "success": False,
                "message": (
                    f"{item_name} is no longer available."
                )
            }

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

        price = float(
            menu.price or 0
        )

        subtotal = (
            price * quantity
        )

        final_items.append({
            "name": menu.name,
            "quantity": quantity,
            "price": price,
            "subtotal": subtotal
        })

        final_total += subtotal

    if not final_items:

        return {
            "success": False,
            "message": (
                "No valid menu items remain "
                "in the order."
            )
        }

    # ========================================================
    # LOAD BUSINESS
    # ========================================================

    from models.business import Business

    business = Business.query.get(
        business_id
    )

    if not business:

        return {
            "success": False,
            "message": (
                "Restaurant could not be found."
            )
        }

    # ========================================================
    # CREATE REAL ORDER
    # ========================================================

    order = Order(
        customer_name=(
            customer.name
            or "Customer"
        ),
        customer_phone=customer.phone,
        delivery_address="Unknown",
        total_price=final_total,
        status="Pending",

        # IMPORTANT:
        # The order is NOT paid yet.
        payment_status="Unpaid",

        payment_method=None,
        payment_token=None,
        payment_transaction_id=None,
        paid_at=None,

        business_id=business_id,
        customer_id=customer.id
    )

    db.session.add(
        order
    )

    db.session.flush()

    # ========================================================
    # CREATE ORDER ITEMS
    # ========================================================

    for item in final_items:

        order_item = OrderItem(
            name=item["name"],
            quantity=item["quantity"],
            price=item["price"],
            subtotal=item["subtotal"],
            order_id=order.id
        )

        db.session.add(
            order_item
        )

        # ========================================================
    # RESERVE INVENTORY
    # ========================================================

    inventory_result = reserve_inventory_for_order(order)

    if not inventory_result.get("success"):

        db.session.rollback()

        ingredient_name = inventory_result.get(
            "ingredient_name"
        )

        if ingredient_name:
            message = (
                f"Sorry, this order cannot be confirmed because "
                f"{ingredient_name} is no longer available."
            )
        else:
            message = (
                "Sorry, this order cannot be confirmed because "
                "one or more ingredients are no longer available."
            )

        return {
            "success": False,
            "message": message,
            "inventory_error": inventory_result.get("reason")
        }

    # ========================================================
    # DEMO PAYMENT
    # ========================================================

    is_demo_order = (
        str(phone).startswith("DEMO-")
    )

    if is_demo_order:

        order.mark_as_paid(
            payment_method="Demo",
            transaction_id=(
                f"DEMO-{order.id}"
            ),
        )

        # Demo payment completes the order immediately, so consume
        # the inventory that was just reserved.
        consumed = consume_inventory_for_order(order.id)

        if consumed < len(inventory_result.get("reservations", [])):
            db.session.rollback()

            return {
                "success": False,
                "message": (
                    "The demo order could not finalize inventory. "
                    "Please try again."
                )
            }

        order.status = "Completed"

        preview.status = "confirmed"

        db.session.commit()

        return {
            "success": True,

            "order": {
                "id": order.id,
                "status": order.status,
                "payment_status": (
                    order.payment_status
                ),
                "total": final_total,
                "currency": "FCFA",
                "items": final_items,
            },

            "payment": {
                "provider": "Demo",
                "status": "Paid",
                "checkout_url": None,
                "token": None,
            },

            "message": (
                f"Demo order #{order.id} "
                "has been confirmed successfully. "
                "Payment was simulated for this demonstration."
            ),
        }


    # ========================================================
    # CREATE REAL PAYDUNYA CHECKOUT
    # ========================================================

    try:

        payment = create_checkout_invoice(
            order=order,
            business=business,
            customer=customer,
            items=final_items
        )

    except Exception as exc:

        print(
            f"PAYDUNYA CHECKOUT ERROR: {exc}"
        )

        db.session.rollback()

        return {
            "success": False,
            "message": (
                "I couldn't create the payment "
                "link right now. Please try again."
            ),
            "payment_error": str(exc)
        }

    if not payment.get(
        "success"
    ):

        db.session.rollback()

        return {
            "success": False,
            "message": (
                "I couldn't create the payment "
                "link right now. Please try again."
            )
        }

    # ========================================================
    # SAVE PAYDUNYA TOKEN
    # ========================================================

    payment_token = payment.get(
        "token"
    )

    checkout_url = payment.get(
        "checkout_url"
    )

    if not payment_token or not checkout_url:

        db.session.rollback()

        return {
            "success": False,
            "message": (
                "PayDunya did not return "
                "a valid payment link."
            )
        }

    order.payment_token = (
        payment_token
    )

    order.payment_status = "Unpaid"

    order.payment_method = (
        "PayDunya"
    )

    # ========================================================
    # COMPLETE PREVIEW
    # ========================================================

    preview.status = "confirmed"

    print(
        "[DEBUG] About to commit PayDunya order",
        flush=True,
    )

    db.session.commit()

    print(
        "[DEBUG] PayDunya order commit completed",
        flush=True,
    )

    # ========================================================
    # RETURN PAYMENT INFORMATION
    # ========================================================

    print(
        "[DEBUG] Returning PayDunya payment result",
        flush=True,
    )

    return {
        "success": True,

        "order": {
            "id": order.id,
            "status": order.status,
            "payment_status": order.payment_status,
            "total": final_total,
            "currency": "FCFA",
            "items": final_items
        },

        "payment": {
            "provider": "PayDunya",
            "status": "Unpaid",
            "checkout_url": checkout_url,
            "token": payment_token
        },

        "message": (
            f"Order #{order.id} has been created. "
            "Payment is required to complete the order."
        )
    }


# ============================================================
# DISCARD PREVIEW
# ============================================================

def discard_pending_order(
    business_id,
    phone
):

    customer = Customer.query.filter_by(
        business_id=business_id,
        phone=phone
    ).first()

    if not customer:

        return {
            "success": False,
            "message": "Customer not found."
        }

    preview = PendingOrder.query.filter_by(
        business_id=business_id,
        customer_id=customer.id,
        status="pending"
    ).order_by(
        PendingOrder.id.desc()
    ).first()

    if not preview:

        return {
            "success": False,
            "message": (
                "No pending order preview."
            )
        }

    preview.status = "cancelled"

    db.session.commit()

    return {
        "success": True,
        "message": (
            "The order preview has been cancelled."
        )
    }


# ============================================================
# MODIFY ACTIVE ORDER
# ============================================================

def modify_active_order(
    business_id,
    phone,
    message
):

    customer = Customer.query.filter_by(
        business_id=business_id,
        phone=phone
    ).first()

    if not customer:

        return {
            "success": False,
            "message": "Customer not found."
        }

    order = Order.query.filter(
        Order.business_id == business_id,
        Order.customer_id == customer.id,
        Order.status.in_([
            "Pending",
            "Preparing"
        ])
    ).order_by(
        Order.id.desc()
    ).first()

    if not order:

        return {
            "success": False,
            "message": (
                "No active order found."
            )
        }

    command = interpret_order_request(
        message,
        order
    )

    if command.get(
        "action"
    ) == "unknown":

        return {
            "success": False,
            "message": (
                "I couldn't understand "
                "the requested change."
            )
        }

    result = execute_order_action(
        business_id,
        order,
        command
    )

    result["order_id"] = order.id

    return result


# ============================================================
# CANCEL ACTIVE ORDER
# ============================================================

def cancel_active_order(
    business_id,
    phone
):

    customer = Customer.query.filter_by(
        business_id=business_id,
        phone=phone
    ).first()

    if not customer:

        return {
            "success": False,
            "message": "Customer not found."
        }

    order = Order.query.filter(
        Order.business_id == business_id,
        Order.customer_id == customer.id,
        Order.status.in_([
            "Pending",
            "Preparing"
        ])
    ).order_by(
        Order.id.desc()
    ).first()

    if not order:

        return {
            "success": False,
            "message": (
                "No active order found."
            )
        }

    order.status = "Cancelled"

    db.session.commit()

    return {
        "success": True,
        "order_id": order.id,
        "status": "Cancelled",
        "message": (
            f"Order #{order.id} has been cancelled."
        )
    }