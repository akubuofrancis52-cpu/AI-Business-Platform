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

    query_text = str(
        query or ""
    ).strip().lower()

    # ========================================================
    # FULL MENU REQUEST
    # ========================================================

    # ========================================================
    # FULL MENU DETECTION — MULTILINGUAL
    # ========================================================

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
        "what do you serve",
        "list menu",
        "list the menu",
        "available menu",
        "full menu",

        # French
        "menu",
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

    # Normalize punctuation before checking.
    menu_query = (
        query_text
        .lower()
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

    # Collapse repeated whitespace after punctuation cleanup.
    menu_query = " ".join(
        menu_query.split()
    )

    # ========================================================
    # ROBUST MULTILINGUAL FULL MENU DETECTION
    # ========================================================

    menu_words = (
        "menu",
        "menú",
        "menü",
        "menyu",
    )

    full_menu_intent_phrases = (
        # English
        "show me",
        "show your",
        "send me",
        "give me",
        "see",
        "view",
        "list",
        "what is on",
        "what do you have",
        "what do you serve",

        # French
        "je peux voir",
        "je veux voir",
        "je voudrais voir",
        "puis je voir",
        "puis-je voir",
        "montre moi",
        "montre-moi",
        "montrez moi",
        "montrez-moi",
        "donne moi",
        "donne-moi",
        "donnez moi",
        "donnez-moi",
        "voir",
        "quel est",
        "qu est ce qu il y a",
        "qu'est-ce qu'il y a",

        # Spanish
        "puedo ver",
        "quiero ver",
        "muéstrame",
        "muestrame",
        "dame",
        "ver",

        # Portuguese
        "posso ver",
        "quero ver",
        "mostre",
        "mostre-me",
        "me mostre",
        "me dê",
        "me de",

        # Italian
        "posso vedere",
        "voglio vedere",
        "mostrami",
        "fammi vedere",

        # German
        "kann ich",
        "zeig mir",
        "zeige mir",
        "was gibt es",
    )

    # If the customer explicitly mentions the menu together
    # with a request to see/show/list it, return the full menu.
    has_menu_word = any(
        word in menu_query
        for word in menu_words
    )

    has_full_menu_intent = any(
        phrase in menu_query
        for phrase in full_menu_intent_phrases
    )

    is_full_menu = (
        menu_query in full_menu_phrases
        or (
            has_menu_word
            and has_full_menu_intent
        )
    )
    
    if is_full_menu:

        from services.ai.menu_intelligence import (
            get_menu_items
        )

        menu_items = get_menu_items(
            business_id
        )

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
                    item.price
                ),
                "currency": "FCFA"
            })

        return {
            "success": True,
            "results": results,
            "full_menu": True
        }

    # ========================================================
    # NORMAL MENU SEARCH
    # ========================================================

    results = search_menu(
        business_id,
        query,
        limit=5
    )

    return {
        "success": True,
        "results": results,
        "full_menu": False
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

    extracted = extract_order(
        business_id,
        message
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
    # Remove previous pending preview
    # --------------------------------------------------------

    PendingOrder.query.filter_by(
        business_id=business_id,
        customer_id=customer.id,
        status="pending"
    ).delete(
        synchronize_session=False
    )

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