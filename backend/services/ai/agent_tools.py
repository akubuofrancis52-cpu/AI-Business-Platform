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

    results = search_menu(
        business_id,
        query,
        limit=5
    )

    return {
        "success": True,
        "results": results
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

def get_pending_order(business_id, phone):

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
            "message": "No pending order preview."
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
                "The pending order preview "
                "could not be read."
            )
        }

    return {
        "success": True,
        "preview": {
            "id": preview.id,
            "items": items,
            "total": float(
                preview.total_price or 0
            ),
            "status": preview.status
        }
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

    if not items:

        if unmatched:

            return {
                "success": False,
                "message": (
                    "I couldn't find the requested "
                    "item on the menu."
                ),
                "unmatched": unmatched
            }

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
    # CREATE PAYDUNYA CHECKOUT
    # ========================================================

    try:

        payment = create_checkout_invoice(
            order=order,
            business=business,
            customer=customer,
            items=final_items
        )

    except Exception as exc:

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

    db.session.commit()

    # ========================================================
    # RETURN PAYMENT INFORMATION
    # ========================================================

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