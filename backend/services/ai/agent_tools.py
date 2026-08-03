import json

from database.db import db

from models.customer import Customer
from models.order import Order
from models.order_item import OrderItem
from models.pending_order import PendingOrder

from services.ai.menu_intelligence import (
    search_menu
)

from services.ai.order_extractor import (
    extract_order
)

from services.ai.order_modifier import (
    interpret_order_request
)

from services.ai.order_executor import (
    execute_order_action
)


# ==========================
# CUSTOMER
# ==========================

def get_customer(
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

    return {
        "success": True,
        "customer": {
            "id": customer.id,
            "name": customer.name,
            "phone": customer.phone,
            "language": customer.language
        }
    }


# ==========================
# MENU SEARCH
# ==========================

def tool_search_menu(
    business_id,
    query
):

    results = search_menu(
        business_id,
        query,
        limit=5
    )

    return {
        "success": True,
        "results": results
    }


# ==========================
# ACTIVE ORDER
# ==========================

def get_active_order(
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
            "message": "No active order found."
        }

    return {
        "success": True,
        "order": {
            "id": order.id,
            "status": order.status,
            "total": float(
                order.total_price or 0
            ),
            "items": [
                {
                    "name": item.name,
                    "quantity": item.quantity,
                    "price": float(
                        item.price
                    ),
                    "subtotal": float(
                        item.subtotal
                    )
                }
                for item in order.items
            ]
        }
    }


# ==========================
# PENDING PREVIEW
# ==========================

def get_pending_order(
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
            "message": "No pending order preview."
        }

    try:

        items = json.loads(
            preview.items_json
        )

    except json.JSONDecodeError:

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


# ==========================
# CREATE ORDER PREVIEW
# ==========================

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

        return {
            "success": False,
            "message": (
                "I could not find valid "
                "available menu items."
            ),
            "unmatched": unmatched
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

    # Remove previous pending preview
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
            extracted["total"]
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
                preview.total_price
            ),
            "currency": "FCFA",
            "status": "pending"
        },
        "message": (
            "Order preview created. "
            "Customer confirmation is required."
        )
    }


# ==========================
# CONFIRM ORDER
# ==========================

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

    except json.JSONDecodeError:

        return {
            "success": False,
            "message": (
                "The pending order is invalid."
            )
        }

    if not items:

        return {
            "success": False,
            "message": "The pending order is empty."
        }

    # Revalidate every item against the real menu
    final_items = []
    final_total = 0

    from models.menu import Menu

    for item in items:

        menu = Menu.query.filter(
            Menu.business_id == business_id,
            Menu.name == item["name"],
            Menu.available.is_(True)
        ).first()

        if not menu:

            return {
                "success": False,
                "message": (
                    f"{item['name']} is no longer "
                    "available."
                )
            }

        quantity = int(
            item.get("quantity", 1)
        )

        if quantity < 1:
            quantity = 1

        price = float(
            menu.price
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

    # Create the real order only now
    order = Order(
        customer_name=(
            customer.name
            or "Customer"
        ),
        customer_phone=customer.phone,
        delivery_address="Unknown",
        total_price=final_total,
        status="Pending",
        business_id=business_id,
        customer_id=customer.id
    )

    db.session.add(
        order
    )

    db.session.flush()

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

    preview.status = "confirmed"

    db.session.commit()

    return {
        "success": True,
        "order": {
            "id": order.id,
            "status": order.status,
            "total": final_total,
            "items": final_items
        },
        "message": (
            f"Order #{order.id} has been confirmed."
        )
    }


# ==========================
# DISCARD PREVIEW
# ==========================

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
            "message": "No pending order preview."
        }

    preview.status = "cancelled"

    db.session.commit()

    return {
        "success": True,
        "message": (
            "The order preview has been cancelled."
        )
    }


# ==========================
# MODIFY ACTIVE ORDER
# ==========================

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
            "message": "No active order found."
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


# ==========================
# CANCEL ACTIVE ORDER
# ==========================

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
            "message": "No active order found."
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