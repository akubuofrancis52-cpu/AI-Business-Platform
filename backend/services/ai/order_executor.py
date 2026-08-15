from database.db import db
import traceback

from models.order import Order
from models.order_item import OrderItem
from models.menu import Menu
from models.customer import Customer



def _find_item(
    order,
    item_name
):
    """
    Find an order item using
    case-insensitive matching.
    """

    if not item_name:
        return None

    target = item_name.strip().lower()

    for item in order.items:

        if item.name.strip().lower() == target:
            return item

    # Partial fallback
    for item in order.items:

        if target in item.name.strip().lower():
            return item

    return None


def recalculate_order_total(
    order
):

    total = 0

    for item in order.items:

        item.quantity = int(
            item.quantity or 0
        )

        item.subtotal = (
            float(item.price)
            * item.quantity
        )

        total += item.subtotal

    order.total_price = total

    return total


def execute_order_action(
    business_id,
    order,
    command
):

    action = command.get(
        "action",
        "unknown"
    )

    item_name = command.get(
        "item_name"
    )

    quantity = command.get(
        "quantity"
    )

    new_item_name = command.get(
        "new_item_name"
    )

    # --------------------------
    # CANCEL
    # --------------------------

    if action == "cancel":

        if order.status in {
            "Completed",
            "Cancelled"
        }:

            return {
                "success": False,
                "message": (
                    "This order can no longer "
                    "be cancelled."
                )
            }

        order.status = "Cancelled"

        db.session.commit()

        return {
            "success": True,
            "action": "cancel",
            "message": (
                f"Order #{order.id} has "
                "been cancelled."
            ),
            "total": float(
                order.total_price or 0
            )
        }

    # --------------------------
    # MODIFY CHECK
    # --------------------------

    if order.status not in {
        "Pending",
        "Preparing"
    }:

        return {
            "success": False,
            "message": (
                "This order can no longer "
                "be modified."
            )
        }

    # --------------------------
    # REMOVE ITEM
    # --------------------------

    if action == "remove_item":

        item = _find_item(
            order,
            item_name
        )

        if not item:

            return {
                "success": False,
                "message": (
                    f"I couldn't find "
                    f"'{item_name}' in "
                    "this order."
                )
            }

        db.session.delete(
            item
        )

        db.session.flush()

        remaining_items = OrderItem.query.filter_by(
            order_id=order.id
        ).all()

        if not remaining_items:

            order.total_price = 0

        else:

            recalculate_order_total(
                order
            )

        db.session.commit()

        return {
            "success": True,
            "action": "remove_item",
            "message": (
                f"{item_name} was removed "
                f"from order #{order.id}."
            ),
            "total": float(
                order.total_price or 0
            )
        }

    # --------------------------
    # SET QUANTITY
    # --------------------------

    if action == "set_quantity":

        item = _find_item(
            order,
            item_name
        )

        if not item:

            return {
                "success": False,
                "message": (
                    f"I couldn't find "
                    f"'{item_name}' in "
                    "this order."
                )
            }

        try:
            quantity = int(
                quantity
            )
        except (TypeError, ValueError):

            return {
                "success": False,
                "message": (
                    "The quantity was invalid."
                )
            }

        if quantity < 1:

            return {
                "success": False,
                "message": (
                    "Quantity must be at least 1."
                )
            }

        item.quantity = quantity

        recalculate_order_total(
            order
        )

        db.session.commit()

        return {
            "success": True,
            "action": "set_quantity",
            "message": (
                f"{item.name} is now "
                f"{quantity}."
            ),
            "total": float(
                order.total_price or 0
            )
        }

    # --------------------------
    # ADD ITEM
    # --------------------------

    if action == "add_item":

        if not item_name:

            return {
                "success": False,
                "message": (
                    "I couldn't determine "
                    "which item to add."
                )
            }

        menu = Menu.query.filter(
            Menu.business_id == business_id,
            Menu.available.is_(True),
            Menu.name.ilike(
                item_name
            )
        ).first()

        if not menu:

            return {
                "success": False,
                "message": (
                    f"'{item_name}' is not "
                    "available on the menu."
                )
            }

        try:
            quantity = int(
                quantity or 1
            )
        except (TypeError, ValueError):
            quantity = 1

        if quantity < 1:
            quantity = 1

        existing = _find_item(
            order,
            menu.name
        )

        if existing:

            existing.quantity += quantity

        else:

            order_item = OrderItem(
                name=menu.name,
                quantity=quantity,
                price=float(menu.price),
                subtotal=(
                    float(menu.price)
                    * quantity
                ),
                order_id=order.id
            )

            db.session.add(
                order_item
            )

        db.session.flush()

        recalculate_order_total(
            order
        )

        db.session.commit()

        return {
            "success": True,
            "action": "add_item",
            "message": (
                f"{quantity} × {menu.name} "
                "added to the order."
            ),
            "total": float(
                order.total_price or 0
            )
        }

    # --------------------------
    # REPLACE ITEM
    # --------------------------

    if action == "replace_item":

        old_item = _find_item(
            order,
            item_name
        )

        if not old_item:

            return {
                "success": False,
                "message": (
                    f"I couldn't find "
                    f"'{item_name}' in "
                    "this order."
                )
            }

        if not new_item_name:

            return {
                "success": False,
                "message": (
                    "I couldn't determine "
                    "the replacement item."
                )
            }

        menu = Menu.query.filter(
            Menu.business_id == business_id,
            Menu.available.is_(True),
            Menu.name.ilike(
                new_item_name
            )
        ).first()

        if not menu:

            return {
                "success": False,
                "message": (
                    f"'{new_item_name}' is "
                    "not available."
                )
            }

        old_quantity = (
            old_item.quantity or 1
        )

        old_item.name = menu.name
        old_item.price = float(
            menu.price
        )
        old_item.quantity = old_quantity
        old_item.subtotal = (
            float(menu.price)
            * old_quantity
        )

        recalculate_order_total(
            order
        )

        db.session.commit()

        return {
            "success": True,
            "action": "replace_item",
            "message": (
                f"{item_name} was replaced "
                f"with {menu.name}."
            ),
            "total": float(
                order.total_price or 0
            )
        }

    return {
        "success": False,
        "message": (
            "I couldn't understand "
            "that order request."
        )
    }

# --------------------------
# CONFIRM PENDING ORDER
# --------------------------

def confirm_pending_order(business_id: int, customer_phone: str, draft_items: list):
    """
    Takes validated items from a draft order and creates permanent
    Order & OrderItem DB records.
    """
    try:
        # 1. Get or create customer record
        customer = Customer.query.filter_by(
            phone=customer_phone,
            business_id=business_id
        ).first()

        if not customer:
            customer = Customer(
                name="WhatsApp Customer",
                phone=customer_phone,
                language="English",
                business_id=business_id
            )
            db.session.add(customer)
            db.session.flush()

        # 2. Recalculate totals using live database prices
        final_items = []
        total_price = 0.0

        for item in draft_items:
            menu_item = Menu.query.filter(
                Menu.business_id == business_id,
                Menu.name.ilike(item["name"]),
                Menu.available == True
            ).first()

            if not menu_item:
                continue

            quantity = int(item.get("quantity", 1))

            if quantity <= 0:
                continue

            subtotal = float(menu_item.price) * quantity

            final_items.append({
                "name": menu_item.name,
                "quantity": quantity,
                "price": float(menu_item.price),
                "subtotal": subtotal
            })

            total_price += subtotal

        # 3. Make sure at least one valid item remains
        if not final_items:
            return {
                "success": False,
                "message": "Items are no longer available in the menu."
            }

        # 4. Create the primary Order
        order = Order(
            customer_name=customer.name,
            customer_phone=customer.phone,
            delivery_address="Pending pickup/delivery info",
            total_price=total_price,
            status="Pending",
            business_id=business_id,
            customer_id=customer.id
        )

        db.session.add(order)
        db.session.flush()

        # 5. Create OrderItem records
        for item in final_items:
            order_item = OrderItem(
                name=item["name"],
                quantity=item["quantity"],
                price=item["price"],
                subtotal=item["subtotal"],
                order_id=order.id
            )

            db.session.add(order_item)

        # 6. Save everything
        db.session.commit()

        return {
            "success": True,
            "order_id": order.id,
            "total_price": total_price,
            "items": final_items,
            "message": (
                f"Order #{order.id} confirmed successfully "
                f"for {total_price} FCFA."
            )
        }

    except Exception as e:
        db.session.rollback()

        import traceback
        traceback.print_exc()

        return {
            "success": False,
            "message": f"Database error during order confirmation: {str(e)}"
        }
            