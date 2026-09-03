from database.db import db
from models.menu import Menu
from models.ingredient import Ingredient
from models.menu_ingredient import MenuIngredient
from models.inventory_reservation import InventoryReservation


ACTIVE_RESERVATION_STATUSES = {"reserved"}


def get_reserved_quantity(ingredient_id):
    row = (
        db.session.query(
            db.func.coalesce(
                db.func.sum(InventoryReservation.quantity_reserved),
                0,
            )
        )
        .filter(
            InventoryReservation.ingredient_id == ingredient_id,
            InventoryReservation.status.in_(ACTIVE_RESERVATION_STATUSES),
        )
        .scalar()
    )

    return float(row or 0)


def get_available_ingredient_quantity(ingredient):
    reserved = get_reserved_quantity(ingredient.id)
    return max(float(ingredient.stock_quantity or 0) - reserved, 0.0)


def check_menu_item_stock(menu_id, quantity=1):
    menu = db.session.get(Menu, menu_id)

    if not menu:
        return {
            "available": False,
            "inventory_configured": False,
            "reason": "menu_item_not_found",
            "missing": [],
            "requirements": [],
        }

    if not menu.available:
        return {
            "available": False,
            "inventory_configured": False,
            "reason": "menu_item_unavailable",
            "missing": [],
            "requirements": [],
        }

    recipe_rows = (
        MenuIngredient.query
        .filter_by(menu_id=menu.id)
        .all()
    )

    if not recipe_rows:
        return {
            "available": True,
            "inventory_configured": False,
            "reason": None,
            "missing": [],
            "requirements": [],
        }

    requirements = []
    missing = []

    requested_quantity = max(int(quantity), 1)

    for recipe in recipe_rows:
        ingredient = recipe.ingredient

        required = (
            float(recipe.quantity_required)
            * requested_quantity
        )

        available_quantity = get_available_ingredient_quantity(
            ingredient
        )

        row = {
            "ingredient_id": ingredient.id,
            "ingredient_name": ingredient.name,
            "required": required,
            "available": available_quantity,
            "unit": ingredient.unit,
        }

        requirements.append(row)

        if (
            not ingredient.active
            or available_quantity < required
        ):
            missing.append(row)

    return {
        "available": len(missing) == 0,
        "inventory_configured": True,
        "reason": (
            "insufficient_inventory"
            if missing
            else None
        ),
        "missing": missing,
        "requirements": requirements,
    }


def filter_inventory_available_menu_items(menu_items, quantity=1):
    """
    Return menu items that are currently orderable.

    Manual Menu.available=False items are already excluded by the
    normal menu layer. For items with configured ingredient recipes,
    also exclude items whose currently available inventory is
    insufficient.

    Items without configured inventory remain available.
    """

    filtered = []

    for item in menu_items:

        result = check_menu_item_stock(
            item.id,
            quantity,
        )

        if (
            result.get("inventory_configured")
            and not result.get("available")
        ):
            continue

        filtered.append(item)

    return filtered

def check_menu_items_stock(items):
    results = []

    for item in items:
        result = check_menu_item_stock(
            item["menu_id"],
            item.get("quantity", 1),
        )

        results.append({
            "menu_id": item["menu_id"],
            "quantity": item.get("quantity", 1),
            **result,
        })

    return results


def reserve_inventory_for_order(order):
    """
    Reserve ingredient stock for an order.

    IMPORTANT:
    - Does not deduct physical stock.
    - Existing active reservations are considered unavailable.
    - Uses row locks where supported (PostgreSQL).
    - Idempotent for an already-reserved order.
    """

    existing = (
        InventoryReservation.query
        .filter_by(
            order_id=order.id,
            status="reserved",
        )
        .first()
    )

    if existing:
        return {
            "success": True,
            "already_reserved": True,
            "reservations": [],
        }

    requirements = {}

    for item in order.items:

        menu = (
            Menu.query
            .filter_by(
                business_id=order.business_id,
                name=item.name,
            )
            .first()
        )

        if not menu:
            continue

        recipe_rows = (
            MenuIngredient.query
            .filter_by(menu_id=menu.id)
            .all()
        )

        for recipe in recipe_rows:
            ingredient_id = recipe.ingredient_id

            requirements[ingredient_id] = (
                requirements.get(ingredient_id, 0)
                + (
                    float(recipe.quantity_required)
                    * int(item.quantity)
                )
            )

    if not requirements:
        return {
            "success": True,
            "already_reserved": False,
            "inventory_configured": False,
            "reservations": [],
        }

    reservations = []

    for ingredient_id, required_quantity in requirements.items():

        ingredient_query = Ingredient.query.filter_by(
            id=ingredient_id
        )

        try:
            ingredient = ingredient_query.with_for_update().one()
        except Exception:
            ingredient = db.session.get(
                Ingredient,
                ingredient_id,
            )

        if not ingredient or not ingredient.active:
            return {
                "success": False,
                "reason": "ingredient_unavailable",
                "ingredient_id": ingredient_id,
            }

        reserved = get_reserved_quantity(
            ingredient.id
        )

        available = max(
            float(ingredient.stock_quantity or 0)
            - reserved,
            0.0,
        )

        if available < required_quantity:
            return {
                "success": False,
                "reason": "insufficient_inventory",
                "ingredient_id": ingredient.id,
                "ingredient_name": ingredient.name,
                "required": required_quantity,
                "available": available,
                "unit": ingredient.unit,
            }

        reservation = InventoryReservation(
            order_id=order.id,
            ingredient_id=ingredient.id,
            quantity_reserved=required_quantity,
            status="reserved",
        )

        db.session.add(reservation)
        reservations.append(reservation)

    db.session.flush()

    return {
        "success": True,
        "already_reserved": False,
        "inventory_configured": True,
        "reservations": reservations,
    }


def release_inventory_for_order(order_id):
    reservations = (
        InventoryReservation.query
        .filter_by(
            order_id=order_id,
            status="reserved",
        )
        .all()
    )

    for reservation in reservations:
        reservation.status = "released"

    db.session.flush()

    return len(reservations)


def consume_inventory_for_order(order_id):
    reservations = (
        InventoryReservation.query
        .filter_by(
            order_id=order_id,
            status="reserved",
        )
        .all()
    )

    consumed = 0

    for reservation in reservations:

        ingredient = (
            Ingredient.query
            .filter_by(id=reservation.ingredient_id)
            .with_for_update()
            .first()
        )

        if not ingredient:
            continue

        ingredient.stock_quantity = max(
            float(ingredient.stock_quantity or 0)
            - float(reservation.quantity_reserved),
            0.0,
        )

        reservation.status = "consumed"
        consumed += 1

    db.session.flush()

    return consumed
