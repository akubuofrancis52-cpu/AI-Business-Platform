from datetime import datetime, timezone

from database.db import db


class InventoryReservation(db.Model):
    __tablename__ = "inventory_reservations"

    id = db.Column(db.Integer, primary_key=True)

    order_id = db.Column(
        db.Integer,
        db.ForeignKey("order.id"),
        nullable=False,
        index=True,
    )

    ingredient_id = db.Column(
        db.Integer,
        db.ForeignKey("ingredients.id"),
        nullable=False,
        index=True,
    )

    quantity_reserved = db.Column(
        db.Float,
        nullable=False,
    )

    status = db.Column(
        db.String(20),
        nullable=False,
        default="reserved",
        index=True,
    )

    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    order = db.relationship(
        "Order",
        backref=db.backref(
            "inventory_reservations",
            lazy=True,
            cascade="all, delete-orphan",
        ),
    )

    ingredient = db.relationship(
        "Ingredient",
        backref=db.backref(
            "inventory_reservations",
            lazy=True,
            cascade="all, delete-orphan",
        ),
    )
