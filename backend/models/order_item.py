from database.db import db


class OrderItem(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )


    name = db.Column(
        db.String(100),
        nullable=False
    )


    quantity = db.Column(
        db.Integer,
        default=1
    )


    price = db.Column(
        db.Float,
        nullable=False
    )


    subtotal = db.Column(
        db.Float,
        nullable=False
    )


    order_id = db.Column(
        db.Integer,
        db.ForeignKey("order.id")
    )


    order = db.relationship(
        "Order",
        backref="items"
    )