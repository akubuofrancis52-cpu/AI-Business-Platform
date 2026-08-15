from datetime import datetime

from database.db import db


class Payment(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    order_id = db.Column(
        db.Integer,
        db.ForeignKey("order.id"),
        nullable=False,
        unique=True
    )

    business_id = db.Column(
        db.Integer,
        db.ForeignKey("business.id"),
        nullable=False
    )

    amount = db.Column(
        db.Float,
        nullable=False
    )

    status = db.Column(
        db.String(50),
        nullable=False,
        default="Pending"
    )

    method = db.Column(
        db.String(50),
        nullable=True
    )

    transaction_id = db.Column(
        db.String(255),
        nullable=True
    )

    paid_at = db.Column(
        db.DateTime,
        nullable=True
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    order = db.relationship(
        "Order",
        backref=db.backref(
            "payment",
            uselist=False
        )
    )

    def mark_paid(
        self,
        method=None,
        transaction_id=None
    ):

        self.status = "Paid"

        if method:
            self.method = method

        if transaction_id:
            self.transaction_id = transaction_id

        self.paid_at = datetime.utcnow()

    def __repr__(self):

        return (
            f"<Payment {self.id} "
            f"- Order {self.order_id} "
            f"- {self.status}>"
        )