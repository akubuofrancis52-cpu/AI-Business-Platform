from datetime import datetime

from database.db import db


class Order(db.Model):

    __tablename__ = "order"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    customer_name = db.Column(
        db.String(100),
        nullable=False
    )

    customer_phone = db.Column(
        db.String(50),
        nullable=False
    )

    delivery_address = db.Column(
        db.String(255),
        nullable=True
    )

    total_price = db.Column(
        db.Float,
        nullable=False
    )

    status = db.Column(
        db.String(50),
        default="Pending"
    )

    # ==========================================================
    # PAYMENT
    # ==========================================================

    payment_status = db.Column(
        db.String(50),
        nullable=False,
        default="Unpaid"
    )

    payment_token = db.Column(
        db.String(150),
        nullable=True
    )

    payment_method = db.Column(
        db.String(50),
        nullable=True
    )

    payment_transaction_id = db.Column(
        db.String(150),
        nullable=True
    )

    paid_at = db.Column(
        db.DateTime,
        nullable=True
    )

    # ==========================================================
    # TIMESTAMPS
    # ==========================================================

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=True
    )

    # ==========================================================
    # RELATIONSHIPS
    # ==========================================================

    customer_id = db.Column(
        db.Integer,
        db.ForeignKey("customer.id"),
        nullable=True
    )

    business_id = db.Column(
        db.Integer,
        db.ForeignKey("business.id"),
        nullable=False
    )

    # ==========================================================
    # HELPERS
    # ==========================================================

    @property
    def is_paid(self):
        """
        True only when the payment has been confirmed.
        """

        return (
            self.payment_status or ""
        ).lower() == "paid"

    @property
    def is_unpaid(self):
        return not self.is_paid

    def mark_as_paid(
        self,
        payment_method=None,
        transaction_id=None
    ):
        """
        Mark this order as paid.

        Revenue should only be counted after this method
        has been called following verified payment confirmation.
        """

        self.payment_status = "Paid"

        if not self.paid_at:
            self.paid_at = datetime.utcnow()

        if payment_method:
            self.payment_method = payment_method

        if transaction_id:
            self.payment_transaction_id = transaction_id

    def mark_as_unpaid(self):
        """
        Mark the order as unpaid.

        Normally used before payment has been confirmed.
        """

        self.payment_status = "Unpaid"
        self.paid_at = None

    def __repr__(self):

        return (
            f"<Order {self.id} - "
            f"{self.status} - "
            f"{self.payment_status}>"
        )