from datetime import datetime

from database.db import db


class Order(db.Model):

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

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=True
    )

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

    def __repr__(self):

        return f"<Order {self.id} - {self.status}>"