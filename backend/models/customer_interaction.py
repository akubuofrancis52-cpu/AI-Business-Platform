from datetime import datetime

from database.db import db


class CustomerInteraction(db.Model):
    __tablename__ = "customer_interactions"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    customer_id = db.Column(
        db.Integer,
        db.ForeignKey("customer.id"),
        nullable=False,
        index=True
    )

    business_id = db.Column(
        db.Integer,
        db.ForeignKey("business.id"),
        nullable=False,
        index=True
    )

    order_id = db.Column(
        db.Integer,
        db.ForeignKey("order.id"),
        nullable=True,
        index=True
    )

    interaction_type = db.Column(
        db.String(50),
        nullable=False,
        index=True
    )

    sentiment = db.Column(
        db.String(20),
        nullable=True,
        index=True
    )

    message = db.Column(
        db.Text,
        nullable=False
    )

    metadata_json = db.Column(
        db.Text,
        nullable=True
    )

    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True
    )

    customer = db.relationship(
        "Customer",
        backref=db.backref(
            "interactions",
            lazy=True,
            cascade="all, delete-orphan"
        )
    )

    order = db.relationship(
        "Order",
        backref=db.backref(
            "customer_interactions",
            lazy=True
        )
    )

    def __repr__(self):
        return (
            f"<CustomerInteraction "
            f"{self.interaction_type} "
            f"{self.sentiment}>"
        )
