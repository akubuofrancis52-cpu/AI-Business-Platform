from datetime import datetime

from database.db import db


class CustomerPreference(db.Model):
    __tablename__ = "customer_preferences"

    id = db.Column(db.Integer, primary_key=True)

    customer_id = db.Column(
        db.Integer,
        db.ForeignKey("customer.id"),
        nullable=False,
        index=True,
    )

    business_id = db.Column(
        db.Integer,
        db.ForeignKey("business.id"),
        nullable=False,
        index=True,
    )

    preference_type = db.Column(
        db.String(50),
        nullable=False,
        index=True,
    )

    preference_value = db.Column(
        db.String(255),
        nullable=False,
    )

    strength = db.Column(
        db.Integer,
        nullable=False,
        default=1,
    )

    source = db.Column(
        db.String(50),
        nullable=True,
    )

    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    updated_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    customer = db.relationship(
        "Customer",
        backref=db.backref(
            "preferences",
            lazy=True,
            cascade="all, delete-orphan",
        ),
    )

    def __repr__(self):
        return (
            f"<CustomerPreference "
            f"{self.preference_type}: {self.preference_value}>"
        )
