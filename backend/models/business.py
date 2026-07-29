from database.db import db


class Business(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(150),
        nullable=False
    )

    business_type = db.Column(
        db.String(50),
        nullable=False
    )

    address = db.Column(
        db.String(255)
    )

    phone = db.Column(
        db.String(30)
    )

    owner_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )


    menus = db.relationship(
        "Menu",
        backref="business",
        lazy=True,
        cascade="all, delete-orphan"
    )


    orders = db.relationship(
        "Order",
        backref="business",
        lazy=True,
        cascade="all, delete-orphan"
    )


    customers = db.relationship(
        "Customer",
        backref="business",
        lazy=True,
        cascade="all, delete-orphan"
    )