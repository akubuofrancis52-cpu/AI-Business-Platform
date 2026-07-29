from database.db import db


class Menu(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(150), nullable=False)

    description = db.Column(db.String(500))

    price = db.Column(db.Float, nullable=False)

    category = db.Column(db.String(100))

    available = db.Column(db.Boolean, default=True)

    business_id = db.Column(
        db.Integer,
        db.ForeignKey("business.id"),
        nullable=False
    )