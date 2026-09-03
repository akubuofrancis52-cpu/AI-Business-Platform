from database.db import db


class Ingredient(db.Model):
    __tablename__ = "ingredients"

    id = db.Column(
        db.Integer,
        primary_key=True,
    )

    name = db.Column(
        db.String(120),
        nullable=False,
    )

    unit = db.Column(
        db.String(30),
        nullable=False,
        default="unit",
    )

    business_id = db.Column(
        db.Integer,
        db.ForeignKey("business.id"),
        nullable=False,
        index=True,
    )

    active = db.Column(
        db.Boolean,
        nullable=False,
        default=True,
    )

    stock_quantity = db.Column(
        db.Float,
        nullable=False,
        default=0,
    )

    low_stock_threshold = db.Column(
        db.Float,
        nullable=False,
        default=0,
    )

    def __repr__(self):
        return (
            f"<Ingredient "
            f"{self.name}: {self.stock_quantity} {self.unit}>"
        )
