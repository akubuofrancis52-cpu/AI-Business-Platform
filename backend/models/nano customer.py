from database.db import db


class Customer(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(100),
        nullable=True
    )

    phone = db.Column(
        db.String(50),
        nullable=False
    )

    language = db.Column(
        db.String(20),
        default="English"
    )

    business_id = db.Column(
        db.Integer,
        db.ForeignKey("business.id"),
        nullable=False
    )


    def __repr__(self):
        return f"<Customer {self.phone}>"w