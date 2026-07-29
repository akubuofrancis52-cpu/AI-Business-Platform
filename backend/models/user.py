from database.db import db


class User(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(db.String(100), unique=True, nullable=False)

    email = db.Column(db.String(120), unique=True, nullable=False)

    password = db.Column(db.String(255), nullable=False)

    language = db.Column(
        db.String(20),
        default="English"
    )

    businesses = db.relationship(
        "Business",
        backref="owner",
        lazy=True
    )