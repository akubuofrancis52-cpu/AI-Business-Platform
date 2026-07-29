from database.db import db


class Conversation(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )


    message = db.Column(
        db.Text,
        nullable=False
    )


    response = db.Column(
        db.Text,
        nullable=False
    )


    created_at = db.Column(
        db.DateTime,
        server_default=db.func.now()
    )


    customer_id = db.Column(
        db.Integer,
        db.ForeignKey("customer.id"),
        nullable=False
    )