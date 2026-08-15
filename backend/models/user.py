from database.db import db
from werkzeug.security import generate_password_hash, check_password_hash


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

    def set_password(self, password):
        """Hash and store a user's password."""
        self.password = generate_password_hash(password)

    def check_password(self, password):
        """Verify a plaintext password against the stored hash."""
        return check_password_hash(self.password, password)
