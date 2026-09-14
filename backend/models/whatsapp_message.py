from datetime import datetime, timezone

from database.db import db


class WhatsAppMessage(db.Model):
    """
    Durable idempotency record for inbound WhatsApp messages.

    A WhatsApp message ID (wamid) is globally unique for our purposes,
    while business_id keeps the record explicitly tenant-scoped.
    """

    __tablename__ = "whatsapp_messages"

    id = db.Column(db.Integer, primary_key=True)

    message_id = db.Column(
        db.String(255),
        nullable=False,
        unique=True,
        index=True,
    )

    business_id = db.Column(
        db.Integer,
        nullable=False,
        index=True,
    )

    from_phone = db.Column(
        db.String(255),
        nullable=True,
    )

    status = db.Column(
        db.String(32),
        nullable=False,
        default="processing",
        index=True,
    )

    attempts = db.Column(
        db.Integer,
        nullable=False,
        default=1,
    )

    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    updated_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    completed_at = db.Column(
        db.DateTime(timezone=True),
        nullable=True,
    )

    error = db.Column(
        db.Text,
        nullable=True,
    )

    def mark_completed(self):
        self.status = "completed"
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.error = None

    def mark_failed(self, error=None):
        self.status = "failed"
        self.updated_at = datetime.now(timezone.utc)
        self.error = str(error)[:4000] if error else None

    def __repr__(self):
        return (
            f"<WhatsAppMessage {self.message_id} "
            f"status={self.status}>"
        )
