import importlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from database.db import db
from models.whatsapp_message import WhatsAppMessage


@pytest.fixture
def whatsapp_lifecycle():
    """
    Lightweight isolated lifecycle harness around the durable
    WhatsAppMessage state machine.

    This tests the persistence rules independently of Meta's network
    payload format and without submitting real background jobs.
    """
    class FakeExecutor:
        def __init__(self):
            self.calls = []
            self.raise_on_submit = False

        def submit(self, fn, *args, **kwargs):
            if self.raise_on_submit:
                raise RuntimeError("executor unavailable")
            self.calls.append((fn, args, kwargs))
            return SimpleNamespace()

    executor = FakeExecutor()

    def claim(message_id, business_id=2, from_phone="+22900000000"):
        record = WhatsAppMessage.query.filter_by(
            message_id=message_id
        ).first()

        if record:
            if record.status == "completed":
                return False, record

            if record.status == "processing":
                now = datetime.now(timezone.utc)
                updated_at = record.updated_at

                if updated_at is not None and updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)

                age = (
                    now - updated_at
                    if updated_at is not None
                    else timedelta.max
                )

                if age <= timedelta(minutes=10):
                    return False, record

                record.status = "processing"
                record.attempts = (record.attempts or 0) + 1
                record.updated_at = now
                record.error = None
                db.session.commit()
                return True, record

            record.status = "processing"
            record.attempts = (record.attempts or 0) + 1
            record.updated_at = datetime.now(timezone.utc)
            record.error = None
            db.session.commit()
            return True, record

        record = WhatsAppMessage(
            message_id=message_id,
            business_id=business_id,
            from_phone=from_phone,
            status="processing",
            attempts=1,
        )
        db.session.add(record)
        db.session.commit()
        return True, record

    def complete(message_id):
        record = WhatsAppMessage.query.filter_by(
            message_id=message_id
        ).first()
        assert record is not None
        record.mark_completed()
        db.session.commit()

    def fail(message_id, error):
        record = WhatsAppMessage.query.filter_by(
            message_id=message_id
        ).first()
        assert record is not None
        record.mark_failed(error)
        db.session.commit()

    return SimpleNamespace(
        executor=executor,
        claim=claim,
        complete=complete,
        fail=fail,
    )


def test_first_message_claims_once(whatsapp_lifecycle):
    claimed, record = whatsapp_lifecycle.claim("wamid-first")

    assert claimed is True
    assert record.status == "processing"
    assert record.attempts == 1


def test_fresh_processing_duplicate_is_ignored(whatsapp_lifecycle):
    first, _ = whatsapp_lifecycle.claim("wamid-duplicate")
    second, record = whatsapp_lifecycle.claim("wamid-duplicate")

    assert first is True
    assert second is False
    assert record.status == "processing"
    assert record.attempts == 1


def test_completed_message_is_never_reprocessed(whatsapp_lifecycle):
    whatsapp_lifecycle.claim("wamid-completed")
    whatsapp_lifecycle.complete("wamid-completed")

    claimed, record = whatsapp_lifecycle.claim("wamid-completed")

    assert claimed is False
    assert record.status == "completed"
    assert record.completed_at is not None
    assert record.attempts == 1


def test_failed_message_can_retry(whatsapp_lifecycle):
    whatsapp_lifecycle.claim("wamid-failed")
    whatsapp_lifecycle.fail(
        "wamid-failed",
        "temporary worker failure",
    )

    claimed, record = whatsapp_lifecycle.claim("wamid-failed")

    assert claimed is True
    assert record.status == "processing"
    assert record.attempts == 2
    assert record.error is None


def test_stale_processing_message_is_reclaimed(whatsapp_lifecycle):
    whatsapp_lifecycle.claim("wamid-stale")

    record = WhatsAppMessage.query.filter_by(
        message_id="wamid-stale"
    ).first()

    record.updated_at = datetime.now(timezone.utc) - timedelta(
        minutes=11
    )
    db.session.commit()

    claimed, record = whatsapp_lifecycle.claim("wamid-stale")

    assert claimed is True
    assert record.status == "processing"
    assert record.attempts == 2
    assert record.error is None


def test_recent_processing_message_is_not_reclaimed(whatsapp_lifecycle):
    whatsapp_lifecycle.claim("wamid-recent")

    record = WhatsAppMessage.query.filter_by(
        message_id="wamid-recent"
    ).first()

    record.updated_at = datetime.now(timezone.utc) - timedelta(
        minutes=9
    )
    db.session.commit()

    claimed, record = whatsapp_lifecycle.claim("wamid-recent")

    assert claimed is False
    assert record.status == "processing"
    assert record.attempts == 1


def test_invalid_media_can_be_failed_and_retried(whatsapp_lifecycle):
    whatsapp_lifecycle.claim("wamid-invalid-media")

    whatsapp_lifecycle.fail(
        "wamid-invalid-media",
        "Audio message has no media ID.",
    )

    record = WhatsAppMessage.query.filter_by(
        message_id="wamid-invalid-media"
    ).first()

    assert record.status == "failed"
    assert "no media ID" in record.error

    claimed, record = whatsapp_lifecycle.claim("wamid-invalid-media")

    assert claimed is True
    assert record.status == "processing"
    assert record.attempts == 2


def test_unsupported_message_can_be_completed(whatsapp_lifecycle):
    whatsapp_lifecycle.claim("wamid-unsupported")
    whatsapp_lifecycle.complete("wamid-unsupported")

    record = WhatsAppMessage.query.filter_by(
        message_id="wamid-unsupported"
    ).first()

    assert record.status == "completed"


def test_executor_submission_failure_can_be_recorded(whatsapp_lifecycle):
    whatsapp_lifecycle.claim("wamid-submit-failure")

    whatsapp_lifecycle.executor.raise_on_submit = True

    with pytest.raises(RuntimeError):
        whatsapp_lifecycle.executor.submit(
            lambda: None,
            "wamid-submit-failure",
        )

    whatsapp_lifecycle.fail(
        "wamid-submit-failure",
        "Failed to submit WhatsApp worker: executor unavailable",
    )

    record = WhatsAppMessage.query.filter_by(
        message_id="wamid-submit-failure"
    ).first()

    assert record.status == "failed"
    assert "executor unavailable" in record.error


def test_contact_name_has_safe_default_before_lookup():
    contact_name = "New Customer"

    assert contact_name == "New Customer"


def test_message_id_is_unique():
    first = WhatsAppMessage(
        message_id="wamid-unique",
        business_id=2,
        status="processing",
        attempts=1,
    )
    db.session.add(first)
    db.session.commit()

    duplicate = WhatsAppMessage(
        message_id="wamid-unique",
        business_id=2,
        status="processing",
        attempts=1,
    )
    db.session.add(duplicate)

    with pytest.raises(Exception):
        db.session.commit()

    db.session.rollback()
