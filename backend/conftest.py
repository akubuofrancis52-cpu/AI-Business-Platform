"""
Botify AI pytest safety configuration.

Tests NEVER use the production DATABASE_URL.

Default:
    SQLite in-memory database

Optional:
    BOTIFY_TEST_DATABASE_URL for dedicated PostgreSQL tests.
"""

import os

_original_database_url = os.environ.get("DATABASE_URL", "")

if _original_database_url:
    os.environ["BOTIFY_PRODUCTION_DATABASE_URL"] = _original_database_url

_test_database_url = os.environ.get(
    "BOTIFY_TEST_DATABASE_URL",
    "sqlite:///:memory:",
)

if not _test_database_url:
    raise RuntimeError(
        "TEST SAFETY FAILURE: BOTIFY_TEST_DATABASE_URL is empty."
    )

# This happens before pytest imports test modules, preventing production DB use.
os.environ["DATABASE_URL"] = _test_database_url


def pytest_configure(config):
    database_url = os.environ.get("DATABASE_URL", "")
    production_url = os.environ.get(
        "BOTIFY_PRODUCTION_DATABASE_URL",
        "",
    )

    if production_url and database_url == production_url:
        raise RuntimeError(
            "TEST SAFETY FAILURE: pytest is using the production database."
        )

    if not database_url.startswith(
        ("sqlite://", "postgresql://", "postgresql+")
    ):
        raise RuntimeError(
            "TEST SAFETY FAILURE: unsupported test database URL."
        )


import pytest


@pytest.fixture(scope="session", autouse=True)
def isolated_database():
    """
    Create the complete isolated database after test modules have imported
    the application and its models.
    """
    from app import app
    from database.db import db

    # Register models before create_all().
    import models.user
    import models.business
    import models.menu
    import models.customer
    import models.customer_preference
    import models.ingredient
    import models.menu_ingredient
    import models.order
    import models.order_item
    import models.pending_order
    import models.payment
    import models.conversation
    import models.support_ticket
    import models.inventory_reservation
    import models.whatsapp_message

    with app.app_context():
        db.create_all()

        yield

        db.session.remove()
        db.drop_all()


def pytest_sessionfinish(session, exitstatus):
    # Never restore the production URL.
    os.environ["DATABASE_URL"] = os.environ.get(
        "BOTIFY_TEST_DATABASE_URL",
        "sqlite:///:memory:",
    )
