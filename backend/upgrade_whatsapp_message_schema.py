"""
Idempotent production migration for WhatsApp durable idempotency.

Safe to run multiple times.

Adds:
    whatsapp_messages

The migration intentionally uses SQLAlchemy inspection/DDL rather than
db.create_all(), so it can be run explicitly against production.
"""

import os
import sys
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    inspect,
    text,
)


TABLE_NAME = "whatsapp_messages"


def get_database_url():
    url = os.environ.get("DATABASE_URL")

    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Refusing to run migration."
        )

    # Render may expose postgres:// while SQLAlchemy expects postgresql://.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]

    # This project uses psycopg v3, not psycopg2.
    # Explicitly select the installed PostgreSQL driver.
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]

    return url


def migrate():
    database_url = get_database_url()

    if not database_url.startswith(("postgresql://", "postgresql+")):
        raise RuntimeError(
            "Refusing to run production migration against a non-PostgreSQL "
            f"database: {database_url.split(':', 1)[0]}"
        )

    print("=" * 64)
    print("BOTIFY AI — WHATSAPP MESSAGE SCHEMA MIGRATION")
    print("=" * 64)
    print()

    engine = create_engine(
        database_url,
        pool_pre_ping=True,
    )

    inspector = inspect(engine)

    print("Database:")
    print(f"  {engine.url.get_backend_name()}")
    print()

    existing_tables = inspector.get_table_names()

    if TABLE_NAME in existing_tables:
        print(f"✓ {TABLE_NAME} already exists.")
        print("  No table creation required.")
        print()

        columns = {
            column["name"]: column
            for column in inspector.get_columns(TABLE_NAME)
        }

        required_columns = {
            "id",
            "message_id",
            "business_id",
            "from_phone",
            "status",
            "attempts",
            "created_at",
            "updated_at",
            "completed_at",
            "error",
        }

        missing = required_columns - set(columns)

        if missing:
            raise RuntimeError(
                "Existing whatsapp_messages table is missing columns: "
                + ", ".join(sorted(missing))
            )

        print("✓ Required columns verified.")

    else:
        print(f"Creating {TABLE_NAME}...")

        metadata = MetaData()

        Table(
            TABLE_NAME,
            metadata,

            Column(
                "id",
                Integer,
                primary_key=True,
            ),

            Column(
                "message_id",
                String(255),
                nullable=False,
                unique=True,
                index=True,
            ),

            Column(
                "business_id",
                Integer,
                nullable=False,
                index=True,
            ),

            Column(
                "from_phone",
                String(255),
                nullable=True,
            ),

            Column(
                "status",
                String(32),
                nullable=False,
                server_default="processing",
                index=True,
            ),

            Column(
                "attempts",
                Integer,
                nullable=False,
                server_default="1",
            ),

            Column(
                "created_at",
                DateTime(timezone=True),
                nullable=False,
                server_default=text("CURRENT_TIMESTAMP"),
            ),

            Column(
                "updated_at",
                DateTime(timezone=True),
                nullable=False,
                server_default=text("CURRENT_TIMESTAMP"),
            ),

            Column(
                "completed_at",
                DateTime(timezone=True),
                nullable=True,
            ),

            Column(
                "error",
                Text,
                nullable=True,
            ),
        )

        metadata.create_all(engine)

        print(f"✓ Created {TABLE_NAME}.")

    print()
    print("=== VERIFYING PRODUCTION SCHEMA ===")

    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = :table_name
                ORDER BY ordinal_position
                """
            ),
            {"table_name": TABLE_NAME},
        )

        rows = result.fetchall()

        if not rows:
            raise RuntimeError(
                f"Migration verification failed: {TABLE_NAME} not found."
            )

        for row in rows:
            print(
                f"  {row.column_name}: "
                f"{row.data_type}, nullable={row.is_nullable}"
            )

        print()

        unique_constraints = conn.execute(
            text(
                """
                SELECT
                    tc.constraint_name,
                    tc.constraint_type
                FROM information_schema.table_constraints tc
                WHERE tc.table_name = :table_name
                  AND tc.constraint_type IN ('UNIQUE', 'PRIMARY KEY')
                ORDER BY tc.constraint_name
                """
            ),
            {"table_name": TABLE_NAME},
        ).fetchall()

        print("Constraints:")

        for constraint in unique_constraints:
            print(
                f"  ✓ {constraint.constraint_name} "
                f"({constraint.constraint_type})"
            )

    engine.dispose()

    print()
    print("=" * 64)
    print("✓ WHATSAPP MESSAGE MIGRATION VERIFIED")
    print("=" * 64)


if __name__ == "__main__":
    try:
        migrate()
    except Exception as exc:
        print()
        print("!" * 64)
        print("MIGRATION FAILED")
        print("!" * 64)
        print(str(exc))
        sys.exit(1)
