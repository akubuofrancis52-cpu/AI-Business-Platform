"""
Botify AI — Payment schema upgrade

Idempotently upgrades an existing PostgreSQL database for the
hardened PayDunya payment model.

Safe to run multiple times.
"""

import os
import sys

from sqlalchemy import create_engine, inspect, text


def normalize_database_url(url):
    if url.startswith("postgres://"):
        return url.replace(
            "postgres://",
            "postgresql+psycopg://",
            1,
        )

    if url.startswith("postgresql://"):
        return url.replace(
            "postgresql://",
            "postgresql+psycopg://",
            1,
        )

    return url


def main():
    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        print("ERROR: DATABASE_URL is not set.")
        return 1

    database_url = normalize_database_url(database_url)

    engine = create_engine(
        database_url,
        pool_pre_ping=True,
    )

    inspector = inspect(engine)

    if "payment" not in inspector.get_table_names():
        print("ERROR: payment table does not exist.")
        print("Initialize the database schema first.")
        return 1

    columns = {
        column["name"]
        for column in inspector.get_columns("payment")
    }

    with engine.begin() as conn:

        # ----------------------------------------------------
        # checkout_url
        # ----------------------------------------------------

        if "checkout_url" not in columns:
            print("Adding payment.checkout_url...")

            conn.execute(
                text(
                    """
                    ALTER TABLE payment
                    ADD COLUMN checkout_url VARCHAR(500)
                    """
                )
            )

            print("Added payment.checkout_url.")
        else:
            print("payment.checkout_url already exists.")

        # ----------------------------------------------------
        # transaction_id uniqueness
        # ----------------------------------------------------

        constraints = inspector.get_unique_constraints(
            "payment"
        )

        unique_transaction = any(
            set(constraint.get("column_names") or []) == {
                "transaction_id"
            }
            for constraint in constraints
        )

        if unique_transaction:
            print(
                "Unique constraint on "
                "payment.transaction_id already exists."
            )

        else:
            duplicate = conn.execute(
                text(
                    """
                    SELECT transaction_id, COUNT(*)
                    FROM payment
                    WHERE transaction_id IS NOT NULL
                    GROUP BY transaction_id
                    HAVING COUNT(*) > 1
                    LIMIT 1
                    """
                )
            ).first()

            if duplicate:
                print(
                    "ERROR: Duplicate transaction_id detected:"
                )
                print(
                    f"  transaction_id={duplicate[0]!r}, "
                    f"count={duplicate[1]}"
                )
                print(
                    "The unique constraint was NOT created."
                )
                return 1

            print(
                "Creating unique constraint "
                "uq_payment_transaction_id..."
            )

            conn.execute(
                text(
                    """
                    ALTER TABLE payment
                    ADD CONSTRAINT uq_payment_transaction_id
                    UNIQUE (transaction_id)
                    """
                )
            )

            print(
                "Created uq_payment_transaction_id."
            )

    print()
    print("Payment schema upgrade completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
