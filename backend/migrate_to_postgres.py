import argparse
import os

from sqlalchemy import create_engine, text

from app import app, db
from models.user import User
from models.business import Business
from models.menu import Menu
from models.customer import Customer
from models.order import Order
from models.order_item import OrderItem
from models.conversation import Conversation
from models.pending_order import PendingOrder


SOURCE_BUSINESS_ID = 2
SOURCE_USER_ID = 1

EXCLUDE_PHONE_PREFIXES = (
    "DEMO-",
    "TEST-",
)


def row_dict(obj):
    """
    Convert an ORM model instance into a plain
    column/value dictionary.
    """
    return {
        column.name: getattr(
            obj,
            column.name,
        )
        for column in obj.__table__.columns
    }


def should_exclude_customer(customer):
    phone = (
        str(customer.phone or "")
        .strip()
        .upper()
    )

    return phone.startswith(
        EXCLUDE_PHONE_PREFIXES
    )


def reset_postgres_sequences(
    engine
):
    """
    Reset PostgreSQL serial/identity sequences after
    inserting explicit primary-key values.
    """

    table_names = (
        "user",
        "business",
        "menu",
        "customer",
        "order",
        "order_item",
        "conversation",
        "pending_order",
    )

    with engine.begin() as connection:

        for table_name in table_names:

            try:

                result = connection.execute(
                    text(
                        f"""
                        SELECT pg_get_serial_sequence(
                            :table_name,
                            'id'
                        )
                        """
                    ),
                    {
                        "table_name": table_name
                    },
                )

                sequence_name = (
                    result.scalar()
                )

                if not sequence_name:
                    continue

                result = connection.execute(
                    text(
                        f"""
                        SELECT COALESCE(
                            MAX(id),
                            0
                        )
                        FROM "{table_name}"
                        """
                    )
                )

                max_id = int(
                    result.scalar() or 0
                )

                connection.execute(
                    text(
                        f"""
                        SELECT setval(
                            :sequence_name,
                            :max_id,
                            true
                        )
                        """
                    ),
                    {
                        "sequence_name": sequence_name,
                        "max_id": max_id,
                    },
                )

            except Exception as exc:

                print(
                    f"Sequence warning for "
                    f"{table_name}: {exc}"
                )


def migrate(
    target_database_url,
    apply=False,
):
    with app.app_context():
        source_engine = db.engine

    if target_database_url.startswith(
        "postgres://"
    ):
        target_database_url = target_database_url.replace(
            "postgres://",
            "postgresql+psycopg://",
            1,
        )

    elif target_database_url.startswith(
        "postgresql://"
    ):
        target_database_url = target_database_url.replace(
            "postgresql://",
            "postgresql+psycopg://",
            1,
        )

    target_engine = create_engine(
        target_database_url,
        pool_pre_ping=True,
    )

    print("\n=== SOURCE ===")
    print(
        "Database:",
        source_engine.url.render_as_string(
            hide_password=True
        )
    )

    print("\n=== TARGET ===")
    print(
        "Database:",
        target_engine.url.render_as_string(
            hide_password=True
        )
    )

    # --------------------------------------------------------
    # CREATE TARGET SCHEMA
    # --------------------------------------------------------

    db.metadata.create_all(
        bind=target_engine
    )

    print(
        "\nTarget schema is ready."
    )

    SourceSession = db.session.__class__

    # Use a clean source session.
    with app.app_context():

        source_session = (
            db.session
        )

        # ----------------------------------------------------
        # LOAD SOURCE DATA
        # ----------------------------------------------------

        source_user = source_session.get(
            User,
            SOURCE_USER_ID,
        )

        if not source_user:

            raise RuntimeError(
                "Source user id=1 was not found."
            )

        source_business = (
            source_session.get(
                Business,
                SOURCE_BUSINESS_ID,
            )
        )

        if not source_business:

            raise RuntimeError(
                "Source business id=2 was not found."
            )

        customers = (
            Customer.query
            .filter_by(
                business_id=SOURCE_BUSINESS_ID
            )
            .order_by(
                Customer.id
            )
            .all()
        )

        customers = [
            customer
            for customer in customers
            if not should_exclude_customer(
                customer
            )
        ]

        customer_ids = {
            customer.id
            for customer in customers
        }

        menus = (
            Menu.query
            .filter_by(
                business_id=SOURCE_BUSINESS_ID
            )
            .order_by(
                Menu.id
            )
            .all()
        )

        orders = (
            Order.query
            .filter_by(
                business_id=SOURCE_BUSINESS_ID
            )
            .order_by(
                Order.id
            )
            .all()
        )

        orders = [
            order
            for order in orders
            if (
                order.customer_id
                in customer_ids
            )
        ]

        order_ids = {
            order.id
            for order in orders
        }

        order_items = (
            OrderItem.query
            .filter(
                OrderItem.order_id.in_(
                    order_ids
                )
            )
            .order_by(
                OrderItem.id
            )
            .all()
        ) if order_ids else []

        conversations = (
            Conversation.query
            .filter(
                Conversation.customer_id.in_(
                    customer_ids
                )
            )
            .order_by(
                Conversation.id
            )
            .all()
        ) if customer_ids else []

        print("\n=== MIGRATION PLAN ===")
        print(
            "Users:",
            1
        )
        print(
            "Businesses:",
            1
        )
        print(
            "Menus:",
            len(menus)
        )
        print(
            "Customers:",
            len(customers)
        )
        print(
            "Orders:",
            len(orders)
        )
        print(
            "Order items:",
            len(order_items)
        )
        print(
            "Conversations:",
            len(conversations)
        )

        if not apply:

            print(
                "\nDRY RUN ONLY."
            )

            print(
                "Nothing has been inserted."
            )

            return

        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        with target_engine.begin() as target:

            existing_user = target.execute(
                text(
                    """
                    SELECT id
                    FROM "user"
                    WHERE email = :email
                    """
                ),
                {
                    "email": source_user.email
                },
            ).fetchone()

            if existing_user:

                raise RuntimeError(
                    "Production database already "
                    "contains the KAMSI account. "
                    "Migration stopped."
                )

            existing_business = target.execute(
                text(
                    """
                    SELECT id
                    FROM business
                    WHERE id = :id
                    """
                ),
                {
                    "id": SOURCE_BUSINESS_ID
                },
            ).fetchone()

            if existing_business:

                raise RuntimeError(
                    "Production database already "
                    "contains business id=2. "
                    "Migration stopped."
                )

        # ----------------------------------------------------
        # INSERT USER
        # ----------------------------------------------------

        with target_engine.begin() as target:

            target.execute(
                db.metadata.tables[
                    "user"
                ].insert().values(
                    **row_dict(
                        source_user
                    )
                )
            )

            # ------------------------------------------------
            # BUSINESS
            # ------------------------------------------------

            target.execute(
                db.metadata.tables[
                    "business"
                ].insert().values(
                    **row_dict(
                        source_business
                    )
                )
            )

            # ------------------------------------------------
            # MENUS
            # ------------------------------------------------

            for menu in menus:

                target.execute(
                    db.metadata.tables[
                        "menu"
                    ].insert().values(
                        **row_dict(menu)
                    )
                )

            # ------------------------------------------------
            # CUSTOMERS
            # ------------------------------------------------

            for customer in customers:

                target.execute(
                    db.metadata.tables[
                        "customer"
                    ].insert().values(
                        **row_dict(
                            customer
                        )
                    )
                )

            # ------------------------------------------------
            # ORDERS
            # ------------------------------------------------

            for order in orders:

                target.execute(
                    db.metadata.tables[
                        "order"
                    ].insert().values(
                        **row_dict(order)
                    )
                )

            # ------------------------------------------------
            # ORDER ITEMS
            # ------------------------------------------------

            for item in order_items:

                target.execute(
                    db.metadata.tables[
                        "order_item"
                    ].insert().values(
                        **row_dict(item)
                    )
                )

            # ------------------------------------------------
            # CONVERSATIONS
            # ------------------------------------------------

            for conversation in conversations:

                target.execute(
                    db.metadata.tables[
                        "conversation"
                    ].insert().values(
                        **row_dict(
                            conversation
                        )
                    )
                )

        reset_postgres_sequences(
            target_engine
        )

        print(
            "\nMigration completed successfully."
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write data to PostgreSQL.",
    )

    args = parser.parse_args()

    target_database_url = os.environ.get(
        "TARGET_DATABASE_URL"
    )

    if not target_database_url:

        raise SystemExit(
            "TARGET_DATABASE_URL is not set."
        )

    migrate(
        target_database_url,
        apply=args.apply,
    )