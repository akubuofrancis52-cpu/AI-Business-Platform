import pytest

from app import app
from database.db import db
from models.user import User
from models.business import Business
from models.menu import Menu
from services.ai.order_extractor import extract_order


def seed_order_pipeline_database():
    """Create the minimum deterministic menu needed by this test."""

    user = User.query.first()

    if not user:
        user = User(
            username="order_pipeline_test_user",
            email="order-pipeline-test@example.com",
        )
        user.set_password("test-password")
        db.session.add(user)
        db.session.flush()

    business = db.session.get(Business, 2)

    if not business:
        business = Business(
            id=2,
            name="Test Restaurant",
            business_type="Restaurant",
            owner_id=user.id,
        )
        db.session.add(business)
        db.session.flush()

    menu_items = {
        "Signature Lebanese Shawarma": 2500,
        "Classic Margherita Pizza": 3000,
        "Fries": 1000,
        "Lemonade": 900,
        "Chocolate Overload Scoop": 1500,
        "Burger": 2000,
    }

    for name, price in menu_items.items():
        existing = (
            Menu.query
            .filter_by(business_id=business.id, name=name)
            .first()
        )

        if not existing:
            db.session.add(
                Menu(
                    name=name,
                    description=f"Test menu item: {name}",
                    price=price,
                    category="Test",
                    available=True,
                    business_id=business.id,
                )
            )

    db.session.commit()


@pytest.fixture(autouse=True)
def setup_order_pipeline_database():
    with app.app_context():
        seed_order_pipeline_database()


@pytest.mark.parametrize(
    "message",
    [
        "I want 2 Signature Lebanese Shawarma",
        "I'd like a Classic Margherita Pizza and some fries",
        "Give me 3 Lemonades",
        "I want a Chocolate Overload Scoop",
        "I want a burger",
        "I want a Classic Margherita Pizza and a burger",
    ],
)
def test_order_extraction_pipeline(message):
    with app.app_context():
        result = extract_order(
            business_id=2,
            customer_message=message,
        )

        print("\n" + "=" * 70)
        print("MESSAGE:", message)
        print("RESULT:", result)

        assert result is not None
