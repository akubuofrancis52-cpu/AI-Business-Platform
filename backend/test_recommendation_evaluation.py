from app import app
from database.db import db
from models.user import User
from models.business import Business
from models.menu import Menu
from models.ingredient import Ingredient
from models.menu_ingredient import MenuIngredient
from models.customer import Customer
from models.customer_preference import CustomerPreference
from models.ingredient import Ingredient
from services.ai.menu_intelligence import recommend_menu


BUSINESS_ID = 2
PHONE = "DEMO-AUTOMATED-RECOMMENDATION-EVAL"


def seed_recommendation_database():
    """Create the minimum deterministic dataset required by this test."""

    owner = User.query.first()

    if owner is None:
        owner = User(
            username="pytest_recommendation_owner",
            email="pytest-recommendation@example.test",
            password="pytest-password",
        )
        db.session.add(owner)
        db.session.flush()

    business = Business.query.filter_by(id=BUSINESS_ID).first()

    if business is None:
        business = Business(
            id=BUSINESS_ID,
            name="Pytest Restaurant",
            business_type="Restaurant",
            owner_id=owner.id,
        )
        db.session.add(business)
        db.session.flush()

    lemonade = Menu.query.filter_by(
        business_id=BUSINESS_ID,
        name="Lemonade",
    ).first()

    if lemonade is None:
        lemonade = Menu(
            name="Lemonade",
            description="Fresh lemonade",
            price=1000,
            category="Drinks",
            available=True,
            business_id=BUSINESS_ID,
        )
        db.session.add(lemonade)

    shawarma = Menu.query.filter_by(
        business_id=BUSINESS_ID,
        name="Signature Lebanese Shawarma",
    ).first()

    if shawarma is None:
        shawarma = Menu(
            name="Signature Lebanese Shawarma",
            description="Signature shawarma",
            price=3500,
            category="Main",
            available=True,
            business_id=BUSINESS_ID,
        )
        db.session.add(shawarma)

    db.session.flush()

    ingredient = Ingredient.query.filter_by(
        business_id=BUSINESS_ID,
        name="Test Shawarma Ingredient",
    ).first()

    if ingredient is None:
        ingredient = Ingredient(
            name="Test Shawarma Ingredient",
            unit="unit",
            business_id=BUSINESS_ID,
            active=True,
            stock_quantity=10,
            low_stock_threshold=1,
        )
        db.session.add(ingredient)
        db.session.flush()

    relationship = MenuIngredient.query.filter_by(
        menu_id=shawarma.id,
        ingredient_id=ingredient.id,
    ).first()

    if relationship is None:
        relationship = MenuIngredient(
            menu_id=shawarma.id,
            ingredient_id=ingredient.id,
            quantity_required=1,
        )
        db.session.add(relationship)

    db.session.commit()




def test_recommendation_system():
    with app.app_context():
        seed_recommendation_database()

        customer = Customer.query.filter_by(
            business_id=BUSINESS_ID,
            phone=PHONE,
        ).first()

        if customer:
            CustomerPreference.query.filter_by(
                customer_id=customer.id,
                business_id=BUSINESS_ID,
            ).delete(synchronize_session=False)

            db.session.delete(customer)
            db.session.commit()

        customer = Customer(
            business_id=BUSINESS_ID,
            phone=PHONE,
            name="Automated Recommendation Evaluation",
        )

        db.session.add(customer)
        db.session.commit()

        cold = recommend_menu(
            business_id=BUSINESS_ID,
            customer_id=customer.id,
            limit=5,
        )

        assert cold
        assert all(
            isinstance(item, dict)
            for item in cold
        )

        db.session.add(
            CustomerPreference(
                customer_id=customer.id,
                business_id=BUSINESS_ID,
                preference_type="explicit_like",
                preference_value="Lemonade",
                strength=3,
                source="automated_test",
            )
        )

        db.session.commit()

        positive = recommend_menu(
            business_id=BUSINESS_ID,
            customer_id=customer.id,
            limit=5,
        )

        assert positive
        assert positive[0]["name"] == "Lemonade"

        db.session.add(
            CustomerPreference(
                customer_id=customer.id,
                business_id=BUSINESS_ID,
                preference_type="explicit_dislike",
                preference_value="Lemonade",
                strength=5,
                source="automated_test",
            )
        )

        db.session.commit()

        disliked = recommend_menu(
            business_id=BUSINESS_ID,
            customer_id=customer.id,
            limit=5,
        )

        assert all(
            item["name"] != "Lemonade"
            for item in disliked
        )

        ingredient = Ingredient.query.filter_by(
            business_id=BUSINESS_ID,
            name="Test Shawarma Ingredient",
        ).first()

        original_stock = None

        if ingredient:
            original_stock = ingredient.stock_quantity
            ingredient.stock_quantity = 0
            db.session.commit()

            inventory_result = recommend_menu(
                business_id=BUSINESS_ID,
                customer_id=customer.id,
                limit=20,
            )

            assert all(
                item["name"] != "Signature Lebanese Shawarma"
                for item in inventory_result
            )

            ingredient.stock_quantity = original_stock
            db.session.commit()

        CustomerPreference.query.filter_by(
            customer_id=customer.id,
            business_id=BUSINESS_ID,
        ).delete(synchronize_session=False)

        db.session.delete(customer)
        db.session.commit()
