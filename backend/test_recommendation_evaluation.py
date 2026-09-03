from app import app
from database.db import db
from models.customer import Customer
from models.customer_preference import CustomerPreference
from models.ingredient import Ingredient
from services.ai.menu_intelligence import recommend_menu


BUSINESS_ID = 2
PHONE = "DEMO-AUTOMATED-RECOMMENDATION-EVAL"


def test_recommendation_system():
    with app.app_context():
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
