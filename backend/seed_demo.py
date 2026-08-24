from app import app, db
from models.user import User
from models.business import Business
from models.menu import Menu


DEMO_EMAIL = "demo@botify.local"
DEMO_USERNAME = "botify_demo"


DEMO_MENU = [
    {
        "name": "Lemonade",
        "description": "Fresh lemonade.",
        "price": 200,
        "category": "Drinks",
    },
    {
        "name": "Assorted Club Sandwich",
        "description": "Toasted club sandwich with chicken and fresh toppings.",
        "price": 3500,
        "category": "Fast Food & Restaurant Specialties",
    },
    {
        "name": "Classic Margherita Pizza",
        "description": "Classic pizza with tomato, mozzarella, and basil.",
        "price": 5500,
        "category": "Fast Food & Restaurant Specialties",
    },
    {
        "name": "Crispy French Fries",
        "description": "Crispy golden French fries.",
        "price": 4500,
        "category": "Fast Food & Restaurant Specialties",
    },
    {
        "name": "Loaded Chicken & Cheese Pizza",
        "description": "Loaded chicken and cheese pizza.",
        "price": 6000,
        "category": "Fast Food & Restaurant Specialties",
    },
    {
        "name": "Signature Lebanese Shawarma",
        "description": "Signature Lebanese-style shawarma.",
        "price": 2500,
        "category": "Fast Food & Restaurant Specialties",
    },
    {
        "name": "Caramel Crunch Sundae",
        "description": "Caramel dessert sundae.",
        "price": 6000,
        "category": "Italian Ice Creams & Desserts",
    },
    {
        "name": "Chocolate Overload Scoop / Cone",
        "description": "Chocolate ice cream served as a scoop or cone.",
        "price": 1500,
        "category": "Italian Ice Creams & Desserts",
    },
    {
        "name": "Classic Vanilla Cornetto Cone",
        "description": "Classic vanilla Cornetto cone.",
        "price": 2000,
        "category": "Italian Ice Creams & Desserts",
    },
    {
        "name": "Mixed Fruit-Flavored Ice Cream Cup",
        "description": "Mixed fruit-flavored ice cream cup.",
        "price": 5000,
        "category": "Italian Ice Creams & Desserts",
    },
    {
        "name": "Strawberry Fruit Gelato",
        "description": "Strawberry fruit gelato.",
        "price": 2000,
        "category": "Italian Ice Creams & Desserts",
    },
]


def seed_demo():
    with app.app_context():

        db.create_all()

        user = User.query.filter_by(
            email=DEMO_EMAIL
        ).first()

        if not user:

            user = User(
                username=DEMO_USERNAME,
                email=DEMO_EMAIL,
                language="English",
            )

            user.set_password(
                "demo-disabled-login"
            )

            db.session.add(user)

            db.session.flush()

        business = Business.query.filter_by(
            name="Italian Ice Cream Cornetto"
        ).first()

        if not business:

            business = Business(
                name="Italian Ice Cream Cornetto",
                business_type="restaurant",
                address="Ganhito, Notre Dame, Cotonou",
                phone="60101010",
                opening_hours=(
                    "Monday - Saturday: 9:00 AM - 10:00 PM\n"
                    "Sunday: 12:00 PM - 8:00 PM"
                ),
                delivery_policy=(
                    "Delivery is available within Cotonou."
                ),
                owner_id=user.id,
            )

            db.session.add(business)

            db.session.flush()

        existing_count = Menu.query.filter_by(
            business_id=business.id
        ).count()

        if existing_count == 0:

            for item in DEMO_MENU:

                db.session.add(
                    Menu(
                        name=item["name"],
                        description=item["description"],
                        price=item["price"],
                        category=item["category"],
                        available=True,
                        business_id=business.id,
                    )
                )

        db.session.commit()

        print(
            f"Demo seed ready: business_id={business.id}"
        )


if __name__ == "__main__":
    seed_demo()