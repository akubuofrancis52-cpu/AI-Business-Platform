from app import app
from services.ai.order_extractor import extract_order


app.app_context().push()

tests = [
    "I want 2 Signature Lebanese Shawarma",
    "I'd like a Classic Margherita Pizza and some fries",
    "Give me 3 Lemonades",
    "I want a Chocolate Overload Scoop",
    "I want a burger",
    "I want a Classic Margherita Pizza and a burger",
]

for message in tests:

    print("\n" + "=" * 70)
    print("MESSAGE:", message)

    try:
        result = extract_order(
            business_id=2,
            customer_message=message,
        )

        print("RESULT:")
        print(result)

    except Exception as e:

        print("ERROR:", repr(e))
