from models.business import Business
from models.menu import Menu
from models.user import User


def build_restaurant_prompt(business_id, customer_message, language="English"):

    business = Business.query.get(business_id)

    if not business:
        return "Business not found."

    menu = Menu.query.filter_by(
        business_id=business_id,
        available=True
    ).all()

    menu_text = ""

    for item in menu:

        menu_text += (
            f"- {item.name}\n"
            f"  Category: {item.category}\n"
            f"  Price: ${item.price:.2f}\n"
        )

        if item.description:
            menu_text += (
                f"  Description: {item.description}\n"
            )

        menu_text += "\n"


    # Get owner only when needed
    owner = User.query.get(
        business.owner_id
    )


    return f"""
You are the AI assistant for this business.

You must answer customers in {language}.

Business Information:

Business Name:
{business.name}

Business Type:
{business.business_type}

Address:
{business.address}

Phone:
{business.phone}


AVAILABLE MENU

{menu_text}


RULES

1. Only recommend items that exist in the menu.
2. Never invent menu items or prices.
3. If a customer asks for something unavailable, politely explain.
4. Help customers place orders.
5. Be friendly, professional, and concise.
6. Confirm order details before finalizing.
7. Use the customer's language preference.


Customer Message:

{customer_message}
"""