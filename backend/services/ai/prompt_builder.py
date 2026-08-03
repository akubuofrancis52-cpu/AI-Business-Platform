from datetime import datetime

from models.business import Business
from models.menu import Menu

from services.ai.menu_intelligence import (
    search_menu
)


# ==========================
# RESTAURANT PROMPT BUILDER
# ==========================

def build_restaurant_prompt(
    business_id,
    customer_message,
    language="English",
    customer_name=None
):

    # ==========================
    # BUSINESS
    # ==========================

    business = Business.query.get(
        business_id
    )

    if not business:
        return "Business not found."

    # ==========================
    # AVAILABLE MENU
    # ==========================

    menu = Menu.query.filter_by(
        business_id=business_id,
        available=True
    ).order_by(
        Menu.category.asc(),
        Menu.name.asc()
    ).all()

    # ==========================
    # BUILD STRUCTURED MENU
    # ==========================

    menu_text = ""

    if menu:

        current_category = None

        for item in menu:

            category = (
                item.category
                or "Other"
            )

            # New category
            if category != current_category:

                current_category = category

                menu_text += f"""

========== {category.upper()} ==========

"""

            menu_text += f"""
ITEM: {item.name}
PRICE: {item.price:,.0f} FCFA
"""

            if item.description:

                menu_text += (
                    "DESCRIPTION: "
                    f"{item.description}\n"
                )

            menu_text += (
                "-------------------------\n"
            )

    else:

        menu_text = (
            "No menu items are "
            "currently available."
        )

    # ==========================
    # MENU INTELLIGENCE
    # ==========================

    # IMPORTANT:
    # Reuse the menu list already loaded above.
    # This avoids another database query.

    relevant_matches = search_menu(
        business_id,
        customer_message,
        limit=5,
        menu_items=menu
    )

    intelligence_text = ""

    if relevant_matches:

        intelligence_text = """
RELEVANT MENU INTELLIGENCE

These are the available menu items
most relevant to the customer's request.

"""

        for item in relevant_matches:

            intelligence_text += f"""
ITEM: {item["name"]}
CATEGORY: {item["category"]}
PRICE: {item["price"]:,.0f} FCFA
"""

            if item["description"]:

                intelligence_text += (
                    "DESCRIPTION: "
                    f"{item['description']}\n"
                )

            intelligence_text += (
                "MATCH SCORE: "
                f"{item['score']}\n"
            )

            intelligence_text += (
                "-------------------------\n"
            )

    else:

        intelligence_text = """
RELEVANT MENU INTELLIGENCE

No strong menu match was found for
the customer's current request.
"""

    # ==========================
    # CURRENT DATE / TIME
    # ==========================

    current_time = datetime.now().strftime(
        "%A, %d %B %Y %I:%M %p"
    )

    # ==========================
    # FINAL PROMPT
    # ==========================

    return f"""
You are the AI customer assistant for {business.name}.

You behave like a professional restaurant employee
on WhatsApp, chat, or the restaurant website.

PERSONALITY

- Friendly
- Fast
- Professional
- Natural
- Concise
- Helpful

CURRENT DATE AND TIME

{current_time}

RESTAURANT

Name:
{business.name}

Business Type:
{business.business_type}

Address:
{business.address or "Not provided"}

Phone:
{business.phone or "Not provided"}

CUSTOMER

Name:
{customer_name or "Unknown"}

Language:
{language}

AVAILABLE MENU

{menu_text}

{intelligence_text}

IMPORTANT MENU INTELLIGENCE RULES

1. The AVAILABLE MENU is the ultimate source of truth.

2. Never invent a menu item.

3. Never invent a price.

4. Never claim an unavailable item exists.

5. Use RELEVANT MENU INTELLIGENCE to help identify
   what the customer is referring to.

6. Customers may use:
   - informal names
   - abbreviations
   - spelling mistakes
   - singular/plural variations
   - descriptions
   - categories

7. When a match is clearly supported by a real
   menu item, use that real menu item.

8. If several real menu items could match equally well,
   ask a short clarification question.

9. Do not guess when the request is ambiguous.

10. Recommendations must use ONLY real available
    menu items.

11. When recommending an item, use its exact real
    menu name and real database price.

12. You may explain an item's suitability using its
    real menu description.

13. Never create a product based only on what the
    customer describes.

14. All prices must be displayed in FCFA.

15. Never invent:
    - discounts
    - promotions
    - opening hours
    - delivery policies
    - restaurant services

ORDERING RULES

When a customer wants to order:

- Identify the requested real menu items.
- Confirm quantities.
- Use real menu prices.
- Do not invent totals.
- Ask for missing information when necessary.
- Keep the conversation natural.

RECOMMENDATION RULES

When the customer asks:

"What do you recommend?"

"What's good?"

"What should I get?"

or similar:

- Recommend available menu items only.
- Prefer items relevant to the customer's request.
- Mention real prices.
- Keep recommendations concise.
- Never invent ingredients or properties that aren't
  supported by the menu description.

SEARCH RULES

When the customer asks whether the restaurant has
something:

- Use the available menu and relevant matches.
- Return only real matching items.
- If nothing matches, say that clearly.
- Do not invent alternatives.

CONVERSATION RULES

- Always reply in {language}.
- Match the customer's language.
- Keep normal responses short.
- Use emojis naturally.
- Be helpful and professional.
- Protect the restaurant's reputation.
- If information is unavailable, say so clearly.

CUSTOMER MESSAGE

{customer_message}

AI RESPONSE
"""