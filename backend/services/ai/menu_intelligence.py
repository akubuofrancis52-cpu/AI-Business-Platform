import re
from difflib import SequenceMatcher

from models.menu import Menu


# ==========================
# TEXT NORMALIZATION & CLEANING
# ==========================

def normalize_text(text):
    """
    Normalize text for reliable menu matching.
    """

    if not text:
        return ""

    text = str(text).lower().strip()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


def clean_query(query):
    """
    Remove stray quantity words or articles 
    if accidentally included in item lookups.
    """
    if not query:
        return ""

    query = str(query).lower().strip()

    stopwords = {
        "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten",
        "a", "an", "the", "some"
    }

    words = query.split()
    filtered = [
        w for w in words 
        if w not in stopwords
    ]

    return " ".join(filtered) if filtered else query


# ==========================
# STRING SIMILARITY
# ==========================

def similarity(first, second):
    """
    Compare two strings and return a score from 0 to 1.
    """

    first = normalize_text(first)
    second = normalize_text(second)

    if not first or not second:
        return 0.0

    if first == second:
        return 1.0

    if first in second:
        return 0.95

    if second in first:
        return 0.95

    return SequenceMatcher(
        None,
        first,
        second
    ).ratio()


# ==========================
# TOKEN OVERLAP
# ==========================

def token_overlap(first, second):
    """
    Compare shared words between two strings.
    """

    first_tokens = set(
        normalize_text(first).split()
    )

    second_tokens = set(
        normalize_text(second).split()
    )

    if not first_tokens or not second_tokens:
        return 0.0

    return len(
        first_tokens & second_tokens
    ) / max(
        len(first_tokens),
        len(second_tokens)
    )


# ==========================
# SCORE MENU ITEM
# ==========================

def score_menu_item(
    query,
    item
):
    """
    Calculate how well a menu item matches a query.

    Priority:
    - name
    - description
    - category
    """

    cleaned_query = clean_query(query)
    name_norm = normalize_text(item.name)

    # Direct or substring match boost
    if cleaned_query == name_norm:
        return 1.0

    if cleaned_query in name_norm or name_norm in cleaned_query:
        return 0.95

    name_score = max(
        similarity(
            cleaned_query,
            item.name
        ),
        token_overlap(
            cleaned_query,
            item.name
        )
    )

    description_score = 0.0

    if item.description:

        description_score = max(
            similarity(
                cleaned_query,
                item.description
            ),
            token_overlap(
                cleaned_query,
                item.description
            )
        )

    category_score = 0.0

    if item.category:

        category_score = max(
            similarity(
                cleaned_query,
                item.category
            ),
            token_overlap(
                cleaned_query,
                item.category
            )
        )

    return (
        name_score * 0.60
        + description_score * 0.25
        + category_score * 0.15
    )


# ==========================
# LOAD MENU
# ==========================

def get_menu_items(
    business_id
):
    """
    Load all available menu items
    for one business.
    """

    return Menu.query.filter_by(
        business_id=business_id,
        available=True
    ).order_by(
        Menu.category.asc(),
        Menu.name.asc()
    ).all()


# ==========================
# SEARCH MENU
# ==========================

def search_menu(
    business_id,
    query,
    limit=5,
    menu_items=None
):
    """
    Search available menu items.

    If menu_items is already provided,
    no additional database query is made.
    """

    query = str(
        query or ""
    ).strip()

    if not query:
        return []

    if menu_items is None:

        menu_items = get_menu_items(
            business_id
        )

    if not menu_items:
        return []

    results = []

    for item in menu_items:

        score = score_menu_item(
            query,
            item
        )

        # Ignore very weak matches
        if score < 0.30:
            continue

        results.append({
            "id": item.id,
            "name": item.name,
            "description": (
                item.description
                or ""
            ),
            "category": (
                item.category
                or "Other"
            ),
            "price": float(
                item.price
            ),
            "currency": "FCFA",
            "score": round(
                score,
                3
            )
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:limit]


# ==========================
# RESOLVE MENU ITEM
# ==========================

def resolve_menu_item(
    business_id,
    name,
    minimum_score=0.55,
    menu_items=None
):
    """
    Return a structured match for a
    customer-provided menu name.
    """

    results = search_menu(
        business_id,
        name,
        limit=1,
        menu_items=menu_items
    )

    if not results:
        return None

    best = results[0]

    if best["score"] < minimum_score:
        return None

    return best


# ==========================
# RESOLVE MENU OBJECT
# ==========================

def resolve_menu_object(
    business_id,
    name,
    minimum_score=0.55,
    menu_items=None
):
    """
    Return the actual SQLAlchemy Menu object.

    This avoids a second database query when
    the menu list was already loaded.
    """

    if menu_items is None:

        menu_items = get_menu_items(
            business_id
        )

    if not menu_items:
        return None

    best_item = None
    best_score = 0.0

    for item in menu_items:

        score = score_menu_item(
            name,
            item
        )

        if score > best_score:

            best_score = score
            best_item = item

    if best_score < minimum_score:
        return None

    return best_item


# ==========================
# RECOMMEND MENU
# ==========================

def recommend_menu(
    business_id,
    query=None,
    limit=3,
    menu_items=None
):
    """
    Return menu recommendations.

    With a query:
        return the strongest matches.

    Without a query:
        return the first available items.
    """

    if menu_items is None:

        menu_items = get_menu_items(
            business_id
        )

    if not menu_items:
        return []

    if query:

        return search_menu(
            business_id,
            query,
            limit=limit,
            menu_items=menu_items
        )

    recommendations = []

    for item in menu_items[:limit]:

        recommendations.append({
            "id": item.id,
            "name": item.name,
            "description": (
                item.description
                or ""
            ),
            "category": (
                item.category
                or "Other"
            ),
            "price": float(
                item.price
            ),
            "currency": "FCFA",
            "score": 1.0
        })

    return recommendations