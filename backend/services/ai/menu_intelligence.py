import re
from difflib import SequenceMatcher

from sqlalchemy.orm import load_only

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

# ==========================
# STRING SIMILARITY
# ==========================

def _similarity_normalized(
    first,
    second,
):
    """
    Compare already-normalized strings.
    """

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
        second,
    ).ratio()


def similarity(
    first,
    second,
):
    """
    Compare two strings and return a score from 0 to 1.
    """

    first = normalize_text(first)
    second = normalize_text(second)

    return _similarity_normalized(
        first,
        second,
    )


# ==========================
# TOKEN OVERLAP
# ==========================

def _token_overlap_normalized(
    first,
    second,
):
    """
    Compare token overlap between already-normalized strings.
    """

    first_tokens = set(
        first.split()
    )

    second_tokens = set(
        second.split()
    )

    if not first_tokens or not second_tokens:
        return 0.0

    return len(
        first_tokens & second_tokens
    ) / max(
        len(first_tokens),
        len(second_tokens),
    )


def token_overlap(
    first,
    second,
):
    """
    Compare shared words between two strings.
    """

    first = normalize_text(first)
    second = normalize_text(second)

    return _token_overlap_normalized(
        first,
        second,
    )


# ==========================
# SCORE MENU ITEM
# ==========================

def score_menu_item(
    query,
    item,
    cleaned_query=None,
):
    """
    Calculate how well a menu item matches a query.

    Priority:
    - name
    - description
    - category

    Normalization is performed once per field so the same
    strings are not repeatedly processed.
    """

    if cleaned_query is None:
        cleaned_query = clean_query(
            query
        )

    name_normalized = normalize_text(
        item.name
    )

    # --------------------------------------------------------
    # DIRECT NAME MATCH
    # --------------------------------------------------------

    if cleaned_query == name_normalized:
        return 1.0

    if (
        cleaned_query in name_normalized
        or name_normalized in cleaned_query
    ):
        return 0.95

    name_score = max(
        _similarity_normalized(
            cleaned_query,
            name_normalized,
        ),
        _token_overlap_normalized(
            cleaned_query,
            name_normalized,
        ),
    )

    # --------------------------------------------------------
    # DESCRIPTION
    # --------------------------------------------------------

    description_score = 0.0

    if item.description:

        description_normalized = normalize_text(
            item.description
        )

        description_score = max(
            _similarity_normalized(
                cleaned_query,
                description_normalized,
            ),
            _token_overlap_normalized(
                cleaned_query,
                description_normalized,
            ),
        )

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    category_score = 0.0

    if item.category:

        category_normalized = normalize_text(
            item.category
        )

        category_score = max(
            _similarity_normalized(
                cleaned_query,
                category_normalized,
            ),
            _token_overlap_normalized(
                cleaned_query,
                category_normalized,
            ),
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
    business_id,
):
    """
    Load only the menu fields used by the AI menu
    intelligence functions.
    """

    return (
        Menu.query
        .options(
            load_only(
                Menu.id,
                Menu.name,
                Menu.description,
                Menu.category,
                Menu.price,
            )
        )
        .filter(
            Menu.business_id == business_id,
            Menu.available.is_(True),
        )
        .order_by(
            Menu.category.asc(),
            Menu.name.asc(),
        )
        .all()
    )


# ==========================
# SEARCH MENU
# ==========================

def search_menu(
    business_id,
    query,
    limit=5,
    menu_items=None,
):
    """
    Search available menu items.

    Reuses the loaded menu when supplied and avoids
    unnecessary full-result sorting when only the top
    few matches are needed.
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

    cleaned_query = clean_query(
        query
    )

    results = []

    for item in menu_items:

        score = score_menu_item(
            query,
            item,
            cleaned_query=cleaned_query,
        )

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
                item.price or 0
            ),
            "currency": "FCFA",
            "score": round(
                score,
                3,
            ),
        })

    if not results:
        return []

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
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
    menu_items=None,
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

    cleaned_query = clean_query(
        name
    )

    best_item = None
    best_score = 0.0

    for item in menu_items:

        score = score_menu_item(
            name,
            item,
            cleaned_query=cleaned_query,
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