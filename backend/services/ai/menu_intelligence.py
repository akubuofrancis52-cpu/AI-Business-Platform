import re
from difflib import SequenceMatcher

from sqlalchemy.orm import load_only

from models.menu import Menu
from models.customer_preference import CustomerPreference
from services.inventory import filter_inventory_available_menu_items


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
# NATURAL QUERY HINTS
# ==========================

QUERY_HINTS = {
    "filling": {
        "hearty",
        "generous",
        "abundant",
        "stacked",
        "chunks",
    },
    "light": {
        "light",
        "cooling",
        "refreshing",
    },
    "refreshing": {
        "refreshing",
        "cooling",
        "cold",
        "fresh",
    },
    "sweet": {
        "sweet",
        "caramel",
        "chocolate",
        "fruit",
    },
    "quick": {
        "quick",
        "snack",
        "fast",
    },
    "chocolate": {
        "chocolate",
        "cocoa",
        "fudge",
    },
    "fruity": {
        "fruit",
        "strawberry",
        "seasonal",
    },
}


def natural_query_score(
    query,
    item,
):
    """Score how well a natural request is supported by menu text."""

    normalized_query = normalize_text(
        query
    )

    item_text = normalize_text(
        " ".join(
            [
                str(item.name or ""),
                str(item.description or ""),
                str(item.category or ""),
            ]
        )
    )

    if not normalized_query:
        return 0.0

    score = 0.0

    for request, hints in QUERY_HINTS.items():

        if request not in normalized_query:
            continue

        matches = sum(
            1
            for hint in hints
            if normalize_text(hint) in item_text
        )

        if matches:
            score += min(
                0.50,
                0.20 * matches,
            )

    return score


# ==========================
# CUSTOMER PREFERENCE MATCHING
# ==========================

PREFERENCE_SYNONYMS = {
    "chicken": {
        "chicken",
    },
    "pizza": {
        "pizza",
    },
    "shawarma": {
        "shawarma",
    },
    "sandwich": {
        "sandwich",
        "club",
    },
    "fries": {
        "fries",
        "french fries",
    },
    "drink": {
        "drink",
        "drinks",
        "beverage",
        "beverages",
        "lemonade",
    },
    "dessert": {
        "dessert",
        "desserts",
        "ice cream",
        "gelato",
        "sundae",
        "cone",
    },
    "spicy": {
        "spicy",
        "chili",
        "pepper",
    },
    "cheese": {
        "cheese",
        "mozzarella",
    },
}


def preference_matches_menu(
    preference_value,
    item,
):
    # Return whether a preference matches a menu item.

    preference = normalize_text(
        preference_value
    )

    if not preference:
        return False

    item_text = normalize_text(
        " ".join(
            [
                str(item.name or ""),
                str(item.description or ""),
                str(item.category or ""),
            ]
        )
    )

    if preference in item_text:
        return True

    for canonical, synonyms in PREFERENCE_SYNONYMS.items():

        if preference != normalize_text(canonical):
            continue

        normalized_synonyms = {
            normalize_text(value)
            for value in synonyms
        }

        return any(
            synonym in item_text
            for synonym in normalized_synonyms
        )

    return False


# ==========================
# RECOMMEND MENU
# ==========================

def recommend_menu(
    business_id,
    query=None,
    limit=3,
    menu_items=None,
    customer_id=None,
):
    """
    Return menu recommendations with optional customer-memory ranking.

    Query-specific searches remain primarily driven by the customer's
    current request. When there is no query, customer preferences can
    influence the recommendation ranking.
    """

    if menu_items is None:

        menu_items = get_menu_items(
            business_id
        )

    if not menu_items:
        return []

    # ========================================================
    # LIVE INVENTORY FILTER
    # ========================================================
    #
    # Menu.available handles manual availability.
    # Inventory handles ingredient-level availability.
    #
    # Businesses without configured ingredient recipes are
    # intentionally left unchanged by the inventory filter.
    menu_items = filter_inventory_available_menu_items(
        menu_items,
        quantity=1,
    )

    if not menu_items:
        return []

    preferences = []

    if customer_id:

        try:
            preferences = (
                CustomerPreference.query
                .filter_by(
                    customer_id=customer_id,
                    business_id=business_id,
                )
                .order_by(
                    CustomerPreference.strength.desc(),
                )
                .limit(12)
                .all()
            )

        except Exception:
            preferences = []

    if query:

        lexical_results = search_menu(
            business_id,
            query,
            limit=max(
                limit * 2,
                10,
            ),
            menu_items=menu_items,
        )

        lexical_by_id = {
            item["id"]: item
            for item in lexical_results
        }

        hybrid_results = []

        for position, menu_item in enumerate(menu_items):

            lexical_score = float(
                lexical_by_id.get(
                    menu_item.id,
                    {},
                ).get(
                    "score",
                    0.0,
                )
            )

            semantic_score = natural_query_score(
                query,
                menu_item,
            )

            preference_score = 0.0
            matched_preferences = []

            for preference in preferences:

                if not preference_matches_menu(
                    preference.preference_value,
                    menu_item,
                ):
                    continue

                strength = max(
                    int(preference.strength or 1),
                    1,
                )

                preference_type = (
                    preference.preference_type
                    or ""
                )

                matched_preferences.append({
                    "type": preference_type,
                    "value": preference.preference_value,
                    "strength": strength,
                })

                if preference_type == "explicit_dislike":
                    preference_score -= min(
                        0.75,
                        0.45 + (0.05 * strength),
                    )

                elif preference_type == "favorite_item":
                    preference_score += min(
                        0.40,
                        0.25 + (0.05 * strength),
                    )

                elif preference_type == "frequent_item":
                    preference_score += min(
                        0.30,
                        0.15 + (0.04 * strength),
                    )

                elif preference_type == "explicit_preference":
                    preference_score += min(
                        0.30,
                        0.18 + (0.04 * strength),
                    )

                elif preference_type == "explicit_like":
                    preference_score += min(
                        0.25,
                        0.15 + (0.03 * strength),
                    )

                elif preference_type == "ordered_item":
                    preference_score += min(
                        0.12,
                        0.06 + (0.02 * strength),
                    )

            # The customer's current request has priority.
            # Customer memory only boosts an item when the item is
            # already reasonably relevant to the current request.
            if (
                semantic_score == 0
                and lexical_score < 0.50
            ):
                preference_score = 0.0
            else:
                preference_score *= 0.35

            combined_score = (
                lexical_score
                + semantic_score
                + preference_score
            )

            if (
                lexical_score >= 0.30
                or semantic_score > 0
                or preference_score > 0
            ):
                hybrid_results.append({
                    "id": menu_item.id,
                    "name": menu_item.name,
                    "description": (
                        menu_item.description
                        or ""
                    ),
                    "category": (
                        menu_item.category
                        or "Other"
                    ),
                    "price": float(
                        menu_item.price or 0
                    ),
                    "currency": "FCFA",
                    "score": round(
                        combined_score,
                        3,
                    ),
                    "reason": {
                        "source": "hybrid",
                        "lexical_score": round(
                            lexical_score,
                            3,
                        ),
                        "semantic_score": round(
                            semantic_score,
                            3,
                        ),
                        "preference_score": round(
                            preference_score,
                            3,
                        ),
                        "matched_preferences": matched_preferences,
                    },
                })

        hybrid_results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return hybrid_results[:limit]

    if customer_id:

        try:
            preferences = (
                CustomerPreference.query
                .filter_by(
                    customer_id=customer_id,
                    business_id=business_id,
                )
                .order_by(
                    CustomerPreference.strength.desc(),
                )
                .limit(12)
                .all()
            )

        except Exception:
            preferences = []

    def item_text(item):
        return normalize_text(
            " ".join(
                [
                    str(item.name or ""),
                    str(item.description or ""),
                    str(item.category or ""),
                ]
            )
        )

    scored_items = []

    for position, item in enumerate(menu_items):

        text = item_text(item)
        score = 1.0
        matched_preferences = []

        for preference in preferences:

            preference_value = normalize_text(
                preference.preference_value
            )

            if not preference_value:
                continue

            preference_type = (
                preference.preference_type or ""
            )

            strength = max(
                int(preference.strength or 1),
                1,
            )

            preference_tokens = [
                token
                for token in preference_value.split()
                if len(token) >= 3
                and token not in {
                    "food",
                    "item",
                    "items",
                    "meal",
                    "meals",
                    "dish",
                    "dishes",
                    "stuff",
                }
            ]

            matched = preference_matches_menu(
                preference_value,
                item,
            )

            if not matched:
                matched = any(
                    token in text
                    for token in preference_tokens
                )

            if not matched:
                continue

            matched_preferences.append({
                "type": preference_type,
                "value": preference.preference_value,
                "strength": strength,
            })

            # An explicit customer dislike is a hard exclusion for
            # preference-based recommendations. Penalizing the score
            # is not sufficient because the item could still appear
            # when the recommendation list is large enough.
            if preference_type == "explicit_dislike":
                score = None
                break

            elif preference_type == "favorite_item":

                score += min(
                    0.40,
                    0.25 + (0.05 * strength),
                )

            elif preference_type == "frequent_item":

                score += min(
                    0.30,
                    0.15 + (0.04 * strength),
                )

            elif preference_type == "explicit_preference":

                score += min(
                    0.30,
                    0.18 + (0.04 * strength),
                )

            elif preference_type == "explicit_like":

                score += min(
                    0.25,
                    0.15 + (0.03 * strength),
                )

            elif preference_type == "ordered_item":

                score += min(
                    0.12,
                    0.06 + (0.02 * strength),
                )

        # Explicit dislikes are hard exclusions.
        if score is None:
            continue

        scored_items.append(
            (
                score,
                position,
                item,
                matched_preferences,
            )
        )

    scored_items.sort(
        key=lambda entry: (
            entry[0],
            -entry[1],
        ),
        reverse=True,
    )

    recommendations = []

    for score, _, item, matched_preferences in scored_items[:limit]:

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
            "score": round(
                score,
                3,
            ),
            "reason": {
                "source": "preference_ranked",
                "base_score": 1.0,
                "final_score": round(
                    score,
                    3,
                ),
                "matched_preferences": matched_preferences,
            },
        })

    return recommendations
