import re
from collections import Counter

from database.db import db
from models.customer_preference import CustomerPreference
from models.order import Order


def _upsert_preference(
    customer_id,
    business_id,
    preference_type,
    preference_value,
    strength,
    source,
):
    preference = CustomerPreference.query.filter_by(
        customer_id=customer_id,
        business_id=business_id,
        preference_type=preference_type,
        preference_value=preference_value,
    ).first()

    if preference:
        preference.strength = strength
        preference.source = source
        return preference

    preference = CustomerPreference(
        customer_id=customer_id,
        business_id=business_id,
        preference_type=preference_type,
        preference_value=preference_value,
        strength=strength,
        source=source,
    )

    db.session.add(preference)
    return preference


def learn_customer_preferences_from_order(order):
    """Learn durable customer preferences from completed orders."""

    if not order or not order.customer_id:
        return []

    completed_orders = (
        Order.query
        .filter_by(
            customer_id=order.customer_id,
            business_id=order.business_id,
            status="Completed",
        )
        .order_by(Order.id.asc())
        .all()
    )

    order_counts = Counter()

    for completed_order in completed_orders:

        items_seen_in_order = set()

        for item in completed_order.items:
            name = (item.name or "").strip()

            if not name:
                continue

            items_seen_in_order.add(name)

        for name in items_seen_in_order:
            order_counts[name] += 1

    if not order_counts:
        return []

    learned = []

    # --------------------------------------------------------
    # ORDERED ITEMS
    # --------------------------------------------------------

    for item_name, order_count in order_counts.items():

        learned.append(
            _upsert_preference(
                customer_id=order.customer_id,
                business_id=order.business_id,
                preference_type="ordered_item",
                preference_value=item_name,
                strength=order_count,
                source="completed_order",
            )
        )

    # --------------------------------------------------------
    # FREQUENT ITEMS
    # --------------------------------------------------------

    for item_name, order_count in order_counts.items():

        if order_count >= 2:

            learned.append(
                _upsert_preference(
                    customer_id=order.customer_id,
                    business_id=order.business_id,
                    preference_type="frequent_item",
                    preference_value=item_name,
                    strength=order_count,
                    source="completed_order",
                )
            )

    # --------------------------------------------------------
    # FAVORITE ITEM
    # --------------------------------------------------------

    repeated_items = {
        name: count
        for name, count in order_counts.items()
        if count >= 2
    }

    if repeated_items:

        favorite_item, favorite_count = max(
            repeated_items.items(),
            key=lambda pair: pair[1],
        )

        favorite = CustomerPreference.query.filter_by(
            customer_id=order.customer_id,
            business_id=order.business_id,
            preference_type="favorite_item",
        ).first()

        if favorite:
            favorite.preference_value = favorite_item
            favorite.strength = favorite_count
            favorite.source = "completed_order"
        else:
            favorite = _upsert_preference(
                customer_id=order.customer_id,
                business_id=order.business_id,
                preference_type="favorite_item",
                preference_value=favorite_item,
                strength=favorite_count,
                source="completed_order",
            )

        learned.append(favorite)

    return learned


def _clean_explicit_preference(value):
    """Clean an explicit preference phrase into a useful memory value."""

    value = value.strip(
        " \t\n.,!?;:()[]{}\"'"
    )

    value = re.split(
        r"\b(?:because|but|although|however|since)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    if len(value) < 2:
        return None

    if len(value) > 100:
        return None

    return value


def extract_explicit_preference(message):
    """
    Extract one explicit customer food preference.

    Returns:
        {
            "preference_type": "...",
            "preference_value": "..."
        }
        or None.
    """

    if not message:
        return None

    text = " ".join(
        str(message).strip().split()
    )

    normalized = text.lower()

    patterns = [
        (
            "explicit_dislike",
            [
                r"^i don't like (.+)$",
                r"^i dont like (.+)$",
                r"^i do not like (.+)$",
                r"^i hate (.+)$",
                r"^i can't stand (.+)$",
                r"^i cannot stand (.+)$",
                r"^no (.+?) for me$",
                r"^i don't want (.+)$",
                r"^i dont want (.+)$",
                r"^i generally avoid (.+)$",
                r"^i usually avoid (.+)$",
                r"^i tend to avoid (.+)$",
                r"^i never eat (.+)$",
            ],
        ),
        (
            "explicit_preference",
            [
                r"^i prefer (.+)$",
                r"^i'd prefer (.+)$",
                r"^id prefer (.+)$",
                r"^my preference is (.+)$",
                r"^i'm more of a (.+?) person$",
                r"^im more of a (.+?) person$",
                r"^i am more of a (.+?) person$",
                r"^i generally prefer (.+)$",
                r"^i usually prefer (.+)$",
                r"^i tend to prefer (.+)$",
            ],
        ),
        (
            "explicit_like",
            [
                r"^i like (.+)$",
                r"^i love (.+)$",
                r"^i really like (.+)$",
                r"^i really love (.+)$",
                r"^i always go for (.+)$",
                r"^i usually go for (.+)$",
                r"^i always order (.+)$",
                r"^i'm a big fan of (.+)$",
                r"^im a big fan of (.+)$",
                r"^i am a big fan of (.+)$",
                r"^i tend to go for (.+)$",
            ],
        ),
    ]

    for preference_type, expressions in patterns:

        for expression in expressions:

            match = re.match(
                expression,
                normalized,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            value = _clean_explicit_preference(
                match.group(1)
            )

            if not value:
                return None

            return {
                "preference_type": preference_type,
                "preference_value": value,
            }

    return None


def learn_explicit_customer_preference(
    customer,
    business_id,
    message,
):
    """Persist an explicitly stated customer preference and resolve conflicts."""

    if not customer or not message:
        return None

    extracted = extract_explicit_preference(
        message
    )

    if not extracted:
        return None

    preference_type = extracted["preference_type"]
    preference_value = extracted["preference_value"]

    # --------------------------------------------------------
    # CONFLICT RESOLUTION
    # --------------------------------------------------------
    #
    # A newer explicit like/dislike overrides the opposite
    # preference for the same value.
    # --------------------------------------------------------

    opposite_types = {
        "explicit_like": "explicit_dislike",
        "explicit_dislike": "explicit_like",
    }

    opposite_type = opposite_types.get(
        preference_type
    )

    if opposite_type:

        opposite = CustomerPreference.query.filter_by(
            customer_id=customer.id,
            business_id=business_id,
            preference_type=opposite_type,
            preference_value=preference_value,
        ).first()

        if opposite:
            db.session.delete(
                opposite
            )

    # --------------------------------------------------------
    # SAVE / STRENGTHEN CURRENT PREFERENCE
    # --------------------------------------------------------

    existing = CustomerPreference.query.filter_by(
        customer_id=customer.id,
        business_id=business_id,
        preference_type=preference_type,
        preference_value=preference_value,
    ).first()

    if existing:

        existing.strength = max(
            int(existing.strength or 1) + 1,
            1,
        )

        existing.source = "explicit_message"

        return existing

    preference = CustomerPreference(
        customer_id=customer.id,
        business_id=business_id,
        preference_type=preference_type,
        preference_value=preference_value,
        strength=1,
        source="explicit_message",
    )

    db.session.add(
        preference
    )

    return preference



def get_customer_relationship_stage(
    customer_id,
    business_id,
):
    """Return a simple relationship stage based on completed orders."""

    if not customer_id:
        return "New"

    completed_order_count = (
        Order.query
        .filter_by(
            customer_id=customer_id,
            business_id=business_id,
            status="Completed",
        )
        .count()
    )

    if completed_order_count <= 0:
        return "New"

    if completed_order_count == 1:
        return "First-time"

    if completed_order_count <= 3:
        return "Returning"

    return "Loyal"


def get_customer_reengagement_status(
    customer_id,
    business_id,
    minimum_days=3,
):
    """Determine whether a customer is eligible for gentle re-engagement."""

    from datetime import datetime

    from models.customer_interaction import CustomerInteraction

    if not customer_id:
        return {
            "eligible": False,
            "reason": "missing_customer",
        }

    completed_orders = (
        Order.query
        .filter_by(
            customer_id=customer_id,
            business_id=business_id,
            status="Completed",
        )
        .order_by(Order.id.desc())
        .all()
    )

    if len(completed_orders) < 2:
        return {
            "eligible": False,
            "reason": "not_enough_completed_orders",
            "completed_orders": len(completed_orders),
        }

    if has_recent_reengagement(
        customer_id=customer_id,
        business_id=business_id,
    ):
        return {
            "eligible": False,
            "reason": "recent_reengagement",
        }

    last_completed_order = completed_orders[0]

    if not last_completed_order.created_at:
        return {
            "eligible": False,
            "reason": "missing_last_order_timestamp",
        }

    active_order = (
        Order.query
        .filter(
            Order.customer_id == customer_id,
            Order.business_id == business_id,
            Order.status.in_(
                [
                    "Pending",
                    "Preparing",
                ]
            ),
        )
        .order_by(Order.id.desc())
        .first()
    )

    if active_order:
        return {
            "eligible": False,
            "reason": "active_order",
            "active_order_id": active_order.id,
        }

    latest_feedback = (
        CustomerInteraction.query
        .filter_by(
            customer_id=customer_id,
            business_id=business_id,
            interaction_type="feedback",
        )
        .order_by(
            CustomerInteraction.created_at.desc()
        )
        .first()
    )

    if (
        latest_feedback
        and latest_feedback.sentiment == "negative"
    ):
        return {
            "eligible": False,
            "reason": "recent_negative_feedback",
        }

    now = datetime.now(timezone.utc)

    days_since_order = (
        now - last_completed_order.created_at
    ).total_seconds() / 86400

    if days_since_order < minimum_days:
        return {
            "eligible": False,
            "reason": "order_too_recent",
            "days_since_order": round(
                days_since_order,
                2,
            ),
        }

    return {
        "eligible": True,
        "reason": "eligible",
        "completed_orders": len(completed_orders),
        "last_order_id": last_completed_order.id,
        "last_order_at": last_completed_order.created_at,
        "days_since_order": round(
            days_since_order,
            2,
        ),
    }


def build_reengagement_candidate(
    customer_id,
    business_id,
):
    """Build a personalized re-engagement candidate without sending it."""

    from models.customer_interaction import CustomerInteraction
    from models.customer import Customer

    customer = Customer.query.filter_by(
        id=customer_id,
        business_id=business_id,
    ).first()

    if not customer:
        return {
            "eligible": False,
            "reason": "customer_not_found",
        }

    eligibility = get_customer_reengagement_status(
        customer_id=customer_id,
        business_id=business_id,
    )

    if not eligibility.get("eligible"):
        return eligibility

    last_order = (
        Order.query
        .filter_by(
            id=eligibility["last_order_id"],
            customer_id=customer_id,
            business_id=business_id,
        )
        .first()
    )

    if not last_order:
        return {
            "eligible": False,
            "reason": "last_order_not_found",
        }

    preferences = (
        CustomerPreference.query
        .filter_by(
            customer_id=customer_id,
            business_id=business_id,
        )
        .order_by(
            CustomerPreference.strength.desc(),
            CustomerPreference.updated_at.desc(),
        )
        .limit(8)
        .all()
    )

    preference_lines = [
        f"{p.preference_type}: {p.preference_value}"
        for p in preferences
        if p.preference_value
    ]

    recent_feedback = (
        CustomerInteraction.query
        .filter_by(
            customer_id=customer_id,
            business_id=business_id,
            interaction_type="feedback",
        )
        .order_by(
            CustomerInteraction.created_at.desc()
        )
        .first()
    )

    items = [
        item.name
        for item in last_order.items
        if item.name
    ]

    relationship_stage = get_customer_relationship_stage(
        customer_id=customer_id,
        business_id=business_id,
    )

    return {
        "eligible": True,
        "customer_id": customer.id,
        "customer_name": customer.name,
        "phone": customer.phone,
        "relationship_stage": relationship_stage,
        "last_order_id": last_order.id,
        "last_order_items": items,
        "days_since_order": eligibility["days_since_order"],
        "preferences": preference_lines,
        "last_feedback_sentiment": (
            recent_feedback.sentiment
            if recent_feedback
            else None
        ),
        "last_feedback_message": (
            recent_feedback.message
            if recent_feedback
            else None
        ),
    }


def find_reengagement_candidates(
    business_id,
    minimum_days=3,
):
    """Return eligible customers without sending any messages."""

    from models.customer import Customer

    customers = (
        Customer.query
        .filter_by(
            business_id=business_id,
        )
        .all()
    )

    candidates = []

    for customer in customers:

        result = build_reengagement_candidate(
            customer_id=customer.id,
            business_id=business_id,
        )

        if result.get("eligible"):
            candidates.append(result)

    return candidates


def generate_reengagement_message(
    provider,
    candidate,
    language="English",
):
    """Generate a subtle personalized re-engagement message."""

    if not candidate or not candidate.get("eligible"):
        return None

    items = candidate.get(
        "last_order_items",
        [],
    )

    preferences = candidate.get(
        "preferences",
        [],
    )

    item_text = (
        ", ".join(items[:3])
        if items
        else "their previous meal"
    )

    preference_text = (
        "; ".join(preferences[:4])
        if preferences
        else "No specific preferences are known."
    )

    prompt = f"""
You are a restaurant customer relationship assistant.

Write ONE short WhatsApp re-engagement message for a returning customer.

CUSTOMER:
Name: {candidate.get("customer_name") or "Customer"}
Relationship stage: {candidate.get("relationship_stage")}
Days since last completed order: {candidate.get("days_since_order")}
Previous order items: {item_text}
Customer preferences: {preference_text}
Last feedback sentiment: {candidate.get("last_feedback_sentiment") or "none"}
Last feedback message: {candidate.get("last_feedback_message") or "none"}

RULES:
- Be warm, natural, and personal.
- Do not sound like an advertisement.
- Do not pressure the customer to order.
- Do not claim that the restaurant "missed" the customer or was waiting for them.
- Do not imply that the customer is currently craving anything.
- Do not invent discounts, promotions, availability, events, or menu facts.
- You may reference a previous item only as something they ordered before.
- Only say the customer liked, loved, enjoyed, preferred, or craved a specific item
  when the provided feedback message explicitly supports that specific item.
- A positive feedback sentiment alone does not prove which item the customer liked.
- Never turn a general positive sentiment into a specific item preference.
- Do not say "we missed you", "we were thinking about you", or imply the restaurant
  is personally waiting for the customer.
- Do not mention customer analytics, memory systems, relationship stages, or AI.
- Keep it to 1 or 2 short sentences.
- Reply in {language}.
"""

    try:

        response = provider.generate(
            prompt,
            temperature=0.65,
            max_tokens=100,
        )

        response = str(
            response or ""
        ).strip()

        return response or None

    except Exception:
        return None


def has_recent_reengagement(
    customer_id,
    business_id,
    cooldown_days=14,
):
    """Return True when a recent re-engagement was already sent."""

    from datetime import datetime, timedelta

    from models.customer_interaction import CustomerInteraction

    cutoff = datetime.now(timezone.utc) - timedelta(
        days=cooldown_days
    )

    interaction = (
        CustomerInteraction.query
        .filter(
            CustomerInteraction.customer_id == customer_id,
            CustomerInteraction.business_id == business_id,
            CustomerInteraction.interaction_type == "reengagement",
            CustomerInteraction.created_at >= cutoff,
        )
        .order_by(
            CustomerInteraction.created_at.desc()
        )
        .first()
    )

    return interaction is not None


def send_customer_reengagement(
    customer_id,
    business_id,
    provider,
    send_function,
    language="English",
    dry_run=True,
):
    """Generate and optionally send one eligible re-engagement message."""

    candidate = build_reengagement_candidate(
        customer_id=customer_id,
        business_id=business_id,
    )

    if not candidate.get("eligible"):
        return {
            "success": False,
            "sent": False,
            "reason": candidate.get("reason"),
            "candidate": candidate,
        }

    message = generate_reengagement_message(
        provider=provider,
        candidate=candidate,
        language=language,
    )

    if not message:
        return {
            "success": False,
            "sent": False,
            "reason": "message_generation_failed",
            "candidate": candidate,
        }

    # --------------------------------------------------------
    # DRY RUN
    # --------------------------------------------------------

    if dry_run:
        return {
            "success": True,
            "sent": False,
            "dry_run": True,
            "message": message,
            "candidate": candidate,
        }

    # --------------------------------------------------------
    # ACTUAL WHATSAPP SEND
    # --------------------------------------------------------

    try:

        sent = bool(
            send_function(
                candidate["phone"],
                message,
            )
        )

    except Exception:
        sent = False

    if not sent:
        return {
            "success": False,
            "sent": False,
            "reason": "whatsapp_send_failed",
            "message": message,
            "candidate": candidate,
        }

    # Record the interaction only after successful send.
    from models.customer_interaction import CustomerInteraction

    try:

        interaction = CustomerInteraction(
            customer_id=customer_id,
            business_id=business_id,
            order_id=candidate.get("last_order_id"),
            interaction_type="reengagement",
            sentiment=None,
            message=message,
            metadata_json=None,
        )

        db.session.add(interaction)
        db.session.commit()

    except Exception:

        db.session.rollback()

        return {
            "success": False,
            "sent": True,
            "reason": "interaction_record_failed",
            "message": message,
            "candidate": candidate,
        }

    return {
        "success": True,
        "sent": True,
        "message": message,
        "candidate": candidate,
    }


def preview_reengagement_message(
    provider,
    candidate,
    language="English",
):
    """Generate a re-engagement message from a validated candidate without sending."""

    if not candidate or not candidate.get("eligible"):
        return {
            "success": False,
            "message": None,
            "reason": "candidate_not_eligible",
        }

    message = generate_reengagement_message(
        provider=provider,
        candidate=candidate,
        language=language,
    )

    if not message:
        return {
            "success": False,
            "message": None,
            "reason": "message_generation_failed",
        }

    return {
        "success": True,
        "message": message,
        "reason": "preview",
    }


def preview_all_reengagements(
    provider,
    business_id,
    language="English",
):
    """Preview all currently eligible re-engagement messages."""

    candidates = find_reengagement_candidates(
        business_id=business_id,
        minimum_days=3,
    )

    results = []

    for candidate in candidates:

        message = generate_reengagement_message(
            provider=provider,
            candidate=candidate,
            language=language,
        )

        results.append({
            "customer_id": candidate["customer_id"],
            "customer_name": candidate.get("customer_name"),
            "phone": candidate.get("phone"),
            "relationship_stage": candidate.get(
                "relationship_stage"
            ),
            "last_order_id": candidate.get(
                "last_order_id"
            ),
            "days_since_order": candidate.get(
                "days_since_order"
            ),
            "message": message,
        })

    return results
