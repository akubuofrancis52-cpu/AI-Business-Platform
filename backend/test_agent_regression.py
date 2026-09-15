import pytest
from services.ai.agent import classify_message_fast, recommendation_opted_out


def test_modify_intent_routing():
    assert classify_message_fast("Add a Lemonade") == "modify_order"
    assert classify_message_fast("I want to add a Lemonade") == "modify_order"
    assert classify_message_fast("Actually remove the Lemonade") == "modify_order"
    assert classify_message_fast("Remove Lemonade") == "modify_order"
    assert classify_message_fast("Take off the Lemonade") == "modify_order"


def test_order_intent_stays_order():
    assert (
        classify_message_fast(
            "I want a Signature Lebanese Shawarma"
        )
        == "order"
    )


def test_recommendation_opt_out():
    assert recommendation_opted_out("No thanks") is True
    assert recommendation_opted_out("Nothing else") is True
    assert recommendation_opted_out("That's all") is True
    assert recommendation_opted_out("Don't recommend anything") is True
    assert recommendation_opted_out("I want a shawarma") is False

@pytest.mark.parametrize(
    "message",
    [
        "make it two",
        "make that three",
        "make this four",
        "change it to two",
        "change that to three",
        "set it to four",
        "remove that one",
        "remove the second item",
        "take off the first one",
        "mets-en deux",
        "mets le deuxième à trois",
        "supprime le premier",
        "retire celui-là",
    ],
)
def test_conversational_modify_messages_route_to_modify_order(message):
    assert classify_message_fast(message) == "modify_order"


@pytest.mark.parametrize(
    "message",
    [
        "I want a shawarma",
        "Give me two lemonades",
        "I want a Classic Margherita Pizza",
        "Je veux un shawarma",
        "Donne-moi deux limonades",
    ],
)
def test_explicit_food_orders_stay_order_intent(message):
    assert classify_message_fast(message) == "order"

