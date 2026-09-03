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
