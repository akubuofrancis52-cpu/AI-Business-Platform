from services.ai.agent import recommendation_opted_out


def test_recommendation_opt_out():
    assert recommendation_opted_out("No thanks") is True
    assert recommendation_opted_out("Nothing else") is True
    assert recommendation_opted_out("That's all") is True
    assert recommendation_opted_out("Thats all") is True
    assert recommendation_opted_out("That is all") is True
    assert recommendation_opted_out("Don't recommend anything") is True
    assert recommendation_opted_out("Dont recommend anything") is True
    assert recommendation_opted_out("Do not recommend anything") is True
    assert recommendation_opted_out("I want a shawarma") is False
