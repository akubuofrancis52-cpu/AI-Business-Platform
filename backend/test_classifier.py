from services.ai.agent import classify_message_fast

tests = [
    "I want two pizzas",
    "Can I have a burger?",
    "Give me fries",
    "I want to place an order",
    "Can I have your opening hours?",
    "What are your opening hours?",
    "When do you close?",
    "Are you open today?",
    "I want information about delivery",
    "Do you deliver?",
    "Where are you located?",
    "Show me the menu",
    "What pizzas do you have?",
    "Cancel my order",
    "Remove the burger from my order",
    "Hello",
    "Hi",
    "Good afternoon",
    "What is good here?",
    "What do you recommend?",
    "What should I get?",
    "Thanks",
]

for message in tests:
    result = classify_message_fast(message)
    print(f"{message} -> {result}")
