
import random
import re
from datetime import datetime
from typing import Optional, Tuple

memory = {"name": None}


def simple_if_else_reply(text: str) -> Optional[str]:
    text = text.lower()
    if "hello" in text or "hi" in text:
        return "Hello! (matched via if-else)"
    elif "bye" in text:
        return "Goodbye! (matched via if-else)"
    else:
        return None  # fall through to the pattern-matching engine


def handle_name(match: re.Match) -> str:
    name = match.group(1) or match.group(2)
    memory["name"] = name.capitalize()
    return f"Nice to meet you, {memory['name']}! I'll remember that."


def recall_name(match: re.Match) -> str:
    if memory["name"]:
        return f"Your name is {memory['name']}, right?"
    return "You haven't told me your name yet!"


def tell_time(match: re.Match) -> str:
    return f"I don't have a live clock, but your system time is {datetime.now().strftime('%H:%M:%S')}."


RULES = [
    ("greeting", re.compile(r"\b(hi|hello|hey|yo|howdy)\b", re.I),
        ["Hello! How can I help you today?",
         "Hi there! What can I do for you?",
         "Hey! Ask me anything."]),

    ("wellbeing", re.compile(r"\bhow (are|r) (you|u)\b", re.I),
        ["I'm just a bunch of rules, but I'm doing great! How about you?",
         "Running smoothly, thanks for asking!"]),

    ("name_intro", re.compile(r"\bmy name is (\w+)\b|\bi'?m (\w+)\b", re.I),
        handle_name),

    ("recall_name", re.compile(r"\bwhat'?s? my name\b", re.I),
        recall_name),

    ("bot_identity", re.compile(r"\b(who are you|what are you)\b", re.I),
        ["I'm a simple rule-based chatbot built with Python and regex pattern matching."]),

    ("thanks", re.compile(r"\b(thanks|thank you|thx)\b", re.I),
        ["You're welcome!", "Anytime!", "No problem at all."]),

    ("weather", re.compile(r"\bweather\b", re.I),
        ["I can't check live weather since I'm rule-based, but I hope it's nice outside!"]),

    ("joke", re.compile(r"\bjoke\b", re.I),
        ["Why don't programmers like nature? It has too many bugs.",
         "Why do Java developers wear glasses? Because they don't see sharp.",
         "I told my computer I needed a break, and it froze immediately."]),

    ("time", re.compile(r"\b(what time is it|current time)\b", re.I),
        tell_time),

    ("help", re.compile(r"\bhelp\b", re.I),
        ["Try things like: 'hello', 'tell me a joke', 'my name is Alex', "
         "'what's my name', 'what time is it', or 'bye'."]),

    ("affirmative", re.compile(r"^\s*(yes|yeah|yep|sure)\s*$", re.I),
        ["Great!", "Good to hear it."]),

    ("negative", re.compile(r"^\s*(no|nope|nah)\s*$", re.I),
        ["Alright, no worries.", "Okay, fair enough."]),

    ("farewell", re.compile(r"\b(bye|goodbye|see ya|exit|quit)\b", re.I),
        ["Goodbye! Have a great day!", "See you later!", "Bye! Come back anytime."]),

    ("fallback", re.compile(r".*"),
        ["I'm not sure I understand. Could you rephrase that?",
         "Hmm, I don't have a rule for that yet. Try typing 'help'.",
         "That's outside my current rule set, sorry!"]),
]

def get_response(user_text: str) -> Tuple[str, str]:
    """
    Returns (reply, matched_rule_name).
    First tries the plain if-else layer, then falls back to the
    regex rule engine (first match wins).
    """
    quick_reply = simple_if_else_reply(user_text)
    if quick_reply:
        return quick_reply, "if-else"

    for rule_name, pattern, action in RULES:
        match = pattern.search(user_text)
        if match:
            if callable(action):
                return action(match), rule_name
            return random.choice(action), rule_name

    return "...", "none"


def main():
    print("=" * 55)
    print(" RuleBot - Rule-Based Chatbot (type 'quit' to exit)")
    print("=" * 55)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break

        if not user_input:
            continue

        reply, rule_name = get_response(user_input)
        print(f"Bot: {reply}")
        print(f"     [matched rule: {rule_name}]")

        if rule_name == "farewell":
            break


if __name__ == "__main__":
    main()
