# Chatbot with Rule-Based Responses

A simple chatbot that identifies user intent and replies using
**predefined rules**, built with plain Python — no external
libraries or ML models required.

## Objective

Build a chatbot that responds to user inputs based on predefined
rules, using **if-else statements** and **pattern matching**
(regular expressions) to identify queries and generate appropriate
responses — a first step toward understanding NLP and conversation
flow.

## How It Works

The bot uses two complementary techniques, both requested by the task:

1. **If-else keyword checking** (`simple_if_else_reply`)
   A minimal example showing the classic approach: check if a
   keyword like `"hello"` or `"bye"` appears in the input and
   return a fixed reply.

2. **Regex-based pattern matching** (`RULES` list + `get_response`)
   Instead of writing dozens of nested if-else blocks, each rule is
   stored as `(name, regex_pattern, response)`. The engine loops
   through the rules **top to bottom** and returns the reply for the
   **first pattern that matches** — this is the standard technique
   used in classic rule-based bots like ELIZA.

   A rule's response can be either:
   - a **list of strings** (a random one is chosen — adds variety), or
   - a **function** (for dynamic replies, e.g. remembering the
     user's name or reporting the current time).

   The last rule (`fallback`) matches *anything* (`.*`) and is kept
   at the end of the list, so the bot always has something to say
   even when no specific rule applies.

3. **Simple memory** — a `memory` dictionary lets the bot remember
   information (like your name) across turns of the conversation,
   demonstrating basic conversational state.

## Supported Rules

| You say (examples)              | Bot behavior                          |
|----------------------------------|----------------------------------------|
| `hi`, `hello`, `hey`             | Greeting                               |
| `how are you`                    | Wellbeing response                     |
| `my name is Alex` / `I'm Alex`   | Stores your name                       |
| `what's my name`                 | Recalls stored name                    |
| `who are you`                    | Bot identity                           |
| `thanks` / `thank you`           | Acknowledgement                        |
| `weather`                        | Canned weather reply                   |
| `tell me a joke`                 | Random joke                            |
| `what time is it`                | Current system time                    |
| `help`                           | Lists example commands                 |
| `yes` / `no`                     | Small talk                             |
| `bye`, `exit`, `quit`            | Ends the conversation                  |
| *(anything else)*                | Fallback / "I don't understand" reply  |

## How to Run

Requires Python 3.7+ (no external packages needed).

```bash
python chatbot.py
```

Example session:

```
=======================================================
 RuleBot - Rule-Based Chatbot (type 'quit' to exit)
=======================================================
You: hi
Bot: Hello! How can I help you today?
     [matched rule: if-else]
You: my name is Alex
Bot: Nice to meet you, Alex! I'll remember that.
     [matched rule: name_intro]
You: what's my name
Bot: Your name is Alex, right?
     [matched rule: recall_name]
You: tell me a joke
Bot: Why don't programmers like nature? It has too many bugs.
     [matched rule: joke]
You: bye
Bot: Goodbye! Have a great day!
     [matched rule: farewell]
```

The `[matched rule: ...]` line is printed under each reply so you
(or your reviewer) can see exactly which rule fired — useful for
demoing how the pattern matching works.

## Project Structure

```
.
├── chatbot.py   # Main chatbot logic + console chat loop
└── README.md    # This file
```

## Extending the Bot

To add a new rule, just append a tuple to the `RULES` list in
`chatbot.py`:

```python
("weather_forecast", re.compile(r"\bforecast\b", re.I),
    ["It looks sunny in my rule book, but check a real weather app!"]),
```

Since the fallback rule must always catch unmatched input, make sure
any new rule is added **before** the `fallback` entry at the bottom
of the list.

## Possible Future Improvements

- Load rules from an external JSON/YAML file instead of hardcoding them
- Add a basic web UI (Flask/Streamlit) on top of the same rule engine
- Track conversation history for multi-turn context
- Add fuzzy matching for typos (e.g. using `difflib`)

## What This Demonstrates

- Reading and parsing user text input
- Two ways of encoding "rules": explicit if-else vs. regex pattern
  tables
- Priority-ordered rule matching with a fallback/default case
- Simple state/memory across a conversation
- Basic building blocks that scale up into real NLP intent-matching
  systems (like those used in early chatbot frameworks)
