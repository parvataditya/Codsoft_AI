# RuleBot - Web Version 

A browser-based version of the rule-based chatbot. Pure HTML, CSS,
and JavaScript — no build tools, servers, or external libraries
required. Open the file and start chatting.

> This is a bonus/demo version. The primary project deliverable is
> the Python console app (`chatbot.py`) in the root of the repo.
> This file exists to showcase the same rule-matching logic with an
> interactive chat UI.

## How to Run

Just open `chatbot.html` in any modern web browser:

- **Double-click** the file, or
- Right-click → **Open with** → your browser, or
- Drag the file into an open browser window

No installation, server, or internet connection needed — everything
runs client-side in the browser.

## How It Works

The logic mirrors the Python version, translated into JavaScript:

1. **Rules array** — each rule has:
   - a `pattern` (a JavaScript regular expression), and
   - either a fixed list of `responses` (a random one is picked), or
     a `respond()` function for dynamic replies.

2. **Pattern matching** — `getBotResponse()` loops through the rules
   **top to bottom** and returns the reply for the **first rule
   whose pattern matches** the user's message.

3. **Fallback rule** — the last rule uses `.*` so it always matches,
   giving a graceful "I don't understand" reply when nothing else
   fits.

4. **Simple memory** — a `userName` variable outside the rules is
   set by one rule (`name_intro`) and read by another
   (`recall_name`), showing basic state carried across the
   conversation.

5. **UI** — built with plain HTML/CSS for the chat bubbles and input
   box, and vanilla JS to handle form submission and rendering
   messages. The matched rule name is shown under each bot reply so
   you can see exactly which rule fired.

## Supported Rules

| You say (examples)              | Bot behavior                          |
|----------------------------------|----------------------------------------|
| `hi`, `hello`, `hey`, `yo`       | Greeting                               |
| `how are you`                    | Wellbeing response                     |
| `my name is Alex` / `I'm Alex`   | Stores your name                       |
| `what's my name`                 | Recalls stored name                    |
| `who are you`                    | Bot identity                           |
| `thanks` / `thank you`           | Acknowledgement                        |
| `weather`                        | Canned weather reply                   |
| `tell me a joke`                 | Random joke                            |
| `what time is it`                | Current browser time                   |
| `help`                           | Lists example commands                 |
| `yes` / `no`                     | Small talk                             |
| `bye`, `exit`, `quit`            | Ends the conversation                  |
| *(anything else)*                | Fallback / "I don't understand" reply  |

Quick-start suggestion chips are also included below the chat input
so you can try common messages with one click.

## File Structure

Everything — HTML, CSS, and JavaScript — is contained in the single
`chatbot_web.html` file for easy sharing and zero setup.

## Extending the Bot

To add a new rule, add an entry to the `rules` array in the
`<script>` section:

```javascript
{
  name: "compliment",
  pattern: /\b(you'?re (great|awesome|cool))\b/i,
  responses: ["Aw, thank you!", "That's kind of you to say!"]
}
```

Place new rules **above** the `fallback` rule, since the fallback
must always stay last (it matches everything).

## What This Demonstrates

- Rule/intent matching using regular expressions in JavaScript
- Priority-ordered checks with a fallback/default case
- Simple state management (remembering the user's name) in a
  browser environment
- Building a minimal chat interface with plain HTML/CSS/JS
