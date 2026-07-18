# Tic-Tac-Toe AI (Minimax with Alpha-Beta Pruning)

An unbeatable Tic-Tac-Toe AI agent that plays against a human player
in the terminal. The AI uses the **Minimax algorithm with
Alpha-Beta Pruning** to search the entire game tree and always
choose the mathematically optimal move — meaning it can never lose
(best case for you: a draw).

## Objective

Implement an AI agent that plays the classic game of Tic-Tac-Toe
against a human player, using Minimax (with Alpha-Beta Pruning) to
make the AI unbeatable — as a hands-on introduction to game theory
and basic search algorithms.

## How It Works

### 1. Game Representation
The board is a flat list of 9 cells (`" "` = empty, `"X"` = human,
`"O"` = AI), indexed like a numpad:

```
 0 | 1 | 2
-----------
 3 | 4 | 5
-----------
 6 | 7 | 8
```

### 2. Minimax Algorithm
Minimax explores every possible sequence of future moves from the
current board state, assuming **both players play optimally**:

- The **AI ("O")** is the *maximizer* — it tries to pick moves that
  lead to the highest possible score.
- The **human ("X")** is treated as the *minimizer* — the algorithm
  assumes the opponent will always try to minimize the AI's score
  (i.e., play their best move too).
- Each terminal board state (win/lose/draw) is scored:
  - AI wins → `+10 - depth`
  - Human wins → `depth - 10`
  - Draw → `0`

  Subtracting/adding `depth` makes the AI prefer **faster wins** and
  **slower losses** — e.g., it'll take a win in 2 moves over a win
  in 4 moves, and delay an unavoidable loss as long as possible.

The AI recursively simulates every possible game continuation and
picks the move that leads to the best guaranteed outcome, no matter
what the opponent does.

### 3. Alpha-Beta Pruning
A plain Minimax search explores the *entire* game tree — for
Tic-Tac-Toe that's small enough to brute-force, but Alpha-Beta
Pruning is added to demonstrate the standard optimization technique
used in real game-playing AI (chess engines, etc.):

- `alpha` = the best score the maximizer can guarantee so far
- `beta` = the best score the minimizer can guarantee so far
- Whenever `beta <= alpha`, the remaining branches at that node are
  **skipped** ("pruned") because a rational opponent would never
  allow that path to be reached — it cuts down the number of board
  states evaluated without changing the final decision.

## How to Run

Requires Python 3.7+ (no external packages needed).

```bash
python tic_tac_toe.py
```

You'll be asked whether you want to go first, then prompted to enter
a position (1–9) on your turn. Example:

```
========================================
 TIC-TAC-TOE  —  You (X) vs Unbeatable AI (O)
========================================
Positions are numbered like this:

 1 | 2 | 3
---+---+---
 4 | 5 | 6
---+---+---
 7 | 8 | 9

Do you want to go first? (y/n): y

   |   |
---+---+---
   |   |
---+---+---
   |   |

Your move (1-9, numbered left-to-right, top-to-bottom): 5
AI is thinking...
AI plays position 1
...
```

## Verified Unbeatable

The AI was tested against **200 simulated games** where the "human"
player made random moves. Result: the AI **never lost a single
game** — only wins or draws — confirming the Minimax + Alpha-Beta
implementation is correct.

## Project Structure

```
.
├── tic_tac_toe.py   # Game logic, Minimax + Alpha-Beta AI, console loop
└── README.md        # This file
```

## Key Functions

| Function | Purpose |
|---|---|
| `check_winner(board)` | Checks all 8 winning lines; returns `'X'`, `'O'`, `'Draw'`, or `None` |
| `minimax(board, depth, is_maximizing, alpha, beta)` | Recursively scores a board state assuming optimal play from both sides |
| `best_ai_move(board)` | Tries every available move, runs minimax on each, and returns the move with the highest score |

## Possible Future Improvements

- Add a difficulty setting (e.g., a "beatable" mode where the AI
  occasionally plays a random suboptimal move)
- Build a GUI version (Tkinter, Pygame, or a web-based version)
- Extend to larger boards (e.g., 4x4 or "N-in-a-row") with a move
  ordering heuristic for deeper pruning efficiency
- Add a transposition table to cache repeated board evaluations

## What This Demonstrates

- Game tree search and recursive backtracking
- The Minimax algorithm for two-player zero-sum games
- Alpha-Beta Pruning as a search optimization technique
- Core game theory concepts: optimal play, adversarial search, and
  guaranteed outcomes
