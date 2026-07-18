"""
TIC-TAC-TOE AI
--------------
A console-based Tic-Tac-Toe game where the human plays against an
unbeatable AI. The AI uses the Minimax algorithm with Alpha-Beta
Pruning to search the game tree and always pick the optimal move.

Run this file directly to play:
    python tic_tac_toe.py
"""

import math

HUMAN = "X"
AI = "O"
EMPTY = " "

WINNING_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
]


# ------------------------------------------------------------------
# Board helpers
# ------------------------------------------------------------------
def print_board(board):
    print()
    for r in range(3):
        row = board[r * 3: r * 3 + 3]
        print(" " + " | ".join(row))
        if r < 2:
            print("---+---+---")
    print()


def available_moves(board):
    return [i for i, cell in enumerate(board) if cell == EMPTY]


def check_winner(board):
    """Returns 'X', 'O', 'Draw', or None (game still in progress)."""
    for a, b, c in WINNING_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    if EMPTY not in board:
        return "Draw"
    return None


# ------------------------------------------------------------------
# Minimax with Alpha-Beta Pruning
# ------------------------------------------------------------------
def minimax(board, depth, is_maximizing, alpha, beta):
    """
    Returns the best achievable score for the current board state.
    AI ('O') is the maximizer, Human ('X') is the minimizer.
    Scores are adjusted by depth so the AI prefers faster wins and
    slower losses.
    """
    result = check_winner(board)
    if result == AI:
        return 10 - depth
    if result == HUMAN:
        return depth - 10
    if result == "Draw":
        return 0

    if is_maximizing:
        best_score = -math.inf
        for move in available_moves(board):
            board[move] = AI
            score = minimax(board, depth + 1, False, alpha, beta)
            board[move] = EMPTY
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break  # Beta cutoff (prune remaining branches)
        return best_score
    else:
        best_score = math.inf
        for move in available_moves(board):
            board[move] = HUMAN
            score = minimax(board, depth + 1, True, alpha, beta)
            board[move] = EMPTY
            best_score = min(best_score, score)
            beta = min(beta, best_score)
            if beta <= alpha:
                break  # Alpha cutoff (prune remaining branches)
        return best_score


def best_ai_move(board):
    """Finds the optimal move for the AI using minimax + alpha-beta."""
    best_score = -math.inf
    move_choice = None
    for move in available_moves(board):
        board[move] = AI
        score = minimax(board, 0, False, -math.inf, math.inf)
        board[move] = EMPTY
        if score > best_score:
            best_score = score
            move_choice = move
    return move_choice


# ------------------------------------------------------------------
# Game loop
# ------------------------------------------------------------------
def get_human_move(board):
    while True:
        raw = input("Your move (1-9, numbered left-to-right, top-to-bottom): ").strip()
        if not raw.isdigit() or not (1 <= int(raw) <= 9):
            print("Please enter a number between 1 and 9.")
            continue
        pos = int(raw) - 1
        if board[pos] != EMPTY:
            print("That cell is already taken. Try again.")
            continue
        return pos


def choose_first_player():
    while True:
        choice = input("Do you want to go first? (y/n): ").strip().lower()
        if choice in ("y", "yes"):
            return HUMAN
        if choice in ("n", "no"):
            return AI
        print("Please answer y or n.")


def print_position_guide():
    print("Positions are numbered like this:")
    print_board(["1", "2", "3", "4", "5", "6", "7", "8", "9"])


def main():
    print("=" * 40)
    print(" TIC-TAC-TOE  —  You (X) vs Unbeatable AI (O)")
    print("=" * 40)
    print_position_guide()

    board = [EMPTY] * 9
    turn = choose_first_player()

    while True:
        print_board(board)
        winner = check_winner(board)
        if winner:
            break

        if turn == HUMAN:
            pos = get_human_move(board)
            board[pos] = HUMAN
            turn = AI
        else:
            print("AI is thinking...")
            pos = best_ai_move(board)
            board[pos] = AI
            print(f"AI plays position {pos + 1}")
            turn = HUMAN

    print_board(board)
    winner = check_winner(board)
    if winner == "Draw":
        print("It's a draw! The AI can't be beaten, only tied.")
    elif winner == AI:
        print("The AI wins! Better luck next time.")
    else:
        print("Wait... you beat the AI? That shouldn't be possible with a bug-free minimax!")


if __name__ == "__main__":
    main()
