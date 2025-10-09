"""Command-line Tic-Tac-Toe game.

This module contains everything required to play a game of tic-tac-toe in
the terminal. It supports both human vs. human and human vs. computer play
using a simple minimax-based AI opponent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


Player = str


WINNING_COMBINATIONS: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


def print_board(board: Sequence[Player | None]) -> None:
    """Render the board to the terminal."""

    def format_cell(value: Optional[Player]) -> str:
        return value if value is not None else " "

    rows = [
        " | ".join(format_cell(board[i + j]) for j in range(3))
        for i in range(0, 9, 3)
    ]
    divider = "\n---------\n"
    print(divider.join(rows))


def available_moves(board: Sequence[Player | None]) -> List[int]:
    """Return the indexes of all empty cells."""

    return [index for index, value in enumerate(board) if value is None]


def check_winner(board: Sequence[Player | None]) -> Optional[Player]:
    """Return the winner symbol ("X" or "O") if there is one."""

    for combo in WINNING_COMBINATIONS:
        first, second, third = combo
        if (
            board[first] is not None
            and board[first] == board[second] == board[third]
        ):
            return board[first]
    return None


def is_draw(board: Sequence[Player | None]) -> bool:
    """Return True if the board is full and there is no winner."""

    return all(cell is not None for cell in board) and check_winner(board) is None


def minimax(board: List[Player | None], player: Player, ai_player: Player) -> Tuple[int, Optional[int]]:
    """Minimax algorithm that returns the best score and move for the AI."""

    opponent = "O" if player == "X" else "X"
    winner = check_winner(board)

    if winner == ai_player:
        return 1, None
    if winner == opponent:
        return -1, None
    if is_draw(board):
        return 0, None

    best_move: Optional[int] = None
    best_score = float("-inf") if player == ai_player else float("inf")

    for move in available_moves(board):
        board[move] = player
        score, _ = minimax(board, opponent, ai_player)
        board[move] = None

        if player == ai_player:
            if score > best_score:
                best_score, best_move = score, move
        else:
            if score < best_score:
                best_score, best_move = score, move

    return best_score, best_move


def get_human_move(board: Sequence[Player | None]) -> int:
    """Prompt the user for a valid move."""

    valid_moves = available_moves(board)
    while True:
        choice = input(f"Choose your move {valid_moves}: ")
        if not choice.isdigit():
            print("Please enter a number corresponding to an empty square.")
            continue

        index = int(choice)
        if index in valid_moves:
            return index
        print("That square is not available. Try again.")


@dataclass
class GameState:
    board: List[Player | None]
    current_player: Player
    against_computer: bool
    human_player: Player


def switch_player(player: Player) -> Player:
    return "O" if player == "X" else "X"


def play_game() -> None:
    """Run the game loop."""

    print("Welcome to Tic-Tac-Toe!")
    mode = ""
    while mode not in {"1", "2"}:
        print("1. Play against another human")
        print("2. Play against the computer")
        mode = input("Select an option (1 or 2): ")

    against_computer = mode == "2"
    human_player = "X"
    if against_computer:
        choice = ""
        while choice not in {"X", "O"}:
            choice = input("Do you want to be X or O? ").upper()
        human_player = choice

    state = GameState(board=[None] * 9, current_player="X", against_computer=against_computer, human_player=human_player)

    while True:
        print("\nCurrent board:")
        print_board(state.board)

        winner = check_winner(state.board)
        if winner or is_draw(state.board):
            break

        if state.against_computer and state.current_player != state.human_player:
            print("Computer is thinking...")
            _, move = minimax(state.board[:], state.current_player, state.current_player)
            assert move is not None
        else:
            move = get_human_move(state.board)

        state.board[move] = state.current_player
        state.current_player = switch_player(state.current_player)

    print("\nFinal board:")
    print_board(state.board)

    if winner := check_winner(state.board):
        if state.against_computer and winner != state.human_player:
            print("Computer wins!")
        elif state.against_computer and winner == state.human_player:
            print("You win!")
        else:
            print(f"Player {winner} wins!")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    play_game()

