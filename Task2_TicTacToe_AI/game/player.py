"""
player.py
---------
Defines the player hierarchy using OOP:
    Player (abstract base)
      ├── HumanPlayer  -- gets moves via validated terminal input
      └── AIPlayer     -- gets moves via Easy/Medium/Hard strategies

Using a common base class lets the Game class treat both players
polymorphically: it just calls `player.get_move(board)` without caring
whether a human or an AI is actually deciding.
"""

import random
from abc import ABC, abstractmethod
from game.board import Board
from game.ai_algorithms import find_best_move

# Supported AI difficulty levels
EASY = "EASY"
MEDIUM = "MEDIUM"
HARD = "HARD"


class Player(ABC):
    """Abstract base class for any Tic-Tac-Toe player (human or AI)."""

    def __init__(self, name: str, mark: str):
        self.name = name
        self.mark = mark  # 'X' or 'O'

    @abstractmethod
    def get_move(self, board: Board) -> int:
        """Returns a 0-8 board index representing this player's chosen move."""
        raise NotImplementedError


class HumanPlayer(Player):
    """
    A human player whose moves come from terminal input. Handles all
    input validation here so the Game loop doesn't need to worry about
    malformed input (e.g., letters, out-of-range numbers, occupied cells).
    """

    def get_move(self, board: Board) -> int:
        while True:
            raw = input(f"{self.name} ({self.mark}), enter your move (1-9): ").strip()

            if not raw:
                print("⚠  Input cannot be empty. Please enter a number from 1-9.")
                continue

            if not raw.isdigit():
                print("⚠  Invalid input. Please enter a NUMBER between 1 and 9.")
                continue

            position = int(raw) - 1  # convert 1-indexed (user-friendly) to 0-indexed

            if not (0 <= position <= 8):
                print("⚠  Out of range. Please choose a number between 1 and 9.")
                continue

            if not board.is_valid_move(position):
                print("⚠  That cell is already taken. Choose an empty cell.")
                continue

            return position


class AIPlayer(Player):
    """
    The AI opponent. Supports three difficulty levels:

        EASY   -> Picks a uniformly random legal move (very beatable).
        MEDIUM -> Mixed strategy: blocks immediate human wins / takes
                  immediate AI wins when available, otherwise plays
                  randomly (beatable, but not careless).
        HARD   -> Full Minimax with Alpha-Beta Pruning over the entire
                  remaining game tree (mathematically unbeatable -- best
                  possible outcome for AI is win or draw, never a loss).
    """

    def __init__(self, name: str, mark: str, opponent_mark: str, difficulty: str = HARD):
        super().__init__(name, mark)
        self.opponent_mark = opponent_mark
        self.difficulty = difficulty

    def get_move(self, board: Board) -> int:
        if self.difficulty == EASY:
            return self._easy_move(board)
        elif self.difficulty == MEDIUM:
            return self._medium_move(board)
        else:
            return self._hard_move(board)

    # ------------------------------------------------------------------
    # EASY: pure randomness
    # ------------------------------------------------------------------
    def _easy_move(self, board: Board) -> int:
        return random.choice(board.get_available_moves())

    # ------------------------------------------------------------------
    # MEDIUM: simple heuristic, no full search
    # ------------------------------------------------------------------
    def _medium_move(self, board: Board) -> int:
        # 1) Take a winning move right now if one exists.
        winning_move = self._find_immediate_win(board, self.mark)
        if winning_move is not None:
            return winning_move

        # 2) Otherwise, block the opponent's immediate winning move.
        blocking_move = self._find_immediate_win(board, self.opponent_mark)
        if blocking_move is not None:
            return blocking_move

        # 3) Otherwise, prefer the center, then corners, then edges.
        preferred_order = [4, 0, 2, 6, 8, 1, 3, 5, 7]
        for cell in preferred_order:
            if cell in board.get_available_moves():
                # Add a touch of randomness so Medium isn't 100% deterministic
                if random.random() < 0.7:
                    return cell

        return random.choice(board.get_available_moves())

    def _find_immediate_win(self, board: Board, mark: str) -> int:
        """Checks if placing `mark` in any available cell wins immediately."""
        for move in board.get_available_moves():
            board.place_mark(move, mark)
            is_win = board.check_winner() == mark
            board.undo_move(move)
            if is_win:
                return move
        return None

    # ------------------------------------------------------------------
    # HARD: Minimax + Alpha-Beta Pruning (unbeatable)
    # ------------------------------------------------------------------
    def _hard_move(self, board: Board) -> int:
        best_move, _ = find_best_move(
            board, ai_mark=self.mark, opponent_mark=self.opponent_mark, use_pruning=True
        )
        return best_move
