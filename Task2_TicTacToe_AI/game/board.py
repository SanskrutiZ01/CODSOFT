"""
board.py
--------
Defines the Board class: the data structure and rules engine for the
Tic-Tac-Toe grid. The Board knows nothing about players or AI -- it only
manages state (cell values), validates moves, and detects terminal states
(win/draw). This separation of concerns keeps game rules independent of
how moves are decided, which is what allows the same Board to be reused
by both HumanPlayer and AIPlayer.
"""

from typing import List, Optional, Tuple

EMPTY = " "

# All 8 winning lines: 3 rows, 3 columns, 2 diagonals (as index triplets)
WINNING_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
)


class Board:
    """
    Represents a 3x3 Tic-Tac-Toe board using a flat list of 9 cells
    (index 0-8), which simplifies win-checking and minimax recursion
    compared to a nested 2D list.

    Cell layout (indices):
         0 | 1 | 2
        -----------
         3 | 4 | 5
        -----------
         6 | 7 | 8
    """

    def __init__(self):
        self.cells: List[str] = [EMPTY] * 9

    def reset(self) -> None:
        """Clears the board back to all-empty cells."""
        self.cells = [EMPTY] * 9

    def get_available_moves(self) -> List[int]:
        """Returns indices of all empty cells (i.e., legal moves)."""
        return [i for i, cell in enumerate(self.cells) if cell == EMPTY]

    def is_valid_move(self, position: int) -> bool:
        """Checks whether a move at `position` is legal."""
        return 0 <= position <= 8 and self.cells[position] == EMPTY

    def place_mark(self, position: int, mark: str) -> bool:
        """
        Attempts to place `mark` ('X' or 'O') at `position`.
        Returns True if successful, False if the move was invalid.
        """
        if not self.is_valid_move(position):
            return False
        self.cells[position] = mark
        return True

    def undo_move(self, position: int) -> None:
        """Reverts a cell back to empty. Essential for Minimax backtracking."""
        self.cells[position] = EMPTY

    def check_winner(self) -> Optional[str]:
        """Returns 'X' or 'O' if that player has a winning line, else None."""
        for a, b, c in WINNING_LINES:
            if self.cells[a] != EMPTY and self.cells[a] == self.cells[b] == self.cells[c]:
                return self.cells[a]
        return None

    def is_full(self) -> bool:
        """Returns True if there are no empty cells left."""
        return EMPTY not in self.cells

    def is_terminal(self) -> bool:
        """A board state is terminal if there's a winner OR it's full (draw)."""
        return self.check_winner() is not None or self.is_full()

    def is_draw(self) -> bool:
        """A draw is a full board with no winner."""
        return self.is_full() and self.check_winner() is None

    def clone(self) -> "Board":
        """
        Returns a deep-enough copy of the board. Used defensively in
        scenarios where we want a snapshot without affecting recursion
        state (not required by the current Minimax implementation, which
        uses undo_move, but useful for testing/extension).
        """
        new_board = Board()
        new_board.cells = self.cells.copy()
        return new_board

    def __str__(self) -> str:
        """
        Renders the board as a clean ASCII grid for the terminal,
        showing 1-indexed position numbers in empty cells as a
        helpful hint for human players.
        """
        rows = []
        for r in range(3):
            row_cells = []
            for c in range(3):
                idx = r * 3 + c
                value = self.cells[idx] if self.cells[idx] != EMPTY else str(idx + 1)
                row_cells.append(f" {value} ")
            rows.append("|".join(row_cells))
        return "\n-----------\n".join(rows)
