"""
utils.py
--------
Small terminal-UI helper functions shared across the project, kept here
so display formatting logic doesn't clutter the Game/Player classes.
"""


def print_banner(text: str) -> None:
    """Prints a bordered banner heading."""
    border = "=" * (len(text) + 4)
    print(border)
    print(f"| {text} |")
    print(border)


def print_divider(char: str = "-", length: int = 36) -> None:
    print(char * length)


def print_board(board) -> None:
    """Prints the board with a small top/bottom margin for readability."""
    print()
    print(board)
    print()
