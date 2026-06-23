"""
main.py
-------
Entry point for the Tic-Tac-Toe AI project. Keeps the entry script
minimal -- all real logic lives in the `game` package -- which is good
software engineering practice (thin entry point, thick well-tested core).
"""

from game.game import Game


def main() -> None:
    try:
        game = Game()
        game.run()
    except Exception as e:
        # Top-level safety net: the program should never crash with a
        # raw traceback in front of the user.
        print(f"\n❌ An unexpected error occurred: {e}")
        print("The application will now close.")


if __name__ == "__main__":
    main()
