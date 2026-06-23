"""
game.py
-------
Defines the Game class: the orchestrator that ties Board, Player
(Human/AI), and Scoreboard together into a complete playable session.

Responsibilities:
    - Initial setup (mode selection, difficulty selection, marks)
    - Running a single round to completion
    - Alternating turns between the two players
    - Detecting and announcing win/draw outcomes
    - Handling the replay loop and scoreboard updates
"""

from game.board import Board
from game.player import HumanPlayer, AIPlayer, EASY, MEDIUM, HARD
from game.scoreboard import Scoreboard
from game.utils import print_banner, print_divider, print_board


class Game:
    """Top-level controller for a full Tic-Tac-Toe session (with replay)."""

    def __init__(self):
        self.board = Board()
        self.player_x = None
        self.player_o = None
        self.scoreboard = None

    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------
    def _choose_game_mode(self) -> str:
        print_divider("=")
        print("SELECT GAME MODE")
        print("  [1] Human vs AI")
        print("  [2] Human vs Human")
        print_divider("=")
        while True:
            choice = input("Enter choice (1-2): ").strip()
            if choice in ("1", "2"):
                return choice
            print("⚠  Invalid choice. Please enter 1 or 2.")

    def _choose_difficulty(self) -> str:
        print_divider("=")
        print("SELECT AI DIFFICULTY")
        print("  [1] Easy   (Random Moves)")
        print("  [2] Medium (Mixed Strategy)")
        print("  [3] Hard   (Minimax + Alpha-Beta -- Unbeatable)")
        print_divider("=")
        mapping = {"1": EASY, "2": MEDIUM, "3": HARD}
        while True:
            choice = input("Enter choice (1-3): ").strip()
            if choice in mapping:
                return mapping[choice]
            print("⚠  Invalid choice. Please enter 1, 2, or 3.")

    def _get_player_name(self, prompt: str) -> str:
        name = input(prompt).strip()
        return name if name else "Player"

    def setup_players(self) -> None:
        """Runs all pre-game prompts to configure players for the session."""
        print_banner("TIC-TAC-TOE: MINIMAX AI EDITION")
        mode = self._choose_game_mode()

        if mode == "1":  # Human vs AI
            human_name = self._get_player_name("Enter your name: ")
            difficulty = self._choose_difficulty()

            human_mark, ai_mark = "X", "O"
            self.player_x = HumanPlayer(human_name, human_mark)
            self.player_o = AIPlayer("RuleBot-AI", ai_mark, opponent_mark=human_mark,
                                      difficulty=difficulty)
        else:  # Human vs Human
            name1 = self._get_player_name("Enter Player 1 name (X): ")
            name2 = self._get_player_name("Enter Player 2 name (O): ")
            self.player_x = HumanPlayer(name1, "X")
            self.player_o = HumanPlayer(name2, "O")

        self.scoreboard = Scoreboard(self.player_x.name, self.player_o.name)

    # ------------------------------------------------------------------
    # SINGLE ROUND
    # ------------------------------------------------------------------
    def play_round(self) -> None:
        """Plays exactly one full round of Tic-Tac-Toe to completion."""
        self.board.reset()
        current_player = self.player_x  # X always starts, by convention

        print_board(self.board)

        while True:
            try:
                move = current_player.get_move(self.board)
            except (EOFError, KeyboardInterrupt):
                print("\n⚠  Game interrupted. Returning to menu.")
                return

            placed = self.board.place_mark(move, current_player.mark)
            if not placed:
                # Defensive check -- HumanPlayer already validates, but
                # AIPlayer logic bugs (if any) should never crash the game.
                print("⚠  Internal error: invalid move attempted. Retrying turn.")
                continue

            print_board(self.board)

            winner_mark = self.board.check_winner()
            if winner_mark is not None:
                winner = current_player
                print(f"🎉 {winner.name} ({winner.mark}) wins this round!")
                self.scoreboard.record_win(winner.name)
                return

            if self.board.is_draw():
                print("🤝 It's a draw!")
                self.scoreboard.record_draw()
                return

            # Switch turns
            current_player = self.player_o if current_player is self.player_x else self.player_x

    # ------------------------------------------------------------------
    # REPLAY LOOP
    # ------------------------------------------------------------------
    def _ask_play_again(self) -> bool:
        while True:
            choice = input("Play another round? (y/n): ").strip().lower()
            if choice in ("y", "yes"):
                return True
            if choice in ("n", "no"):
                return False
            print("⚠  Please answer 'y' or 'n'.")

    def run(self) -> None:
        """Main public method: sets up players, then loops rounds with replay."""
        self.setup_players()

        while True:
            self.play_round()
            self.scoreboard.display()

            if not self._ask_play_again():
                break

        print_banner("FINAL RESULTS")
        self.scoreboard.display()
        print("Thanks for playing! 👋")
