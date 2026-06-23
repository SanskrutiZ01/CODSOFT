"""
scoreboard.py
-------------
Tracks and displays match statistics (wins/losses/draws) across multiple
rounds within a single session, supporting the "Replay" feature.
"""


class Scoreboard:
    """Encapsulates score-tracking state for a multi-round session."""

    def __init__(self, player_one_name: str, player_two_name: str):
        self.player_one_name = player_one_name
        self.player_two_name = player_two_name
        self.scores = {
            player_one_name: 0,
            player_two_name: 0,
            "Draws": 0,
        }
        self.rounds_played = 0

    def record_win(self, winner_name: str) -> None:
        """Increments the win count for the given player and the round counter."""
        if winner_name in self.scores:
            self.scores[winner_name] += 1
        self.rounds_played += 1

    def record_draw(self) -> None:
        """Increments the draw count and the round counter."""
        self.scores["Draws"] += 1
        self.rounds_played += 1

    def display(self) -> None:
        """Prints a formatted scoreboard summary."""
        print("\n+" + "-" * 32 + "+")
        print("|          SCOREBOARD           |")
        print("+" + "-" * 32 + "+")
        print(f"| Rounds Played: {self.rounds_played:<15}|")
        for name, score in self.scores.items():
            print(f"| {name:<20}: {score:<8}|")
        print("+" + "-" * 32 + "+\n")
