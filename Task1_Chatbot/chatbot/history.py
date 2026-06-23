"""
history.py
----------
Manages conversation history for the chatbot session.

Demonstrates:
    - Object-Oriented Programming (class-based design)
    - File I/O (saving conversation logs)
    - Encapsulation of state (history list is private to the class)
"""

import json
import os
from datetime import datetime


class ConversationHistory:
    """
    Stores each (user_input, bot_response, timestamp) exchange in memory,
    and can persist the full conversation to a JSON log file on disk.
    """

    def __init__(self):
        self._records = []  # list of dicts: {timestamp, user, bot}

    def add_entry(self, user_text: str, bot_response: str) -> None:
        """Adds a single conversation turn to the in-memory history."""
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user_text,
            "bot": bot_response,
        }
        self._records.append(entry)

    def get_history(self) -> list:
        """Returns the full list of conversation records."""
        return self._records

    def show_history(self) -> None:
        """Prints a formatted view of the conversation so far."""
        if not self._records:
            print("No conversation history yet.")
            return

        print("\n----- CONVERSATION HISTORY -----")
        for i, record in enumerate(self._records, start=1):
            print(f"[{i}] ({record['timestamp']})")
            print(f"    You : {record['user']}")
            print(f"    Bot : {record['bot']}")
        print("---------------------------------\n")

    def save_to_file(self, folder: str = "logs") -> str:
        """
        Saves the entire conversation history to a timestamped JSON file
        inside the given folder. Returns the file path created.
        """
        os.makedirs(folder, exist_ok=True)
        filename = f"chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(folder, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._records, f, indent=4)

        return filepath

    def clear(self) -> None:
        """Clears the in-memory conversation history."""
        self._records = []
