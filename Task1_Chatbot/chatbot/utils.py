"""
utils.py
--------
Generic helper functions used across the chatbot project.
Separating utilities keeps engine.py focused only on response logic.
"""

import re
import string
from datetime import datetime


def normalize_text(text: str) -> str:
    """
    Cleans and standardizes user input for reliable keyword matching.

    Steps:
        1. Convert to lowercase
        2. Strip leading/trailing whitespace
        3. Remove punctuation (keeps apostrophes for words like "what's")
        4. Collapse multiple spaces into one

    Example:
        "  What IS   AI??  " -> "what is ai"
    """
    text = text.lower().strip()
    punctuation_to_remove = string.punctuation.replace("'", "")
    text = text.translate(str.maketrans("", "", punctuation_to_remove))
    text = re.sub(r"\s+", " ", text)
    return text


def contains_keyword(user_text: str, keywords: tuple) -> bool:
    """
    Checks whether ANY keyword/phrase from `keywords` appears inside
    the normalized user_text as a WHOLE WORD/PHRASE match (not a raw
    substring). This avoids false positives like "internship" matching
    the greeting keyword "hi" (since "hi" is literally inside "ship").

    Implementation: wraps each keyword with \\b (word boundary) regex
    anchors, so multi-word phrases like "what is ai" still match
    correctly across word boundaries on both ends.
    """
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, user_text):
            return True
    return False


def get_current_datetime_string() -> str:
    """Returns a human-friendly current date & time string."""
    now = datetime.now()
    return now.strftime("%A, %d %B %Y | %I:%M %p")


def get_time_based_greeting() -> str:
    """Returns 'Good Morning/Afternoon/Evening' based on current hour."""
    hour = datetime.now().hour
    if hour < 12:
        return "Good Morning"
    elif hour < 17:
        return "Good Afternoon"
    else:
        return "Good Evening"


def print_divider(char: str = "-", length: int = 60) -> None:
    """Prints a clean divider line for terminal UI formatting."""
    print(char * length)


def print_banner(text: str) -> None:
    """Prints a boxed banner -- used for headers in the terminal UI."""
    border = "=" * (len(text) + 4)
    print(border)
    print(f"| {text} |")
    print(border)
