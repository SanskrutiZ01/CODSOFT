"""
engine.py
---------
The core Rule-Based Response Engine.

This module is responsible for ALL decision-making logic of the chatbot:
    1. Built-in command detection (help, exit, history, time, etc.)
    2. Small-talk pattern matching (greetings, thanks, farewells)
    3. Keyword-based knowledge domain matching
    4. Fallback handling when no rule matches

Design Pattern Used: This follows a simplified "Chain of Responsibility"
pattern -- the user input passes through a sequence of rule-checkers,
and the first one that matches handles the response.
"""

from chatbot.knowledge_base import (
    KNOWLEDGE_DOMAINS,
    GREETING_PATTERNS,
    THANKS_PATTERNS,
    FAREWELL_PATTERNS,
    NAME_QUERY_PATTERNS,
    MOOD_QUERY_PATTERNS,
)
from chatbot.utils import (
    normalize_text,
    contains_keyword,
    get_current_datetime_string,
    get_time_based_greeting,
)


class ChatbotEngine:
    """
    Encapsulates the matching logic and keeps track of simple session
    state (e.g., user's name) needed for personalization.
    """

    def __init__(self, user_name: str = "Friend"):
        self.user_name = user_name
        self.last_matched_domain = None  # used for analytics / smarter fallback

    # ------------------------------------------------------------------
    # PUBLIC ENTRY POINT
    # ------------------------------------------------------------------
    def generate_response(self, raw_user_input: str) -> str:
        """
        Main pipeline: takes raw user text, normalizes it, then tries
        each rule category in priority order until one matches.
        """
        if not raw_user_input or not raw_user_input.strip():
            return "I didn't receive any input. Could you please type something?"

        text = normalize_text(raw_user_input)

        # Priority 1: Small talk (greetings, thanks, farewells, identity)
        small_talk_response = self._check_small_talk(text)
        if small_talk_response:
            return small_talk_response

        # Priority 2: Date / Time queries
        if contains_keyword(text, ("time", "date", "today", "what day is it")):
            return self._handle_datetime_query(text)

        # Priority 3: Knowledge domain matching (AI, ML, DL, DS, Python, etc.)
        domain_response = self._check_knowledge_domains(text)
        if domain_response:
            return domain_response

        # Priority 4: Fallback -- no rule matched
        return self._fallback_response()

    # ------------------------------------------------------------------
    # RULE CATEGORY 1: SMALL TALK
    # ------------------------------------------------------------------
    def _check_small_talk(self, text: str) -> str:
        if contains_keyword(text, GREETING_PATTERNS):
            greeting = get_time_based_greeting()
            return f"{greeting}, {self.user_name}! How can I help you today?"

        if contains_keyword(text, THANKS_PATTERNS):
            return "You're welcome! Happy to help."

        if contains_keyword(text, NAME_QUERY_PATTERNS):
            return "I'm RuleBot -- your rule-based AI/Data Science learning assistant."

        if contains_keyword(text, MOOD_QUERY_PATTERNS):
            return "I'm just a program, but I'm running perfectly! Ready to help you learn."

        if contains_keyword(text, FAREWELL_PATTERNS):
            return f"__EXIT__"  # special signal handled by main.py

        return None

    # ------------------------------------------------------------------
    # RULE CATEGORY 2: DATE / TIME
    # ------------------------------------------------------------------
    def _handle_datetime_query(self, text: str) -> str:
        return f"Current date & time: {get_current_datetime_string()}"

    # ------------------------------------------------------------------
    # RULE CATEGORY 3: KNOWLEDGE DOMAINS
    # ------------------------------------------------------------------
    def _check_knowledge_domains(self, text: str) -> str:
        """
        Loops through every domain dictionary and checks if the user's
        text matches any keyword tuple. Keywords are sorted by length
        (longest first) so more specific phrases are checked before
        shorter, more generic ones -- avoiding incorrect early matches.
        """
        for domain_name, domain_dict in KNOWLEDGE_DOMAINS.items():
            sorted_keys = sorted(domain_dict.keys(), key=lambda k: -len(k[0]))
            for keyword_tuple in sorted_keys:
                if contains_keyword(text, keyword_tuple):
                    self.last_matched_domain = domain_name
                    return domain_dict[keyword_tuple]
        return None

    # ------------------------------------------------------------------
    # RULE CATEGORY 4: FALLBACK
    # ------------------------------------------------------------------
    def _fallback_response(self) -> str:
        suggestion = ""
        if self.last_matched_domain:
            suggestion = f" Perhaps ask another question about {self.last_matched_domain}?"

        return (
            "I'm not sure I understood that. I can answer questions about AI, "
            "Machine Learning, Deep Learning, Data Science, Python, Career "
            f"Guidance, or Study Tips. Type 'help' to see what I can do.{suggestion}"
        )
