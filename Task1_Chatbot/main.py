"""
main.py
-------
Entry point for the Rule-Based Chatbot application.

Provides:
    - Personalized greeting (asks for user's name)
    - A clean, menu-driven terminal UI
    - Built-in commands: help, history, save, clear, exit
    - Robust error handling so the program never crashes on bad input
"""

from chatbot.engine import ChatbotEngine
from chatbot.history import ConversationHistory
from chatbot.utils import print_banner, print_divider, get_current_datetime_string


HELP_TEXT = """
Available Commands:
  help      - Show this help menu
  history   - View the full conversation history of this session
  save      - Save the conversation history to a JSON log file
  clear     - Clear the current conversation history
  time      - Show current date and time
  exit/quit - End the conversation and close the chatbot

Knowledge Domains You Can Ask About:
  1. Artificial Intelligence   (e.g., "What is AI?")
  2. Machine Learning          (e.g., "What is overfitting?")
  3. Deep Learning             (e.g., "What is a CNN?")
  4. Data Science              (e.g., "What is EDA?")
  5. Python Programming        (e.g., "What is a dictionary?")
  6. Career Guidance           (e.g., "How to get an internship?")
  7. Study Tips                (e.g., "Best way to learn ML?")
"""


def display_main_menu() -> None:
    """Displays the initial menu options to the user."""
    print_divider("=")
    print("MAIN MENU")
    print("  [1] Start Chatting")
    print("  [2] View Help / Command List")
    print("  [3] Exit")
    print_divider("=")


def get_user_name() -> str:
    """Safely prompts for and returns the user's name (with error handling)."""
    try:
        name = input("Before we begin, what's your name? ").strip()
        return name if name else "Friend"
    except (EOFError, KeyboardInterrupt):
        return "Friend"


def run_chat_loop(engine: ChatbotEngine, history: ConversationHistory) -> None:
    """
    Runs the main interactive chat loop. Handles all in-chat commands
    and routes regular text to the ChatbotEngine for a rule-based response.
    """
    print("\nType 'help' anytime to see commands, or 'exit' to leave the chat.\n")

    while True:
        try:
            user_input = input(f"{engine.user_name}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Session interrupted. Goodbye!")
            break

        if not user_input:
            print("Bot: Please type something so I can help you.")
            continue

        lowered = user_input.lower()

        # ---- Built-in command handling (checked before the engine) ----
        if lowered in ("help", "menu"):
            print(HELP_TEXT)
            continue

        if lowered == "history":
            history.show_history()
            continue

        if lowered == "save":
            try:
                path = history.save_to_file()
                print(f"Bot: Conversation saved to '{path}'.")
            except OSError as e:
                print(f"Bot: Could not save history due to an error: {e}")
            continue

        if lowered == "clear":
            history.clear()
            print("Bot: Conversation history cleared.")
            continue

        if lowered in ("exit", "quit", "bye", "goodbye"):
            print(f"Bot: Goodbye, {engine.user_name}! Keep learning and stay curious. 👋")
            break

        # ---- Route everything else through the rule-based engine ----
        try:
            response = engine.generate_response(user_input)
        except Exception as e:
            # Defensive error handling: the chatbot should never crash
            response = f"Oops, something went wrong while processing that: {e}"

        if response == "__EXIT__":
            print(f"Bot: Goodbye, {engine.user_name}! Keep learning and stay curious. 👋")
            history.add_entry(user_input, "Goodbye message sent.")
            break

        print(f"Bot: {response}")
        history.add_entry(user_input, response)


def main() -> None:
    """Application entry point."""
    print_banner("RULEBOT - AI/DS RULE-BASED CHATBOT")
    print(f"Session started: {get_current_datetime_string()}\n")

    user_name = get_user_name()
    engine = ChatbotEngine(user_name=user_name)
    history = ConversationHistory()

    print(f"\nNice to meet you, {user_name}! 🤖")

    while True:
        display_main_menu()
        try:
            choice = input("Enter your choice (1-3): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Goodbye!")
            break

        if choice == "1":
            run_chat_loop(engine, history)
            break  # chat loop itself handles exit; end program after it returns
        elif choice == "2":
            print(HELP_TEXT)
        elif choice == "3":
            print(f"Goodbye, {user_name}! 👋")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.\n")


if __name__ == "__main__":
    main()
