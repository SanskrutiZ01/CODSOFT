# 🤖 RuleBot — A Modular Rule-Based AI/Data Science Chatbot

RuleBot is a **rule-based chatbot** built entirely in Python using dictionaries, keyword detection, and pattern matching — **no machine learning models or external APIs involved**. It is designed to answer questions across seven knowledge domains relevant to AI & Data Science students, while demonstrating clean, modular, production-style Python architecture.

---

## 📌 Project Overview

This chatbot was built to fulfill the internship task: **"Chatbot with Rule-Based Responses."** Rather than a few scattered `if-else` statements, RuleBot implements a structured **rule-based response engine** with prioritized rule categories, a multi-domain knowledge base, conversation history logging, and a menu-driven terminal interface — reflecting how a junior engineer would design a small, maintainable NLP-adjacent system.

---

## ✨ Features

- 🧠 **7 Knowledge Domains**: Artificial Intelligence, Machine Learning, Deep Learning, Data Science, Python Programming, Career Guidance, Study Tips
- 🔍 **Keyword & Pattern Matching Engine** using regex word-boundary matching (avoids false positives like "internship" matching "hi")
- 🗂️ **Modular Architecture** — knowledge base, engine, utilities, and history are cleanly separated into independent modules
- 💬 **Conversation History** — view, clear, or save full session history to a timestamped JSON log file
- 📋 **Menu-Driven Interface** — start chatting, view help, or exit from a clean main menu
- 🆘 **Help Command** — lists all commands and example questions per domain
- 👋 **Personalized Greeting** — asks for the user's name and uses it throughout the session, with time-of-day-aware greetings (Good Morning/Afternoon/Evening)
- 🕒 **Date & Time Support** — responds to "what's the time/date" queries
- 🛡️ **Robust Error Handling** — catches empty input, keyboard interrupts, and unexpected exceptions without crashing
- 🎨 **Clean Terminal UI** — banners, dividers, and consistent formatting
- 📝 **Fully Commented Code** — every function and module is documented with docstrings explaining purpose and design rationale

---

## 📁 Folder Structure

```
rule_based_chatbot/
│
├── main.py                  # Entry point: menu-driven terminal UI
├── requirements.txt         # Dependency file (standard library only)
├── README.md                # Project documentation
│
├── chatbot/                 # Core package
│   ├── __init__.py          # Package initializer
│   ├── knowledge_base.py    # All domain knowledge (dictionaries)
│   ├── utils.py             # Text normalization & helper functions
│   ├── history.py           # Conversation history manager (class-based)
│   └── engine.py            # Rule-based response engine (core logic)
│
└── logs/                    # Auto-created folder for saved chat logs (JSON)
```

---

## 🧩 Module-by-Module Explanation

| Module | Responsibility |
|---|---|
| **`knowledge_base.py`** | Stores all rule data as dictionaries: `{keyword_tuple: response}` for each of the 7 domains, plus small-talk patterns (greetings, thanks, farewells). Pure data — no logic — so new knowledge can be added without touching code elsewhere. |
| **`utils.py`** | Stateless helper functions: `normalize_text()` cleans user input (lowercase, punctuation removal), `contains_keyword()` performs word-boundary regex matching, and datetime/UI helpers support the terminal display. |
| **`history.py`** | A `ConversationHistory` class that encapsulates the session's chat log: add entries, display them, save to a timestamped JSON file, or clear them. Demonstrates OOP and file I/O. |
| **`engine.py`** | The heart of the chatbot — `ChatbotEngine` class. Implements a Chain-of-Responsibility style pipeline: small-talk check → date/time check → knowledge-domain check → fallback. Keeps track of the last matched domain to make fallback responses smarter. |
| **`main.py`** | Orchestrates everything: greets the user, shows the main menu, runs the chat loop, and routes built-in commands (`help`, `history`, `save`, `clear`, `exit`) versus regular questions (sent to the engine). |

---

## ▶️ Sample Execution

```
======================================
| RULEBOT - AI/DS RULE-BASED CHATBOT |
======================================
Session started: Saturday, 20 June 2026 | 07:25 AM

Before we begin, what's your name? Alex

Nice to meet you, Alex! 🤖
============================================================
MAIN MENU
  [1] Start Chatting
  [2] View Help / Command List
  [3] Exit
============================================================
Enter your choice (1-3): 1

Type 'help' anytime to see commands, or 'exit' to leave the chat.

Alex: hi
Bot: Good Morning, Alex! How can I help you today?
Alex: what is overfitting
Bot: Overfitting occurs when a model learns the training data too well, including noise, and performs poorly on new, unseen data.
Alex: how to get internship
Bot: To land internships: build a strong GitHub portfolio, work on real projects, sharpen DSA and core CS fundamentals, and tailor your resume to the role you're targeting.
Alex: time
Bot: Current date & time: Saturday, 20 June 2026 | 07:25 AM
Alex: save
Bot: Conversation saved to 'logs/chat_log_20260620_072500.json'.
Alex: exit
Bot: Goodbye, Alex! Keep learning and stay curious. 👋
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/rule-based-chatbot.git
cd rule-based-chatbot/rule_based_chatbot

# 2. Run the chatbot (no installation needed — standard library only)
python main.py
```

---

## ✅ Why This Project Satisfies the Internship Task

| Internship Requirement | How RuleBot Fulfills It |
|---|---|
| Build a chatbot that responds based on predefined rules | All responses come from static dictionaries in `knowledge_base.py` — zero ML/LLM involved |
| Use if-else / pattern matching to identify queries | `engine.py` uses a prioritized rule pipeline plus regex-based keyword/pattern matching in `utils.py` |
| Demonstrate basic NLP understanding | Implements text normalization (lowercasing, punctuation stripping), tokenization-adjacent keyword detection, and word-boundary pattern matching — core NLP preprocessing concepts |
| Demonstrate conversation flow | Multi-turn conversation loop with context (last matched domain influences fallback), history tracking, and natural exit handling |

The project stays **strictly rule-based** (no `sklearn`, no transformers, no API keys) while going well beyond a basic 10-line if-else script — showing depth through modular design, multiple domains, OOP, file I/O, and defensive error handling.

---

## 🔮 Future Enhancements

- Add a **GUI** using Tkinter or a **web interface** using Flask/Streamlit
- Support **fuzzy matching** (e.g., `difflib`) to handle typos in user queries
- Add a **quiz mode** that tests the user on AI/DS concepts using the same knowledge base
- Allow **multi-language support** for Hindi/regional language queries
- Add a **synonym mapping layer** so more phrasings map to the same rule
- Persist user profiles across sessions (favorite domain, frequently asked topics)
- Add **unit tests** (pytest) for the engine and utils modules to ensure rule-matching reliability

---

## 🏷️ Tech Stack

`Python 3` · Standard Library only (`re`, `string`, `datetime`, `json`, `os`) · OOP · Regex Pattern Matching

---

## 📄 License

This project is open-source and free to use for educational purposes.
