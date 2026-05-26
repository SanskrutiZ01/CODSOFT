# ==========================================
# CodSoft AI Internship - Task 1
# Chatbot with Rule-Based Responses
# Author: Sanskruti Zode
# ==========================================

print("Hello! I am StudyBot 🤖")
print("I can answer basic questions about AI, Python and study tips.")
print("Type 'bye' anytime to exit.\n")

while True:
    user = input("You: ").lower().strip()

    # Greetings
    if "hello" in user or "hi" in user or "hey" in user:
        print("Bot: Hello! How can I help you?")

    # AI questions
    elif "ai" in user or "artificial intelligence" in user:
        print("Bot: AI stands for Artificial Intelligence. It helps machines solve problems and make decisions.")

    # Machine Learning
    elif "machine learning" in user:
        print("Bot: Machine Learning is a part of AI where computers learn from data.")

    # Python / Coding
    elif "python" in user or "coding" in user or "programming" in user:
        print("Bot: Python is a beginner-friendly language widely used in AI and Data Science.")

    # Study help
    elif "study" in user or "exam" in user or "tips" in user:
        print("Bot: Study regularly, revise important concepts and practice coding every day.")

    # Help
    elif "help" in user:
        print("Bot: You can ask me about AI, machine learning, Python or study tips.")

    # Exit
    elif "bye" in user:
        print("Bot: Goodbye! Keep learning and all the best.")
        break

    # Default reply
    else:
        print("Bot: Sorry, I didn't understand. Please ask about AI, Python or study tips.")