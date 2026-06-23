"""
knowledge_base.py
------------------
Stores all domain knowledge used by the rule-based response engine.

Each domain is represented as a dictionary where:
    key   -> a tuple of keywords/patterns that trigger the rule
    value -> the response string returned when the keyword is matched

Keeping the knowledge separate from the matching logic (engine.py) follows
the Single Responsibility Principle and makes the chatbot easy to extend:
adding a new fact only requires adding a new entry here, no engine changes.
"""

# ---------------------------------------------------------------------------
# 1. ARTIFICIAL INTELLIGENCE DOMAIN
# ---------------------------------------------------------------------------
AI_DOMAIN = {
    ("what is ai", "what is artificial intelligence", "define ai"):
        "Artificial Intelligence (AI) is the simulation of human intelligence "
        "in machines that are programmed to think, learn, and solve problems.",

    ("types of ai", "kinds of ai", "ai categories"):
        "AI is broadly categorized into: 1) Narrow AI (task-specific), "
        "2) General AI (human-level reasoning), and 3) Super AI (theoretical, "
        "beyond human intelligence).",

    ("applications of ai", "uses of ai", "ai applications"):
        "AI is used in chatbots, recommendation systems, self-driving cars, "
        "fraud detection, healthcare diagnostics, robotics, and virtual assistants.",

    ("turing test",):
        "The Turing Test, proposed by Alan Turing, checks whether a machine's "
        "responses are indistinguishable from a human's during a conversation.",
}

# ---------------------------------------------------------------------------
# 2. MACHINE LEARNING DOMAIN
# ---------------------------------------------------------------------------
ML_DOMAIN = {
    ("what is machine learning", "what is ml", "define ml"):
        "Machine Learning (ML) is a subset of AI where systems learn patterns "
        "from data and improve their performance without being explicitly programmed.",

    ("types of machine learning", "ml types", "categories of ml"):
        "Machine Learning has three main types: Supervised Learning, "
        "Unsupervised Learning, and Reinforcement Learning.",

    ("supervised learning",):
        "Supervised Learning trains a model on labeled data, where the correct "
        "output is already known (e.g., predicting house prices from features).",

    ("unsupervised learning",):
        "Unsupervised Learning finds hidden patterns or groupings in unlabeled "
        "data, such as customer segmentation using clustering.",

    ("reinforcement learning",):
        "Reinforcement Learning trains an agent to make decisions by rewarding "
        "good actions and penalizing bad ones, commonly used in robotics and game AI.",

    ("overfitting",):
        "Overfitting occurs when a model learns the training data too well, "
        "including noise, and performs poorly on new, unseen data.",

    ("underfitting",):
        "Underfitting happens when a model is too simple to capture the "
        "underlying pattern in the data, leading to poor performance overall.",
}

# ---------------------------------------------------------------------------
# 3. DEEP LEARNING DOMAIN
# ---------------------------------------------------------------------------
DL_DOMAIN = {
    ("what is deep learning", "define deep learning"):
        "Deep Learning is a subset of Machine Learning that uses multi-layered "
        "neural networks to automatically learn complex patterns from large datasets.",

    ("neural network", "what is a neural network"):
        "A Neural Network is a computational model inspired by the human brain, "
        "made up of interconnected layers of nodes (neurons) that process data.",

    ("cnn", "convolutional neural network"):
        "A Convolutional Neural Network (CNN) is mainly used for image data; "
        "it uses convolutional layers to automatically detect spatial features.",

    ("rnn", "recurrent neural network"):
        "A Recurrent Neural Network (RNN) is designed for sequential data like "
        "text or time series, as it retains memory of previous inputs.",

    ("activation function",):
        "An activation function (like ReLU, Sigmoid, or Softmax) introduces "
        "non-linearity into a neural network, enabling it to learn complex patterns.",
}

# ---------------------------------------------------------------------------
# 4. DATA SCIENCE DOMAIN
# ---------------------------------------------------------------------------
DS_DOMAIN = {
    ("what is data science", "define data science"):
        "Data Science is an interdisciplinary field that uses statistics, "
        "programming, and domain knowledge to extract insights from data.",

    ("data science lifecycle", "data science process"):
        "The typical Data Science lifecycle includes: Problem Definition, "
        "Data Collection, Data Cleaning, Exploratory Data Analysis, Modeling, "
        "Evaluation, and Deployment.",

    ("eda", "exploratory data analysis"):
        "Exploratory Data Analysis (EDA) is the process of summarizing and "
        "visualizing data to uncover patterns, trends, and anomalies before modeling.",

    ("data cleaning", "data preprocessing"):
        "Data Cleaning involves handling missing values, removing duplicates, "
        "fixing inconsistent formats, and treating outliers to prepare quality data.",

    ("pandas",):
        "Pandas is a Python library used for data manipulation and analysis, "
        "providing powerful structures like DataFrames and Series.",

    ("numpy",):
        "NumPy is a Python library for numerical computing, offering support "
        "for multi-dimensional arrays and a wide range of mathematical functions.",
}

# ---------------------------------------------------------------------------
# 5. PYTHON PROGRAMMING DOMAIN
# ---------------------------------------------------------------------------
PYTHON_DOMAIN = {
    ("what is python", "define python"):
        "Python is a high-level, interpreted programming language known for "
        "its simplicity, readability, and extensive libraries for AI/ML and Data Science.",

    ("list vs tuple", "difference between list and tuple"):
        "A list is mutable (can be changed after creation) while a tuple is "
        "immutable (cannot be changed). Lists use [], tuples use ().",

    ("what is a function", "define function in python"):
        "A function is a reusable block of code defined using the 'def' keyword "
        "that performs a specific task and can accept inputs (parameters) and return outputs.",

    ("what is oop", "object oriented programming"):
        "Object-Oriented Programming (OOP) is a paradigm based on objects and "
        "classes, built on four pillars: Encapsulation, Abstraction, Inheritance, and Polymorphism.",

    ("what is a dictionary", "python dictionary"):
        "A dictionary in Python is an unordered, mutable collection of key-value "
        "pairs, defined using curly braces {} and accessed using keys.",

    ("exception handling", "try except"):
        "Exception handling in Python uses try-except blocks to catch and "
        "manage runtime errors gracefully, preventing the program from crashing.",
}

# ---------------------------------------------------------------------------
# 6. CAREER GUIDANCE DOMAIN
# ---------------------------------------------------------------------------
CAREER_DOMAIN = {
    ("career in ai", "ai career path", "how to become ai engineer"):
        "To build a career in AI: master Python, statistics, and ML/DL "
        "fundamentals, build projects, contribute to open-source, and apply for "
        "internships to gain real-world experience.",

    ("skills required for data science", "data science skills"):
        "Key Data Science skills include Python/R, SQL, statistics, data "
        "visualization, machine learning, and strong communication for presenting insights.",

    ("internship tips", "how to get internship"):
        "To land internships: build a strong GitHub portfolio, work on real "
        "projects, sharpen DSA and core CS fundamentals, and tailor your resume "
        "to the role you're targeting.",

    ("resume tips", "how to build resume"):
        "A strong tech resume should highlight projects with measurable impact, "
        "relevant skills, internships, and certifications -- kept concise to one page.",

    ("interview tips", "how to prepare for interview"):
        "Prepare for interviews by practicing DSA problems, revising core "
        "subjects (OS, DBMS, CN), reviewing your own projects deeply, and "
        "practicing clear communication of your thought process.",
}

# ---------------------------------------------------------------------------
# 7. STUDY TIPS DOMAIN
# ---------------------------------------------------------------------------
STUDY_DOMAIN = {
    ("study tips", "how to study effectively"):
        "Effective study tips: use active recall, spaced repetition, the "
        "Pomodoro technique, and teach concepts to others to deepen understanding.",

    ("how to learn machine learning", "best way to learn ml"):
        "To learn ML effectively: build strong math foundations (linear algebra, "
        "statistics), learn Python, take a structured course, then build projects "
        "applying each concept immediately.",

    ("time management",):
        "Good time management involves prioritizing tasks (e.g., using the "
        "Eisenhower Matrix), setting realistic deadlines, and avoiding multitasking.",

    ("how to stay consistent", "consistency tips"):
        "To stay consistent: set small daily goals, track progress visibly, "
        "join a study group or community, and remind yourself of your long-term 'why'.",
}

# ---------------------------------------------------------------------------
# MASTER REGISTRY: maps a domain name to its dictionary
# This allows the engine to loop through domains dynamically instead of
# hardcoding each one, which keeps the engine code domain-agnostic.
# ---------------------------------------------------------------------------
KNOWLEDGE_DOMAINS = {
    "Artificial Intelligence": AI_DOMAIN,
    "Machine Learning": ML_DOMAIN,
    "Deep Learning": DL_DOMAIN,
    "Data Science": DS_DOMAIN,
    "Python Programming": PYTHON_DOMAIN,
    "Career Guidance": CAREER_DOMAIN,
    "Study Tips": STUDY_DOMAIN,
}

# ---------------------------------------------------------------------------
# SMALL-TALK / GREETING RULES (separate from knowledge domains)
# ---------------------------------------------------------------------------
GREETING_PATTERNS = ("hi", "hello", "hey", "good morning", "good evening", "good afternoon")
THANKS_PATTERNS = ("thanks", "thank you", "thx", "appreciate it")
FAREWELL_PATTERNS = ("bye", "goodbye", "see you", "exit", "quit")
NAME_QUERY_PATTERNS = ("what is your name", "who are you")
MOOD_QUERY_PATTERNS = ("how are you", "how are you doing")
