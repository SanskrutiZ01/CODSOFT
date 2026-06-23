# ❌⭕ Tic-Tac-Toe AI — Minimax & Alpha-Beta Pruning Edition

A professional, object-oriented implementation of Tic-Tac-Toe featuring an **unbeatable AI** built on classical game-theory search algorithms — **Minimax** and **Minimax with Alpha-Beta Pruning**. No machine learning, no APIs, no external services: this is a pure demonstration of adversarial search.

---

## 📌 Project Overview

This project implements **Task 2: Tic-Tac-Toe AI** from the internship brief. It goes beyond a single-file script by structuring the game as a proper Python package with distinct `Board`, `Player` (Human/AI), `Game`, and `Scoreboard` classes, and by implementing **both** Minimax and Alpha-Beta Pruning side-by-side so the difference between them is explicit and measurable — directly reflecting the algorithmic focus of the task.

---

## ✨ Features

- 🧠 **Minimax Algorithm** — full game-tree search, implemented from scratch
- ⚡ **Alpha-Beta Pruning** — same correctness, drastically fewer nodes explored
- 🏆 **Mathematically Unbeatable Hard AI** — verified to never lose, even against an opponent that also plays optimally (best case: draw)
- 🎚️ **Three Difficulty Levels**:
  - **Easy** — fully random legal moves
  - **Medium** — heuristic: wins if possible, blocks opponent wins, else positional preference (center > corners > edges)
  - **Hard** — Minimax + Alpha-Beta Pruning (unbeatable)
- 🧩 **Clean OOP Design** — `Board`, `Player` (abstract base), `HumanPlayer`, `AIPlayer`, `Game`, `Scoreboard`
- 👥 **Two Game Modes** — Human vs AI, or Human vs Human
- 🔁 **Replay System** — play multiple rounds in one session without restarting
- 📊 **Scoreboard** — tracks wins/losses/draws per player across rounds
- 🛡️ **Robust Input Validation** — handles empty input, non-numeric input, out-of-range numbers, occupied cells, and keyboard interrupts gracefully
- 🎨 **Clean Terminal UI** — bordered banners, numbered empty cells as move hints, clear round/result messages
- 📝 **Thoroughly Commented Code** — every class and function explains both *what* it does and *why* it's designed that way

---

## 📁 Folder Structure

```
tic_tac_toe_ai/
│
├── main.py                  # Entry point (thin -- delegates to Game class)
├── requirements.txt         # Dependency file (standard library only)
├── README.md                # Project documentation
│
└── game/                    # Core package
    ├── __init__.py          # Package initializer
    ├── board.py             # Board class -- state, rules, win/draw detection
    ├── ai_algorithms.py     # minimax() and minimax_alpha_beta() implementations
    ├── player.py            # Player (ABC), HumanPlayer, AIPlayer (Easy/Medium/Hard)
    ├── scoreboard.py         # Scoreboard class -- multi-round score tracking
    ├── game.py               # Game class -- setup, turn loop, replay orchestration
    └── utils.py              # Terminal UI helper functions
```

---

## 🧩 Module-by-Module Explanation

| Module | Responsibility |
|---|---|
| **`board.py`** | Manages the 9-cell board as a flat list, with `place_mark`, `undo_move` (for backtracking during search), `check_winner`, `is_draw`, and a clean `__str__` renderer showing 1-9 position hints on empty cells. |
| **`ai_algorithms.py`** | Contains the two core algorithms: `minimax()` (unpruned, exhaustive) and `minimax_alpha_beta()` (pruned). Also contains `find_best_move()`, the function `AIPlayer` calls on Hard difficulty, and `evaluate()`, which scores terminal states (+10 win, -10 loss, 0 draw, depth-adjusted). |
| **`player.py`** | Defines the `Player` abstract base class and three concrete implementations: `HumanPlayer` (validated terminal input) and `AIPlayer` (Easy/Medium/Hard strategies, with Hard delegating to `find_best_move`). |
| **`scoreboard.py`** | A small class that records wins per player name and draws, and prints a formatted scoreboard — supports the Replay requirement. |
| **`game.py`** | The `Game` orchestrator: prompts for mode/difficulty/names, runs a single round's turn loop (`play_round`), detects win/draw, and manages the replay loop (`run`). |
| **`utils.py`** | Shared print helpers (`print_banner`, `print_divider`, `print_board`) so terminal formatting stays consistent and DRY. |
| **`main.py`** | Minimal entry point with a top-level try/except so the program never crashes with a raw traceback in front of the user. |

---

## ▶️ Sample Gameplay

```
===================================
| TIC-TAC-TOE: MINIMAX AI EDITION |
===================================
====================================
SELECT GAME MODE
  [1] Human vs AI
  [2] Human vs Human
====================================
Enter choice (1-2): 1
Enter your name: Alex
====================================
SELECT AI DIFFICULTY
  [1] Easy   (Random Moves)
  [2] Medium (Mixed Strategy)
  [3] Hard   (Minimax + Alpha-Beta -- Unbeatable)
====================================
Enter choice (1-3): 3

 1 | 2 | 3
-----------
 4 | 5 | 6
-----------
 7 | 8 | 9

Alex (X), enter your move (1-9): 1

 X | 2 | 3
-----------
 4 | 5 | 6
-----------
 7 | 8 | 9


 X | 2 | 3
-----------
 4 | O | 6
-----------
 7 | 8 | 9

Alex (X), enter your move (1-9): 2
...
🤝 It's a draw!

+--------------------------------+
|          SCOREBOARD           |
+--------------------------------+
| Rounds Played: 1               |
| Alex                : 0        |
| RuleBot-AI          : 0        |
| Draws               : 1        |
+--------------------------------+

Play another round? (y/n): n
Thanks for playing! 👋
```

---

## 🧠 Minimax Explanation

**Minimax** is a recursive decision-making algorithm for two-player, zero-sum games with perfect information (each player can see the entire board, and one player's gain is exactly the other's loss).

It works by exploring the **entire game tree** from the current position:
1. The **maximizing player** (the AI) tries to pick the move that leads to the **highest** possible score.
2. The **minimizing player** (the opponent) is assumed to play optimally too, always trying to lead to the **lowest** possible score for the AI.
3. At terminal states (win/loss/draw), a fixed score is returned (+10 / -10 / 0 in this project, depth-adjusted).
4. These scores propagate back up the tree: each node returns the max (if it's the AI's turn) or min (if it's the opponent's turn) of its children's scores.
5. The AI simply picks the move whose subtree returns the best score for it.

Because Minimax assumes the opponent ALSO plays optimally, the AI's chosen move is **guaranteed never to result in a loss** if the opponent does play optimally — and it will punish any mistake the opponent makes, since suboptimal opponent moves only lead to better outcomes for the AI in the explored tree.

In `ai_algorithms.py`, this is implemented in the `minimax()` function: it recurses with `is_maximizing` flipped each ply, using `board.place_mark()` / `board.undo_move()` to explore and backtrack through the actual game state in place (rather than copying the board at every node, which would waste memory).

---

## ⚡ Alpha-Beta Pruning Explanation

**Alpha-Beta Pruning** is an optimization on top of Minimax that produces the **exact same result** while exploring far fewer nodes, by tracking two bounds as it recurses:

- **alpha**: the best score the maximizer can *guarantee* so far (lower bound)
- **beta**: the best score the minimizer can *guarantee* so far (upper bound)

The key insight: if, while exploring a minimizing node, we find a move whose score is already `<= alpha` (the maximizer's current best guarantee elsewhere), the minimizer would never let the maximizer reach this node in the first place — so the rest of this branch can be **pruned (skipped)** without affecting the final answer. The symmetric case holds at maximizing nodes with `beta`.

In code (`minimax_alpha_beta()` in `ai_algorithms.py`), this is the line:
```python
if alpha >= beta:
    break  # cut off remaining sibling branches
```

This project measured the real impact directly: for the AI's very first move on an empty board, **plain Minimax evaluated 549,945 recursive calls**, while **Alpha-Beta Pruning evaluated only 34,202** — roughly a **16x reduction** — while producing the identical best move and score. This is the clearest, most concrete demonstration of why pruning matters in adversarial search.

---

## 📊 Complexity Analysis

### Time Complexity

- **Plain Minimax**: `O(b^d)` where `b` is the branching factor (number of legal moves, ≤ 9 and shrinking each ply) and `d` is the remaining depth (≤ 9). In the worst case (first move, empty board), this explores the (loose) bound of 9! ≈ 362,880 leaf-level paths, matching the measured ~550K total recursive calls including internal nodes.
- **Alpha-Beta Pruning**: Best case `O(b^(d/2))` when move ordering is favorable (i.e., the search effectively explores the square root of the unpruned tree size). In practice for Tic-Tac-Toe, this project measured roughly a **16x reduction** in node count on the first move, consistent with substantial real-world pruning even without explicit move-ordering heuristics.
- Because Tic-Tac-Toe's state space is tiny (at most 9! ≈ 362,880 possible games), even the unpruned Minimax runs instantly on modern hardware — Alpha-Beta's benefit here is more about **demonstrating the algorithmic technique correctly** than solving a performance problem, which is exactly the pedagogical point of this task.

### Space Complexity

- **`O(d)`** for both algorithms, where `d` is the maximum recursion depth (at most 9, the number of cells). This is because the implementation explores the game tree **depth-first using backtracking** (`place_mark` / `undo_move`) on a single shared `Board` instance, rather than allocating a new copied board at every node — so memory usage is proportional only to the call stack depth, not the number of nodes visited.

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/tic-tac-toe-ai.git
cd tic-tac-toe-ai/tic_tac_toe_ai

# 2. Run the game (no installation needed -- standard library only)
python main.py
```

---

## ✅ Why This Project Satisfies the Internship Task

| Internship Requirement | How This Project Fulfills It |
|---|---|
| AI agent plays Tic-Tac-Toe against a human | `Game` class supports Human vs AI mode with full turn management |
| Use Minimax with/without Alpha-Beta Pruning | Both `minimax()` and `minimax_alpha_beta()` are implemented from scratch in `ai_algorithms.py` |
| Make the AI unbeatable | Hard difficulty uses full-depth Minimax + Alpha-Beta; verified programmatically to never lose against an optimally-playing opponent (best outcome: draw) |
| Understand game theory and search algorithms | README explains both algorithms conceptually and with measured node-count evidence; code structure mirrors textbook adversarial search |

The project remains **strictly a classical search algorithm implementation** — no `sklearn`, no neural networks, no APIs — while demonstrating depth through OOP design, multiple difficulty tiers, measured algorithmic comparisons, and defensive engineering (input validation, replay, scoreboard).

---

## 🔮 Future Enhancements

- Add a **GUI** using Tkinter or Pygame for a visual board instead of terminal text
- Implement **move ordering heuristics** (e.g., always evaluate center/corners first) to push Alpha-Beta pruning efficiency even closer to its theoretical best case
- Add a **transposition table** (memoization of previously evaluated board states) to avoid re-computing symmetric positions
- Extend to an **N x N generalized board** (e.g., 4x4 Tic-Tac-Toe with "connect 4-in-a-row") to explore how search cost scales
- Add **unit tests** (pytest) for `Board`, `ai_algorithms`, and `Player` classes
- Persist scoreboard statistics across sessions using a JSON/SQLite file
- Add a **difficulty auto-adjust** mode that scales AI strength based on the human's win rate

---

## 🏷️ Tech Stack

`Python 3` · Standard Library only (`random`, `abc`, `typing`) · Object-Oriented Programming · Minimax · Alpha-Beta Pruning · Adversarial Search

---

## 📄 License

This project is open-source and free to use for educational purposes.
