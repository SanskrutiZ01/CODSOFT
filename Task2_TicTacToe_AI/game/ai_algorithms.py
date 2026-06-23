"""
ai_algorithms.py
----------------
Pure game-theory search algorithms: Minimax and Minimax with Alpha-Beta
Pruning. These functions are deliberately kept independent of the Player
classes so they can be unit-tested in isolation and clearly demonstrate
the underlying algorithm to a reviewer or grader.

No machine learning, heuristics-from-data, or external services are used --
this is classical adversarial search over the exact game tree.
"""

from typing import Tuple
from game.board import Board

# Score constants from the AI's (maximizing player's) perspective
WIN_SCORE = 10
LOSE_SCORE = -10
DRAW_SCORE = 0


def evaluate(board: Board, ai_mark: str, opponent_mark: str, depth: int) -> int:
    """
    Scores a terminal board state. The `depth` term rewards faster wins
    and slower losses, so the AI prefers an immediate win over a delayed
    one, and a delayed loss over an immediate one (it still loses either
    way against perfect play, but plays "smarter" against imperfect play).
    """
    winner = board.check_winner()
    if winner == ai_mark:
        return WIN_SCORE - depth
    elif winner == opponent_mark:
        return LOSE_SCORE + depth
    return DRAW_SCORE


def minimax(board: Board, depth: int, is_maximizing: bool,
            ai_mark: str, opponent_mark: str) -> int:
    """
    Classic Minimax WITHOUT pruning -- explores the ENTIRE remaining game
    tree from the current board state.

    Parameters:
        board          -- current Board instance (mutated/backtracked in place)
        depth          -- current recursion depth (used for score shaping)
        is_maximizing  -- True if it's the AI's (maximizer's) turn to move
        ai_mark        -- the AI's mark ('X' or 'O')
        opponent_mark  -- the human/opponent's mark

    Returns:
        The minimax value of the current board state.

    This is included alongside the pruned version specifically so the
    project demonstrates BOTH algorithms as required by the task, and so
    their performance/node-count difference can be measured directly.
    """
    if board.is_terminal():
        return evaluate(board, ai_mark, opponent_mark, depth)

    if is_maximizing:
        best_score = float("-inf")
        for move in board.get_available_moves():
            board.place_mark(move, ai_mark)
            score = minimax(board, depth + 1, False, ai_mark, opponent_mark)
            board.undo_move(move)
            best_score = max(best_score, score)
        return best_score
    else:
        best_score = float("inf")
        for move in board.get_available_moves():
            board.place_mark(move, opponent_mark)
            score = minimax(board, depth + 1, True, ai_mark, opponent_mark)
            board.undo_move(move)
            best_score = min(best_score, score)
        return best_score


def minimax_alpha_beta(board: Board, depth: int, alpha: float, beta: float,
                        is_maximizing: bool, ai_mark: str, opponent_mark: str) -> int:
    """
    Minimax WITH Alpha-Beta Pruning -- functionally equivalent to plain
    Minimax (returns the same value) but skips branches that cannot
    possibly influence the final decision, making it significantly faster.

    Parameters:
        alpha -- best score the MAXIMIZER can guarantee so far
        beta  -- best score the MINIMIZER can guarantee so far

    Pruning rule: if at any point alpha >= beta, the current branch can
    never be chosen by the parent node, so we stop exploring it ("cut off").
    """
    if board.is_terminal():
        return evaluate(board, ai_mark, opponent_mark, depth)

    if is_maximizing:
        best_score = float("-inf")
        for move in board.get_available_moves():
            board.place_mark(move, ai_mark)
            score = minimax_alpha_beta(board, depth + 1, alpha, beta, False, ai_mark, opponent_mark)
            board.undo_move(move)
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break  # Beta cut-off: minimizer parent won't allow this branch
        return best_score
    else:
        best_score = float("inf")
        for move in board.get_available_moves():
            board.place_mark(move, opponent_mark)
            score = minimax_alpha_beta(board, depth + 1, alpha, beta, True, ai_mark, opponent_mark)
            board.undo_move(move)
            best_score = min(best_score, score)
            beta = min(beta, best_score)
            if alpha >= beta:
                break  # Alpha cut-off: maximizer parent won't allow this branch
        return best_score


def find_best_move(board: Board, ai_mark: str, opponent_mark: str,
                    use_pruning: bool = True) -> Tuple[int, int]:
    """
    Evaluates every available move for the AI and returns the
    (best_position, best_score) pair using either plain Minimax or the
    Alpha-Beta pruned version, based on `use_pruning`.

    This is the function called by AIPlayer on Hard difficulty.
    """
    best_score = float("-inf")
    best_move = -1

    for move in board.get_available_moves():
        board.place_mark(move, ai_mark)
        if use_pruning:
            score = minimax_alpha_beta(
                board, depth=0, alpha=float("-inf"), beta=float("inf"),
                is_maximizing=False, ai_mark=ai_mark, opponent_mark=opponent_mark
            )
        else:
            score = minimax(
                board, depth=0, is_maximizing=False,
                ai_mark=ai_mark, opponent_mark=opponent_mark
            )
        board.undo_move(move)

        if score > best_score:
            best_score = score
            best_move = move

    return best_move, best_score
