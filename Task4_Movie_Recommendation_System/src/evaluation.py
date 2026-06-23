"""
evaluation.py
---------------
Evaluation Metrics: Precision@K and Recall@K.

Definitions
-----------
For a given user and cut-off K:
    - "Relevant" items = items the user rated >= relevance_threshold in
      the held-out TEST set.
    - "Recommended@K" = the top-K items the model would recommend
      (computed only over the candidates the user has a known test
      interaction with -- standard practice for offline evaluation,
      since we cannot verify relevance for items with no ground truth).

    Precision@K = |Recommended@K ∩ Relevant| / K
    Recall@K    = |Recommended@K ∩ Relevant| / |Relevant|

Precision@K answers: "Of what we recommended, how much was actually
relevant?" Recall@K answers: "Of all relevant items, how many did we
manage to surface?" Both are averaged across all test users to give a
single project-level score.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from src.exceptions import InvalidParameterError, require
from src.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Computes Precision@K / Recall@K for any recommender exposing a
    `.recommend(user_id, top_n)` method returning a DataFrame with a
    `movieId` column.
    """

    def __init__(self, test_df: pd.DataFrame, relevance_threshold: float = 3.5):
        self.test_df = test_df
        self.relevance_threshold = relevance_threshold

    def _relevant_items(self, user_id: int) -> set:
        user_test = self.test_df[self.test_df["userId"] == user_id]
        relevant = user_test[user_test["rating"] >= self.relevance_threshold]
        return set(relevant["movieId"])

    def evaluate_user(self, recommend_fn: Callable[[int, int], pd.DataFrame],
                       user_id: int, k: int) -> Dict[str, float]:
        relevant = self._relevant_items(user_id)
        if not relevant:
            return {"precision": np.nan, "recall": np.nan}

        try:
            recs = recommend_fn(user_id, k)
        except Exception as exc:  # noqa: BLE001 - log and skip user gracefully
            logger.warning("Skipping userId=%s during evaluation: %s", user_id, exc)
            return {"precision": np.nan, "recall": np.nan}

        recommended = set(recs["movieId"].head(k))
        hits = recommended & relevant

        precision = len(hits) / k if k > 0 else 0.0
        recall = len(hits) / len(relevant) if relevant else 0.0
        return {"precision": precision, "recall": recall}

    def evaluate(self, recommend_fn: Callable[[int, int], pd.DataFrame],
                 k: int = 10, user_ids: List[int] | None = None) -> Dict[str, float]:
        """Average Precision@K / Recall@K across all (or given) test users."""
        require(k > 0, InvalidParameterError, "k must be a positive integer.")

        if user_ids is None:
            user_ids = sorted(self.test_df["userId"].unique())

        precisions, recalls = [], []
        for uid in user_ids:
            scores = self.evaluate_user(recommend_fn, uid, k)
            if not np.isnan(scores["precision"]):
                precisions.append(scores["precision"])
                recalls.append(scores["recall"])

        result = {
            f"Precision@{k}": round(float(np.mean(precisions)), 4) if precisions else 0.0,
            f"Recall@{k}": round(float(np.mean(recalls)), 4) if recalls else 0.0,
            "evaluated_users": len(precisions),
        }
        logger.info("Evaluation result (k=%d): %s", k, result)
        return result
