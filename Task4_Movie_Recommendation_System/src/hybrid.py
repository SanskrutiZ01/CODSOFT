"""
hybrid.py
----------
Hybrid Recommendation Approach.

Combines Content-Based and Collaborative Filtering scores using a
weighted-sum (linear hybridization) strategy:

    hybrid_score = alpha * normalized_content_score
                 + (1 - alpha) * normalized_collaborative_score

Both component scores are min-max normalised to [0, 1] before combining
since they live on different scales (cosine similarity vs. predicted
1-5 star rating). `alpha` lets you tune how much weight to give each
signal -- e.g. alpha=1.0 reduces to pure content-based, alpha=0.0 to
pure collaborative filtering.

This addresses the classic weaknesses of each individual approach:
    - Content-based alone -> over-specialises, "filter bubble" effect.
    - Collaborative alone -> fails on new/unrated items & users (cold start).
A hybrid mitigates both: collaborative captures community taste signal,
content-based provides cold-start coverage for items with few ratings.
"""

from __future__ import annotations

import pandas as pd

from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.exceptions import InvalidParameterError, require
from src.logger import get_logger

logger = get_logger(__name__)


def _normalize(series: pd.Series) -> pd.Series:
    if series.max() == series.min():
        return series.apply(lambda _: 1.0)
    return (series - series.min()) / (series.max() - series.min())


class HybridRecommender:
    """Weighted combination of content-based and collaborative filtering scores."""

    def __init__(self, content_model: ContentBasedRecommender,
                 collaborative_model: CollaborativeFilteringRecommender,
                 alpha: float = 0.5):
        require(0.0 <= alpha <= 1.0, InvalidParameterError, "alpha must be in [0, 1].")
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.alpha = alpha

    def recommend(self, user_id: int, top_n: int = 10, candidate_pool: int = 50) -> pd.DataFrame:
        """Blend both models' scores over a wider candidate pool, then re-rank."""
        require(top_n > 0, InvalidParameterError, "top_n must be a positive integer.")

        content_df = self.content_model.recommend(user_id, top_n=candidate_pool)
        collab_df = self.collaborative_model.recommend(user_id, top_n=candidate_pool)

        merged = pd.merge(
            content_df[["movieId", "title", "genres", "content_score"]],
            collab_df[["movieId", "predicted_rating"]],
            on="movieId", how="outer"
        )
        # fill missing scores (movie surfaced by only one model) with that
        # model's minimum observed score rather than 0/NaN, to avoid unfairly
        # penalising items only one model had visibility into.
        merged["content_score"] = merged["content_score"].fillna(merged["content_score"].min() if merged["content_score"].notna().any() else 0)
        merged["predicted_rating"] = merged["predicted_rating"].fillna(merged["predicted_rating"].min() if merged["predicted_rating"].notna().any() else 0)
        merged["title"] = merged["title"].fillna(
            merged["movieId"].map(self.content_model.movies["title"])
        )
        merged["genres"] = merged["genres"].fillna(
            merged["movieId"].map(self.content_model.movies["genres"])
        )

        merged["norm_content"] = _normalize(merged["content_score"])
        merged["norm_collab"] = _normalize(merged["predicted_rating"])
        merged["hybrid_score"] = (
            self.alpha * merged["norm_content"] + (1 - self.alpha) * merged["norm_collab"]
        )

        result = merged.sort_values("hybrid_score", ascending=False).head(top_n)
        result = result[["movieId", "title", "genres", "content_score",
                          "predicted_rating", "hybrid_score"]]
        result["hybrid_score"] = result["hybrid_score"].round(3)
        return result.reset_index(drop=True)

    def explain(self, user_id: int, movie_id: int) -> str:
        content_part = self.content_model.explain(user_id, movie_id)
        collab_part = self.collaborative_model.explain(user_id, movie_id)
        return (f"[Hybrid explanation, alpha={self.alpha}]\n"
                f"  Content-based signal: {content_part}\n"
                f"  Collaborative signal: {collab_part}")
