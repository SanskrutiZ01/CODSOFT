"""
collaborative_filtering.py
----------------------------
Collaborative Filtering recommender with two interchangeable strategies:

1. User-Based CF: find users with similar rating *patterns* (Pearson /
   cosine similarity over the user-item matrix) and recommend what
   similar users liked.

2. Model-Based CF (Matrix Factorization via Truncated SVD): decompose
   the sparse user-item matrix into low-rank user-latent and
   item-latent factor matrices, then reconstruct predicted ratings for
   every (user, movie) pair. This captures *latent* taste dimensions
   that aren't explicit in the raw data (e.g. "dark, slow-burn dramas")
   and scales far better than pairwise similarity on large catalogs.

No external/paid ML services are used -- only scikit-learn's
`TruncatedSVD`, which runs entirely locally.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from src.exceptions import ModelNotTrainedError, UserNotFoundError, require, InvalidParameterError
from src.logger import get_logger

logger = get_logger(__name__)


class CollaborativeFilteringRecommender:
    """Matrix-factorization (SVD) and user-based collaborative filtering."""

    def __init__(self, user_item_matrix: pd.DataFrame, movies: pd.DataFrame,
                 n_factors: int = 20, strategy: Literal["svd", "user_based"] = "svd"):
        self.raw_matrix = user_item_matrix
        self.movies = movies.set_index("movieId", drop=False)
        self.n_factors = n_factors
        self.strategy = strategy

        self.user_means: pd.Series | None = None
        self.filled_matrix: pd.DataFrame | None = None
        self.predicted_ratings: pd.DataFrame | None = None
        self.user_similarity: pd.DataFrame | None = None
        self._is_fitted = False

    def fit(self) -> "CollaborativeFilteringRecommender":
        """Mean-centre missing ratings, then factorize / build similarities."""
        self.user_means = self.raw_matrix.mean(axis=1)
        # Fill missing entries with each user's own mean rating (a standard,
        # simple imputation for sparse rating matrices) rather than 0, which
        # would wrongly imply "actively disliked".
        self.filled_matrix = self.raw_matrix.apply(
            lambda row: row.fillna(self.user_means[row.name]), axis=1
        )

        n_components = min(self.n_factors, min(self.filled_matrix.shape) - 1)
        n_components = max(n_components, 1)

        if self.strategy == "svd":
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            user_factors = svd.fit_transform(self.filled_matrix.values)
            item_factors = svd.components_
            reconstructed = user_factors @ item_factors
            self.predicted_ratings = pd.DataFrame(
                reconstructed, index=self.filled_matrix.index,
                columns=self.filled_matrix.columns
            ).clip(0.5, 5.0)
            logger.info("Fitted SVD collaborative model with %d latent factors "
                        "(explained variance ratio sum=%.3f)",
                        n_components, svd.explained_variance_ratio_.sum())
        else:  # user_based
            sim = cosine_similarity(self.filled_matrix.values)
            self.user_similarity = pd.DataFrame(
                sim, index=self.filled_matrix.index, columns=self.filled_matrix.index
            )
            logger.info("Fitted user-based collaborative model over %d users",
                        len(self.filled_matrix))

        self._is_fitted = True
        return self

    def _check_fitted(self):
        if not self._is_fitted:
            raise ModelNotTrainedError("Call fit() before generating recommendations.")

    def _predict_user_based(self, user_id: int, top_k_neighbors: int = 10) -> pd.Series:
        neighbors = self.user_similarity[user_id].drop(index=user_id).sort_values(ascending=False)
        neighbors = neighbors.head(top_k_neighbors)
        neighbor_ratings = self.filled_matrix.loc[neighbors.index]
        weights = neighbors.values.reshape(-1, 1)
        weighted_sum = (neighbor_ratings.values * weights).sum(axis=0)
        weight_total = np.abs(weights).sum() or 1e-9
        return pd.Series(weighted_sum / weight_total, index=self.filled_matrix.columns)

    def recommend(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        require(top_n > 0, InvalidParameterError, "top_n must be a positive integer.")
        self._check_fitted()
        require(user_id in self.raw_matrix.index, UserNotFoundError,
                f"userId={user_id} not found in the ratings data.")

        if self.strategy == "svd":
            predictions = self.predicted_ratings.loc[user_id]
        else:
            predictions = self._predict_user_based(user_id)

        seen = self.raw_matrix.loc[user_id].dropna().index
        candidates = predictions.drop(index=seen, errors="ignore").sort_values(ascending=False)
        top = candidates.head(top_n)

        result = self.movies.loc[top.index, ["movieId", "title", "genres"]].copy()
        result["predicted_rating"] = top.values.round(2)
        return result.reset_index(drop=True)

    def explain(self, user_id: int, movie_id: int) -> str:
        self._check_fitted()
        title = self.movies.loc[movie_id, "title"]
        if self.strategy == "svd":
            score = self.predicted_ratings.loc[user_id, movie_id]
            return (f"'{title}' was recommended because the matrix-factorization model "
                    f"predicts you would rate it {score:.2f}/5, based on latent taste "
                    f"patterns learned from users with similar rating behaviour.")
        neighbors = self.user_similarity[user_id].drop(index=user_id).sort_values(ascending=False).head(3)
        return (f"'{title}' was recommended because it was highly rated by users most "
                f"similar to you (top similar userIds: {list(neighbors.index)}).")
