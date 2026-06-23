"""
content_based.py
------------------
Content-Based Filtering recommender.

Concept
-------
Each movie is represented as a feature vector built from its genres
(TF-IDF weighted). A user's "taste profile" vector is the average of
the feature vectors of movies they rated highly. Recommendations are
the unseen movies whose vectors are most *similar* (cosine similarity)
to the user's taste profile.

Why TF-IDF instead of plain one-hot genres?
TF-IDF down-weights very common genres (e.g. "Drama", which appears on
hundreds of movies and therefore carries little discriminative signal)
and up-weights rarer, more distinctive genres -- giving a sharper,
more personalised similarity signal.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.exceptions import ItemNotFoundError, ModelNotTrainedError, UserNotFoundError, require, InvalidParameterError
from src.logger import get_logger

logger = get_logger(__name__)


class ContentBasedRecommender:
    """Recommends movies based on genre similarity to a user's liked movies."""

    def __init__(self, movies: pd.DataFrame, ratings: pd.DataFrame,
                 like_threshold: float = 3.5):
        self.movies = movies.set_index("movieId", drop=False)
        self.ratings = ratings
        self.like_threshold = like_threshold

        self._tfidf_matrix = None
        self._similarity_matrix: pd.DataFrame | None = None
        self._movie_ids: List[int] | None = None
        self._is_fitted = False

    def fit(self) -> "ContentBasedRecommender":
        """Build the TF-IDF genre matrix and the item-item similarity matrix."""
        genre_corpus = self.movies["genres"].str.replace("|", " ", regex=False)
        vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+")
        self._tfidf_matrix = vectorizer.fit_transform(genre_corpus)
        self._movie_ids = list(self.movies["movieId"])

        sim = cosine_similarity(self._tfidf_matrix)
        self._similarity_matrix = pd.DataFrame(
            sim, index=self._movie_ids, columns=self._movie_ids
        )
        self._is_fitted = True
        logger.info("ContentBasedRecommender fitted on %d movies (TF-IDF dim=%d)",
                    len(self._movie_ids), self._tfidf_matrix.shape[1])
        return self

    def _check_fitted(self):
        if not self._is_fitted:
            raise ModelNotTrainedError("Call fit() before generating recommendations.")

    def build_user_profile(self, user_id: int) -> pd.Series:
        """Average similarity-weighted vector over the user's liked movies."""
        self._check_fitted()
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        require(len(user_ratings) > 0, UserNotFoundError,
                f"No ratings found for userId={user_id}.")

        liked = user_ratings[user_ratings["rating"] >= self.like_threshold]
        if liked.empty:  # fall back to the user's top-rated movies if none clear "like" threshold
            liked = user_ratings.sort_values("rating", ascending=False).head(3)

        liked_ids = [m for m in liked["movieId"] if m in self._similarity_matrix.index]
        require(len(liked_ids) > 0, UserNotFoundError,
                f"None of userId={user_id}'s rated movies exist in the catalog.")

        # weight each liked movie's similarity row by how much the user liked it
        weights = liked.set_index("movieId").loc[liked_ids, "rating"]
        weighted = self._similarity_matrix.loc[liked_ids].T.mul(weights.values, axis=1)
        profile = weighted.sum(axis=1) / weights.sum()
        return profile

    def recommend(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """Return the top-N unseen movies most similar to the user's taste profile."""
        require(top_n > 0, InvalidParameterError, "top_n must be a positive integer.")
        self._check_fitted()

        profile = self.build_user_profile(user_id)
        seen = set(self.ratings.loc[self.ratings["userId"] == user_id, "movieId"])
        candidates = profile.drop(labels=seen, errors="ignore").sort_values(ascending=False)
        top = candidates.head(top_n)

        result = self.movies.loc[top.index, ["movieId", "title", "genres"]].copy()
        result["content_score"] = top.values
        return result.reset_index(drop=True)

    def explain(self, user_id: int, movie_id: int) -> str:
        """Human-readable explanation of why a movie was recommended."""
        self._check_fitted()
        require(movie_id in self.movies.index, ItemNotFoundError,
                f"movieId={movie_id} not found in catalog.")

        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        liked = user_ratings[user_ratings["rating"] >= self.like_threshold]
        if liked.empty:
            liked = user_ratings.sort_values("rating", ascending=False).head(3)

        sims: List[Tuple[int, float]] = []
        for liked_id in liked["movieId"]:
            if liked_id in self._similarity_matrix.index and liked_id != movie_id:
                sims.append((liked_id, self._similarity_matrix.loc[liked_id, movie_id]))
        sims.sort(key=lambda x: x[1], reverse=True)

        target_title = self.movies.loc[movie_id, "title"]
        target_genres = self.movies.loc[movie_id, "genres"]
        if not sims:
            return f"'{target_title}' was suggested based on its genres: {target_genres}."

        best_id, best_score = sims[0]
        best_title = self.movies.loc[best_id, "title"]
        return (f"'{target_title}' ({target_genres}) was recommended because it shares "
                f"strong genre similarity (score={best_score:.2f}) with '{best_title}', "
                f"which you rated highly.")
