"""
recommender_system.py
-----------------------
`RecommenderSystem` is the top-level facade class that wires together
the data pipeline, the three recommendation strategies, and evaluation
-- this is the single entry point the CLI / notebooks / tests interact
with, following the Facade design pattern to hide internal complexity.
"""

from __future__ import annotations

import os
from typing import Literal

import pandas as pd

from src.data_preprocessing import DataPreprocessor
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.hybrid import HybridRecommender
from src.evaluation import Evaluator
from src.exceptions import InvalidParameterError, UserNotFoundError, require
from src.logger import get_logger

logger = get_logger(__name__)

Strategy = Literal["content", "collaborative", "hybrid"]


class RecommenderSystem:
    """Facade exposing a single, simple API over all three recommenders."""

    def __init__(self, movies_path: str, ratings_path: str,
                 cf_strategy: str = "svd", hybrid_alpha: float = 0.5):
        self.movies_path = movies_path
        self.ratings_path = ratings_path
        self.hybrid_alpha = hybrid_alpha
        self.cf_strategy = cf_strategy

        self.preprocessor = DataPreprocessor(movies_path, ratings_path)
        self.content_model: ContentBasedRecommender | None = None
        self.collaborative_model: CollaborativeFilteringRecommender | None = None
        self.hybrid_model: HybridRecommender | None = None
        self._is_ready = False

    # ------------------------------------------------------------------ #
    def build(self) -> "RecommenderSystem":
        """Run the full pipeline: load data -> engineer features -> fit models."""
        self.preprocessor.load()
        genre_matrix = self.preprocessor.build_genre_matrix()  # noqa: F841 (used internally by content model)
        user_item_matrix = self.preprocessor.build_user_item_matrix()

        self.content_model = ContentBasedRecommender(
            self.preprocessor.movies, self.preprocessor.ratings
        ).fit()

        self.collaborative_model = CollaborativeFilteringRecommender(
            user_item_matrix, self.preprocessor.movies, strategy=self.cf_strategy
        ).fit()

        self.hybrid_model = HybridRecommender(
            self.content_model, self.collaborative_model, alpha=self.hybrid_alpha
        )

        self._is_ready = True
        logger.info("RecommenderSystem build complete - all 3 models ready.")
        return self

    def _require_ready(self):
        require(self._is_ready, InvalidParameterError,
                "Call build() before requesting recommendations.")

    def known_users(self) -> list:
        self._require_ready()
        return sorted(self.preprocessor.ratings["userId"].unique().tolist())

    # ------------------------------------------------------------------ #
    def recommend(self, user_id: int, strategy: Strategy = "hybrid",
                  top_n: int = 10) -> pd.DataFrame:
        self._require_ready()
        require(user_id in self.known_users(), UserNotFoundError,
                f"userId={user_id} does not exist in the dataset. "
                f"Use the 'new user' flow instead (see input_new_user_preferences).")

        if strategy == "content":
            return self.content_model.recommend(user_id, top_n)
        if strategy == "collaborative":
            return self.collaborative_model.recommend(user_id, top_n)
        if strategy == "hybrid":
            return self.hybrid_model.recommend(user_id, top_n)
        raise InvalidParameterError(
            f"Unknown strategy '{strategy}'. Use 'content', 'collaborative' or 'hybrid'."
        )

    def explain(self, user_id: int, movie_id: int, strategy: Strategy = "hybrid") -> str:
        self._require_ready()
        if strategy == "content":
            return self.content_model.explain(user_id, movie_id)
        if strategy == "collaborative":
            return self.collaborative_model.explain(user_id, movie_id)
        if strategy == "hybrid":
            return self.hybrid_model.explain(user_id, movie_id)
        raise InvalidParameterError(f"Unknown strategy '{strategy}'.")

    # ------------------------------------------------------------------ #
    def recommend_for_new_user(self, liked_genres: list[str], top_n: int = 10) -> pd.DataFrame:
        """Cold-start path: a brand-new user with no rating history supplies
        their favourite genres directly -> pure content-based recommendation.
        This demonstrates handling the classic collaborative-filtering
        cold-start problem.
        """
        self._require_ready()
        require(len(liked_genres) > 0, InvalidParameterError,
                "Provide at least one preferred genre.")

        movies = self.preprocessor.movies
        genre_matrix = self.preprocessor.genre_matrix
        valid_genres = [g for g in liked_genres if g in genre_matrix.columns]
        require(len(valid_genres) > 0, InvalidParameterError,
                f"None of {liked_genres} match known genres: {list(genre_matrix.columns)}")

        scores = genre_matrix[valid_genres].sum(axis=1)
        top = scores.sort_values(ascending=False).head(top_n)
        result = movies.set_index("movieId").loc[top.index, ["title", "genres"]].copy()
        result["match_score"] = top.values
        return result.reset_index().rename(columns={"index": "movieId"})

    # ------------------------------------------------------------------ #
    def evaluate(self, k: int = 10, strategy: Strategy = "hybrid",
                test_size: float = 0.2) -> dict:
        """Re-trains the pipeline on a train split and measures Precision@K /
        Recall@K on the held-out test split.
        """
        self._require_ready()
        train_df, test_df = self.preprocessor.train_test_split(test_size=test_size)

        # Rebuild models on the TRAIN split only, so evaluation reflects
        # genuine generalisation rather than memorised training ratings.
        train_user_item = train_df.pivot_table(index="userId", columns="movieId", values="rating")
        content_eval = ContentBasedRecommender(self.preprocessor.movies, train_df).fit()
        collab_eval = CollaborativeFilteringRecommender(
            train_user_item, self.preprocessor.movies, strategy=self.cf_strategy
        ).fit()
        hybrid_eval = HybridRecommender(content_eval, collab_eval, alpha=self.hybrid_alpha)

        model_map = {"content": content_eval, "collaborative": collab_eval, "hybrid": hybrid_eval}
        require(strategy in model_map, InvalidParameterError, f"Unknown strategy '{strategy}'.")
        model = model_map[strategy]

        evaluator = Evaluator(test_df)
        # Only evaluate users who exist in the train split (others would
        # have no model to score them with).
        valid_users = [u for u in test_df["userId"].unique() if u in train_user_item.index]
        return evaluator.evaluate(model.recommend, k=k, user_ids=valid_users)

    # ------------------------------------------------------------------ #
    def save_recommendations_to_csv(self, recommendations: pd.DataFrame,
                                     output_path: str) -> str:
        require(not recommendations.empty, InvalidParameterError,
                "Cannot save an empty recommendations DataFrame.")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        recommendations.to_csv(output_path, index=False)
        logger.info("Saved %d recommendations to %s", len(recommendations), output_path)
        return output_path
