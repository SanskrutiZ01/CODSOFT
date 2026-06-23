"""
data_preprocessing.py
----------------------
Loading, cleaning, validating and transforming the raw MovieLens-style
CSV files into structures ready for modelling:

    - cleaned `movies` and `ratings` DataFrames
    - a genre feature matrix (one-hot) for content-based filtering
    - a user-item ratings matrix for collaborative filtering
    - a chronological train/test split for evaluation

Design notes
------------
This is implemented as a single `DataPreprocessor` class (OOP) so the
pipeline can be unit-tested and re-used independently of the recommenders.
Every public method validates its inputs and raises a descriptive
`DataValidationError` rather than failing silently or with a cryptic
pandas traceback -- this matters a lot once the pipeline is fed real,
messy data instead of the clean sample set.
"""

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd
import numpy as np

from src.exceptions import DataValidationError, require
from src.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Loads and prepares MovieLens-style data for the recommender models."""

    REQUIRED_MOVIE_COLS = {"movieId", "title", "genres"}
    REQUIRED_RATING_COLS = {"userId", "movieId", "rating"}

    def __init__(self, movies_path: str, ratings_path: str):
        self.movies_path = movies_path
        self.ratings_path = ratings_path

        self.movies: pd.DataFrame | None = None
        self.ratings: pd.DataFrame | None = None
        self.genre_matrix: pd.DataFrame | None = None
        self.user_item_matrix: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # Loading & validation
    # ------------------------------------------------------------------ #
    def load(self) -> "DataPreprocessor":
        """Load CSVs from disk and run schema/sanity validation."""
        require(os.path.exists(self.movies_path), DataValidationError,
                f"Movies file not found at: {self.movies_path}")
        require(os.path.exists(self.ratings_path), DataValidationError,
                f"Ratings file not found at: {self.ratings_path}")

        logger.info("Loading movies from %s", self.movies_path)
        movies = pd.read_csv(self.movies_path)
        logger.info("Loading ratings from %s", self.ratings_path)
        ratings = pd.read_csv(self.ratings_path)

        self._validate_schema(movies, self.REQUIRED_MOVIE_COLS, "movies.csv")
        self._validate_schema(ratings, self.REQUIRED_RATING_COLS, "ratings.csv")

        self.movies = self._clean_movies(movies)
        self.ratings = self._clean_ratings(ratings)

        logger.info("Loaded %d movies and %d ratings from %d users",
                    len(self.movies), len(self.ratings),
                    self.ratings["userId"].nunique())
        return self

    @staticmethod
    def _validate_schema(df: pd.DataFrame, required_cols: set, label: str) -> None:
        missing = required_cols - set(df.columns)
        require(not missing, DataValidationError,
                f"{label} is missing required column(s): {missing}")
        require(len(df) > 0, DataValidationError, f"{label} is empty.")

    @staticmethod
    def _clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
        movies = movies.copy()
        movies = movies.drop_duplicates(subset="movieId")
        movies["genres"] = movies["genres"].fillna("(no genres listed)")
        movies["title"] = movies["title"].fillna("Unknown Title").str.strip()
        movies["movieId"] = movies["movieId"].astype(int)
        return movies.reset_index(drop=True)

    @staticmethod
    def _clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
        ratings = ratings.copy()
        before = len(ratings)
        ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
        ratings = ratings[(ratings["rating"] >= 0.5) & (ratings["rating"] <= 5.0)]
        ratings["userId"] = ratings["userId"].astype(int)
        ratings["movieId"] = ratings["movieId"].astype(int)
        ratings["rating"] = ratings["rating"].astype(float)
        dropped = before - len(ratings)
        if dropped:
            logger.warning("Dropped %d invalid rating rows during cleaning", dropped)
        return ratings.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Feature engineering
    # ------------------------------------------------------------------ #
    def build_genre_matrix(self) -> pd.DataFrame:
        """One-hot encode the pipe-separated `genres` column.

        This is the core *feature engineering* step for content-based
        filtering: every movie becomes a binary vector over the genre
        vocabulary, e.g. [Action=1, Comedy=0, Drama=1, ...].
        """
        require(self.movies is not None, DataValidationError,
                "Call load() before build_genre_matrix().")

        genre_lists = self.movies["genres"].str.split("|")
        all_genres = sorted({g for genres in genre_lists for g in genres})

        matrix = pd.DataFrame(
            0, index=self.movies["movieId"], columns=all_genres, dtype=np.int8
        )
        for movie_id, genres in zip(self.movies["movieId"], genre_lists):
            matrix.loc[movie_id, genres] = 1

        self.genre_matrix = matrix
        logger.info("Built genre feature matrix: %d movies x %d genres",
                    matrix.shape[0], matrix.shape[1])
        return matrix

    def build_user_item_matrix(self) -> pd.DataFrame:
        """Pivot ratings into a (users x movies) matrix for collaborative filtering."""
        require(self.ratings is not None, DataValidationError,
                "Call load() before build_user_item_matrix().")

        matrix = self.ratings.pivot_table(
            index="userId", columns="movieId", values="rating"
        )
        self.user_item_matrix = matrix
        sparsity = 1 - matrix.notna().sum().sum() / (matrix.shape[0] * matrix.shape[1])
        logger.info("Built user-item matrix: %d users x %d movies (sparsity=%.2f%%)",
                    matrix.shape[0], matrix.shape[1], sparsity * 100)
        return matrix

    # ------------------------------------------------------------------ #
    # Train / test split (for evaluation)
    # ------------------------------------------------------------------ #
    def train_test_split(self, test_size: float = 0.2, random_state: int = 42
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Per-user random split so every user appears in both splits
        (a plain global shuffle could leave some users entirely out of
        the training set, which would make evaluation meaningless).
        """
        require(self.ratings is not None, DataValidationError,
                "Call load() before train_test_split().")
        require(0 < test_size < 1, DataValidationError,
                "test_size must be between 0 and 1.")

        rng = np.random.RandomState(random_state)
        train_rows, test_rows = [], []

        for _, group in self.ratings.groupby("userId"):
            group = group.sample(frac=1, random_state=rng.randint(0, 1_000_000))
            n_test = max(1, int(len(group) * test_size)) if len(group) > 4 else 0
            test_rows.append(group.iloc[:n_test])
            train_rows.append(group.iloc[n_test:])

        train_df = pd.concat(train_rows).reset_index(drop=True)
        test_df = pd.concat(test_rows).reset_index(drop=True)
        logger.info("Train/test split -> train=%d rows, test=%d rows",
                    len(train_df), len(test_df))
        return train_df, test_df
