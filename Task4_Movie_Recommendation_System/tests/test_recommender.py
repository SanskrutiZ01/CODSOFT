"""
test_recommender.py
---------------------
Lightweight unit tests using pytest. Run with:  pytest -v

These cover: data validation, feature engineering, all three
recommendation strategies, error handling, and evaluation metrics.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_preprocessing import DataPreprocessor
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.hybrid import HybridRecommender
from src.recommender_system import RecommenderSystem
from src.exceptions import (
    DataValidationError, UserNotFoundError, InvalidParameterError, ItemNotFoundError
)

MOVIES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "movies.csv")
RATINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "ratings.csv")


@pytest.fixture(scope="module")
def system():
    return RecommenderSystem(MOVIES_PATH, RATINGS_PATH).build()


# --------------------------- Data preprocessing --------------------------- #

def test_load_valid_data():
    pre = DataPreprocessor(MOVIES_PATH, RATINGS_PATH).load()
    assert len(pre.movies) > 0
    assert len(pre.ratings) > 0


def test_missing_file_raises():
    with pytest.raises(DataValidationError):
        DataPreprocessor("nonexistent_movies.csv", RATINGS_PATH).load()


def test_genre_matrix_shape():
    pre = DataPreprocessor(MOVIES_PATH, RATINGS_PATH).load()
    matrix = pre.build_genre_matrix()
    assert matrix.shape[0] == len(pre.movies)
    assert matrix.values.max() <= 1 and matrix.values.min() >= 0


def test_train_test_split_covers_users():
    pre = DataPreprocessor(MOVIES_PATH, RATINGS_PATH).load()
    train, test = pre.train_test_split(test_size=0.2)
    assert len(train) + len(test) == len(pre.ratings)


# --------------------------- Content-based --------------------------- #

def test_content_based_recommend(system):
    recs = system.content_model.recommend(user_id=1, top_n=5)
    assert len(recs) == 5
    assert "content_score" in recs.columns


def test_content_based_unknown_user_raises(system):
    with pytest.raises(UserNotFoundError):
        system.content_model.recommend(user_id=99999, top_n=5)


def test_content_based_explain_unknown_movie_raises(system):
    with pytest.raises(ItemNotFoundError):
        system.content_model.explain(user_id=1, movie_id=999999)


# --------------------------- Collaborative --------------------------- #

def test_collaborative_recommend(system):
    recs = system.collaborative_model.recommend(user_id=1, top_n=5)
    assert len(recs) == 5
    assert "predicted_rating" in recs.columns
    assert recs["predicted_rating"].between(0.5, 5.0).all()


def test_collaborative_invalid_topn_raises(system):
    with pytest.raises(InvalidParameterError):
        system.collaborative_model.recommend(user_id=1, top_n=0)


# --------------------------- Hybrid --------------------------- #

def test_hybrid_recommend(system):
    recs = system.hybrid_model.recommend(user_id=1, top_n=5)
    assert len(recs) == 5
    assert "hybrid_score" in recs.columns
    assert recs["hybrid_score"].is_monotonic_decreasing


def test_hybrid_invalid_alpha_raises(system):
    with pytest.raises(InvalidParameterError):
        HybridRecommender(system.content_model, system.collaborative_model, alpha=1.5)


# --------------------------- Facade / cold start / evaluation --------------------------- #

def test_recommend_invalid_strategy(system):
    with pytest.raises(InvalidParameterError):
        system.recommend(user_id=1, strategy="not_a_real_strategy")  # type: ignore


def test_cold_start_new_user(system):
    recs = system.recommend_for_new_user(["Action", "Comedy"], top_n=5)
    assert len(recs) <= 5
    assert "match_score" in recs.columns


def test_evaluate_returns_metrics(system):
    results = system.evaluate(k=5, strategy="hybrid")
    assert f"Precision@5" in results
    assert f"Recall@5" in results
    assert 0.0 <= results["Precision@5"] <= 1.0
    assert 0.0 <= results["Recall@5"] <= 1.0


def test_save_recommendations_to_csv(tmp_path, system):
    recs = system.recommend(user_id=1, strategy="hybrid", top_n=5)
    out_path = os.path.join(tmp_path, "test_output.csv")
    saved_path = system.save_recommendations_to_csv(recs, out_path)
    assert os.path.exists(saved_path)
