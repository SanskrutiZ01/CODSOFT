"""
exceptions.py
-------------
Custom exception hierarchy and lightweight validation helpers used
throughout the recommender system. Centralising errors here keeps
error handling consistent and makes the codebase easier to test.
"""

from __future__ import annotations


class RecommenderError(Exception):
    """Base class for all recommender-system specific errors."""


class DataValidationError(RecommenderError):
    """Raised when input data fails schema / sanity checks."""


class UserNotFoundError(RecommenderError):
    """Raised when a requested userId does not exist in the dataset."""


class ItemNotFoundError(RecommenderError):
    """Raised when a requested movieId does not exist in the dataset."""


class ModelNotTrainedError(RecommenderError):
    """Raised when a prediction is requested before the model is fit."""


class InvalidParameterError(RecommenderError):
    """Raised when a method receives an out-of-range / invalid argument."""


def require(condition: bool, exception: type[RecommenderError], message: str) -> None:
    """Raise `exception(message)` if `condition` is False.

    A tiny helper to keep validation one-liners readable, e.g.:
        require(k > 0, InvalidParameterError, "k must be a positive integer")
    """
    if not condition:
        raise exception(message)
