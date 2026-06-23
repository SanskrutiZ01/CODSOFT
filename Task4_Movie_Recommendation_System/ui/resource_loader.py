"""
resource_loader.py
--------------------
Wraps the existing `RecommenderSystem` backend (unchanged business logic
from the CLI project) behind Streamlit's resource cache, so the data
pipeline + model fitting (TF-IDF, SVD, etc.) runs exactly once per
session/deployment instead of on every widget interaction.
"""

from __future__ import annotations

import streamlit as st

from src.recommender_system import RecommenderSystem


@st.cache_resource(show_spinner="Loading data and training models...")
def load_recommender_system(movies_path: str, ratings_path: str,
                            cf_strategy: str = "svd",
                            hybrid_alpha: float = 0.5) -> RecommenderSystem:
    """Build (and cache) the RecommenderSystem facade.

    Cached by Streamlit on (movies_path, ratings_path, cf_strategy,
    hybrid_alpha) -- changing any of these (e.g. via the sidebar) will
    correctly trigger a rebuild instead of serving a stale model.
    """
    system = RecommenderSystem(
        movies_path, ratings_path, cf_strategy=cf_strategy, hybrid_alpha=hybrid_alpha
    ).build()
    return system
