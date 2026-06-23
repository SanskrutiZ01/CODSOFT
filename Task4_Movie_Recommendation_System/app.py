"""
app.py
-------
Streamlit web application for the Movie Recommender System.

Run with:
    streamlit run app.py

Architecture
------------
This file only handles page routing + global setup. ALL recommendation
logic lives in `src/` (unchanged from the CLI project) and is reused
as-is through the `RecommenderSystem` facade -- the UI is a thin,
swappable presentation layer (`StreamlitApp` class below), following
the same OOP / separation-of-concerns principle as the rest of the
codebase.
"""

from __future__ import annotations

import os

import streamlit as st

from src.exceptions import RecommenderError
from ui.resource_loader import load_recommender_system
from ui.pages import HomePage, RecommendPage, ColdStartPage, EvaluationPage, AboutPage

DEFAULT_MOVIES_PATH = "data/sample/movies.csv"
DEFAULT_RATINGS_PATH = "data/sample/ratings.csv"
CSS_PATH = "assets/style.css"


class StreamlitApp:
    """Top-level application controller: page registry + sidebar routing."""

    def __init__(self):
        self.pages = [
            HomePage(),
            RecommendPage(),
            ColdStartPage(),
            EvaluationPage(),
            AboutPage(),
        ]

    # ------------------------------------------------------------------ #
    def configure_page(self) -> None:
        st.set_page_config(
            page_title="Movie Recommender System",
            page_icon="🎬",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        self._inject_css()

    @staticmethod
    def _inject_css() -> None:
        if os.path.exists(CSS_PATH):
            with open(CSS_PATH, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # ------------------------------------------------------------------ #
    def render_sidebar(self) -> tuple[str, str, float]:
        """Renders dataset path + model config controls; returns the
        current selection used to (re)build the cached RecommenderSystem.
        """
        with st.sidebar:
            st.markdown("## 🎬 Movie Recommender")
            st.caption("Content-Based · Collaborative · Hybrid")
            st.markdown("---")

            page_label = st.radio(
                "Navigate",
                options=[f"{p.icon}  {p.title}" for p in self.pages],
                label_visibility="collapsed",
            )

            st.markdown("---")
            with st.expander("⚙️ Data & Model Settings"):
                movies_path = st.text_input("Movies CSV path", value=DEFAULT_MOVIES_PATH)
                ratings_path = st.text_input("Ratings CSV path", value=DEFAULT_RATINGS_PATH)
                hybrid_alpha = st.slider(
                    "Hybrid weight (α = content weight)",
                    min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                    help="α=1 → pure content-based, α=0 → pure collaborative filtering",
                )

            st.markdown("---")
            st.caption("Built with Streamlit · scikit-learn · pandas")

        return movies_path, ratings_path, hybrid_alpha, page_label

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        self.configure_page()
        movies_path, ratings_path, hybrid_alpha, page_label = self.render_sidebar()

        try:
            system = load_recommender_system(
                movies_path, ratings_path, hybrid_alpha=hybrid_alpha
            )
        except RecommenderError as exc:
            st.error(f"⚠️ Failed to load data/models: {exc}")
            st.info("Check the file paths in **⚙️ Data & Model Settings** in the sidebar.")
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"⚠️ Unexpected error while building the recommender system: {exc}")
            return

        active_page = next(
            (p for p in self.pages if f"{p.icon}  {p.title}" == page_label), self.pages[0]
        )

        try:
            active_page.render(system)
        except RecommenderError as exc:
            st.error(f"⚠️ {exc}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"⚠️ Unexpected error: {exc}")
            st.exception(exc)


def main():
    StreamlitApp().run()


if __name__ == "__main__":
    main()
