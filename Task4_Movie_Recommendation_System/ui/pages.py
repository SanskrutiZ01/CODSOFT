"""
pages.py
---------
Each screen of the app is a class implementing a common `Page`
interface (`render`). This keeps the app object-oriented and modular:
adding a new screen means adding a new class and one line of routing
in `app.py`, instead of growing a single monolithic script.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
import streamlit as st

from src.recommender_system import RecommenderSystem
from src.exceptions import RecommenderError
from ui.components import (
    render_header, render_metric_row, render_recommendation_table,
    render_download_button, render_explanations, render_error, render_empty_state,
)

STRATEGY_LABELS = {
    "Hybrid (recommended)": "hybrid",
    "Content-Based Filtering": "content",
    "Collaborative Filtering": "collaborative",
}
SCORE_COL_BY_STRATEGY = {
    "hybrid": "hybrid_score",
    "content": "content_score",
    "collaborative": "predicted_rating",
}


class Page(ABC):
    """Common interface every app screen implements."""

    title: str = "Page"
    icon: str = "📄"

    @abstractmethod
    def render(self, system: RecommenderSystem) -> None:
        ...


class HomePage(Page):
    title = "Home"
    icon = "🏠"

    def render(self, system: RecommenderSystem) -> None:
        render_header("🎬 Movie Recommender System",
                      "Content-Based · Collaborative · Hybrid Filtering — built with Python & scikit-learn")

        n_users = len(system.known_users())
        n_movies = len(system.preprocessor.movies)
        n_ratings = len(system.preprocessor.ratings)
        render_metric_row({
            "👥 Users": n_users,
            "🎞️ Movies": n_movies,
            "⭐ Ratings": n_ratings,
            "🎭 Genres": len(system.preprocessor.genre_matrix.columns),
        })

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 🧩 Content-Based")
            st.write("Recommends movies similar in **genre profile** to ones you rated highly, "
                     "using TF-IDF + cosine similarity.")
        with col2:
            st.markdown("#### 🤝 Collaborative")
            st.write("Predicts your ratings using **matrix factorization (SVD)** over patterns "
                     "from users with similar taste.")
        with col3:
            st.markdown("#### ⚡ Hybrid")
            st.write("Blends both signals with a tunable weight, balancing **personalisation** "
                     "and **cold-start coverage**.")

        st.markdown("---")
        st.markdown(
            "👉 Use the sidebar to navigate to **Get Recommendations**, **New User (Cold Start)**, "
            "or **Model Evaluation**."
        )


class RecommendPage(Page):
    title = "Get Recommendations"
    icon = "🎯"

    def render(self, system: RecommenderSystem) -> None:
        render_header("🎯 Get Recommendations", "Recommendations for an existing user, in real time.")

        known_users = system.known_users()

        with st.form("recommend_form"):
            col1, col2 = st.columns([1, 1])
            with col1:
                user_id = st.number_input(
                    "User ID", min_value=int(min(known_users)),
                    max_value=int(max(known_users)), step=1,
                    help=f"Valid range: {min(known_users)}–{max(known_users)} ({len(known_users)} known users)",
                )
            with col2:
                strategy_label = st.selectbox("Recommendation Strategy", list(STRATEGY_LABELS.keys()))

            col3, col4 = st.columns([1, 1])
            with col3:
                top_n = st.slider("Number of recommendations (Top-N)", min_value=3, max_value=25, value=10)
            with col4:
                show_explanations = st.checkbox("Show explanations", value=True)

            submitted = st.form_submit_button("🚀 Generate Recommendations", use_container_width=True)

        if not submitted:
            render_empty_state("Fill in the form above and click **Generate Recommendations**.")
            return

        strategy = STRATEGY_LABELS[strategy_label]
        try:
            with st.spinner("Computing recommendations..."):
                recs = system.recommend(int(user_id), strategy=strategy, top_n=int(top_n))
        except RecommenderError as exc:
            render_error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            render_error(f"Unexpected error: {exc}")
            return

        st.success(f"Generated {len(recs)} recommendations using **{strategy_label}** for user **{int(user_id)}**.")
        score_col = SCORE_COL_BY_STRATEGY[strategy]
        render_recommendation_table(recs, score_col)

        render_download_button(recs, f"recommendations_user{int(user_id)}_{strategy}.csv")

        if show_explanations:
            st.markdown("#### 💡 Why these recommendations?")
            try:
                explanations = [system.explain(int(user_id), mid, strategy=strategy)
                                for mid in recs["movieId"]]
                render_explanations(explanations)
            except RecommenderError as exc:
                render_error(str(exc))


class ColdStartPage(Page):
    title = "New User (Cold Start)"
    icon = "🆕"

    def render(self, system: RecommenderSystem) -> None:
        render_header("🆕 New User Recommendations", "No rating history yet? Tell us what you like.")
        st.caption(
            "This demonstrates handling the classic **cold-start problem**: collaborative "
            "filtering cannot work for users with no history, so we fall back to pure "
            "content-based matching on stated genre preferences."
        )

        all_genres = sorted(system.preprocessor.genre_matrix.columns.tolist())

        with st.form("cold_start_form"):
            genres = st.multiselect("Select your favourite genres", options=all_genres,
                                    default=all_genres[:2] if len(all_genres) >= 2 else all_genres)
            top_n = st.slider("Number of recommendations (Top-N)", min_value=3, max_value=25, value=10)
            submitted = st.form_submit_button("🚀 Get Recommendations", use_container_width=True)

        if not submitted:
            render_empty_state("Select at least one genre and click **Get Recommendations**.")
            return

        try:
            with st.spinner("Matching genres..."):
                recs = system.recommend_for_new_user(genres, top_n=int(top_n))
        except RecommenderError as exc:
            render_error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            render_error(f"Unexpected error: {exc}")
            return

        st.success(f"Found {len(recs)} movies matching: {', '.join(genres)}")
        render_recommendation_table(recs, "match_score")
        render_download_button(recs, "new_user_recommendations.csv")


class EvaluationPage(Page):
    title = "Model Evaluation"
    icon = "📊"

    def render(self, system: RecommenderSystem) -> None:
        render_header("📊 Model Evaluation", "Precision@K and Recall@K, compared across all three strategies.")
        st.caption(
            "Computed on an 80/20 per-user train/test split. 'Relevant' = test-set movies "
            "the user rated ≥ 3.5/5. Models are refit on the train split only, so these "
            "numbers reflect genuine generalisation."
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            k = st.number_input("K", min_value=1, max_value=25, value=10, step=1)
            run = st.button("▶️ Run Evaluation", use_container_width=True)

        if not run:
            render_empty_state("Choose K and click **Run Evaluation**. This re-fits all 3 models on a train split.")
            return

        rows = []
        progress = st.progress(0, text="Evaluating...")
        strategies = ["content", "collaborative", "hybrid"]
        for i, strategy in enumerate(strategies):
            try:
                result = system.evaluate(k=int(k), strategy=strategy)
                result["strategy"] = strategy
                rows.append(result)
            except RecommenderError as exc:
                render_error(f"{strategy}: {exc}")
            progress.progress((i + 1) / len(strategies), text=f"Evaluated {strategy}")
        progress.empty()

        if not rows:
            render_empty_state("No evaluation results were produced.")
            return

        report_df = pd.DataFrame(rows).set_index("strategy")
        st.dataframe(report_df, use_container_width=True)

        precision_col = f"Precision@{int(k)}"
        recall_col = f"Recall@{int(k)}"
        chart_df = report_df[[precision_col, recall_col]]
        st.bar_chart(chart_df)

        render_download_button(report_df.reset_index(), f"evaluation_report_k{int(k)}.csv")


class AboutPage(Page):
    title = "About"
    icon = "ℹ️"

    def render(self, system: RecommenderSystem) -> None:
        render_header("ℹ️ About This Project", "")
        st.markdown(
            """
This **Movie Recommender System** demonstrates three classic recommendation
paradigms, built end-to-end in Python with an object-oriented, testable
architecture:

- **Content-Based Filtering** — TF-IDF genre vectors + cosine similarity
- **Collaborative Filtering** — Matrix Factorization via Truncated SVD
- **Hybrid Recommendation** — tunable weighted blend of both signals

**Engineering highlights**
- Modular `src/` package: data pipeline, three model classes, evaluator, facade
- Custom exception hierarchy + input validation (no silent failures)
- 15 automated `pytest` unit tests
- Offline evaluation: Precision@K / Recall@K on a proper train/test split
- This Streamlit UI is a thin presentation layer — **zero business logic
  duplication**; it calls the exact same `RecommenderSystem` class used by
  the CLI (`main.py`)

Built as a portfolio project for an AI & Data Science internship.
No paid APIs or external services are used — everything runs locally.
            """
        )
        st.markdown("---")
        st.caption("Source: github.com/yourusername/movie-recommender-system")
