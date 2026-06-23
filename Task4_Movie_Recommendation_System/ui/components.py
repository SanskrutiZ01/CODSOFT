"""
components.py
---------------
Reusable, styled UI building blocks shared across pages. Keeping these
in dedicated functions (rather than copy-pasted markup in every page)
is the Streamlit equivalent of componentizing a frontend, and keeps
`app.py` / page modules readable.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="app-header">
            <h1>{title}</h1>
            {f'<p class="app-subtitle">{subtitle}</p>' if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(metrics: dict) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.metric(label, value)


def render_recommendation_table(df: pd.DataFrame, score_col: str) -> None:
    """Render the recommendations as a styled, sorted dataframe with a
    visual score bar so the table reads as a polished product feature,
    not a raw pandas dump.
    """
    display_df = df.copy()
    if score_col in display_df.columns:
        max_score = display_df[score_col].max() or 1
        display_df["match"] = (display_df[score_col] / max_score).clip(0, 1)
    else:
        display_df["match"] = 1.0

    column_config = {
        "movieId": st.column_config.NumberColumn("ID", width="small"),
        "title": st.column_config.TextColumn("Title", width="large"),
        "genres": st.column_config.TextColumn("Genres", width="medium"),
        "match": st.column_config.ProgressColumn(
            "Match", min_value=0, max_value=1, format=""
        ),
    }
    if score_col in display_df.columns:
        column_config[score_col] = st.column_config.NumberColumn(
            score_col.replace("_", " ").title(), format="%.3f"
        )

    ordered_cols = [c for c in ["movieId", "title", "genres", score_col, "match"]
                    if c in display_df.columns]
    st.dataframe(
        display_df[ordered_cols],
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
    )


def render_download_button(df: pd.DataFrame, filename: str, label: str = "Download as CSV") -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇️ {label}",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def render_explanations(explanations: list[str]) -> None:
    for i, text in enumerate(explanations, start=1):
        with st.container(border=True):
            st.markdown(f"**{i}.** {text}")


def render_error(message: str) -> None:
    st.error(f"⚠️ {message}")


def render_empty_state(message: str) -> None:
    st.info(message)
