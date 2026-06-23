# 🎬 Movie Recommender System — Content-Based, Collaborative & Hybrid Filtering (CLI + Streamlit Web App)

A production-style **Recommendation Engine** built in Python, implementing three recommendation paradigms from scratch on top of `scikit-learn`: **Content-Based Filtering**, **Collaborative Filtering (Matrix Factorization via SVD)**, and a **Hybrid model** that blends both. The project ships with both a **command-line interface** and a **Streamlit web application**, sharing 100% of the same underlying business logic — clean OOP architecture, a validated data pipeline, automated tests, and offline evaluation metrics (Precision@K / Recall@K).

> Built as part of an ML/Data Science internship task. Designed to demonstrate genuine understanding of recommender system theory **and** software engineering / product practice — not just a `df.corr()` script.

🔗 **Live demo:** *(deploy via Streamlit Community Cloud — see [Deployment](#-deployment) below)*

---

## 📌 Project Overview

Recommender systems power Netflix, Amazon, Spotify and YouTube. This project re-implements the three foundational approaches used in industry, on MovieLens-style movie rating data, and exposes them through both a scriptable CLI and a polished web UI:

| Approach | Idea | Strength | Weakness |
|---|---|---|---|
| **Content-Based** | Recommend items *similar* to what a user already liked, based on item features (genres) | Works for new items, explainable | Over-specializes, can't capture "wisdom of the crowd" |
| **Collaborative Filtering** | Recommend items liked by users with *similar rating behaviour* | Captures latent taste patterns beyond explicit features | Fails for new users/items (cold start) |
| **Hybrid** | Weighted combination of both | Mitigates both weaknesses above | Slightly more complex to tune |

All three are implemented, evaluated, and compared empirically in this repo (see [Evaluation Metrics](#-evaluation-metrics) below), and explorable interactively in the Streamlit app.

---

## ✨ Features

**Recommendation engine (`src/`)**
- ✅ **Content-Based Filtering** — TF-IDF genre vectors + cosine similarity
- ✅ **Collaborative Filtering** — Matrix Factorization via Truncated SVD (+ optional user-based KNN strategy)
- ✅ **Hybrid Recommender** — tunable weighted-sum blend (`alpha` parameter)
- ✅ **Full data preprocessing pipeline** — schema validation, cleaning, deduplication, sparsity reporting
- ✅ **Feature engineering** — one-hot genre matrix, TF-IDF vectorization, user-item pivot matrix
- ✅ **Cold-start handling** — new users can get recommendations from genre preferences alone, no rating history needed
- ✅ **Top-N recommendation generation** for any user, any strategy
- ✅ **Recommendation explanations** — human-readable "why was this recommended" output for every strategy
- ✅ **Object-Oriented, modular architecture** — Facade pattern (`RecommenderSystem`) over independently testable model classes
- ✅ **Custom exception hierarchy** + input validation throughout (no silent failures)
- ✅ **Evaluation suite** — Precision@K and Recall@K, computed on a proper per-user train/test split
- ✅ **CSV export** of every recommendation run
- ✅ **15 automated unit tests** (`pytest`)
- ✅ Zero paid APIs — runs 100% locally with open-source libraries

**Streamlit web application (`app.py`, `ui/`)**
- ✅ **Modern, themed UI** (custom CSS + Streamlit theme config)
- ✅ **Sidebar navigation** between Home / Recommend / Cold Start / Evaluation / About
- ✅ **Genre selection dropdown** (multi-select) for cold-start users
- ✅ **User ID input** with live valid-range hint
- ✅ **Recommendation strategy selector** (Content / Collaborative / Hybrid)
- ✅ **Top-N slider**
- ✅ **Polished recommendation table** with progress-bar "match" column
- ✅ **Download recommendations as CSV** button on every results screen
- ✅ **Recommendation explanations** rendered as readable cards
- ✅ **Graceful error handling** — invalid users, empty genre selection, bad file paths all surface as friendly UI messages, never a stack trace
- ✅ **OOP architecture** — `StreamlitApp` controller + `Page` interface (`HomePage`, `RecommendPage`, `ColdStartPage`, `EvaluationPage`, `AboutPage`), all reusing the same `RecommenderSystem` facade as the CLI
- ✅ Cached model loading (`st.cache_resource`) so retraining only happens when settings change

---

## 📁 Folder Structure

```
movie-recommender-system/
│
├── app.py                          # 🚀 Streamlit entry point: streamlit run app.py
│
├── ui/                              # Streamlit presentation layer (OOP)
│   ├── __init__.py
│   ├── pages.py                    # Page interface + HomePage/RecommendPage/ColdStartPage/EvaluationPage/AboutPage
│   ├── components.py               # Reusable styled widgets (tables, headers, download buttons, explanations)
│   └── resource_loader.py          # st.cache_resource-wrapped RecommenderSystem loader
│
├── assets/
│   └── style.css                   # Custom CSS for a modern, non-default Streamlit look
│
├── .streamlit/
│   └── config.toml                 # Theme + server configuration
│
├── data/
│   └── sample/
│       ├── movies.csv              # MovieLens-schema sample catalog (40 movies)
│       └── ratings.csv             # 830 synthetic-but-realistic ratings, 60 users
│
├── src/                             # Core recommendation engine (used by BOTH app.py and main.py)
│   ├── __init__.py
│   ├── data_preprocessing.py       # DataPreprocessor: load, clean, validate, feature-engineer
│   ├── content_based.py            # ContentBasedRecommender (TF-IDF + cosine similarity)
│   ├── collaborative_filtering.py  # CollaborativeFilteringRecommender (SVD / user-based)
│   ├── hybrid.py                   # HybridRecommender (weighted blend)
│   ├── evaluation.py               # Evaluator: Precision@K, Recall@K
│   ├── recommender_system.py       # RecommenderSystem facade — single entry point
│   ├── exceptions.py               # Custom exception hierarchy + validation helper
│   └── logger.py                   # Centralised logging configuration
│
├── scripts/
│   └── generate_sample_data.py     # Generates the bundled sample dataset
│
├── tests/
│   ├── __init__.py
│   └── test_recommender.py         # 15 pytest unit tests (engine-level, framework-agnostic)
│
├── outputs/
│   ├── sample_recommendations_user3_hybrid.csv
│   ├── sample_recommendations_user3_content.csv
│   ├── sample_recommendations_user3_collaborative.csv
│   ├── sample_recommendations_newuser.csv
│   └── evaluation_report.csv
│
├── main.py                         # CLI entry point (interactive + scriptable)
├── requirements.txt
├── .gitignore
└── README.md
```

**Key design point:** `app.py` and `main.py` are two independent, thin entry points over the exact same `src/` engine. No recommendation logic is duplicated or reimplemented for the web app — `ui/resource_loader.py` simply caches an instance of the same `RecommenderSystem` class used everywhere else.

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit web app
```bash
streamlit run app.py
```
This opens the app in your browser at `http://localhost:8501`. Use the sidebar to:
1. Pick a page (Home / Get Recommendations / New User / Model Evaluation / About)
2. Adjust dataset paths or the hybrid blend weight under **⚙️ Data & Model Settings**
3. Enter a User ID, pick a strategy and Top-N, and click **Generate Recommendations**

### 3. Run the CLI (unchanged from the original project)
```bash
# Interactive mode
python main.py

# Scriptable mode
python main.py --user-id 3 --strategy hybrid --top-n 10 --explain
python main.py --new-user --genres Action Comedy --top-n 10
python main.py --evaluate --strategy hybrid --k 10
```

### 4. Run the test suite
```bash
pytest tests/ -v
```

### 5. Use a real, larger MovieLens dataset
This project ships with a small synthetic-but-realistic sample dataset so it runs instantly with no setup. To use real data, download **ml-latest-small** or **ml-25m** from [grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens/), then either:
- point the CLI at the real files: `python main.py --movies path/to/movies.csv --ratings path/to/ratings.csv`, or
- type the new paths into the **⚙️ Data & Model Settings** panel in the Streamlit sidebar.

The pipeline is schema-compatible — no code changes needed either way.

---

## 🌐 Deployment

### Deploy to Streamlit Community Cloud (free)
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repo, branch `main`, and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud will install everything from `requirements.txt` automatically.
5. The bundled `data/sample/` CSVs are committed to the repo, so the deployed app works immediately with zero extra configuration.

### Run in Docker (optional, for any other host)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
```bash
docker build -t movie-recommender .
docker run -p 8501:8501 movie-recommender
```

### Run on any other PaaS (Render, Railway, Heroku, etc.)
Set the start command to:
```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

---

## 📊 Dataset Description

The bundled sample dataset (`data/sample/`) follows the **exact MovieLens schema**:

- `movies.csv` → `movieId, title, genres` (pipe-separated genres, e.g. `Action|Adventure|Sci-Fi`)
- `ratings.csv` → `userId, movieId, rating, timestamp` (ratings on a 0.5–5.0 scale)

It contains **40 movies** spanning 18 genres and **~830 ratings** from **60 synthetic users**, each generated with a randomized "genre taste profile" so the ratings carry real, learnable structure (not pure noise) — this is what makes the collaborative and content-based models actually converge on sensible recommendations. Regenerate or resize it any time with:
```bash
python scripts/generate_sample_data.py
```

---

## 🧠 Algorithm Explanation

### Content-Based Filtering
1. Each movie's `genres` string is vectorized using **TF-IDF** (Term Frequency–Inverse Document Frequency), treating each genre like a "word" in a document.
2. TF-IDF down-weights genres that appear on almost every movie (e.g. "Drama") and up-weights rarer, more distinctive genres — producing a sharper similarity signal than plain one-hot encoding.
3. **Cosine similarity** is computed between every pair of movies, producing an item-item similarity matrix.
4. A user's **taste profile** is built by averaging the similarity vectors of the movies they rated highly, weighted by their rating.
5. The unseen movies with the highest similarity to that profile are returned as Top-N recommendations.

### Collaborative Filtering
1. Ratings are pivoted into a sparse **user-item matrix**.
2. Missing entries are imputed with each user's own mean rating (avoids the false signal of treating "unrated" as "rated zero").
3. **Truncated SVD** (matrix factorization) decomposes the matrix into low-rank **user-latent** and **item-latent** factor matrices — discovering hidden taste dimensions (e.g. "dark slow-burn dramas") that aren't explicit in any single feature.
4. Multiplying the factors back together reconstructs a **predicted rating** for every (user, movie) pair, including unseen ones.
5. An alternative **user-based KNN** strategy is also implemented: cosine similarity between users' raw rating vectors, with predictions as the rating-weighted average of the top-K most similar users.

### Hybrid Recommendation
- Both models' scores are **min-max normalized** to [0, 1] (they live on different native scales: cosine similarity vs. 1–5 star predicted rating).
- Combined via a weighted sum: `hybrid_score = α · content_score + (1−α) · collaborative_score`.
- `α` is a tunable hyperparameter — `α=1.0` reduces to pure content-based, `α=0.0` to pure collaborative.
- This directly addresses the **cold-start problem**: brand-new items with few ratings still get a usable content-based score, while established items benefit from the community signal collaborative filtering provides.

### Similarity Measures Used
- **Cosine similarity** — measures the angle between two vectors, ignoring magnitude. Used for both item-item (content-based) and user-user (collaborative) similarity since rating/feature vectors are sparse and direction matters more than scale.
- **Pearson-correlation-style mean-centering** — used implicitly in the collaborative pipeline via per-user mean imputation, to correct for users who simply tend to rate everything higher/lower than average.

### Content-Based vs. Collaborative — Key Differences
| | Content-Based | Collaborative |
|---|---|---|
| Signal used | Item features (genres) | User-user / item-item rating patterns |
| Cold start (new item) | ✅ Handles well | ❌ Cannot recommend until rated |
| Cold start (new user) | ✅ Handles via stated preferences | ❌ Needs rating history |
| Diversity | ❌ Can over-specialize | ✅ Can surface serendipitous items |
| Needs other users' data? | No | Yes |

---

## 📐 Evaluation Metrics

Implemented in `src/evaluation.py`, computed on a **per-user train/test split** (80/20) so every user appears in both splits:

- **Precision@K** = (relevant items in top-K) / K — *"Of what we recommended, how much was actually wanted?"*
- **Recall@K** = (relevant items in top-K) / (total relevant items) — *"Of everything the user actually liked, how much did we surface?"*
- "Relevant" = test-set items rated ≥ 3.5/5 by that user.

### Sample results on the bundled dataset (K=10)
*(see `outputs/evaluation_report.csv` — regenerate via `python main.py --evaluate --strategy <name> --k 10`)*

| Strategy | Precision@10 | Recall@10 |
|---|---|---|
| Content-Based | 0.0697 | 0.5606 |
| Collaborative Filtering | 0.0545 | 0.4242 |
| **Hybrid** | **0.0667** | **0.5303** |

> Note: absolute scores depend heavily on dataset size/sparsity — the bundled sample dataset is intentionally tiny for fast demos. Re-run on the full MovieLens ml-latest-small (100k ratings) for production-representative numbers.

---

## ⏱️ Complexity Discussion

| Component | Time Complexity | Notes |
|---|---|---|
| TF-IDF vectorization | O(n_movies × n_genres) | One-time, at `fit()` |
| Content item-item similarity | O(n_movies²) | Precomputed once; fine up to tens of thousands of items |
| SVD factorization | O(n_users × n_movies × n_factors) per iteration (truncated, so far cheaper than full SVD's O(min(u,m)²·max(u,m))) | Scales to large sparse matrices much better than pairwise similarity |
| User-based KNN | O(n_users²) for the similarity matrix | Becomes a bottleneck at large user counts — SVD strategy is preferred at scale |
| Hybrid recommend | O(candidate_pool) | Linear in the merged candidate set, negligible overhead |
| Precision/Recall@K | O(n_test_users × K) | Cheap; dominated by the underlying model's `recommend()` cost |

**Scalability takeaway:** content-based similarity and user-based CF both scale quadratically in catalog/user size respectively — fine for this project's scope, but the reason real-world systems (Netflix, Spotify) lean on matrix factorization / deep embeddings + approximate nearest-neighbour search (e.g. FAISS, Annoy) at scale.

---

## 🖥️ Sample Output

### Streamlit Web App
The **Get Recommendations** page lets you pick a User ID, strategy, and Top-N from dropdowns/sliders, then renders a styled table with a "match" progress bar per row, a one-click **⬇️ Download as CSV** button, and expandable explanation cards underneath — no terminal required.

The **Model Evaluation** page re-fits all three strategies on a train split live and renders Precision@K / Recall@K side-by-side as both a table and a bar chart.

### CLI
```
$ python main.py --user-id 3 --strategy hybrid --top-n 5 --explain

 movieId                  title                       genres  content_score  predicted_rating  hybrid_score
      22      Fight Club (1999)               Drama|Thriller         0.462              3.35         0.902
      34           Joker (2019)        Crime|Drama|Thriller         0.558              3.16         0.780
      23       Gladiator (2000)      Action|Adventure|Drama         0.422              3.15         0.636
      13       Pulp Fiction (1994) Comedy|Crime|Drama|Thriller       0.562              3.00         0.602
      15              Speed (1994)     Action|Romance|Thriller       0.404              3.10         0.562

Explanations:
- [Hybrid explanation, alpha=0.5]
  Content-based signal: 'Fight Club (1999)' (Drama|Thriller) was recommended because it shares
  strong genre similarity (score=0.78) with 'Se7en (1995)', which you rated highly.
  Collaborative signal: 'Fight Club (1999)' was recommended because the matrix-factorization
  model predicts you would rate it 3.35/5, based on latent taste patterns learned from users
  with similar rating behaviour.
```
Full CSVs of these runs are saved under `outputs/`.

---

## 🔮 Future Enhancements

- [ ] Swap TF-IDF genre vectors for **movie plot embeddings** (e.g. Sentence-BERT) for richer content-based similarity
- [ ] Implement **implicit feedback** modelling (clicks/watch-time) alongside explicit ratings
- [ ] Add **deep learning** collaborative filtering (Neural Collaborative Filtering / autoencoders)
- [ ] Introduce **time-aware** recommendations (recency-weighted ratings, trending items)
- [ ] Add **diversity & novelty** re-ranking to counter content-based filter-bubble effects
- [ ] Wrap the `RecommenderSystem` facade in a **FastAPI** service to power both the Streamlit UI and external integrations
- [ ] Add **A/B testing simulation** to compare strategies on simulated user feedback loops
- [ ] Scale similarity search using **approximate nearest neighbours** (FAISS/Annoy) for large catalogs
- [ ] Add **movie poster thumbnails** (via TMDB's free API) to the Streamlit recommendation table
- [ ] Add **user authentication** so each visitor's ratings/preferences persist across sessions

---

## 🛠️ Tech Stack

`Python 3.10+` · `pandas` · `NumPy` · `scikit-learn` (TF-IDF, cosine similarity, TruncatedSVD) · `Streamlit` (web UI) · `pytest`

No paid APIs, no external recommendation services — fully self-contained and reproducible, locally or deployed.

---

## 📄 License

MIT — free to use, modify, and showcase in your own portfolio.
