"""
generate_sample_data.py
------------------------
Generates a small, realistic, MovieLens-style sample dataset so the
project is runnable out-of-the-box without requiring an external download.

For production-scale results, download the full MovieLens dataset
(ml-latest-small or ml-25m) from https://grouplens.org/datasets/movielens/
and drop `movies.csv` and `ratings.csv` into the `data/` folder. The
pipeline code is 100% compatible with the official MovieLens schema:
    movies.csv  -> movieId, title, genres
    ratings.csv -> userId, movieId, rating, timestamp
"""

import random
import csv
import os

random.seed(42)

MOVIES = [
    (1, "Toy Story (1995)", "Adventure|Animation|Children|Comedy|Fantasy"),
    (2, "Jumanji (1995)", "Adventure|Children|Fantasy"),
    (3, "Heat (1995)", "Action|Crime|Thriller"),
    (4, "GoldenEye (1995)", "Action|Adventure|Thriller"),
    (5, "Braveheart (1995)", "Action|Drama|War"),
    (6, "Apollo 13 (1995)", "Adventure|Drama|IMAX"),
    (7, "Pocahontas (1995)", "Animation|Children|Drama|Musical|Romance"),
    (8, "The Usual Suspects (1995)", "Crime|Mystery|Thriller"),
    (9, "Se7en (1995)", "Mystery|Thriller"),
    (10, "Star Wars: Episode IV (1977)", "Action|Adventure|Sci-Fi"),
    (11, "Forrest Gump (1994)", "Comedy|Drama|Romance|War"),
    (12, "The Lion King (1994)", "Adventure|Animation|Children|Drama|Musical"),
    (13, "Pulp Fiction (1994)", "Comedy|Crime|Drama|Thriller"),
    (14, "The Shawshank Redemption (1994)", "Crime|Drama"),
    (15, "Speed (1994)", "Action|Romance|Thriller"),
    (16, "Interstellar (2014)", "Adventure|Drama|Sci-Fi|IMAX"),
    (17, "The Dark Knight (2008)", "Action|Crime|Drama|Thriller"),
    (18, "Inception (2010)", "Action|Sci-Fi|Thriller|IMAX"),
    (19, "Up (2009)", "Adventure|Animation|Children|Drama"),
    (20, "WALL-E (2008)", "Adventure|Animation|Children|Sci-Fi"),
    (21, "The Matrix (1999)", "Action|Sci-Fi"),
    (22, "Fight Club (1999)", "Drama|Thriller"),
    (23, "Gladiator (2000)", "Action|Adventure|Drama"),
    (24, "Finding Nemo (2003)", "Adventure|Animation|Children|Comedy"),
    (25, "The Incredibles (2004)", "Action|Adventure|Animation|Children|Comedy"),
    (26, "Spirited Away (2001)", "Animation|Adventure|Fantasy"),
    (27, "The Notebook (2004)", "Drama|Romance"),
    (28, "La La Land (2016)", "Comedy|Drama|Musical|Romance"),
    (29, "Mad Max: Fury Road (2015)", "Action|Adventure|Sci-Fi"),
    (30, "Whiplash (2014)", "Drama|Music"),
    (31, "Coco (2017)", "Adventure|Animation|Children|Comedy|Fantasy"),
    (32, "Avengers: Endgame (2019)", "Action|Adventure|Sci-Fi"),
    (33, "Parasite (2019)", "Comedy|Drama|Thriller"),
    (34, "Joker (2019)", "Crime|Drama|Thriller"),
    (35, "Soul (2020)", "Animation|Adventure|Children|Comedy|Fantasy|Musical"),
    (36, "Dune (2021)", "Adventure|Drama|Sci-Fi|IMAX"),
    (37, "Top Gun: Maverick (2022)", "Action|Drama"),
    (38, "Everything Everywhere All at Once (2022)", "Action|Adventure|Comedy|Sci-Fi"),
    (39, "Oppenheimer (2023)", "Biography|Drama|History"),
    (40, "Barbie (2023)", "Adventure|Comedy|Fantasy"),
]

N_USERS = 60
RATING_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# Give each synthetic user a genre "taste profile" so ratings are not random
# noise -- this makes content-based and collaborative filtering behave
# realistically on the sample data.
GENRE_POOL = sorted(set(g for _, _, genres in MOVIES for g in genres.split("|")))


def build_user_profiles(n_users):
    profiles = []
    for uid in range(1, n_users + 1):
        liked_genres = random.sample(GENRE_POOL, k=random.randint(2, 4))
        profiles.append((uid, liked_genres))
    return profiles


def simulate_rating(user_genres, movie_genres):
    overlap = len(set(user_genres) & set(movie_genres.split("|")))
    base = 2.5 + overlap * 0.7
    noise = random.uniform(-0.8, 0.8)
    rating = max(0.5, min(5.0, base + noise))
    # round to nearest 0.5 (MovieLens style)
    return round(rating * 2) / 2


def generate():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
    os.makedirs(out_dir, exist_ok=True)

    movies_path = os.path.join(out_dir, "movies.csv")
    ratings_path = os.path.join(out_dir, "ratings.csv")

    with open(movies_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["movieId", "title", "genres"])
        for movie_id, title, genres in MOVIES:
            writer.writerow([movie_id, title, genres])

    profiles = build_user_profiles(N_USERS)
    rows = []
    timestamp = 1_000_000_000
    for uid, liked_genres in profiles:
        n_rated = random.randint(8, 20)
        rated_movies = random.sample(MOVIES, k=min(n_rated, len(MOVIES)))
        for movie_id, _, genres in rated_movies:
            rating = simulate_rating(liked_genres, genres)
            timestamp += random.randint(1000, 100000)
            rows.append([uid, movie_id, rating, timestamp])

    with open(ratings_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["userId", "movieId", "rating", "timestamp"])
        writer.writerows(rows)

    print(f"Generated {len(MOVIES)} movies -> {movies_path}")
    print(f"Generated {len(rows)} ratings from {N_USERS} users -> {ratings_path}")


if __name__ == "__main__":
    generate()
