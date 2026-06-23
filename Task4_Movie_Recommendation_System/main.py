"""
main.py
--------
Command-line entry point for the Movie Recommender System.

Usage examples
--------------
Interactive mode (prompts for userId / preferences):
    python main.py

Non-interactive mode (for scripting / demos):
    python main.py --user-id 5 --strategy hybrid --top-n 10
    python main.py --new-user --genres Action Comedy --top-n 10
    python main.py --evaluate --strategy hybrid --k 10
"""

from __future__ import annotations

import argparse
import sys

from src.recommender_system import RecommenderSystem
from src.exceptions import RecommenderError
from src.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MOVIES_PATH = "data/sample/movies.csv"
DEFAULT_RATINGS_PATH = "data/sample/ratings.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Movie Recommender System")
    parser.add_argument("--movies", default=DEFAULT_MOVIES_PATH, help="Path to movies CSV")
    parser.add_argument("--ratings", default=DEFAULT_RATINGS_PATH, help="Path to ratings CSV")
    parser.add_argument("--user-id", type=int, help="Existing userId to recommend for")
    parser.add_argument("--strategy", choices=["content", "collaborative", "hybrid"],
                        default="hybrid", help="Recommendation strategy")
    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations")
    parser.add_argument("--new-user", action="store_true",
                        help="Cold-start flow: recommend by genre preference only")
    parser.add_argument("--genres", nargs="+", help="Preferred genres for --new-user mode")
    parser.add_argument("--evaluate", action="store_true", help="Run Precision@K / Recall@K evaluation")
    parser.add_argument("--k", type=int, default=10, help="K for evaluation metrics")
    parser.add_argument("--output", default="outputs/recommendations.csv",
                        help="CSV path to save recommendations")
    parser.add_argument("--explain", action="store_true",
                        help="Print an explanation for each recommended movie")
    return parser.parse_args()


def interactive_prompt(system: RecommenderSystem):
    print("\n=== Movie Recommender System (Interactive Mode) ===")
    print("1. Get recommendations for an existing user")
    print("2. Get recommendations as a new user (cold start, by genre)")
    print("3. Run evaluation (Precision@K / Recall@K)")
    print("4. Exit")
    choice = input("Select an option [1-4]: ").strip()

    if choice == "1":
        known = system.known_users()
        print(f"Known user IDs range: {min(known)}-{max(known)} ({len(known)} users)")
        user_id = int(input("Enter userId: ").strip())
        strategy = input("Strategy [content/collaborative/hybrid] (default hybrid): ").strip() or "hybrid"
        top_n = int(input("How many recommendations? (default 10): ").strip() or 10)
        recs = system.recommend(user_id, strategy=strategy, top_n=top_n)
        print(recs.to_string(index=False))
        path = system.save_recommendations_to_csv(recs, "outputs/recommendations.csv")
        print(f"\nSaved to {path}")

        if input("Show explanations? [y/N]: ").strip().lower() == "y":
            for movie_id in recs["movieId"]:
                print("-", system.explain(user_id, movie_id, strategy=strategy))

    elif choice == "2":
        genres_input = input("Enter favourite genres, comma-separated (e.g. Action,Comedy): ")
        genres = [g.strip() for g in genres_input.split(",") if g.strip()]
        top_n = int(input("How many recommendations? (default 10): ").strip() or 10)
        recs = system.recommend_for_new_user(genres, top_n=top_n)
        print(recs.to_string(index=False))
        path = system.save_recommendations_to_csv(recs, "outputs/new_user_recommendations.csv")
        print(f"\nSaved to {path}")

    elif choice == "3":
        strategy = input("Strategy [content/collaborative/hybrid] (default hybrid): ").strip() or "hybrid"
        k = int(input("K (default 10): ").strip() or 10)
        results = system.evaluate(k=k, strategy=strategy)
        print(results)

    else:
        print("Goodbye!")
        sys.exit(0)


def main():
    args = parse_args()

    print("Building recommender system (loading data, fitting models)...")
    system = RecommenderSystem(args.movies, args.ratings).build()
    print("Ready.\n")

    try:
        if args.evaluate:
            results = system.evaluate(k=args.k, strategy=args.strategy)
            print(f"Evaluation results ({args.strategy}, k={args.k}): {results}")
            return

        if args.new_user:
            if not args.genres:
                print("Error: --new-user requires --genres GENRE1 GENRE2 ...")
                sys.exit(1)
            recs = system.recommend_for_new_user(args.genres, top_n=args.top_n)
            print(recs.to_string(index=False))
            system.save_recommendations_to_csv(recs, args.output)
            print(f"\nSaved to {args.output}")
            return

        if args.user_id is not None:
            recs = system.recommend(args.user_id, strategy=args.strategy, top_n=args.top_n)
            print(recs.to_string(index=False))
            system.save_recommendations_to_csv(recs, args.output)
            print(f"\nSaved to {args.output}")

            if args.explain:
                print("\nExplanations:")
                for movie_id in recs["movieId"]:
                    print("-", system.explain(args.user_id, movie_id, strategy=args.strategy))
            return

        # No flags given -> drop into interactive mode
        interactive_prompt(system)

    except RecommenderError as exc:
        logger.error("Recommender error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
