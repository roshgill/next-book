"""
main.py

Command-line interface for the next-book recommendation system.
Lets you query any of the three models directly from the terminal
without starting the web server.

Usage:
    python main.py --query "Gilead"
    python main.py --query "Gilead" --model classical
    python main.py --query "9780002005883" --model naive --k 5

Attribution: AI-assisted code (Claude Sonnet 4.6, claude.ai/code)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from scripts.predict import VALID_MODELS, BookRecommender


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--query",
        required=True,
        help="Book title (substring match) or isbn13 to use as the query.",
    )
    p.add_argument(
        "--model",
        choices=VALID_MODELS,
        default="deep",
        help="Which recommender to use (default: deep).",
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of recommendations to return (default: 10).",
    )
    p.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/processed/catalog_clean.parquet"),
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models/"),
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()

    print(f"\nLoading recommender ({args.model})…")
    rec = BookRecommender(
        model_name=args.model,
        catalog_path=args.catalog,
        models_dir=args.models_dir,
    )

    resolved = rec.resolve_isbn(args.query)
    if resolved is None:
        matches = rec.search_titles(args.query, limit=5)
        if matches:
            print(f"\nNo exact match for {args.query!r}. Did you mean:")
            for isbn, title in matches:
                print(f"  {title}  (isbn13: {isbn})")
        else:
            print(f"\nNo match found for {args.query!r}.")
        return

    query_book = rec.get_book(resolved)
    print(f"\nQuery: {query_book.title!r} by {query_book.authors}")
    print(f"       [{query_book.categories}]  rating={query_book.average_rating}  year={query_book.published_year}")
    print(f"\nTop {args.k} recommendations (model={args.model}):")
    print("─" * 70)

    for i, r in enumerate(rec.recommend_with_metadata(resolved, k=args.k), 1):
        rating = f"{r.average_rating:.2f}" if r.average_rating is not None else "n/a"
        year = str(r.published_year) if r.published_year is not None else "n/a"
        print(f"{i:2d}. {r.title!r} by {r.authors}")
        print(f"    [{r.categories}]  rating={rating}  year={year}")
        print(f"    {r.description[:120]}…")
        print()


if __name__ == "__main__":
    main()
