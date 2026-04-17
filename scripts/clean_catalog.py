"""
clean_catalog.py

Second-pass cleanup of the processed catalog. Addresses issues surfaced
by the data audit that make_dataset.py's schema-level cleaning does not:

  1. Multiple editions of the same book (same title, same author,
     different ISBN). Keeps the edition with the most ratings.
  2. Rows that share an exact description with another row (publishers
     reuse boilerplate; different books with identical descriptions will
     confuse similarity search).
  3. Descriptions shorter than 100 characters (too sparse to embed
     meaningfully -- raises the floor set in make_dataset.py).
  4. Books with average_rating == 0, which are effectively null ratings.

Usage:
    python scripts/clean_catalog.py \\
        --input data/processed/catalog.parquet \\
        --output data/processed/catalog_clean.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


MIN_DESCRIPTION_LENGTH_STRICT = 100


class CatalogCleaner:
    """Second-pass cleanup on top of make_dataset.py's output."""

    def __init__(self, min_description_length: int = MIN_DESCRIPTION_LENGTH_STRICT):
        self.min_description_length = min_description_length

    def drop_short_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Raise the description-length floor.

        make_dataset.py uses 30 chars to drop obvious nulls; here we go
        stricter because anything below ~100 chars doesn't carry enough
        signal for embedding or TF-IDF to do useful work.
        """
        before = len(df)
        df = df[df["description_length"] >= self.min_description_length].copy()
        logger.info(
            "Dropped %d rows with descriptions < %d chars (%d -> %d)",
            before - len(df),
            self.min_description_length,
            before,
            len(df),
        )
        return df

    def drop_duplicate_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows whose description appears more than once in the catalog.

        Catches publisher boilerplate reuse and residual duplicates that
        the isbn13-level dedup in make_dataset.py misses.
        """
        before = len(df)
        desc_counts = df["description"].value_counts()
        dup_descriptions = desc_counts[desc_counts > 1].index
        df = df[~df["description"].isin(dup_descriptions)].copy()
        logger.info(
            "Dropped %d rows with duplicate descriptions (%d -> %d)",
            before - len(df),
            before,
            len(df),
        )
        return df

    def collapse_editions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collapse multi-edition duplicates to one row per (title, first_author).

        Different ISBNs with the same title and author are almost always
        different editions of the same book. For recommendation these are
        a single "item" -- keeping all editions (a) inflates the catalog,
        (b) causes the top-N to fill with editions of the query book,
        and (c) biases precision@10 upward.

        We keep the edition with the highest ratings_count (most-read
        edition, best metadata signal), breaking ties by most recent year.
        """
        before = len(df)

        df = df.copy()
        df["_first_author_lower"] = df["authors_list"].apply(
            lambda xs: xs[0].lower().strip() if len(xs) > 0 else ""
        )
        df["_title_lower"] = df["title"].str.lower().str.strip()

        # Sort so the winner of each group comes first after dedup.
        df = df.sort_values(
            by=["ratings_count", "published_year"],
            ascending=[False, False],
            na_position="last",
        )
        df = df.drop_duplicates(
            subset=["_title_lower", "_first_author_lower"], keep="first"
        )

        df = df.drop(columns=["_first_author_lower", "_title_lower"])
        df = df.sort_index().reset_index(drop=True)

        logger.info(
            "Collapsed editions: %d -> %d rows (%d edition duplicates removed)",
            before,
            len(df),
            before - len(df),
        )
        return df

    def normalize_zero_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Treat average_rating == 0 as a null rating.

        A book with an average_rating of exactly 0.0 almost certainly has
        no ratings yet, not a genuinely-terrible 0-star average. Setting
        these to NaN prevents the MLP's rating-diff feature from reading
        "distance of 4 stars between the query and this book" when in
        reality the candidate has no rating at all.
        """
        before_zero_count = int((df["average_rating"] == 0).sum())
        df = df.copy()
        df.loc[df["average_rating"] == 0, "average_rating"] = pd.NA
        logger.info("Normalized %d zero-ratings to NaN", before_zero_count)
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full second-pass cleanup."""
        logger.info("Starting second-pass cleanup with %d rows", len(df))
        df = self.drop_short_descriptions(df)
        df = self.drop_duplicate_descriptions(df)
        df = self.collapse_editions(df)
        df = self.normalize_zero_ratings(df)
        logger.info("Second-pass cleanup complete. Final row count: %d", len(df))
        return df


def summarize(df: pd.DataFrame) -> None:
    """Log key stats about the cleaned catalog so the caller can sanity-check."""
    logger.info("=== Cleaned catalog summary ===")
    logger.info("Rows: %d", len(df))
    logger.info("Unique isbn13: %d", df["isbn13"].nunique())
    logger.info("Unique titles: %d", df["title"].nunique())

    logger.info(
        "Length bucket distribution:\n%s",
        df["description_length_bucket"].value_counts().to_string(),
    )
    logger.info(
        "Description length -- mean: %.0f, median: %.0f",
        df["description_length"].mean(),
        df["description_length"].median(),
    )

    all_cats = df["categories_list"].explode()
    logger.info("Unique categories: %d", all_cats.nunique())
    logger.info(
        "Top 10 categories:\n%s",
        all_cats.value_counts().head(10).to_string(),
    )

    # Rating coverage after zero-normalization.
    rated = df["average_rating"].notna().sum()
    logger.info(
        "Books with a valid average_rating: %d (%.1f%%)",
        rated,
        100 * rated / len(df),
    )

    # ratings_count coverage (drives the naive popularity baseline).
    rc_valid = df["ratings_count"].notna().sum()
    logger.info(
        "Books with a valid ratings_count: %d (%.1f%%)",
        rc_valid,
        100 * rc_valid / len(df),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/catalog.parquet"),
        help="Path to the output of make_dataset.py.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/catalog_clean.parquet"),
        help="Output path for the cleaned catalog.",
    )
    p.add_argument(
        "--min-description-length",
        type=int,
        default=MIN_DESCRIPTION_LENGTH_STRICT,
        help="Stricter minimum description length in characters.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input not found: {args.input}. Run make_dataset.py first."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    logger.info("Loaded %d rows from %s", len(df), args.input)

    cleaner = CatalogCleaner(min_description_length=args.min_description_length)
    cleaned = cleaner.clean(df)
    summarize(cleaned)

    cleaned.to_parquet(args.output, index=False)
    logger.info("Wrote cleaned catalog to %s", args.output)


if __name__ == "__main__":
    main()
