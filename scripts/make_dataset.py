"""
make_dataset.py

Load the raw 7k Books with Metadata CSV from data/raw/books.csv,
clean it, parse multi-value fields, and write a processed catalog
ready for feature building and modeling.

The raw CSV is expected to already be present at data/raw/books.csv.

Usage:
    python scripts/make_dataset.py \\
        --input data/raw/books.csv \\
        --output data/processed/catalog.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


# Columns we expect from the raw Kaggle dataset.
REQUIRED_COLUMNS = [
    "isbn13",
    "title",
    "authors",
    "categories",
    "description",
    "thumbnail",
    "published_year",
    "average_rating",
]

# Optional columns — used if present, tolerated if absent.
OPTIONAL_COLUMNS = ["isbn10", "subtitle", "ratings_count", "num_pages"]

# Minimum description length in characters. Below this, embeddings and TF-IDF
# are too sparse to produce meaningful similarity; we drop these rows.
MIN_DESCRIPTION_LENGTH = 30


class CatalogBuilder:
    """Transforms the raw Kaggle CSV into a clean catalog for downstream use."""

    def __init__(self, min_description_length: int = MIN_DESCRIPTION_LENGTH):
        self.min_description_length = min_description_length

    def load(self, path: Path) -> pd.DataFrame:
        """Read the raw CSV and verify required columns are present."""
        logger.info("Loading raw catalog from %s", path)
        df = pd.read_csv(path)

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Raw dataset is missing required columns: {missing}. "
                f"Found columns: {df.columns.tolist()}"
            )

        for col in OPTIONAL_COLUMNS:
            if col not in df.columns:
                logger.warning(
                    "Optional column '%s' not found; downstream code should handle this.",
                    col,
                )

        logger.info("Loaded %d rows, %d columns", len(df), df.shape[1])
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unusable rows and normalize text fields.

        Filtering rules, in order:
          1. Drop rows missing isbn13 (our primary key).
          2. Drop rows missing description or with description shorter than
             the minimum length.
          3. Drop rows missing title or categories.
          4. Deduplicate on isbn13 (keep first).
        """
        n_start = len(df)
        logger.info("Starting clean. %d rows in.", n_start)

        # 1. Primary key must exist.
        df = df.dropna(subset=["isbn13"]).copy()
        df["isbn13"] = df["isbn13"].astype(str).str.strip()
        df = df[df["isbn13"] != ""]
        logger.info("After isbn13 filter: %d rows", len(df))

        # 2. Description must exist and be substantive.
        df = df.dropna(subset=["description"])
        df["description"] = df["description"].astype(str).str.strip()
        df = df[df["description"].str.len() >= self.min_description_length]
        logger.info(
            "After description filter (>= %d chars): %d rows",
            self.min_description_length,
            len(df),
        )

        # 3. Title and categories must exist.
        df = df.dropna(subset=["title", "categories"])
        df["title"] = df["title"].astype(str).str.strip()
        df["categories"] = df["categories"].astype(str).str.strip()
        df = df[(df["title"] != "") & (df["categories"] != "")]
        logger.info("After title/categories filter: %d rows", len(df))

        # 4. Deduplicate on primary key.
        n_before_dedup = len(df)
        df = df.drop_duplicates(subset=["isbn13"], keep="first")
        logger.info(
            "After dedup on isbn13: %d rows (%d duplicates dropped)",
            len(df),
            n_before_dedup - len(df),
        )

        logger.info("Clean complete. %d -> %d rows (%.1f%% kept).",
                    n_start, len(df), 100 * len(df) / n_start)
        return df.reset_index(drop=True)

    def parse_multi_value_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split semicolon-delimited author and category fields into lists.

        Also lowercases and strips whitespace on each element.
        """
        def split_clean(s: str) -> list[str]:
            if not isinstance(s, str):
                return []
            return [part.strip() for part in s.split(";") if part.strip()]

        def split_clean_lower(s: str) -> list[str]:
            return [part.lower() for part in split_clean(s)]

        df = df.copy()
        df["authors_list"] = df["authors"].fillna("").apply(split_clean)
        df["categories_list"] = df["categories"].fillna("").apply(split_clean_lower)

        # Drop rows that ended up with no categories after parsing.
        before = len(df)
        df = df[df["categories_list"].apply(len) > 0].reset_index(drop=True)
        logger.info(
            "Dropped %d rows with no parsable categories; %d remain.",
            before - len(df),
            len(df),
        )

        return df

    def add_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fields needed downstream (length buckets, popularity fallback)."""
        df = df.copy()

        df["description_length"] = df["description"].str.len()

        def bucket(n: int) -> str:
            if n < 200:
                return "short"
            if n < 500:
                return "medium"
            return "long"

        df["description_length_bucket"] = df["description_length"].apply(bucket)

        # Popularity signal: prefer ratings_count if present; otherwise fall
        # back to average_rating. The naive baseline and MLP popularity
        # feature both consume this column.
        if "ratings_count" in df.columns:
            df["popularity"] = pd.to_numeric(
                df["ratings_count"], errors="coerce"
            ).fillna(0)
            df["popularity_source"] = "ratings_count"
        else:
            logger.warning(
                "ratings_count missing; falling back to average_rating for popularity. "
                "Naive baseline will be rating-based rather than volume-based."
            )
            df["popularity"] = pd.to_numeric(
                df["average_rating"], errors="coerce"
            ).fillna(0)
            df["popularity_source"] = "average_rating"

        # Coerce published_year and average_rating to numeric.
        df["published_year"] = pd.to_numeric(
            df["published_year"], errors="coerce"
        )
        df["average_rating"] = pd.to_numeric(
            df["average_rating"], errors="coerce"
        )

        return df

    def build(self, input_path: Path) -> pd.DataFrame:
        """Full pipeline: load -> clean -> parse -> derive."""
        df = self.load(input_path)
        df = self.clean(df)
        df = self.parse_multi_value_fields(df)
        df = self.add_derived_fields(df)
        return df


def summarize(df: pd.DataFrame) -> None:
    """Log key stats about the processed catalog so the caller can sanity-check."""
    logger.info("=== Processed catalog summary ===")
    logger.info("Rows: %d", len(df))
    logger.info("Unique isbn13: %d", df["isbn13"].nunique())
    logger.info(
        "Description length — mean: %.0f, median: %.0f, min: %d, max: %d",
        df["description_length"].mean(),
        df["description_length"].median(),
        df["description_length"].min(),
        df["description_length"].max(),
    )
    logger.info(
        "Length bucket distribution:\n%s",
        df["description_length_bucket"].value_counts().to_string(),
    )

    # Unique categories (flattened).
    all_cats = df["categories_list"].explode()
    logger.info("Unique categories: %d", all_cats.nunique())
    logger.info("Top 10 categories:\n%s", all_cats.value_counts().head(10).to_string())

    logger.info("Thumbnail coverage: %.1f%%",
                100 * df["thumbnail"].notna().mean())
    logger.info("Popularity source: %s", df["popularity_source"].iloc[0])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/books.csv"),
        help="Path to raw CSV (expected at data/raw/books.csv).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/catalog.parquet"),
        help="Output path (parquet recommended; .csv also supported).",
    )
    p.add_argument(
        "--min-description-length",
        type=int,
        default=MIN_DESCRIPTION_LENGTH,
        help="Minimum description length in characters.",
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
        raise FileNotFoundError(f"Input not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    builder = CatalogBuilder(min_description_length=args.min_description_length)
    catalog = builder.build(args.input)
    summarize(catalog)

    # Write output. Parquet preserves list columns (authors_list, categories_list)
    # natively; CSV would stringify them and break downstream parsing.
    if args.output.suffix == ".parquet":
        catalog.to_parquet(args.output, index=False)
    elif args.output.suffix == ".csv":
        logger.warning(
            "Writing to CSV will serialize list columns as strings; "
            "downstream code must re-parse them. Prefer parquet."
        )
        catalog.to_csv(args.output, index=False)
    else:
        raise ValueError(f"Unsupported output extension: {args.output.suffix}")

    logger.info("Wrote processed catalog to %s", args.output)


if __name__ == "__main__":
    main()
