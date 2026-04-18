"""
evaluate.py

Runs all three recommenders over a held-out evaluation set and
computes the three headline metrics defined in CLAUDE.md:

    Precision@10  -- fraction of recs that are "relevant" to the query
                     under the compound proxy (shared category AND
                     |rating_diff| < 0.3).
    Intra-List Diversity (ILD)
                  -- avg pairwise category dissimilarity across the 10
                     recommendations. Counterbalances precision (a
                     system returning 10 near-duplicates scores high on
                     precision but low on diversity).
    Catalog Coverage
                  -- share of the catalog that ever appears in any
                     top-10 across the eval set. Penalizes recommenders
                     that collapse to the same handful of books (the
                     naive popularity baseline should score worst by
                     construction).

Also runs the locked experiment: Precision@10 stratified by query
description-length bucket (short / medium / long), for each of the
three models. Produces one chart and a CSV.

Outputs written to ``data/outputs/``:
    metrics.csv              -- one row per model, three columns of metrics
    metrics.md               -- markdown table, paste-ready for the report
    experiment.csv           -- one row per (model, length_bucket)
    experiment.png           -- chart of Precision@10 vs. bucket per model

Usage:
    python -m scripts.evaluate
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.predict import BookRecommender, VALID_MODELS


logger = logging.getLogger(__name__)


# Eval set size. 500 books gives tight enough confidence intervals for
# the precision metric (~+/- 2-3 percentage points at a typical rate)
# while keeping total runtime under a minute.
DEFAULT_EVAL_SIZE = 500

# Compound relevance proxy threshold. Must match the value used during
# training pair generation in model.py.
RATING_DIFF_THRESHOLD = 0.3

# Random seed for reproducibility of the eval-set sample.
RANDOM_SEED = 42

# Length buckets for the experiment, in characters. Must match the
# buckets used by make_dataset.py so the precomputed
# description_length_bucket column aligns.
LENGTH_BUCKETS = ("short", "medium", "long")


# ---------------------------------------------------------------------------
# Relevance proxy + metric helpers
# ---------------------------------------------------------------------------

class RelevanceProxy:
    """Encapsulates the compound relevance rule used for Precision@10.

    Two books are considered relevant to each other iff:
      1. They share at least one category, AND
      2. Their average ratings differ by less than ``rating_threshold``.

    If either book has a missing rating, the rating-proximity clause is
    considered satisfied (we fall back to the category-only signal
    rather than dropping the pair, which would bias the evaluation).
    """

    def __init__(
        self, catalog: pd.DataFrame, rating_threshold: float = RATING_DIFF_THRESHOLD
    ):
        self.rating_threshold = rating_threshold

        # Pre-extract per-row lookups for speed. A 500-query * 10-rec =
        # 5000-pair loop is small, but nice-to-have cleanliness.
        cat = catalog.reset_index(drop=True)
        self.isbn_to_row: dict[str, int] = {
            str(isbn): i for i, isbn in enumerate(cat["isbn13"])
        }
        self.row_categories = [
            frozenset(lst) for lst in cat["categories_list"].tolist()
        ]
        self.ratings = cat["average_rating"].to_numpy(dtype=np.float32)

    def is_relevant(self, query_isbn: str, candidate_isbn: str) -> bool:
        qi = self.isbn_to_row.get(str(query_isbn))
        ci = self.isbn_to_row.get(str(candidate_isbn))
        if qi is None or ci is None:
            return False

        # Category clause.
        if self.row_categories[qi].isdisjoint(self.row_categories[ci]):
            return False

        # Rating-proximity clause, with NaN-tolerance.
        q_rating, c_rating = self.ratings[qi], self.ratings[ci]
        if np.isnan(q_rating) or np.isnan(c_rating):
            return True
        return abs(q_rating - c_rating) < self.rating_threshold


# ---------------------------------------------------------------------------
# Per-query metric computations
# ---------------------------------------------------------------------------

def precision_at_k(
    query_isbn: str,
    rec_isbns: list[str],
    proxy: RelevanceProxy,
) -> float:
    """Fraction of recs that are relevant to the query under the proxy."""
    if not rec_isbns:
        return 0.0
    return sum(proxy.is_relevant(query_isbn, r) for r in rec_isbns) / len(rec_isbns)


def intra_list_diversity(
    rec_isbns: list[str], catalog_row_categories: dict[str, frozenset]
) -> float:
    """Average pairwise category Jaccard distance across the recommendations.

    Two recs with the same category set contribute distance 0. Two recs
    with completely disjoint categories contribute distance 1. The mean
    of all n*(n-1)/2 pair distances is the ILD.

    Returns 0.0 for fewer than 2 recs.
    """
    n = len(rec_isbns)
    if n < 2:
        return 0.0

    cat_sets = [catalog_row_categories.get(str(isbn), frozenset()) for isbn in rec_isbns]
    distances: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = cat_sets[i], cat_sets[j]
            if not a and not b:
                distances.append(0.0)
                continue
            intersection = len(a & b)
            union = len(a | b)
            jaccard_sim = (intersection / union) if union else 0.0
            distances.append(1.0 - jaccard_sim)
    return float(np.mean(distances)) if distances else 0.0


# ---------------------------------------------------------------------------
# Eval driver
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    model: str
    precision_at_10: float
    ild: float
    catalog_coverage: float
    n_queries: int


class Evaluator:
    """Runs the full eval suite for a given set of models."""

    def __init__(
        self,
        catalog_path: Path,
        models_dir: Path,
        eval_size: int = DEFAULT_EVAL_SIZE,
        k: int = 10,
        seed: int = RANDOM_SEED,
    ):
        self.catalog_path = catalog_path
        self.models_dir = models_dir
        self.eval_size = eval_size
        self.k = k
        self.seed = seed

        # One catalog read, shared across all models. We also rely on
        # each BookRecommender loading its own catalog internally,
        # which is wasteful in wall time but keeps the predict module
        # self-contained. OK for 3 models x 5787 rows.
        self.catalog = pd.read_parquet(catalog_path).reset_index(drop=True)

        self.proxy = RelevanceProxy(self.catalog)
        self.row_cat_lookup: dict[str, frozenset] = dict(
            zip(
                self.catalog["isbn13"].astype(str).tolist(),
                [frozenset(lst) for lst in self.catalog["categories_list"].tolist()],
            )
        )

        # Sample the eval set. Done once so all models are scored on
        # identical queries (fair comparison).
        rng = np.random.default_rng(seed)
        eval_indices = rng.choice(
            len(self.catalog), size=min(eval_size, len(self.catalog)), replace=False
        )
        self.eval_isbns = (
            self.catalog.iloc[eval_indices]["isbn13"].astype(str).tolist()
        )
        self.eval_buckets = (
            self.catalog.iloc[eval_indices]["description_length_bucket"].tolist()
        )
        logger.info(
            "Eval set: %d queries (seed=%d). Bucket split: %s",
            len(self.eval_isbns),
            seed,
            {b: self.eval_buckets.count(b) for b in LENGTH_BUCKETS},
        )

    # -- Per-model evaluation -------------------------------------------

    def evaluate_model(
        self, model_name: str
    ) -> tuple[ModelMetrics, list[dict]]:
        """Score one model on the full eval set.

        Returns:
            (aggregate metrics, per-query records)
        The per-query records contain one row per query with the
        metrics and the query's length bucket, so the length-bucket
        experiment can be computed from the same data.
        """
        logger.info("=== Evaluating model=%s ===", model_name)
        recommender = BookRecommender(
            model_name=model_name,
            catalog_path=self.catalog_path,
            models_dir=self.models_dir,
        )

        precisions: list[float] = []
        ilds: list[float] = []
        all_rec_isbns: set[str] = set()
        per_query_records: list[dict] = []

        for i, (q_isbn, bucket) in enumerate(
            zip(self.eval_isbns, self.eval_buckets), start=1
        ):
            try:
                rec_isbns = recommender.recommend(q_isbn, k=self.k)
            except KeyError:
                logger.warning(
                    "Skipping query %s (not found in model's index)", q_isbn
                )
                continue

            p = precision_at_k(q_isbn, rec_isbns, self.proxy)
            ild = intra_list_diversity(rec_isbns, self.row_cat_lookup)

            precisions.append(p)
            ilds.append(ild)
            all_rec_isbns.update(rec_isbns)
            per_query_records.append(
                dict(
                    model=model_name,
                    query_isbn=q_isbn,
                    length_bucket=bucket,
                    precision_at_10=p,
                    ild=ild,
                )
            )

            if i % 100 == 0:
                logger.info(
                    "  [%s] %d/%d queries processed",
                    model_name,
                    i,
                    len(self.eval_isbns),
                )

        coverage = len(all_rec_isbns) / len(self.catalog)
        metrics = ModelMetrics(
            model=model_name,
            precision_at_10=float(np.mean(precisions)) if precisions else 0.0,
            ild=float(np.mean(ilds)) if ilds else 0.0,
            catalog_coverage=float(coverage),
            n_queries=len(precisions),
        )
        logger.info(
            "  [%s] done. P@10=%.3f  ILD=%.3f  Coverage=%.3f  (n=%d)",
            model_name,
            metrics.precision_at_10,
            metrics.ild,
            metrics.catalog_coverage,
            metrics.n_queries,
        )
        return metrics, per_query_records

    # -- Full run --------------------------------------------------------

    def run(self, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluate every model; return (aggregate_df, per_query_df)."""
        aggregate_rows: list[dict] = []
        all_per_query: list[dict] = []

        for model_name in models:
            metrics, per_query = self.evaluate_model(model_name)
            aggregate_rows.append(
                dict(
                    model=metrics.model,
                    precision_at_10=round(metrics.precision_at_10, 4),
                    ild=round(metrics.ild, 4),
                    catalog_coverage=round(metrics.catalog_coverage, 4),
                    n_queries=metrics.n_queries,
                )
            )
            all_per_query.extend(per_query)

        return pd.DataFrame(aggregate_rows), pd.DataFrame(all_per_query)


# ---------------------------------------------------------------------------
# Experiment: Precision@10 by description-length bucket
# ---------------------------------------------------------------------------

def experiment_by_length_bucket(per_query_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate precision@10 by (model, length_bucket)."""
    grouped = (
        per_query_df.groupby(["model", "length_bucket"])["precision_at_10"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "precision_at_10", "count": "n_queries"})
    )
    grouped["precision_at_10"] = grouped["precision_at_10"].round(4)
    # Order buckets short -> medium -> long for consistent plotting.
    grouped["length_bucket"] = pd.Categorical(
        grouped["length_bucket"], categories=list(LENGTH_BUCKETS), ordered=True
    )
    grouped = grouped.sort_values(["model", "length_bucket"]).reset_index(drop=True)
    return grouped


def plot_experiment(experiment_df: pd.DataFrame, output_path: Path) -> None:
    """Render the per-bucket precision chart to PNG."""
    # Lazy import so evaluate.py doesn't require matplotlib unless
    # the experiment output is actually being generated.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model_name, group in experiment_df.groupby("model"):
        ax.plot(
            group["length_bucket"],
            group["precision_at_10"],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    ax.set_xlabel("Query description length bucket")
    ax.set_ylabel("Precision@10")
    ax.set_title("Precision@10 vs. description length, by model")
    ax.set_ylim(0, max(0.05, experiment_df["precision_at_10"].max() * 1.2))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Model", loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote chart to %s", output_path)


# ---------------------------------------------------------------------------
# Report-friendly output
# ---------------------------------------------------------------------------

def metrics_to_markdown(aggregate_df: pd.DataFrame) -> str:
    """Render the aggregate metrics as a paste-ready markdown table."""
    lines = [
        "| Model | Precision@10 | ILD | Catalog Coverage | n_queries |",
        "|---|---|---|---|---|",
    ]
    for _, row in aggregate_df.iterrows():
        lines.append(
            f"| {row['model']} | "
            f"{row['precision_at_10']:.3f} | "
            f"{row['ild']:.3f} | "
            f"{row['catalog_coverage']:.3f} | "
            f"{row['n_queries']} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
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
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/outputs/"),
    )
    p.add_argument(
        "--eval-size",
        type=int,
        default=DEFAULT_EVAL_SIZE,
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=list(VALID_MODELS),
        choices=VALID_MODELS,
        help="Which models to evaluate (default: all three).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator(
        catalog_path=args.catalog,
        models_dir=args.models_dir,
        eval_size=args.eval_size,
    )
    aggregate_df, per_query_df = evaluator.run(models=args.models)

    # Write aggregate metrics.
    metrics_csv = args.output_dir / "metrics.csv"
    aggregate_df.to_csv(metrics_csv, index=False)
    logger.info("Wrote metrics CSV to %s", metrics_csv)

    metrics_md = args.output_dir / "metrics.md"
    metrics_md.write_text(metrics_to_markdown(aggregate_df) + "\n")
    logger.info("Wrote metrics markdown to %s", metrics_md)

    # Experiment: per-bucket precision.
    experiment_df = experiment_by_length_bucket(per_query_df)
    experiment_csv = args.output_dir / "experiment.csv"
    experiment_df.to_csv(experiment_csv, index=False)
    logger.info("Wrote experiment CSV to %s", experiment_csv)

    experiment_png = args.output_dir / "experiment.png"
    plot_experiment(experiment_df, experiment_png)

    # Also dump per-query records for any ad-hoc analysis the user wants.
    per_query_csv = args.output_dir / "per_query.csv"
    per_query_df.to_csv(per_query_csv, index=False)
    logger.info("Wrote per-query CSV to %s", per_query_csv)

    # Console summary.
    print("\n=== Aggregate metrics ===")
    print(metrics_to_markdown(aggregate_df))
    print("\n=== Precision@10 by description-length bucket ===")
    print(
        experiment_df.to_string(index=False)
    )


if __name__ == "__main__":
    main()
