"""
predict.py

Inference layer for the book recommender. Loads all precomputed
artifacts (catalog, TF-IDF matrix, embeddings, MLP weights, feature
scaler) once at construction time and exposes a single API:

    recommender = BookRecommender(model_name="deep")
    isbns = recommender.recommend(query_isbn="9780061120084", k=10)

This module is imported by ``main.py`` and by ``evaluate.py``. It does
not import ``model.py`` directly -- all the model-definition code it
needs is duplicated from ``model.py`` via reuse of shared logic from
the ``scripts.model`` module.

No training happens here. No side effects. No mutation of artifacts.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

# We intentionally depend on model.py for the MLP class + feature
# builder + artifact loader. This keeps the feature-computation logic
# in one place (no training/serving drift) and keeps predict.py slim.
from scripts.model import (
    FEATURE_NAMES,
    FeatureArtifacts,
    MLP,
    PairFeatureBuilder,
    PopularityRecommender,
    TfidfRecommender,
)


logger = logging.getLogger(__name__)


# Deep model: stage-1 retrieval fetches this many candidates by raw
# embedding cosine similarity; the MLP then re-ranks them.
DEEP_RETRIEVAL_K = 50


# ---------------------------------------------------------------------------
# Deep recommender: MiniLM retrieval + MLP re-rank
# ---------------------------------------------------------------------------

class DeepRecommender:
    """Two-stage: embedding-similarity retrieval, then MLP re-rank."""

    def __init__(
        self,
        artifacts: FeatureArtifacts,
        mlp_weights_path: Path,
        scaler_path: Path,
        retrieval_k: int = DEEP_RETRIEVAL_K,
        device: str | None = None,
    ):
        self.artifacts = artifacts
        self.retrieval_k = retrieval_k
        self.device = torch.device(device) if device else self._pick_device()

        # Load MLP weights.
        n_features = len(FEATURE_NAMES)
        self.mlp = MLP(n_features=n_features).to(self.device)
        state_dict = torch.load(mlp_weights_path, map_location=self.device)
        self.mlp.load_state_dict(state_dict)
        self.mlp.eval()

        # Load feature scaler.
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # Feature builder reused from training -- guarantees no
        # training/serving skew in how features are computed.
        self.feature_builder = PairFeatureBuilder(artifacts)

        logger.info(
            "DeepRecommender ready: %d features, retrieval_k=%d, device=%s",
            n_features,
            retrieval_k,
            self.device,
        )

    @staticmethod
    def _pick_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def recommend(self, query_isbn: str, k: int = 10) -> list[str]:
        """Return top-k isbn13 recommendations for the given query book."""
        idx = self.artifacts.isbn_to_idx.get(query_isbn)
        if idx is None:
            raise KeyError(f"ISBN not found in catalog: {query_isbn}")

        # Stage 1: retrieve top-`retrieval_k` by embedding cosine similarity.
        # Embeddings are L2-normalized, so dot product == cosine similarity.
        emb = self.artifacts.embeddings
        query_emb = emb[idx]
        sims = emb @ query_emb  # (n_books,)
        sims[idx] = -np.inf  # exclude the query itself

        # Partial sort: argpartition is O(n) for picking top-k.
        top_candidate_indices = np.argpartition(-sims, self.retrieval_k)[
            : self.retrieval_k
        ]

        # Stage 2: compute features for each (query, candidate) pair,
        # scale them, and score with the MLP.
        q_arr = np.full(self.retrieval_k, idx, dtype=np.int64)
        c_arr = top_candidate_indices.astype(np.int64)
        feats = self.feature_builder.compute_batch(q_arr, c_arr)
        feats_scaled = self.scaler.transform(feats).astype(np.float32)

        with torch.no_grad():
            x = torch.from_numpy(feats_scaled).to(self.device)
            scores = self.mlp(x).cpu().numpy()

        # Sort candidates by MLP score descending, take top-k.
        reranked = np.argsort(-scores)
        top_k_positions = reranked[:k]
        final_candidate_indices = c_arr[top_k_positions]

        return [
            self.artifacts.catalog.at[int(i), "isbn13"]
            for i in final_candidate_indices
        ]


# ---------------------------------------------------------------------------
# Unified façade: pick any of the three models by name
# ---------------------------------------------------------------------------

VALID_MODELS = ("naive", "classical", "deep")


@dataclass
class RecommendationResult:
    """One recommendation row with the metadata the UI needs to render it."""

    isbn13: str
    title: str
    authors: str
    categories: str
    description: str
    thumbnail: str | None
    average_rating: float | None
    published_year: int | None


class BookRecommender:
    """Façade that loads artifacts once and dispatches to the chosen model.

    Typical use:

        rec = BookRecommender(model_name="deep")
        results = rec.recommend_with_metadata("9780061120084", k=10)

    The heavy work (loading embeddings, the TF-IDF matrix, the MLP
    weights) happens once in ``__init__``. Subsequent ``recommend``
    calls are fast (<50ms typical).
    """

    def __init__(
        self,
        model_name: str = "deep",
        catalog_path: Path = Path("data/processed/catalog_clean.parquet"),
        models_dir: Path = Path("models/"),
    ):
        if model_name not in VALID_MODELS:
            raise ValueError(
                f"Unknown model_name={model_name!r}. "
                f"Valid choices: {VALID_MODELS}"
            )
        self.model_name = model_name
        self.catalog_path = Path(catalog_path)
        self.models_dir = Path(models_dir)

        # Load artifacts (catalog + TF-IDF + embeddings + isbn index).
        self.artifacts = FeatureArtifacts.load(self.catalog_path, self.models_dir)

        # Build the chosen recommender. We only instantiate the one
        # requested; loading all three would waste memory in the app
        # (DeepRecommender in particular loads torch weights).
        self._backend = self._build_backend(model_name)

        # Title lookup (lowercased) for the UI's "search by title" flow.
        titles_lower = self.artifacts.catalog["title"].str.lower()
        self._title_to_isbn: dict[str, str] = dict(
            zip(titles_lower, self.artifacts.catalog["isbn13"].astype(str))
        )

    def _build_backend(self, model_name: str):
        if model_name == "naive":
            return PopularityRecommender(self.artifacts)
        if model_name == "classical":
            return TfidfRecommender(self.artifacts)
        if model_name == "deep":
            return DeepRecommender(
                artifacts=self.artifacts,
                mlp_weights_path=self.models_dir / "mlp.pt",
                scaler_path=self.models_dir / "feature_scaler.pkl",
            )
        raise AssertionError(f"unreachable: model_name={model_name}")

    # -- Query resolution -------------------------------------------------

    def resolve_isbn(self, query: str) -> str | None:
        """Resolve a user query to an isbn13.

        Accepts either an isbn13 directly, or a title (case-insensitive,
        exact match). Returns None if no match.
        """
        query = query.strip()
        if query in self.artifacts.isbn_to_idx:
            return query

        lower = query.lower()
        if lower in self._title_to_isbn:
            return self._title_to_isbn[lower]

        return None

    def search_titles(self, query: str, limit: int = 10) -> list[tuple[str, str]]:
        """Substring-match titles for autocomplete. Returns (isbn13, title) pairs."""
        if not query.strip():
            return []
        q = query.strip().lower()
        cat = self.artifacts.catalog
        mask = cat["title"].str.lower().str.contains(q, regex=False, na=False)
        matches = cat[mask].head(limit)
        return list(
            zip(
                matches["isbn13"].astype(str).tolist(),
                matches["title"].astype(str).tolist(),
            )
        )

    # -- Recommendation entrypoints --------------------------------------

    def recommend(self, query_isbn: str, k: int = 10) -> list[str]:
        """Return k isbn13 recommendations. Raises KeyError on unknown ISBN."""
        return self._backend.recommend(query_isbn, k=k)

    def recommend_with_metadata(
        self, query_isbn: str, k: int = 10
    ) -> list[RecommendationResult]:
        """Return k recommendations as rich metadata rows for UI rendering."""
        isbns = self.recommend(query_isbn, k=k)
        return [self._isbn_to_result(isbn) for isbn in isbns]

    def get_book(self, isbn: str) -> RecommendationResult:
        """Return metadata for a single book. Useful for rendering the query card."""
        return self._isbn_to_result(isbn)

    # -- Internal: row -> rich result ------------------------------------

    def _isbn_to_result(self, isbn: str) -> RecommendationResult:
        idx = self.artifacts.isbn_to_idx.get(isbn)
        if idx is None:
            raise KeyError(f"ISBN not found in catalog: {isbn}")
        row = self.artifacts.catalog.iloc[idx]

        def _opt_float(v) -> float | None:
            return None if pd.isna(v) else float(v)

        def _opt_int(v) -> int | None:
            return None if pd.isna(v) else int(v)

        def _opt_str(v) -> str | None:
            if v is None:
                return None
            if isinstance(v, float) and pd.isna(v):
                return None
            s = str(v).strip()
            return s if s else None

        return RecommendationResult(
            isbn13=str(row["isbn13"]),
            title=str(row["title"]),
            authors=str(row["authors"]) if not pd.isna(row["authors"]) else "",
            categories=str(row["categories"]),
            description=str(row["description"]),
            thumbnail=_opt_str(row["thumbnail"]),
            average_rating=_opt_float(row["average_rating"]),
            published_year=_opt_int(row["published_year"]),
        )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=VALID_MODELS, default="deep")
    p.add_argument("--query", type=str, required=True,
                   help="isbn13 or book title to use as query")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--catalog", type=Path,
                   default=Path("data/processed/catalog_clean.parquet"))
    p.add_argument("--models-dir", type=Path, default=Path("models/"))
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    rec = BookRecommender(
        model_name=args.model,
        catalog_path=args.catalog,
        models_dir=args.models_dir,
    )

    resolved = rec.resolve_isbn(args.query)
    if resolved is None:
        logger.error("Could not resolve query %r to a known book.", args.query)
        matches = rec.search_titles(args.query, limit=5)
        if matches:
            logger.info("Did you mean one of these?")
            for isbn, title in matches:
                logger.info("  %s  %s", isbn, title)
        return

    query_book = rec.get_book(resolved)
    print(f"\nQuery: {query_book.title!r} by {query_book.authors} "
          f"[{query_book.categories}]\n")
    print(f"Top {args.k} recommendations (model={args.model}):")
    print("-" * 80)
    for i, r in enumerate(rec.recommend_with_metadata(resolved, k=args.k), 1):
        rating = f"{r.average_rating:.2f}" if r.average_rating is not None else "n/a"
        year = r.published_year if r.published_year is not None else "n/a"
        print(f"{i:2d}. {r.title!r} by {r.authors}")
        print(f"    [{r.categories}]  rating={rating}  year={year}")
        print(f"    {r.description[:140]}...")
        print()


if __name__ == "__main__":
    main()
