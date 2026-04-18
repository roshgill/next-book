"""
model.py

Defines the three recommender models required by the project and trains
the MLP re-ranker used by the deep model.

Running this script (``python scripts/model.py``) trains the MLP and
saves its weights plus the feature scaler to ``models/``. The naive and
classical recommenders have no trainable parameters, so they are only
defined here (class interfaces) and consumed by ``predict.py`` at
inference time.

Files produced:
    models/mlp.pt               -- trained MLP weights (state dict)
    models/feature_scaler.pkl   -- fitted StandardScaler for the 8 MLP input features

Design notes:
    * All three recommenders share the same ``.recommend(query_isbn, k)``
      API so ``predict.py`` can swap between them cleanly.
    * The MLP is intentionally small (8 -> 32 -> 16 -> 1). The feature
      space is already engineered; the network just learns a non-linear
      weighting over eight informative signals.
    * Training labels come from the compound relevance proxy documented
      in CLAUDE.md (shared category AND |rating_diff| < 0.3 stars).
"""

from __future__ import annotations

import argparse
import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


# Relevance proxy parameters. Two books are positive pairs iff they
# share a category AND their average ratings differ by less than this
# threshold. This proxy is rating-proximity-aware so "Fiction rated
# 4.5" doesn't count as relevant to "Fiction rated 3.0".
RATING_DIFF_THRESHOLD = 0.3

# How many positive/negative pairs to sample per query book during
# training data generation. 20+20 = 40 pairs x ~5800 books ~= 230k
# labeled examples, which is plenty for an 8-feature network.
POSITIVES_PER_QUERY = 20
NEGATIVES_PER_QUERY = 20

# MLP architecture + training hyperparameters.
MLP_HIDDEN_SIZES = (32, 16)
MLP_EPOCHS = 10
MLP_BATCH_SIZE = 512
MLP_LEARNING_RATE = 1e-3

# Used for reproducibility of negative sampling and train/val split.
RANDOM_SEED = 42

# Ordered list of feature names for the MLP input. The order is
# contractual -- ``predict.py`` must compute features in the same order.
#
# Note: ``shared_category_count`` is deliberately omitted. The training
# label is derived from category overlap, so including it as a feature
# would cause label leakage (the MLP would trivially memorize
# ``shared_category_count >= 1 -> positive`` instead of learning which
# category-matching candidates are actually the best matches).
FEATURE_NAMES = [
    "embedding_cosine_sim",
    "tfidf_cosine_sim",
    "abs_rating_diff",
    "abs_year_diff",
    "description_length_ratio",
    "shared_author_flag",
    "log_candidate_popularity",
]


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

@dataclass
class FeatureArtifacts:
    """Bundle of the precomputed artifacts from build_features.py."""

    catalog: pd.DataFrame                  # cleaned catalog, row-aligned with matrices
    isbn_to_idx: dict[str, int]            # isbn13 -> row index in matrices
    tfidf_matrix: sparse.csr_matrix        # (n_books, vocab) sparse
    embeddings: np.ndarray                 # (n_books, 384) L2-normalized float32

    @classmethod
    def load(cls, catalog_path: Path, models_dir: Path) -> "FeatureArtifacts":
        """Load all artifacts and verify row alignment across them."""
        logger.info("Loading catalog from %s", catalog_path)
        catalog = pd.read_parquet(catalog_path).reset_index(drop=True)

        isbn_index = np.load(models_dir / "isbn_index.npy", allow_pickle=True)
        tfidf_matrix = sparse.load_npz(models_dir / "tfidf_matrix.npz")
        embeddings = np.load(models_dir / "embeddings.npy")

        # Alignment check: all three artifacts must have the same number
        # of rows as the catalog, in the same order.
        assert len(catalog) == len(isbn_index), (
            f"Catalog rows ({len(catalog)}) do not match isbn_index "
            f"({len(isbn_index)}). Re-run build_features.py."
        )
        assert tfidf_matrix.shape[0] == len(catalog)
        assert embeddings.shape[0] == len(catalog)
        assert (catalog["isbn13"].values == isbn_index).all(), (
            "Catalog isbn13 order does not match isbn_index. "
            "Re-run build_features.py."
        )

        isbn_to_idx = {isbn: i for i, isbn in enumerate(isbn_index)}

        logger.info(
            "Artifacts loaded: %d books, tfidf shape %s, embeddings shape %s",
            len(catalog),
            tfidf_matrix.shape,
            embeddings.shape,
        )
        return cls(
            catalog=catalog,
            isbn_to_idx=isbn_to_idx,
            tfidf_matrix=tfidf_matrix,
            embeddings=embeddings,
        )


# ---------------------------------------------------------------------------
# Recommender interface + the three implementations
# ---------------------------------------------------------------------------

class BaseRecommender(ABC):
    """Shared interface for every recommender."""

    def __init__(self, artifacts: FeatureArtifacts):
        self.artifacts = artifacts

    @abstractmethod
    def recommend(self, query_isbn: str, k: int = 10) -> list[str]:
        """Return k isbn13 strings, best first, excluding the query itself."""
        raise NotImplementedError


class PopularityRecommender(BaseRecommender):
    """Naive baseline: return the top-k books by ratings_count, ignoring query.

    The query book is excluded from results when it happens to be in the
    popularity list. This baseline is query-agnostic by design -- its job
    is to establish the floor that any real model must beat.
    """

    def __init__(self, artifacts: FeatureArtifacts):
        super().__init__(artifacts)
        catalog = artifacts.catalog
        # Pre-rank once. Books with null ratings_count rank last.
        ranked = catalog.assign(
            _rc=catalog["ratings_count"].fillna(-1)
        ).sort_values("_rc", ascending=False)
        self._ranked_isbns: list[str] = ranked["isbn13"].astype(str).tolist()

    def recommend(self, query_isbn: str, k: int = 10) -> list[str]:
        return [isbn for isbn in self._ranked_isbns if isbn != query_isbn][:k]


class TfidfRecommender(BaseRecommender):
    """Classical content-based: cosine similarity over TF-IDF vectors."""

    def recommend(self, query_isbn: str, k: int = 10) -> list[str]:
        idx = self.artifacts.isbn_to_idx.get(query_isbn)
        if idx is None:
            raise KeyError(f"ISBN not found in catalog: {query_isbn}")

        # Cosine sim via sparse matrix-vector product. TF-IDF rows from
        # sklearn's vectorizer are already L2-normalized, so the dot
        # product equals cosine similarity.
        query_vec = self.artifacts.tfidf_matrix[idx]
        sims = self.artifacts.tfidf_matrix.dot(query_vec.T).toarray().ravel()
        sims[idx] = -np.inf  # exclude self

        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]

        return [
            self.artifacts.catalog.at[int(i), "isbn13"]
            for i in top
        ]


class MLP(nn.Module):
    """Tiny feedforward re-ranker: ``n_features -> 32 -> 16 -> 1``.

    The sigmoid output is interpreted as P(relevant | query, candidate).
    """

    def __init__(
        self,
        n_features: int,
        hidden_sizes: tuple[int, ...] = MLP_HIDDEN_SIZES,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Feature computation (shared between training and inference)
# ---------------------------------------------------------------------------

class PairFeatureBuilder:
    """Computes the 8 MLP input features for (query, candidate) book pairs.

    Kept as a class with an explicit interface so ``predict.py`` can
    reuse it at inference time with the same code path used during
    training -- no drift between training and serving feature logic.
    """

    def __init__(self, artifacts: FeatureArtifacts):
        self.artifacts = artifacts

        # Pre-extract numeric arrays we will hit repeatedly. Indexing a
        # pandas Series in a tight loop is ~20x slower than a numpy array.
        cat = artifacts.catalog
        self.ratings = cat["average_rating"].to_numpy(dtype=np.float32)
        self.years = cat["published_year"].to_numpy(dtype=np.float32)
        self.desc_lengths = cat["description_length"].to_numpy(dtype=np.float32)

        # Popularity: log1p(ratings_count) with null handled as 0 ratings.
        rc = cat["ratings_count"].fillna(0).to_numpy(dtype=np.float32)
        self.log_popularity = np.log1p(rc)

        # Convert categories and authors to frozen sets per row for fast
        # intersection checks. Author comparison is case-insensitive.
        self.categories = [
            frozenset(lst) for lst in cat["categories_list"].tolist()
        ]
        self.authors = [
            frozenset(a.lower().strip() for a in lst)
            for lst in cat["authors_list"].tolist()
        ]

    def compute_batch(
        self, query_indices: np.ndarray, candidate_indices: np.ndarray
    ) -> np.ndarray:
        """Return an (n_pairs, n_features) float32 feature matrix."""
        assert len(query_indices) == len(candidate_indices)
        n = len(query_indices)

        q = query_indices
        c = candidate_indices

        emb = self.artifacts.embeddings
        # Rows are already L2-normalized, so dot product == cosine sim.
        embedding_sim = np.sum(emb[q] * emb[c], axis=1)

        # TF-IDF rows are also L2-normalized by sklearn.
        tfidf_mat = self.artifacts.tfidf_matrix
        tfidf_sim = np.asarray(
            tfidf_mat[q].multiply(tfidf_mat[c]).sum(axis=1)
        ).ravel()

        # Shared-author flag only. Shared-category count is intentionally
        # NOT a feature (see FEATURE_NAMES docstring re: label leakage).
        shared_author = np.empty(n, dtype=np.float32)
        for i in range(n):
            qi, ci = q[i], c[i]
            shared_author[i] = 1.0 if (
                self.authors[qi] & self.authors[ci]
            ) else 0.0

        # Numeric diffs with nan-safe absolute difference. When either
        # side is NaN (e.g., no rating available), we fall back to the
        # dataset mean for that feature. Prevents NaN poisoning the MLP.
        def nan_abs_diff(a: np.ndarray, q_idx: np.ndarray,
                         c_idx: np.ndarray) -> np.ndarray:
            diff = np.abs(a[q_idx] - a[c_idx])
            mean_val = np.nanmean(a)
            return np.where(np.isnan(diff), mean_val, diff)

        abs_rating_diff = nan_abs_diff(self.ratings, q, c)
        abs_year_diff = nan_abs_diff(self.years, q, c)

        # Description length ratio: shorter over longer, always in (0, 1].
        # Symmetric (ratio of a,b == ratio of b,a) and bounded.
        q_len = self.desc_lengths[q]
        c_len = self.desc_lengths[c]
        desc_len_ratio = np.minimum(q_len, c_len) / np.maximum(q_len, c_len)

        log_pop = self.log_popularity[c]

        feats = np.stack(
            [
                embedding_sim.astype(np.float32),
                tfidf_sim.astype(np.float32),
                abs_rating_diff.astype(np.float32),
                abs_year_diff.astype(np.float32),
                desc_len_ratio.astype(np.float32),
                shared_author,
                log_pop,
            ],
            axis=1,
        )
        assert feats.shape == (n, len(FEATURE_NAMES))
        return feats


# ---------------------------------------------------------------------------
# Training-pair generation
# ---------------------------------------------------------------------------

class TrainingPairGenerator:
    """Samples labeled (query, candidate) pairs from the catalog.

    Positives: share >= 1 category AND |rating_diff| < RATING_DIFF_THRESHOLD.
    Negatives: share 0 categories (and any rating).

    The compound positive rule mirrors the evaluation relevance proxy,
    which keeps training-time and eval-time notions of "relevant"
    consistent. This is deliberate: we want the re-ranker to optimize
    exactly the criterion it will be judged on.
    """

    def __init__(
        self,
        artifacts: FeatureArtifacts,
        positives_per_query: int = POSITIVES_PER_QUERY,
        negatives_per_query: int = NEGATIVES_PER_QUERY,
        seed: int = RANDOM_SEED,
    ):
        self.artifacts = artifacts
        self.n_pos = positives_per_query
        self.n_neg = negatives_per_query
        self.rng = np.random.default_rng(seed)

        cat = artifacts.catalog
        self.n_books = len(cat)
        self.ratings = cat["average_rating"].to_numpy(dtype=np.float32)

        # Precompute {category -> set of book indices} for O(1) lookups.
        cat_to_indices: dict[str, list[int]] = {}
        for i, cats in enumerate(cat["categories_list"].tolist()):
            for c in cats:
                cat_to_indices.setdefault(c, []).append(i)
        self.cat_to_indices = {
            k: np.array(v, dtype=np.int64) for k, v in cat_to_indices.items()
        }
        # Row-level category sets for quick membership checks.
        self.row_categories = [
            frozenset(lst) for lst in cat["categories_list"].tolist()
        ]

    def _sample_positives(self, q_idx: int) -> np.ndarray:
        """Sample up to n_pos candidate indices satisfying the positive rule."""
        q_cats = self.row_categories[q_idx]
        # Union of all books that share any category with the query.
        candidate_pool = np.unique(
            np.concatenate(
                [self.cat_to_indices[c] for c in q_cats]
                + [np.empty(0, dtype=np.int64)]
            )
        )
        # Exclude the query itself.
        candidate_pool = candidate_pool[candidate_pool != q_idx]

        if len(candidate_pool) == 0:
            return np.empty(0, dtype=np.int64)

        # Apply rating-proximity filter. If query has no rating, we skip
        # this filter (only category match is required) -- otherwise we
        # would eliminate the query entirely.
        q_rating = self.ratings[q_idx]
        if not np.isnan(q_rating):
            c_ratings = self.ratings[candidate_pool]
            close_enough = np.abs(c_ratings - q_rating) < RATING_DIFF_THRESHOLD
            # NaN ratings on candidate side: treat as non-matches (keep
            # the training signal clean).
            close_enough = close_enough & ~np.isnan(c_ratings)
            candidate_pool = candidate_pool[close_enough]

        if len(candidate_pool) == 0:
            return np.empty(0, dtype=np.int64)

        size = min(self.n_pos, len(candidate_pool))
        return self.rng.choice(candidate_pool, size=size, replace=False)

    def _sample_negatives(self, q_idx: int) -> np.ndarray:
        """Sample n_neg candidate indices that share no category with the query."""
        q_cats = self.row_categories[q_idx]
        # Rejection sampling is fine: for most queries < 50% of books
        # share a category, so a few rounds of sampling suffice.
        chosen: list[int] = []
        attempts = 0
        max_attempts = self.n_neg * 10
        while len(chosen) < self.n_neg and attempts < max_attempts:
            batch = self.rng.integers(0, self.n_books, size=self.n_neg * 2)
            for i in batch:
                if i == q_idx:
                    continue
                if self.row_categories[i].isdisjoint(q_cats):
                    chosen.append(int(i))
                    if len(chosen) >= self.n_neg:
                        break
            attempts += self.n_neg * 2
        return np.array(chosen[: self.n_neg], dtype=np.int64)

    def generate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (query_idx, cand_idx, label) arrays for the full train set."""
        q_list: list[np.ndarray] = []
        c_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []

        for q_idx in range(self.n_books):
            pos = self._sample_positives(q_idx)
            neg = self._sample_negatives(q_idx)
            if len(pos):
                q_list.append(np.full(len(pos), q_idx, dtype=np.int64))
                c_list.append(pos)
                y_list.append(np.ones(len(pos), dtype=np.float32))
            if len(neg):
                q_list.append(np.full(len(neg), q_idx, dtype=np.int64))
                c_list.append(neg)
                y_list.append(np.zeros(len(neg), dtype=np.float32))

            if (q_idx + 1) % 1000 == 0:
                logger.info("Generated pairs for %d/%d books", q_idx + 1, self.n_books)

        queries = np.concatenate(q_list)
        candidates = np.concatenate(c_list)
        labels = np.concatenate(y_list)

        n_pos = int(labels.sum())
        logger.info(
            "Training pairs: %d total (%d positive, %d negative, %.1f%% pos)",
            len(labels),
            n_pos,
            len(labels) - n_pos,
            100 * n_pos / len(labels),
        )
        return queries, candidates, labels


# ---------------------------------------------------------------------------
# MLP training
# ---------------------------------------------------------------------------

class MLPTrainer:
    """Encapsulates MLP training with train/val split and checkpoint saving."""

    def __init__(
        self,
        feature_builder: PairFeatureBuilder,
        models_dir: Path,
        epochs: int = MLP_EPOCHS,
        batch_size: int = MLP_BATCH_SIZE,
        learning_rate: float = MLP_LEARNING_RATE,
        val_fraction: float = 0.1,
        seed: int = RANDOM_SEED,
    ):
        self.feature_builder = feature_builder
        self.models_dir = models_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_fraction = val_fraction
        self.rng = np.random.default_rng(seed)
        self.device = self._pick_device()
        logger.info("MLP trainer using device: %s", self.device)

    @staticmethod
    def _pick_device() -> torch.device:
        """Use MPS on Apple silicon, CUDA if available, else CPU."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def train(
        self,
        query_idx: np.ndarray,
        candidate_idx: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[MLP, StandardScaler]:
        """Compute features, fit a scaler, train the MLP, return both."""
        logger.info("Computing features for %d training pairs", len(labels))
        features = self.feature_builder.compute_batch(query_idx, candidate_idx)

        # Train/val split. Shuffle once before splitting.
        perm = self.rng.permutation(len(labels))
        n_val = int(len(labels) * self.val_fraction)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        X_train_raw = features[train_idx]
        y_train = labels[train_idx]
        X_val_raw = features[val_idx]
        y_val = labels[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_val = scaler.transform(X_val_raw).astype(np.float32)

        # Move to tensors once.
        X_train_t = torch.from_numpy(X_train).to(self.device)
        y_train_t = torch.from_numpy(y_train).to(self.device)
        X_val_t = torch.from_numpy(X_val).to(self.device)
        y_val_t = torch.from_numpy(y_val).to(self.device)

        model = MLP(n_features=len(FEATURE_NAMES)).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.BCELoss()

        n_train = len(y_train)
        for epoch in range(1, self.epochs + 1):
            model.train()
            # Simple in-memory mini-batching.
            order = torch.randperm(n_train, device=self.device)
            running_loss = 0.0
            n_batches = 0
            for start in range(0, n_train, self.batch_size):
                batch = order[start : start + self.batch_size]
                xb = X_train_t[batch]
                yb = y_train_t[batch]

                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
                n_batches += 1

            train_loss = running_loss / max(n_batches, 1)

            # Val loss + accuracy at 0.5 threshold.
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = float(loss_fn(val_pred, y_val_t).item())
                val_acc = float(
                    ((val_pred > 0.5).float() == y_val_t).float().mean().item()
                )

            logger.info(
                "Epoch %2d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.3f",
                epoch,
                self.epochs,
                train_loss,
                val_loss,
                val_acc,
            )

        return model, scaler

    def save(self, model: MLP, scaler: StandardScaler) -> None:
        """Persist MLP weights and feature scaler to ``models_dir``."""
        self.models_dir.mkdir(parents=True, exist_ok=True)

        weights_path = self.models_dir / "mlp.pt"
        torch.save(model.state_dict(), weights_path)
        logger.info("Saved MLP weights to %s", weights_path)

        scaler_path = self.models_dir / "feature_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        logger.info("Saved feature scaler to %s", scaler_path)


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
    p.add_argument("--epochs", type=int, default=MLP_EPOCHS)
    p.add_argument("--batch-size", type=int, default=MLP_BATCH_SIZE)
    p.add_argument("--learning-rate", type=float, default=MLP_LEARNING_RATE)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    artifacts = FeatureArtifacts.load(args.catalog, args.models_dir)

    logger.info("Generating training pairs...")
    generator = TrainingPairGenerator(artifacts)
    q_idx, c_idx, y = generator.generate()

    feature_builder = PairFeatureBuilder(artifacts)
    trainer = MLPTrainer(
        feature_builder=feature_builder,
        models_dir=args.models_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    model, scaler = trainer.train(q_idx, c_idx, y)
    trainer.save(model, scaler)

    logger.info("model.py complete.")


if __name__ == "__main__":
    main()
