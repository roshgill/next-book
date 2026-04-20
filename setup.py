"""
setup.py

End-to-end pipeline: download/clean data, build features, train all models.
Run this once after cloning to reproduce the full project from raw data.

Usage:
    python setup.py                  # full pipeline
    python setup.py --skip-embeddings  # skip MiniLM encoding (faster iteration)

Attribution: AI-assisted code (Claude Sonnet 4.6, claude.ai/code)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(cmd: list[str], step: str) -> None:
    """Run a subprocess command and exit on failure."""
    logger.info("=== %s ===", step)
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("Step '%s' failed with exit code %d", step, result.returncode)
        sys.exit(result.returncode)
    logger.info("Step '%s' complete.\n", step)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip MiniLM embedding computation (uses cached embeddings.npy if present).",
    )
    p.add_argument(
        "--input",
        default="data/raw/books.csv",
        help="Path to raw CSV (default: data/raw/books.csv).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.input).exists():
        logger.error(
            "Raw dataset not found at %s. "
            "Download 7k Books with Metadata from Kaggle and place it there.",
            args.input,
        )
        sys.exit(1)

    py = sys.executable

    # Step 1: Schema cleaning + description filtering
    run(
        [py, "scripts/make_dataset.py", "--input", args.input],
        "Step 1: make_dataset — clean raw CSV",
    )

    # Step 2: Second-pass cleanup (edition dedup, duplicate descriptions, zero ratings)
    run(
        [py, "scripts/clean_catalog.py"],
        "Step 2: clean_catalog — deduplicate editions and descriptions",
    )

    # Step 3: TF-IDF fit + MiniLM embeddings
    embed_args = ["--skip-embeddings"] if args.skip_embeddings else []
    run(
        [py, "scripts/build_features.py"] + embed_args,
        "Step 3: build_features — TF-IDF + embeddings",
    )

    # Step 4: Train MLP re-ranker (naive and TF-IDF have no trainable params)
    run(
        [py, "scripts/model.py"],
        "Step 4: model — train MLP re-ranker",
    )

    logger.info("Pipeline complete. Run `python main.py` to start the recommendation CLI.")
    logger.info("Or run `uvicorn api.main:app --reload --port 8000` to start the web server.")


if __name__ == "__main__":
    main()
