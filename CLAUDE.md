# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**next-book** is a book recommendation system built as a module project for a recommendation systems course. The system recommends books similar to one the user has enjoyed (content-based item-to-item), using a 7K-book catalog with rich metadata and descriptions. The goal is to build, evaluate, and deploy a model that produces high-quality recommendations, while demonstrating sound ML engineering, rigorous evaluation, and a polished user-facing application.

## Project Approach (Locked)

- **System type**: Content-based item-to-item recommender. Input = one book the user liked (by title or ISBN). Output = 10 similar books with LLM-generated reasoning per recommendation.
- **Not building**: collaborative filtering, sequential models, or user-profile-based recommenders. Examples of those in the rubric do not apply here.
- **Dataset**: 7k Books with Metadata (Dylan Castillo, Kaggle). Available fields: `isbn13`, `isbn10`, `title`, `subtitle`, `authors`, `categories`, `description`, `thumbnail`, `published_year`, `average_rating`, `num_pages`, `ratings_count`. 6,458 rows after dropping null/short descriptions.
- **Primary key**: `isbn13`.
- **Important dataset property**: every book has exactly one top-level category. This shapes evaluation (see Evaluation section).

## Models (Locked)

All three must be implemented, documented, and findable in the repo. One becomes the deployed model.

1. **Naive baseline (popularity)** — return the top-10 books by `ratings_count`, query-agnostic (but excluding the query itself). Books with null `ratings_count` are dropped from the baseline's candidate pool. This is a standard and meaningfully hard-to-beat baseline in rec systems: popular books are genuinely liked by more people, so "just recommend what's popular" is a real floor that content-based systems must clear.
2. **Classical ML model** — TF-IDF vectorizer over the `description` field plus cosine similarity for retrieval. Standard content-based baseline; tests whether lexical overlap alone is sufficient.
3. **Deep learning model (feedforward NN re-ranker)** — two-stage:
   - Stage 1: pretrained MiniLM (`sentence-transformers/all-MiniLM-L6-v2`) embeds descriptions; top-50 candidates retrieved by cosine similarity.
   - Stage 2: a small MLP (8 features → 32 → 16 → 1, ReLU activations, sigmoid output) re-ranks the 50 candidates.
   - Features: embedding cosine similarity, TF-IDF cosine similarity, shared category count, absolute rating difference, absolute year difference, description length ratio, shared-author flag, log candidate popularity (`log1p(ratings_count)`).
   - Training: binary cross-entropy on the relevance-proxy labels defined in Evaluation. ~20 positives and ~20 negatives sampled per query book.
   - Maps to the class taxonomy's **feedforward — feature interaction modeling** archetype.

Each model lives in `scripts/model.py` or a dedicated module under `scripts/`, and the README documents where to find each.

## Evaluation (Locked)

- **Relevance proxy (compound)**: two books are "relevant" to each other if they (a) share a category AND (b) their `average_rating` values differ by less than 0.3 stars.
  - A pure shared-category proxy was rejected because every book in this dataset has exactly one top-level category and "Fiction" alone covers 39% of the catalog — under a pure-category proxy, a random baseline would score Precision@10 ≈ 0.39 and the spread between models would collapse.
  - The rating-proximity half adds a quality-matching criterion (a reader who loved a 4.5-star literary novel is unlikely to love a 3.2-star literary novel even in the same genre).
  - This proxy remains independent of the features models train on (description text, embeddings), preserving evaluation validity.
- **Metrics**:
  - **Precision@10** — fraction of the 10 returned recs that are "relevant" to the query under the compound proxy. Primary metric.
  - **Intra-List Diversity (ILD)** — average pairwise category dissimilarity across the 10 recs. Counterbalances Precision@10 (a system returning 10 near-duplicates scores high on precision but low on diversity).
  - **Catalog Coverage** — share of the catalog that ever appears in any top-10 across the eval set. Penalizes recommenders that collapse to the same handful of books. The popularity baseline should score worst on this by construction, which is part of the narrative.
- **Evaluation set**: hold out 500 books as queries; for each, recommend 10 from the remaining ~5,900.

## Experiment (Locked)

- **Question**: Does description length affect recommendation quality, and does the effect differ across the three models?
- **Setup**: bucket query books by description length — short (<200 chars, ~2150 books), medium (200–500 chars, ~2650 books), long (>500 chars, ~1660 books). Measure Precision@10 per bucket for each of the three models.
- **Hypothesis**: TF-IDF degrades sharply on short descriptions because it has too few tokens to work with; the embedding-based model degrades less because the pretrained encoder brings external semantic knowledge. The popularity baseline should be flat across buckets (query-agnostic).
- **Deliverable**: one chart (Precision@10 vs. description-length bucket, three lines for three models) and a paragraph of interpretation.

## Novelty Statement

Most public book recommender systems use collaborative filtering and fail on cold-start (new books with no rating history). This project's contribution is threefold:

1. A purely content-based pipeline that works on any book with a description, including new releases with zero ratings.
2. A learned feedforward re-ranker that combines semantic (embedding), lexical (TF-IDF), and metadata signals — richer than any single-signal content-based system.
3. LLM-generated natural-language reasoning for each recommendation, giving interpretability that black-box similarity search lacks.

## Scope Discipline

- **Target build time**: approximately 3 hours end-to-end excluding the written report.
- **Do not add features** beyond what is specified in this file without explicit approval. Ideation happened before the build; the build is execution.
- **Deploy stack**: TBD — do not start deployment work until models and evaluation are complete. Candidates: Next.js + Vercel, Gradio on Hugging Face Spaces, Streamlit (rubric-risk, only if time collapses).

## Required Report Sections

Problem Statement · Data Sources · Related Work · Evaluation Strategy & Metrics · Modeling Approach · Data Processing Pipeline · Hyperparameter Tuning Strategy · Models Evaluated (all three) · Results · Error Analysis (5 specific mispredictions with root causes and mitigations) · Experiment Write-Up · Conclusions · Future Work · Commercial Viability Statement · Ethics Statement

## Repository Structure

```
├── README.md
├── CLAUDE.md
├── requirements.txt
├── setup.py                  # get data, build features, train models end-to-end
├── main.py                   # entry point / user interface
├── scripts/
│   ├── make_dataset.py       # data acquisition + cleaning
│   ├── build_features.py     # TF-IDF fit, embedding precomputation, feature engineering
│   ├── train.py              # training logic for all three models (fit, save artifacts)
│   └── predict.py            # inference logic for all three models (load artifacts, recommend)
├── models/                   # serialized trained model artifacts (embeddings, TF-IDF matrix, MLP weights)
├── data/
│   ├── raw/                  # raw Kaggle CSV
│   ├── processed/            # cleaned catalog with parsed categories
│   └── outputs/              # predictions, evaluation outputs, experiment results
├── notebooks/                # exploration only — NOT graded
└── .gitignore
```

Jupyter notebooks are only permitted inside `notebooks/`. They are not graded and exist solely for exploration.

## Code Quality Rules

- All code must be modularized into classes and functions — no loose executable code outside functions or `if __name__ == "__main__"` guards.
- Use descriptive variable names, docstrings on every public function/class, and comments where logic is non-obvious.
- Any external code or AI-generated code must be attributed at the top of the file with a link to the original source.

## Git Workflow

- Feature work goes on branches, not directly on `main`.
- All changes must come in via PRs; PRs require a review before merge (self-review is acceptable for solo work but must be substantive).
- Write meaningful PR descriptions and review comments — this is assessed.

## Evaluation Expectations

Metric selection is justified in the Evaluation section above (Precision@10, ILD, Catalog Coverage). Quantitative comparison tables across the three models and visualizations (per-bucket precision chart from the experiment, confusion-style rec-overlap heatmap for error analysis) are expected in the final report.
