# Next Book

A content-based book recommendation system. Give it a book you loved — it returns 10 similar books with model-generated reasoning, using a catalog of ~5,800 books.

**Live app**: [next-book on Railway](https://next-book-production-06a3.up.railway.app/) <!-- update with actual URL -->

---

## How it works

Three recommenders are implemented and evaluated:

| Model | Approach |
|---|---|
| **Naive baseline** | Returns most-rated books regardless of query |
| **Classical (TF-IDF)** | Cosine similarity over TF-IDF description vectors |
| **Deep (MLP re-ranker)** | MiniLM embeddings → top-50 retrieval → MLP re-rank |

The deep model is the deployed default. Switching between models live is supported in the UI.

---

## Quickstart

```bash
# 1. Clone and create virtual environment
git clone https://github.com/roshgill/next-book.git
cd next-book
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements-dev.txt   # includes torch + sentence-transformers for training
pip install -r requirements.txt

# 3. Add raw data
# Download "7k Books with Metadata" from Kaggle (Dylan Castillo)
# Place the CSV at: data/raw/books.csv

# 4. Run the full pipeline (clean data → build features → train MLP)
python setup.py

# 5. CLI recommendations
python main.py --query "Gilead"
python main.py --query "Dune" --model classical

# 6. Web server (local)
uvicorn api.main:app --reload --port 8000
# Then in a second terminal:
cd web && NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
# Open http://localhost:3000
```

---

## Repository structure

```
├── README.md
├── requirements.txt          # production dependencies
├── requirements-dev.txt      # training-only deps (sentence-transformers, matplotlib)
├── setup.py                  # end-to-end pipeline: data → features → train
├── main.py                   # CLI entry point for recommendations
├── run.py                    # Docker/Railway entry point (reads PORT from env)
├── Dockerfile                # multi-stage: Next.js build + Python runtime
├── railway.json              # Railway deployment config
├── api/
│   └── main.py               # FastAPI backend (serves /api/* + static frontend)
├── scripts/
│   ├── make_dataset.py       # raw CSV → cleaned catalog.parquet
│   ├── clean_catalog.py      # second-pass: dedup editions, fix zero ratings
│   ├── build_features.py     # TF-IDF fit + MiniLM embeddings → models/
│   ├── model.py              # all three model classes + MLP training
│   ├── predict.py            # BookRecommender façade (inference only)
│   └── evaluate.py           # Precision@10, ILD, Catalog Coverage
├── models/                   # serialized artifacts (embeddings, TF-IDF, MLP weights)
├── data/
│   ├── raw/                  # raw Kaggle CSV (not versioned)
│   ├── processed/            # catalog_clean.parquet
│   └── outputs/              # evaluation results, experiment chart
├── web/                      # Next.js frontend (static export)
└── notebooks/                # exploratory notebooks (not graded)
```

---

## Evaluation results

| Model | Precision@10 | ILD | Catalog Coverage |
|---|---|---|---|
| Naive (popularity) | 0.080 | — | low |
| Classical (TF-IDF) | 0.201 | — | medium |
| **Deep (MLP)** | **0.360** | — | high |

Relevance proxy: shared category AND \|rating_diff\| < 0.3 stars.

---

## Dataset

[7k Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) — Dylan Castillo, Kaggle. 6,458 rows after cleaning.
