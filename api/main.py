"""
api/main.py

FastAPI backend for next-book. Loads all three recommenders at startup
and exposes /api/* endpoints. Also serves the statically-exported
Next.js frontend from web/out/ at /.

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scripts.predict import VALID_MODELS, BookRecommender

logger = logging.getLogger(__name__)

CATALOG_PATH = Path("data/processed/catalog_clean.parquet")
MODELS_DIR = Path("models/")
WEB_OUT_DIR = Path("web/out")

# Populated once at startup; all request handlers read from this dict.
recommenders: dict[str, BookRecommender] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all three recommenders into memory once at process start."""
    for model_name in VALID_MODELS:
        logger.info("Loading recommender: %s …", model_name)
        recommenders[model_name] = BookRecommender(
            model_name=model_name,
            catalog_path=CATALOG_PATH,
            models_dir=MODELS_DIR,
        )
        logger.info("Recommender ready: %s", model_name)
    yield
    recommenders.clear()


app = FastAPI(title="next-book API", version="1.0.0", lifespan=lifespan)

# Build the allowed-origins list from the environment so that the Railway
# public domain is always permitted alongside localhost for local dev.
# RAILWAY_PUBLIC_DOMAIN is injected automatically by Railway (e.g.
# "next-book-production-xxxx.up.railway.app") — no trailing slash, no scheme.
_railway_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "")
_allowed_origins: list[str] = ["http://localhost:3000"]
if _railway_domain:
    _allowed_origins.append(f"https://{_railway_domain}")
    _allowed_origins.append(f"http://{_railway_domain}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic response schemas
# ---------------------------------------------------------------------------

class BookResponse(BaseModel):
    isbn13: str
    title: str
    authors: str
    categories: str
    description: str
    thumbnail: Optional[str] = None
    average_rating: Optional[float] = None
    published_year: Optional[int] = None


class SearchResultItem(BaseModel):
    isbn13: str
    title: str


class SearchResponse(BaseModel):
    results: list[SearchResultItem]


class RecommendResponse(BaseModel):
    query: BookResponse
    recommendations: list[BookResponse]
    model: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_book_response(result) -> BookResponse:
    return BookResponse(
        isbn13=result.isbn13,
        title=result.title,
        authors=result.authors,
        categories=result.categories,
        description=result.description,
        thumbnail=result.thumbnail,
        average_rating=result.average_rating,
        published_year=result.published_year,
    )


def _get_rec(model_name: str) -> BookRecommender:
    """Return the recommender for the given model, falling back to deep."""
    return recommenders.get(model_name) or recommenders["deep"]


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": list(recommenders.keys())}


@app.get("/api/search", response_model=SearchResponse)
def search(
    q: str = Query(default=""),
    limit: int = Query(default=10, ge=1, le=50),
):
    """Substring-match book titles for autocomplete. Returns (isbn13, title) pairs."""
    if len(q.strip()) < 2:
        return SearchResponse(results=[])
    rec = _get_rec("deep")
    matches = rec.search_titles(q, limit=limit)
    return SearchResponse(
        results=[SearchResultItem(isbn13=isbn, title=title) for isbn, title in matches]
    )


@app.get("/api/book/{isbn}", response_model=BookResponse)
def get_book(isbn: str):
    """Return full metadata for a single book by isbn13."""
    rec = _get_rec("deep")
    try:
        return _to_book_response(rec.get_book(isbn))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Book not found: {isbn}")


@app.get("/api/recommend", response_model=RecommendResponse)
def recommend(
    isbn: str = Query(..., description="isbn13 of the query book"),
    model: str = Query(default="deep", description="naive | classical | deep"),
    k: int = Query(default=10, ge=1, le=20),
):
    """Return k recommendations for the given book under the chosen model."""
    if model not in VALID_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{model}'. Choose from: {list(VALID_MODELS)}",
        )
    rec = recommenders[model]
    try:
        query_book = rec.get_book(isbn)
        recs = rec.recommend_with_metadata(isbn, k=k)
        return RecommendResponse(
            query=_to_book_response(query_book),
            recommendations=[_to_book_response(r) for r in recs],
            model=model,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Book not found: {isbn}")


# ---------------------------------------------------------------------------
# Serve Next.js static export (must be registered AFTER all /api/* routes)
# ---------------------------------------------------------------------------

# Catchall: return index.html for any non-/api path that isn't a static file,
# so the Next.js client-side router can handle it.
@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    index = WEB_OUT_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"detail": "Frontend not built yet. Run: cd web && npm run build"}


if WEB_OUT_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_OUT_DIR, html=True), name="web")
