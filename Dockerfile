# Stage 1: build Next.js static export
FROM node:20-slim AS web-builder
WORKDIR /build
COPY web/package.json web/package-lock.json* ./
RUN npm ci
COPY web/ ./
RUN npm run build
# Produces /build/out

# Stage 2: Python runtime
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch first — must come before requirements.txt so pip doesn't
# later pull in the CUDA build as a dependency. Saves ~2.5GB vs default.
RUN pip install --no-cache-dir torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
# sentence-transformers is NOT installed here — embeddings are precomputed
# in models/embeddings.npy and loaded at runtime. No inference-time encoding.
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/
COPY api/ ./api/
COPY data/processed/catalog_clean.parquet ./data/processed/
COPY models/ ./models/
COPY --from=web-builder /build/out ./web/out

EXPOSE 8000
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
