# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage Dockerfile for AskMyDocs Backend
# Target: Render (Docker), slim image < 2 GB
#
# Stage 1 (builder): install all deps and compile wheels
# Stage 2 (runtime): copy only the installed packages + app code,
#                    pre-download ML models so no cold-start on Render
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install OS build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install CPU-only PyTorch first (no CUDA, significantly smaller image)
RUN pip install --no-cache-dir \
    "torch>=2.5.0" \
    --index-url https://download.pytorch.org/whl/cpu


# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt



# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Install runtime OS libs only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin


# Set Python path
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# ── Pre-download ML models at build time ──────────────────────────────────────
# This bakes the model weights into the image layer so that Render workers
# never need to download them on cold start (avoids 60-90s startup delay).
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
print('Downloading embedding model...'); \
SentenceTransformer('${EMBEDDING_MODEL}'); \
print('Downloading reranker model...'); \
CrossEncoder('${RERANKER_MODEL}'); \
print('Models ready.')"

# Copy application source code + startup script
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY start.sh .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app \
    && chmod +x /app/start.sh
USER appuser

EXPOSE 8000

# Health check (matches GET /health endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# start.sh: runs alembic upgrade head, then uvicorn
CMD ["/app/start.sh"]
