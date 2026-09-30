# Pinned versions: both stages use the SAME python image, so the venv built in stage 1
# runs on exactly the same interpreter in stage 2. Bump versions here deliberately.
ARG PYTHON_IMAGE=python:3.12.14-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.9.30

FROM ${UV_IMAGE} AS uv

# --- Stage 1: uv resolves and installs dependencies into /app/.venv ---
FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Embedding model is downloaded at build time, so the container starts without network.
# Must match EMBEDDING_MODEL at runtime (compose passes the same value as a build arg).
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV HF_HOME=/opt/hf
RUN /app/.venv/bin/python -c "import sys; from sentence_transformers import SentenceTransformer as M; M(sys.argv[1], device='cpu')" "$EMBEDDING_MODEL"

# --- Dev stage: runtime venv + dev group (pytest, ruff) for `docker compose run --rm test` ---
# Only built on demand (target: dev); the default build target is still the runtime stage below.
# The code is not copied: compose mounts the repo at /src, so tests see the working tree.
FROM builder AS dev
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project
WORKDIR /src
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RUFF_NO_CACHE=true \
    HF_HUB_OFFLINE=1
CMD ["sh", "-c", "ruff check . && ruff format --check . && pytest -q -p no:cacheprovider"]

# --- Stage 2: runtime — only the venv and the code, no uv, no build cache ---
FROM ${PYTHON_IMAGE}

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TQDM_DISABLE=1

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /opt/hf /opt/hf
COPY migrations ./migrations
# The corpus (~100 KB) for the ingest CLI where nothing mounts it (the k8s ingest Job);
# docker compose still mounts ./data over it, so editing a document needs no rebuild.
COPY data/corpus ./data/corpus
COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
