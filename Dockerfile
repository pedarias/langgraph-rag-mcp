# Use Python 3.13 slim image as a lean base
FROM python:3.13-slim AS builder

# Set the working directory in the container
WORKDIR /app

# Set environment variables for python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies required for building some python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt pyproject.toml uv.lock README.md ./
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /bin/uv
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy

# Install CPU-only PyTorch. sentence-transformers requires it, and this is an optimization to keep the image size small.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra embeddings --no-dev --no-install-project

# Install the rest of the Python dependencies
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra embeddings --no-dev --no-editable

# Copy the rest of your application code
FROM python:3.13-slim
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANGGRAPH_MCP_DATA_DIR=/data \
    HF_HOME=/models
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app \
    && mkdir /data /models && chown app:app /data /models
COPY --from=builder /app/.venv /app/.venv
COPY langgraph-mcp.py .
USER app

# Set the default command to run the MCP server
ENTRYPOINT ["langgraph-rag"]
CMD ["serve"]