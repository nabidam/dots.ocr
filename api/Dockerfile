FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/workspace/dots.ocr

WORKDIR /app

COPY api/pyproject.toml /app/pyproject.toml
RUN uv sync --no-dev

COPY api/app /app/app
COPY api/config.yaml /app/config.yaml
COPY dots_ocr /workspace/dots.ocr/dots_ocr

EXPOSE 8080

CMD ["uv", "run", "--no-sync", "python", "-m", "app.run"]
