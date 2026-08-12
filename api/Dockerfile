FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
RUN uv sync --no-dev

COPY app /app/app
COPY config.yaml /app/config.yaml
COPY dots_ocr /app/dots_ocr

EXPOSE 8080

CMD ["uv", "run", "--no-sync", "python", "-m", "app.run"]
