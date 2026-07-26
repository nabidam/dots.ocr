"""Configured Uvicorn launcher for the standalone API."""

from __future__ import annotations

import uvicorn

from app.main import app, settings


def main() -> None:
    """Run the API using host and port from config.yaml or environment."""

    uvicorn.run(
        app,
        host=settings.app.host,
        port=settings.app.port,
        log_config=None,
    )


if __name__ == "__main__":
    main()


__all__ = ["main"]
