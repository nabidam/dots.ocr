"""FastAPI application exposing health and document OCR endpoints."""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal

# The API is intentionally kept under api/, while the engine package is at the
# repository root. Make both `python -m app.run` and `uvicorn app.main:app`
# work when launched from the api directory without requiring PYTHONPATH setup.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if (REPOSITORY_ROOT / "dots_ocr").is_dir() and str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, load_settings
from app.logging_setup import configure_logging
from app.ocr_service import DocumentInputError, OcrInferenceError, OcrService

settings = load_settings()
configure_logging(settings.app.log_level, settings.logging)
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Liveness response."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"status": "ok", "service": "dots-ocr-api"}]
        }
    )

    status: str
    service: str


class OcrPageResponse(BaseModel):
    """OCR response for one document page."""

    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(
        ge=1,
        description="1-based page number in the original upload.",
    )
    image_base64: str = Field(
        description="PNG bytes for this page, base64 encoded without a data-URL prefix.",
    )
    image_media_type: Literal["image/png"] = Field(
        description="Media type of image_base64. The API normalizes page images to PNG.",
    )
    ocr_result: Any = Field(
        description=(
            "Parsed JSON returned by dots.ocr. If the model returns invalid JSON, "
            "the raw model text is returned instead."
        ),
    )


class OcrResponse(BaseModel):
    """Complete OCR response in source-page order."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "filename": "invoice.pdf",
                    "total_pages": 2,
                    "pages": [
                        {
                            "page_number": 1,
                            "image_base64": (
                                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
                                "+A8AAQUBAScY42YAAAAASUVORK5CYII="
                            ),
                            "image_media_type": "image/png",
                            "ocr_result": [
                                {
                                    "bbox": [72, 72, 540, 120],
                                    "category": "Text",
                                    "text": "Example document text.",
                                }
                            ],
                        },
                        {
                            "page_number": 2,
                            "image_base64": (
                                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
                                "+A8AAQUBAScY42YAAAAASUVORK5CYII="
                            ),
                            "image_media_type": "image/png",
                            "ocr_result": [
                                {
                                    "bbox": [72, 72, 540, 120],
                                    "category": "Text",
                                    "text": "Second page text.",
                                }
                            ],
                        },
                    ],
                }
            ]
        }
    )

    filename: str = Field(description="Original uploaded filename.")
    total_pages: int = Field(ge=1, description="Number of pages processed.")
    pages: list[OcrPageResponse] = Field(
        min_length=1,
        description="OCR results in original page order.",
    )


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    application.state.settings = settings
    application.state.ocr_service = OcrService(settings)
    logger.info(
        "Starting %s with vLLM target %s://%s:%s model=%s prompt_mode=%s",
        settings.app.name,
        settings.vllm.protocol,
        settings.vllm.host,
        settings.vllm.port,
        settings.vllm.model_name,
        settings.ocr.prompt_mode,
    )
    yield
    logger.info("Stopping %s", settings.app.name)


OPENAPI_TAGS = [
    {
        "name": "system",
        "description": "Service liveness and operational endpoints.",
    },
    {
        "name": "ocr",
        "description": "Image and PDF OCR powered by dots.ocr through vLLM.",
    },
]

app = FastAPI(
    title="dots.ocr API",
    version="0.1.0",
    description=(
        "A small HTTP API for processing images and PDFs with dots.ocr.\n\n"
        "Upload one file as multipart form field `file`. PDF pages are rendered "
        "and processed individually; the response preserves source page order and "
        "includes both the rendered page image and that page's OCR result.\n\n"
        "The default `prompt_layout_all_en` mode returns structured layout/OCR JSON. "
        "Configure `OCR_PROMPT_MODE` when a different dots.ocr task is required."
    ),
    openapi_tags=OPENAPI_TAGS,
    contact={"name": "dots.ocr API maintainers"},
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next: Any) -> JSONResponse:
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    started_at = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled request error request_id=%s path=%s", request_id, request.url.path)
        raise
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.exception_handler(DocumentInputError)
async def document_input_error_handler(request: Request, exc: DocumentInputError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "request_id": request.state.request_id},
    )


@app.exception_handler(OcrInferenceError)
async def ocr_inference_error_handler(request: Request, exc: OcrInferenceError) -> JSONResponse:
    logger.error("OCR inference error request_id=%s: %s", request.state.request_id, exc, exc_info=True)
    return JSONResponse(
        status_code=502,
        content={"detail": str(exc), "request_id": request.state.request_id},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API liveness",
    description=(
        "Returns `200 OK` when the API process is running. This is a liveness "
        "check and does not make a request to the vLLM server."
    ),
    response_description="Service liveness status.",
    tags=["system"],
)
async def health() -> HealthResponse:
    """Return service liveness."""

    return HealthResponse(status="ok", service=settings.app.name)


@app.post(
    "/ocr",
    response_model=OcrResponse,
    summary="OCR one image or PDF",
    description=(
        "Accepts one image or PDF upload using multipart field `file`. Supported "
        "images include JPEG, PNG, WebP, BMP, and TIFF. PDFs are rendered at the "
        "configured DPI, then each page is sent to dots.ocr sequentially.\n\n"
        "### Example\n\n"
        "```bash\n"
        "curl -X POST http://localhost:8080/ocr \\\n"
        "  -F 'file=@demo/demo_image1.jpg'\n"
        "```\n\n"
        "The response contains one entry in `pages` per source page. "
        "`image_base64` is raw base64 PNG data; prepend `data:image/png;base64,` "
        "when a browser data URL is needed."
    ),
    response_description="Per-page images and dots.ocr JSON results.",
    responses={
        400: {
            "description": "The upload is empty, unsupported, too large, or unreadable.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Only image files and PDFs are supported",
                        "request_id": "9f3f1c5e-5b16-4c4c-bd7e-9e28d7cc2a91",
                    }
                }
            },
        },
        422: {
            "description": "The multipart `file` field was not provided.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "file"],
                                "msg": "Field required",
                                "type": "missing",
                            }
                        ]
                    }
                }
            },
        },
        502: {
            "description": "The vLLM/dots.ocr inference request failed.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": (
                            "dots.ocr inference failed; check the vLLM server and API configuration"
                        ),
                        "request_id": "9f3f1c5e-5b16-4c4c-bd7e-9e28d7cc2a91",
                    }
                }
            },
        },
    },
    tags=["ocr"],
)
async def process_ocr(
    request: Request,
    file: UploadFile = File(
        ...,
        description="One image or PDF to process. Maximum size is configured by `max_upload_size_mb`.",
    ),
) -> OcrResponse:
    """OCR one image or PDF and return one base64 image/result pair per page."""

    filename = file.filename or "upload"
    data = await file.read(settings.max_upload_size_bytes + 1)
    if len(data) > settings.max_upload_size_bytes:
        raise DocumentInputError(
            f"Uploaded file exceeds the {settings.app.max_upload_size_mb} MB limit"
        )

    logger.info(
        "Starting OCR request_id=%s filename=%s content_type=%s size_bytes=%s",
        request.state.request_id,
        filename,
        file.content_type,
        len(data),
    )
    pages = await request.app.state.ocr_service.process(data, file.content_type, filename)
    logger.info(
        "Completed OCR request_id=%s filename=%s pages=%s",
        request.state.request_id,
        filename,
        len(pages),
    )
    return OcrResponse(filename=filename, total_pages=len(pages), pages=pages)


__all__ = ["app"]
