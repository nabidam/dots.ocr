"""FastAPI application exposing health and document OCR endpoints."""

from __future__ import annotations

import json
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
API_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if (API_ROOT / "dots_ocr").is_dir() and str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))
elif (REPOSITORY_ROOT / "dots_ocr").is_dir() and str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, load_settings
from app.logging_setup import configure_logging
from app.ocr_service import DocumentInputError, OcrInferenceError, OcrService

NDJSON_MEDIA_TYPE = "application/x-ndjson"

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
        description="Browser-ready PNG data URL containing this page image.",
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
                                "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
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
                                "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
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


class OcrStreamMeta(BaseModel):
    """First NDJSON line of a streaming response."""

    type: Literal["meta"] = "meta"
    filename: str = Field(description="Original uploaded filename.")
    total_pages: int = Field(ge=1, description="Number of pages that will be streamed.")


class OcrStreamPage(OcrPageResponse):
    """One page result, emitted as soon as that page finishes."""

    type: Literal["page"] = "page"


class OcrStreamDone(BaseModel):
    """Terminal line of a successful streaming response."""

    type: Literal["done"] = "done"
    completed_pages: int = Field(ge=0, description="Number of page lines emitted.")


class OcrStreamError(BaseModel):
    """Terminal line emitted when processing fails after streaming started.

    The HTTP status is already `200` by then, so failures are reported in-band
    instead of through the regular exception handlers.
    """

    type: Literal["error"] = "error"
    detail: str = Field(description="Human readable failure reason.")
    request_id: str = Field(description="Correlation id, also sent as X-Request-ID.")


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.app.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
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


async def _read_upload(request: Request, file: UploadFile) -> tuple[str, bytes]:
    """Read the upload and enforce the configured size limit."""

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
    return filename, data


def _ndjson_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


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
        "`image_base64` is already a browser-ready `data:image/png;base64,...` URL."
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

    filename, data = await _read_upload(request, file)
    pages = await request.app.state.ocr_service.process(data, file.content_type, filename)
    logger.info(
        "Completed OCR request_id=%s filename=%s pages=%s",
        request.state.request_id,
        filename,
        len(pages),
    )
    return OcrResponse(filename=filename, total_pages=len(pages), pages=pages)


@app.post(
    "/ocr/stream",
    summary="OCR one image or PDF, streaming one page at a time",
    description=(
        "Same input as `POST /ocr`, but results are streamed as newline "
        "delimited JSON (`application/x-ndjson`) so a page is delivered as soon "
        "as it finishes instead of after the whole document.\n\n"
        "Line order is: one `meta` line, one `page` line per source page in page "
        "order, then either `done` or `error`.\n\n"
        "Because the response status is committed with the first line, a failure "
        "that happens mid-document is reported as a final `error` line under "
        "HTTP 200. Rejected uploads still fail with 400 before streaming starts.\n\n"
        "### Example\n\n"
        "```bash\n"
        "curl -N -X POST http://localhost:8080/ocr/stream \\\n"
        "  -F 'file=@demo/demo_pdf1.pdf'\n"
        "```\n\n"
        "Clients must treat each line as a complete JSON document and stop at "
        "`done` or `error`. A stream that ends without either line was truncated."
    ),
    response_description="Newline delimited JSON stream of per-page results.",
    responses={
        200: {
            "description": "NDJSON stream of meta, page, and terminal lines.",
            "content": {
                NDJSON_MEDIA_TYPE: {
                    "schema": {
                        "oneOf": [
                            OcrStreamMeta.model_json_schema(),
                            OcrStreamPage.model_json_schema(),
                            OcrStreamDone.model_json_schema(),
                            OcrStreamError.model_json_schema(),
                        ]
                    },
                    "example": (
                        '{"type":"meta","filename":"invoice.pdf","total_pages":2}\n'
                        '{"type":"page","page_number":1,"image_base64":"data:image/png;base64,...",'
                        '"image_media_type":"image/png","ocr_result":[]}\n'
                        '{"type":"page","page_number":2,"image_base64":"data:image/png;base64,...",'
                        '"image_media_type":"image/png","ocr_result":[]}\n'
                        '{"type":"done","completed_pages":2}\n'
                    ),
                }
            },
        },
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
    },
    tags=["ocr"],
)
async def process_ocr_stream(
    request: Request,
    file: UploadFile = File(
        ...,
        description="One image or PDF to process. Maximum size is configured by `max_upload_size_mb`.",
    ),
) -> StreamingResponse:
    """Stream one NDJSON line per page as soon as that page is finished."""

    filename, data = await _read_upload(request, file)
    request_id = request.state.request_id
    # Decoding happens before the response starts so an unusable upload is
    # still answered with a regular 400 by the DocumentInputError handler.
    document = await request.app.state.ocr_service.open_document(
        data, file.content_type, filename
    )

    async def emit() -> AsyncIterator[str]:
        completed_pages = 0
        try:
            yield _ndjson_line(
                OcrStreamMeta(filename=filename, total_pages=document.total_pages).model_dump()
            )
            async for page in document.pages():
                if await request.is_disconnected():
                    logger.info(
                        "Client disconnected request_id=%s filename=%s after_pages=%s",
                        request_id,
                        filename,
                        completed_pages,
                    )
                    return
                completed_pages += 1
                yield _ndjson_line({"type": "page", **page})
            yield _ndjson_line(OcrStreamDone(completed_pages=completed_pages).model_dump())
            logger.info(
                "Completed OCR stream request_id=%s filename=%s pages=%s",
                request_id,
                filename,
                completed_pages,
            )
        except (DocumentInputError, OcrInferenceError) as exc:
            logger.error(
                "OCR stream failed request_id=%s filename=%s after_pages=%s: %s",
                request_id,
                filename,
                completed_pages,
                exc,
                exc_info=True,
            )
            yield _ndjson_line(
                OcrStreamError(detail=str(exc), request_id=request_id).model_dump()
            )
        except Exception:  # the client must not be left hanging on an unexpected fault
            logger.exception(
                "Unexpected OCR stream error request_id=%s filename=%s after_pages=%s",
                request_id,
                filename,
                completed_pages,
            )
            yield _ndjson_line(
                OcrStreamError(
                    detail="Unexpected error while processing the document",
                    request_id=request_id,
                ).model_dump()
            )
        finally:
            document.close()

    return StreamingResponse(
        emit(),
        media_type=NDJSON_MEDIA_TYPE,
        headers={
            "Cache-Control": "no-cache",
            # Tell nginx-style proxies not to buffer, which would defeat streaming.
            "X-Accel-Buffering": "no",
        },
    )


__all__ = ["app"]
