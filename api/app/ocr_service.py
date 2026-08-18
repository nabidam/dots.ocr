"""Document decoding and dots.ocr inference service."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import fitz
from PIL import Image, UnidentifiedImageError

from dots_ocr.model.inference import inference_with_vllm
from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.utils.doc_utils import fitz_doc_to_image
from dots_ocr.utils.format_transformer import fillLayoutJsonPictures
from dots_ocr.utils.image_utils import PILimage_to_base64

from app.config import Settings

logger = logging.getLogger(__name__)


class DocumentInputError(ValueError):
    """Raised when an uploaded document cannot be processed."""


class OcrInferenceError(RuntimeError):
    """Raised when the configured vLLM server cannot produce an OCR result."""


class PageSource(ABC):
    """A decoded upload that renders one page at a time.

    Rendering lazily matters for multi-page PDFs: a 50 page document rendered
    eagerly costs a long CPU stall and holds every page bitmap in memory before
    the first inference can start.
    """

    total_pages: int

    @abstractmethod
    async def render(self, page_number: int) -> Image.Image:
        """Return the 1-based page as an RGB image."""

    def close(self) -> None:
        """Release decoder resources. Safe to call more than once."""


class _ImagePageSource(PageSource):
    """A single-page source backed by an already decoded image."""

    total_pages = 1

    def __init__(self, image: Image.Image) -> None:
        self._image = image

    async def render(self, page_number: int) -> Image.Image:
        return self._image


class _PdfPageSource(PageSource):
    """A PDF rendered page by page at the configured DPI."""

    def __init__(self, document: fitz.Document, dpi: int) -> None:
        self._document = document
        self._dpi = dpi
        # MuPDF documents are not safe to use from several threads at once, so
        # renders are serialized even when pages are processed concurrently.
        self._render_lock = asyncio.Lock()
        self.total_pages = document.page_count

    async def render(self, page_number: int) -> Image.Image:
        async with self._render_lock:
            return await asyncio.to_thread(self._render_page, page_number)

    def _render_page(self, page_number: int) -> Image.Image:
        return fitz_doc_to_image(self._document[page_number - 1], target_dpi=self._dpi)

    def close(self) -> None:
        if not self._document.is_closed:
            self._document.close()


def _decode_image(data: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as opened_image:
            opened_image.verify()
        with Image.open(io.BytesIO(data)) as opened_image:
            return opened_image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise DocumentInputError("The uploaded file is not a readable image") from exc


def _open_pdf(data: bytes, dpi: int) -> _PdfPageSource:
    try:
        document = fitz.open(stream=data, filetype="pdf")
    except (RuntimeError, ValueError) as exc:
        raise DocumentInputError("The uploaded file is not a readable PDF") from exc

    if document.page_count < 1:
        document.close()
        raise DocumentInputError("The uploaded PDF does not contain any pages")
    return _PdfPageSource(document, dpi)


async def _open_page_source(
    data: bytes, content_type: str | None, filename: str, dpi: int
) -> PageSource:
    if not data:
        raise DocumentInputError("The uploaded file is empty")

    lower_filename = filename.lower()
    is_pdf = content_type == "application/pdf" or lower_filename.endswith(".pdf") or data.startswith(b"%PDF-")
    if is_pdf:
        return await asyncio.to_thread(_open_pdf, data, dpi)

    is_image = (content_type or "").startswith("image/") or lower_filename.rsplit(".", 1)[-1] in {
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp",
        "tif",
        "tiff",
    }
    if not is_image:
        raise DocumentInputError("Only image files and PDFs are supported")
    return _ImagePageSource(await asyncio.to_thread(_decode_image, data))


def _image_to_base64(image: Image.Image) -> str:
    """Return a browser-ready PNG data URL."""

    return PILimage_to_base64(image, format="PNG")


def _parse_ocr_result(raw_result: str, image: Image.Image) -> Any:
    """Parse JSON and embed cropped Picture cells as ready-to-use data URLs."""

    if not isinstance(raw_result, str):
        return raw_result

    candidate = raw_result.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = candidate[3:-3].strip()
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].lstrip()
    try:
        parsed_result = json.loads(candidate)
    except json.JSONDecodeError:
        logger.warning("dots.ocr returned non-JSON output; returning it as a string")
        return raw_result

    if not isinstance(parsed_result, list):
        return parsed_result

    try:
        return fillLayoutJsonPictures(image, parsed_result, text_key="text")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        logger.warning(
            "Could not fill Picture cells in dots.ocr output; returning parsed JSON: %s",
            exc,
        )
        return parsed_result


def _build_page_result(page_number: int, image: Image.Image, raw_result: str) -> dict[str, Any]:
    """Assemble one page payload. CPU bound; call from a worker thread."""

    return {
        "page_number": page_number,
        "image_base64": _image_to_base64(image),
        "image_media_type": "image/png",
        "ocr_result": _parse_ocr_result(raw_result, image),
    }


class OcrDocument:
    """An opened upload whose pages are rendered and OCR'd on demand."""

    def __init__(self, service: OcrService, source: PageSource) -> None:
        self._service = service
        self._source = source

    @property
    def total_pages(self) -> int:
        return self._source.total_pages

    async def pages(self) -> AsyncIterator[dict[str, Any]]:
        """Yield one result per page, in source page order."""

        prompt = dict_promptmode_to_prompt[self._service.settings.ocr.prompt_mode]
        for page_number in range(1, self.total_pages + 1):
            yield await self._process_page(page_number, prompt)

    async def _process_page(self, page_number: int, prompt: str) -> dict[str, Any]:
        image = await self._source.render(page_number)
        raw_result = await self._service.infer_page(image, prompt)
        # PNG encoding and Picture cropping are CPU bound and would otherwise
        # block the event loop for every other request while a page is emitted.
        return await asyncio.to_thread(_build_page_result, page_number, image, raw_result)

    def close(self) -> None:
        self._source.close()


class OcrService:
    """Coordinates document decoding and ordered calls to the vLLM-backed engine."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._inference_slots = asyncio.Semaphore(
            settings.app.max_concurrent_inferences
        )
        logger.info(
            "Initializing OcrService with vLLM target: %s://%s:%s/v1 | Model: '%s' | API Key: '%s' | Prompt Mode: '%s'",
            settings.vllm.protocol,
            settings.vllm.host,
            settings.vllm.port,
            settings.vllm.model_name,
            "***" if settings.vllm.api_key and settings.vllm.api_key != "0" else settings.vllm.api_key,
            settings.ocr.prompt_mode,
        )

    async def open_document(
        self, data: bytes, content_type: str | None, filename: str
    ) -> OcrDocument:
        """Decode the upload far enough to know its page count.

        Raises DocumentInputError before any result is produced, so callers can
        still answer with a normal error response.
        """

        source = await _open_page_source(
            data,
            content_type,
            filename,
            dpi=self.settings.ocr.pdf_dpi,
        )
        return OcrDocument(self, source)

    async def process(self, data: bytes, content_type: str | None, filename: str) -> list[dict[str, Any]]:
        document = await self.open_document(data, content_type, filename)
        try:
            return [page async for page in document.pages()]
        finally:
            document.close()

    async def infer_page(self, image: Image.Image, prompt: str) -> str:
        async with self._inference_slots:
            try:
                # dots_ocr.model.inference reads the OpenAI key from API_KEY.
                os.environ["API_KEY"] = self.settings.vllm.api_key or "0"
                result = await asyncio.to_thread(
                    inference_with_vllm,
                    image,
                    prompt,
                    protocol=self.settings.vllm.protocol,
                    ip=self.settings.vllm.host,
                    port=self.settings.vllm.port,
                    temperature=self.settings.ocr.temperature,
                    top_p=self.settings.ocr.top_p,
                    max_completion_tokens=self.settings.ocr.max_completion_tokens,
                    model_name=self.settings.vllm.model_name,
                )
            except Exception as exc:
                logger.error(
                    "vLLM inference request failed for endpoint %s://%s:%s/v1 (model='%s'): %s",
                    self.settings.vllm.protocol,
                    self.settings.vllm.host,
                    self.settings.vllm.port,
                    self.settings.vllm.model_name,
                    exc,
                    exc_info=True,
                )
                raise OcrInferenceError(
                    f"dots.ocr inference failed on {self.settings.vllm.protocol}://{self.settings.vllm.host}:{self.settings.vllm.port} (model='{self.settings.vllm.model_name}'): {exc}"
                ) from exc

        if result is None:
            logger.error(
                "vLLM server returned empty result for endpoint %s://%s:%s/v1 (model='%s')",
                self.settings.vllm.protocol,
                self.settings.vllm.host,
                self.settings.vllm.port,
                self.settings.vllm.model_name,
            )
            raise OcrInferenceError(
                f"dots.ocr returned no result from {self.settings.vllm.protocol}://{self.settings.vllm.host}:{self.settings.vllm.port} (model='{self.settings.vllm.model_name}')"
            )
        return result


__all__ = [
    "DocumentInputError",
    "OcrDocument",
    "OcrInferenceError",
    "OcrService",
]
