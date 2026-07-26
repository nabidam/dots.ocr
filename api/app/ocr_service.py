"""Document decoding and dots.ocr inference service."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import fitz
from PIL import Image, UnidentifiedImageError

from dots_ocr.model.inference import inference_with_vllm
from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.utils.doc_utils import fitz_doc_to_image

from app.config import Settings

logger = logging.getLogger(__name__)


class DocumentInputError(ValueError):
    """Raised when an uploaded document cannot be processed."""


class OcrInferenceError(RuntimeError):
    """Raised when the configured vLLM server cannot produce an OCR result."""


@dataclass(frozen=True)
class PageImage:
    """An ordered page image ready for inference."""

    page_number: int
    image: Image.Image


def _load_image(data: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as opened_image:
            opened_image.verify()
        with Image.open(io.BytesIO(data)) as opened_image:
            return opened_image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise DocumentInputError("The uploaded file is not a readable image") from exc


def _load_pdf_pages(data: bytes, dpi: int) -> list[PageImage]:
    try:
        document = fitz.open(stream=data, filetype="pdf")
    except (RuntimeError, ValueError) as exc:
        raise DocumentInputError("The uploaded file is not a readable PDF") from exc

    pages: list[PageImage] = []
    try:
        for page_number, page in enumerate(document, start=1):
            pages.append(
                PageImage(
                    page_number=page_number,
                    image=fitz_doc_to_image(page, target_dpi=dpi),
                )
            )
    finally:
        document.close()

    if not pages:
        raise DocumentInputError("The uploaded PDF does not contain any pages")
    return pages


def _decode_pages(data: bytes, content_type: str | None, filename: str, dpi: int) -> list[PageImage]:
    if not data:
        raise DocumentInputError("The uploaded file is empty")

    lower_filename = filename.lower()
    is_pdf = content_type == "application/pdf" or lower_filename.endswith(".pdf") or data.startswith(b"%PDF-")
    if is_pdf:
        return _load_pdf_pages(data, dpi)

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
    return [PageImage(page_number=1, image=_load_image(data))]


def _image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _parse_ocr_result(raw_result: str) -> Any:
    """Parse JSON model output while preserving non-JSON output for diagnostics."""

    if not isinstance(raw_result, str):
        return raw_result

    candidate = raw_result.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = candidate[3:-3].strip()
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].lstrip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        logger.warning("dots.ocr returned non-JSON output; returning it as a string")
        return raw_result


class OcrService:
    """Coordinates document decoding and ordered calls to the vLLM-backed engine."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._inference_slots = asyncio.Semaphore(
            settings.app.max_concurrent_inferences
        )

    async def process(self, data: bytes, content_type: str | None, filename: str) -> list[dict[str, Any]]:
        pages = _decode_pages(
            data,
            content_type,
            filename,
            dpi=self.settings.ocr.pdf_dpi,
        )
        prompt = dict_promptmode_to_prompt[self.settings.ocr.prompt_mode]
        results: list[dict[str, Any]] = []

        for page in pages:
            raw_result = await self._infer_page(page.image, prompt)
            results.append(
                {
                    "page_number": page.page_number,
                    "image_base64": _image_to_base64(page.image),
                    "image_media_type": "image/png",
                    "ocr_result": _parse_ocr_result(raw_result),
                }
            )
        return results

    async def _infer_page(self, image: Image.Image, prompt: str) -> str:
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
                raise OcrInferenceError(
                    "dots.ocr inference failed; check the vLLM server and API configuration"
                ) from exc

        if result is None:
            raise OcrInferenceError(
                "dots.ocr returned no result; check the vLLM server and API configuration"
            )
        return result


__all__ = [
    "DocumentInputError",
    "OcrInferenceError",
    "OcrService",
]
