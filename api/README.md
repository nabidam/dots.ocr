# dots.ocr API

Standalone FastAPI service for processing one image or PDF with dots.ocr through an OpenAI-compatible vLLM server.

The service is intentionally isolated under `api/`; it does not modify the repository's existing demos or engine implementation.

## API documentation

When the service is running:

- Swagger UI: <http://localhost:8080/docs>
- ReDoc: <http://localhost:8080/redoc>
- OpenAPI JSON: <http://localhost:8080/openapi.json>

The Swagger page documents the multipart upload, response structure, base64 page images, success examples, and 400/422/502 error examples.

## Endpoints

### `GET /health`

Returns API liveness. It does not contact vLLM.

```json
{
  "status": "ok",
  "service": "dots-ocr-api"
}
```

### `POST /ocr`

Accepts one multipart file using the field name `file`.

Supported input formats:

- PDF
- JPEG/JPG
- PNG
- WebP
- BMP
- TIFF

Example:

```bash
curl -X POST http://localhost:8080/ocr \
  -F 'file=@../demo/demo_image1.jpg'
```

The response contains one ordered entry per source page:

```json
{
  "filename": "invoice.pdf",
  "total_pages": 1,
  "pages": [
    {
      "page_number": 1,
      "image_base64": "data:image/png;base64,...",
      "image_media_type": "image/png",
      "ocr_result": [
        {
          "bbox": [72, 72, 540, 120],
          "category": "Text",
          "text": "Example document text."
        }
      ]
    }
  ]
}
```

`image_base64` is already a browser-ready `data:image/png;base64,...` data URL and can be assigned directly to an image `src` attribute.

Page images dominate the response size for multi-page documents. A client that already has the source document can request `?include_images=false` on `/ocr` and `/ocr/stream`; `image_base64` and `image_media_type` are then `null` and only the OCR JSON is returned. Images are included by default.

For structured layout results, `Picture` cells are post-processed with dots.ocr's `fillLayoutJsonPictures` helper. Their `text` value contains a cropped, browser-ready PNG data URL for that picture.

The default prompt is `prompt_layout_all_en`, which asks dots.ocr for structured layout/OCR JSON. If the model returns valid JSON, `ocr_result` is a JSON object or array. If it returns invalid JSON, the raw model text is preserved as a string.

## `POST /ocr/stream`

Same input as `POST /ocr`, but results are streamed as newline delimited JSON (`application/x-ndjson`) so each page arrives as soon as it is finished. Time to first page no longer scales with document length.

```bash
curl -N -X POST http://localhost:8080/ocr/stream \
  -F 'file=@../demo/demo_pdf1.pdf'
```

Each line is a complete JSON document:

```
{"type":"meta","filename":"invoice.pdf","total_pages":2}
{"type":"page","page_number":1,"image_base64":"data:image/png;base64,...","image_media_type":"image/png","ocr_result":[]}
{"type":"page","page_number":2,"image_base64":"data:image/png;base64,...","image_media_type":"image/png","ocr_result":[]}
{"type":"done","completed_pages":2}
```

`page` lines carry the same fields as an entry in the `/ocr` `pages` array and arrive in source page order.

Rejected uploads still fail with `400` before the stream starts. Once the first line is sent the status is committed to `200`, so a later failure is reported as a terminal `{"type":"error","detail":...,"request_id":...}` line instead. A stream that ends without a `done` or `error` line was truncated and should be treated as failed.

Closing the connection stops the remaining pages from being processed.

When running behind a reverse proxy, disable response buffering for this route (`proxy_buffering off;` in nginx); the response already sets `X-Accel-Buffering: no`.

## Configuration

Defaults are in [`config.yaml`](config.yaml). Environment variables override YAML values. Copy [`.env.example`](.env.example) to `.env` for local secrets and overrides; `.env` is ignored by git.

Important settings:

| Environment variable | YAML setting | Default | Purpose |
| --- | --- | --- | --- |
| `API_HOST` | `app.host` | `0.0.0.0` | API bind host |
| `API_PORT` | `app.port` | `8080` | API bind port |
| `LOG_LEVEL` | `app.log_level` | `INFO` | Console/file log level |
| `MAX_UPLOAD_SIZE_MB` | `app.max_upload_size_mb` | `50` | Maximum upload size |
| `MAX_CONCURRENT_INFERENCES` | `app.max_concurrent_inferences` | `1` | Inference concurrency limit |
| `CORS_ORIGINS` | `app.cors_origins` | `["*"]` | Allowed CORS origins (comma-separated or list) |
| `VLLM_PROTOCOL` | `vllm.protocol` | `http` | vLLM protocol |
| `VLLM_HOST` | `vllm.host` | `localhost` | vLLM hostname |
| `VLLM_PORT` | `vllm.port` | `8000` | vLLM custom port |
| `VLLM_MODEL_NAME` | `vllm.model_name` | `model` | vLLM served model name |
| `VLLM_API_KEY` | — | `0` | API key passed to vLLM |
| `OCR_PROMPT_MODE` | `ocr.prompt_mode` | `prompt_layout_all_en` | dots.ocr task prompt |
| `OCR_TEMPERATURE` | `ocr.temperature` | `0.1` | Sampling temperature |
| `OCR_TOP_P` | `ocr.top_p` | `0.9` | Nucleus sampling value |
| `OCR_MAX_COMPLETION_TOKENS` | `ocr.max_completion_tokens` | `32768` | Maximum model output tokens |
| `PDF_DPI` | `ocr.pdf_dpi` | `200` | PDF page rendering DPI |
| `LOG_FILE` | `logging.file` | `logs/dots-ocr-api.log` | Rotating log path |
| `LOG_MAX_BYTES` | `logging.max_bytes` | `10485760` | Maximum size per log file |
| `LOG_BACKUP_COUNT` | `logging.backup_count` | `5` | Number of rotated log files |

Supported prompt modes are defined by dots.ocr, including `prompt_layout_all_en`, `prompt_layout_only_en`, `prompt_ocr`, `prompt_web_parsing`, `prompt_scene_spotting`, and `prompt_image_to_svg`.

The vLLM model name must match its `--served-model-name` value. The repository README's vLLM example uses `model`.

## Run locally with uv

From the repository root:

```bash
cd api
cp .env.example .env
uv sync
uv run python -m app.run
```

The launcher automatically adds the repository root to Python's import path so the local `dots_ocr` engine package is available. If launching Uvicorn directly, run it from `api/`:

```bash
uv run uvicorn app.main:app
```

If `ModuleNotFoundError: No module named 'dots_ocr'` still appears, confirm that the current directory is `api/` and that this is the repository containing the `dots_ocr/` directory.

The API expects vLLM to already be running. For local vLLM, set values such as:

```dotenv
VLLM_HOST=localhost
VLLM_PORT=8000
VLLM_MODEL_NAME=model
VLLM_API_KEY=0
```

## Run with Docker Compose

The Compose file starts the API only; it assumes vLLM is running separately on the host or another reachable machine.

```bash
cp api/.env.example api/.env
docker compose -f api/docker-compose.yml up --build
```

By default, the container connects to `host.docker.internal:8000`. Override `VLLM_HOST`, `VLLM_PORT`, and `VLLM_MODEL_NAME` in the shell or `api/.env`.

Rotating logs are written to `api/logs/` through the Compose volume mount.

## Error responses

- `400`: empty, unsupported, oversized, or unreadable upload.
- `422`: missing or invalid multipart request fields.
- `502`: vLLM or dots.ocr inference failure.

Error responses include a `detail` message and, for application errors, a `request_id`. Every response also includes the `X-Request-ID` header; clients may provide their own request ID for tracing.

## Operational notes

- PDF pages are rendered one at a time, on demand, and results are always emitted in source page order.
- `MAX_CONCURRENT_INFERENCES` also sets how many pages of one document are processed in parallel; raise it only if the vLLM backend has spare capacity.
- API responses can be large because every page image is returned as base64.
- The API has no client authentication layer. Place it behind an authenticated gateway or add authentication before exposing it publicly.
- `VLLM_API_KEY` is only used for the vLLM OpenAI-compatible client and is never written to logs.
- The service does not run the vLLM model server or download model weights.
