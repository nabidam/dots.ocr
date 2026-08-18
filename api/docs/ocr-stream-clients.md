# Consuming `POST /ocr/stream`

`POST /ocr/stream` returns newline delimited JSON (`application/x-ndjson`). One page result is written as soon as that page finishes, so a client sees page 1 after a single inference instead of after the whole document.

This guide covers the wire protocol and a working client in curl, fetch, axios, PHP cURL, Python, and Go.

## Request

| | |
| --- | --- |
| Method | `POST` |
| Path | `/ocr/stream` |
| Body | `multipart/form-data`, one file in field `file` |
| Query | `include_images` (default `true`) |
| Response | `200` + `application/x-ndjson`, or `400`/`422` before the stream starts |

`include_images=false` drops `image_base64` from every page line. For a 50 page PDF this is the difference between a ~100 MB response and a few hundred kilobytes, so use it whenever the client already holds the source document.

## Line protocol

Every line is one complete JSON document. Lines arrive in this order:

| `type` | When | Fields |
| --- | --- | --- |
| `meta` | Once, first | `filename`, `total_pages` |
| `page` | Once per page, in page order | `page_number`, `image_base64`, `image_media_type`, `image_width`, `image_height`, `source_width`, `source_height`, `ocr_result` |
| `done` | Once, last, on success | `completed_pages` |
| `error` | Once, last, on failure | `detail`, `request_id` |

```
{"type":"meta","filename":"invoice.pdf","total_pages":2}
{"type":"page","page_number":1,"image_base64":"data:image/png;base64,...","image_media_type":"image/png","image_width":1654,"image_height":2339,"source_width":1654,"source_height":2339,"ocr_result":[]}
{"type":"page","page_number":2,"image_base64":"data:image/png;base64,...","image_media_type":"image/png","image_width":1654,"image_height":2339,"source_width":1654,"source_height":2339,"ocr_result":[]}
{"type":"done","completed_pages":2}
```

### Rules every client must follow

1. **A `200` status is not success.** The status is committed with the first line, so a failure part way through a document arrives as a final `error` line under `200`.
2. **Treat a missing terminal line as failure.** A stream that ends without `done` or `error` was truncated by a network or proxy fault.
3. **Buffer partial lines.** A chunk boundary can land in the middle of a line, and a page line carrying a base64 image is often several megabytes. Split on `\n` and keep the remainder.
4. **Do not set a whole-request timeout.** A 50 page document legitimately takes minutes. Use an idle or between-chunks timeout instead.
5. **Rejected uploads still fail normally.** An empty, oversized, or unsupported file returns `400` with a JSON body before any line is written, so check the status first.
6. **Closing the connection stops the work.** The server checks for a disconnected client between pages and abandons the remaining inference.
7. **`ocr_result` coordinates refer to `source_width`/`source_height`.** When `ocr.image_max_dimension` downscales the returned image, scale boxes by `source_width / image_width` before drawing them on `image_base64`.

## curl

```bash
curl -N -X POST "http://localhost:8080/ocr/stream" \
  -F "file=@document.pdf"
```

`-N` disables curl's output buffering. Without it lines are held back and the stream looks like a normal blocking response.

Piped through jq, printing progress instead of megabytes of base64:

```bash
curl -sN -X POST "http://localhost:8080/ocr/stream?include_images=false" \
  -F "file=@document.pdf" \
| jq -c --unbuffered 'if .type == "page" then {type, page_number} else . end'
```

Writing each page to its own file:

```bash
curl -sN -X POST "http://localhost:8080/ocr/stream?include_images=false" \
  -F "file=@document.pdf" \
| while IFS= read -r line; do
    case "$(printf '%s' "$line" | jq -r .type)" in
      page)  printf '%s' "$line" | jq -c .ocr_result > "page-$(printf '%s' "$line" | jq -r .page_number).json" ;;
      error) printf '%s' "$line" | jq -r '"failed: \(.detail) (request \(.request_id))"' >&2; exit 1 ;;
      done)  printf '%s' "$line" | jq -r '"finished \(.completed_pages) pages"' ;;
    esac
  done
```

## fetch (browser and Node 18+)

`fetch` exposes the body as a `ReadableStream`, which is the most direct fit for NDJSON.

```js
async function* readNdjson(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let newline;
    while ((newline = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) yield JSON.parse(line);
    }
  }

  const tail = (buffer + decoder.decode()).trim();
  if (tail) yield JSON.parse(tail);
}

export async function ocrStream(file, { includeImages = true, signal, onPage } = {}) {
  const body = new FormData();
  body.append("file", file);

  const response = await fetch(
    `http://localhost:8080/ocr/stream?include_images=${includeImages}`,
    { method: "POST", body, signal },
  );

  // Rejected uploads fail before the stream starts, with a normal JSON body.
  if (!response.ok) {
    const { detail } = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(detail);
  }

  let totalPages = 0;
  let finished = false;

  for await (const message of readNdjson(response)) {
    switch (message.type) {
      case "meta":
        totalPages = message.total_pages;
        break;
      case "page":
        onPage?.(message, totalPages);
        break;
      case "error":
        throw new Error(`${message.detail} (request ${message.request_id})`);
      case "done":
        finished = true;
        break;
    }
  }

  // No terminal line means the connection died part way through.
  if (!finished) throw new Error("OCR stream ended before completion");
  return totalPages;
}
```

Rendering pages as they arrive, and cancelling the request (which stops server-side work):

```js
const controller = new AbortController();
document.querySelector("#cancel").onclick = () => controller.abort();

await ocrStream(fileInput.files[0], {
  signal: controller.signal,
  onPage: (page, totalPages) => {
    const scale = page.source_width / page.image_width; // 1 unless downscaled
    console.log(`page ${page.page_number}/${totalPages}, bbox scale ${scale}`);
    const img = document.createElement("img");
    img.src = page.image_base64;
    document.body.append(img);
  },
});
```

In Node the same function works with `fetch` and a `File`/`Blob`:

```js
import { openAsBlob } from "node:fs";

const file = await openAsBlob("document.pdf", { type: "application/pdf" });
await ocrStream(file, { includeImages: false, onPage: (p) => console.log(p.page_number) });
```

## axios

### Node

```js
import axios from "axios";
import FormData from "form-data";
import fs from "node:fs";
import readline from "node:readline";

const form = new FormData();
form.append("file", fs.createReadStream("document.pdf"));

const response = await axios.post(
  "http://localhost:8080/ocr/stream",
  form,
  {
    headers: form.getHeaders(),
    params: { include_images: false },
    responseType: "stream",
    timeout: 0, // a whole-request timeout would abort long documents
    maxContentLength: Infinity,
    maxBodyLength: Infinity,
  },
);

// readline handles chunk boundaries and multi-megabyte lines.
const lines = readline.createInterface({ input: response.data, crlfDelay: Infinity });

let finished = false;
for await (const line of lines) {
  if (!line.trim()) continue;
  const message = JSON.parse(line);

  if (message.type === "page") console.log(`page ${message.page_number}`);
  if (message.type === "error") throw new Error(message.detail);
  if (message.type === "done") finished = true;
}
if (!finished) throw new Error("OCR stream ended before completion");
```

Errors raised before the stream starts arrive as a normal axios error, but with `responseType: "stream"` the body is a stream rather than parsed JSON:

```js
try {
  /* request as above */
} catch (error) {
  const body = error.response?.data;
  if (body?.readable) {
    const chunks = [];
    for await (const chunk of body) chunks.push(chunk);
    console.error(JSON.parse(Buffer.concat(chunks).toString()).detail);
  } else {
    throw error;
  }
}
```

### Browser

The default XHR adapter buffers the whole response, which defeats streaming. Use the fetch adapter (axios 1.7+) and consume `response.data` as a `ReadableStream`:

```js
const body = new FormData();
body.append("file", file);

const response = await axios.post("http://localhost:8080/ocr/stream", body, {
  adapter: "fetch",
  responseType: "stream",
  params: { include_images: true },
});

for await (const message of readNdjson({ body: response.data })) { // readNdjson from the fetch section
  console.log(message.type);
}
```

On older axios versions, use `fetch` directly rather than `onDownloadProgress`, which re-delivers the accumulated response text on every event.

## PHP (cURL)

`CURLOPT_WRITEFUNCTION` receives chunks as they arrive. Return the byte count from the callback, otherwise cURL aborts the transfer.

```php
<?php

declare(strict_types=1);

$buffer = '';
$finished = false;

$handleLine = static function (array $message) use (&$finished): void {
    switch ($message['type']) {
        case 'meta':
            fwrite(STDERR, "pages: {$message['total_pages']}\n");
            break;
        case 'page':
            file_put_contents(
                sprintf('page-%d.json', $message['page_number']),
                json_encode($message['ocr_result'], JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE)
            );
            break;
        case 'error':
            throw new RuntimeException("{$message['detail']} (request {$message['request_id']})");
        case 'done':
            $finished = true;
            break;
    }
};

$curl = curl_init();
curl_setopt_array($curl, [
    CURLOPT_URL => 'http://localhost:8080/ocr/stream?include_images=false',
    CURLOPT_POST => true,
    CURLOPT_POSTFIELDS => [
        'file' => new CURLFile('document.pdf', 'application/pdf', 'document.pdf'),
    ],
    // Must stay false: RETURNTRANSFER would collect the whole body first.
    CURLOPT_RETURNTRANSFER => false,
    CURLOPT_TIMEOUT => 0,          // no overall limit
    CURLOPT_LOW_SPEED_LIMIT => 1,  // but do fail on a stalled connection
    CURLOPT_LOW_SPEED_TIME => 600,
    CURLOPT_WRITEFUNCTION => static function ($curl, string $chunk) use (&$buffer, $handleLine): int {
        $buffer .= $chunk;
        while (($newline = strpos($buffer, "\n")) !== false) {
            $line = trim(substr($buffer, 0, $newline));
            $buffer = substr($buffer, $newline + 1);
            if ($line !== '') {
                $handleLine(json_decode($line, true, 512, JSON_THROW_ON_ERROR));
            }
        }
        return strlen($chunk);
    },
]);

curl_exec($curl);
$status = curl_getinfo($curl, CURLINFO_RESPONSE_CODE);
$error = curl_error($curl);
curl_close($curl);

if ($status >= 400) {
    throw new RuntimeException("upload rejected with HTTP {$status}");
}
if ($error !== '') {
    throw new RuntimeException("transfer failed: {$error}");
}
if (!$finished) {
    throw new RuntimeException('OCR stream ended before completion');
}
```

When relaying the stream to a browser from PHP-FPM, also disable output buffering (`ob_end_flush()` then `flush()` per line) and send `X-Accel-Buffering: no` on your own response.

## Python

### requests

```python
import json

import requests

URL = "http://localhost:8080/ocr/stream"


def ocr_stream(path: str, include_images: bool = False) -> int:
    with open(path, "rb") as upload:
        with requests.post(
            URL,
            files={"file": (path, upload, "application/pdf")},
            params={"include_images": str(include_images).lower()},
            stream=True,
            # (connect timeout, read timeout). The read timeout applies between
            # chunks, so a slow document is fine but a stalled one still fails.
            timeout=(10, 300),
        ) as response:
            response.raise_for_status()  # rejected uploads fail here, before any line

            total_pages = 0
            finished = False
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                message = json.loads(raw_line)

                if message["type"] == "meta":
                    total_pages = message["total_pages"]
                elif message["type"] == "page":
                    print(f"page {message['page_number']}/{total_pages}")
                elif message["type"] == "error":
                    raise RuntimeError(f"{message['detail']} (request {message['request_id']})")
                elif message["type"] == "done":
                    finished = True

            if not finished:
                raise RuntimeError("OCR stream ended before completion")
            return total_pages
```

`stream=True` is what makes this incremental; without it `requests` downloads the whole body before returning.

### httpx (async)

```python
import json

import httpx


async def ocr_stream(path: str) -> None:
    timeout = httpx.Timeout(300.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        with open(path, "rb") as upload:
            async with client.stream(
                "POST",
                "http://localhost:8080/ocr/stream",
                files={"file": (path, upload, "application/pdf")},
                params={"include_images": "false"},
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise RuntimeError(response.json()["detail"])

                async for line in response.aiter_lines():
                    if line.strip():
                        message = json.loads(line)
                        print(message["type"], message.get("page_number", ""))
```

## Go

Use `bufio.Reader.ReadBytes`, not `bufio.Scanner`: the scanner's default 64 KB token limit is far smaller than a page line carrying a base64 image.

```go
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
)

type streamLine struct {
	Type string `json:"type"`

	// meta
	Filename   string `json:"filename"`
	TotalPages int    `json:"total_pages"`

	// page
	PageNumber  int             `json:"page_number"`
	ImageBase64 string          `json:"image_base64"`
	ImageWidth  int             `json:"image_width"`
	SourceWidth int             `json:"source_width"`
	OCRResult   json.RawMessage `json:"ocr_result"`

	// done
	CompletedPages int `json:"completed_pages"`

	// error
	Detail    string `json:"detail"`
	RequestID string `json:"request_id"`
}

func ocrStream(ctx context.Context, endpoint, path string) error {
	// Stream the upload through a pipe so a large PDF is never fully buffered.
	pipeReader, pipeWriter := io.Pipe()
	form := multipart.NewWriter(pipeWriter)

	go func() {
		defer pipeWriter.Close()

		file, err := os.Open(path)
		if err != nil {
			pipeWriter.CloseWithError(err)
			return
		}
		defer file.Close()

		part, err := form.CreateFormFile("file", path)
		if err != nil {
			pipeWriter.CloseWithError(err)
			return
		}
		if _, err := io.Copy(part, file); err != nil {
			pipeWriter.CloseWithError(err)
			return
		}
		pipeWriter.CloseWithError(form.Close())
	}()

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint+"?include_images=false", pipeReader)
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", form.FormDataContentType())

	// No client Timeout: it would cap the whole body read. Cancel via ctx instead.
	response, err := (&http.Client{}).Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()

	if response.StatusCode >= 400 {
		body, _ := io.ReadAll(response.Body)
		return fmt.Errorf("upload rejected with HTTP %d: %s", response.StatusCode, body)
	}

	reader := bufio.NewReaderSize(response.Body, 1<<20)
	finished := false

	for {
		line, readErr := reader.ReadBytes('\n')

		if len(bytes.TrimSpace(line)) > 0 {
			var message streamLine
			if err := json.Unmarshal(bytes.TrimSpace(line), &message); err != nil {
				return fmt.Errorf("malformed stream line: %w", err)
			}

			switch message.Type {
			case "meta":
				fmt.Printf("%s: %d pages\n", message.Filename, message.TotalPages)
			case "page":
				fmt.Printf("page %d\n", message.PageNumber)
			case "error":
				return fmt.Errorf("%s (request %s)", message.Detail, message.RequestID)
			case "done":
				finished = true
			}
		}

		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return readErr
		}
	}

	if !finished {
		return fmt.Errorf("OCR stream ended before completion")
	}
	return nil
}

func main() {
	if err := ocrStream(context.Background(), "http://localhost:8080/ocr/stream", "document.pdf"); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
```

Cancelling the `context` closes the connection, which stops the server from processing the remaining pages.

## Pitfalls

| Symptom | Cause | Fix |
| --- | --- | --- |
| Everything arrives at once, at the end | Proxy response buffering | `proxy_buffering off;` for this route; the response already sets `X-Accel-Buffering: no` |
| Everything arrives at once, only in curl | curl output buffering | Add `-N` |
| Everything arrives at once, only in a browser | axios default XHR adapter | Use `fetch`, or axios `adapter: "fetch"` |
| Truncated JSON parse errors | Splitting chunks without a line buffer | Keep the remainder after the last `\n` |
| `bufio.Scanner: token too long` | 64 KB scanner limit vs. multi-megabyte lines | Use `bufio.Reader.ReadBytes('\n')` |
| Request dies after N seconds on long PDFs | Whole-request timeout | Use an idle/read timeout, or a cancellable context |
| Failures look like successes | Only checking the HTTP status | Require a `done` line; treat `error` and truncation as failure |
| Response is huge | Page images returned by default | Add `include_images=false`, or configure `OCR_IMAGE_FORMAT=webp` and `OCR_IMAGE_MAX_DIMENSION` |
| Bounding boxes are offset | Image downscaled by `image_max_dimension` | Scale by `source_width / image_width` |

## See also

- [`README.md`](../README.md) for configuration, error responses, and the non-streaming `POST /ocr`.
- `GET /docs` on a running instance for the generated OpenAPI reference.
