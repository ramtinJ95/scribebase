# ScribeBase

ScribeBase is a local-first knowledge node for turning documents, notes, and web articles into cited Markdown and searchable RAG context.

It can ingest PDFs, scanned pages, images, handwritten notes, Markdown, plain text, and automation-submitted articles; extract or OCR them into Markdown; chunk and embed the text locally; index it in Weaviate; and retrieve grounded context for agents.

Local-first means extraction, OCR, embeddings, indexing, and retrieval run on your machine. ScribeBase does not call a generation model; consuming agents use the cited context it returns.

## What ScribeBase does

- Extracts true-text PDFs with PyMuPDF/PyMuPDF4LLM.
- OCRs scanned PDFs and images with local providers.
- Ingests Markdown and plain-text sources without OCR.
- Accepts article/text JSON submissions over HTTP for external automations.
- Stores originals, rendered pages, Markdown, manifests, page metadata, and chunks under a local data directory.
- Preserves generic metadata such as URL, origin, publisher, author, tags, collection, and source dates.
- Creates embeddings through a local llama.cpp-compatible `/v1/embeddings` server.
- Indexes chunks into local Weaviate using self-provided vectors.
- Searches with hybrid retrieval and metadata filters.
- Returns cited context packs for consuming agents.

No Ollama dependency is used.

## Requirements

- Python 3.11+
- `uv`
- Docker, for local Weaviate
- `llama-server`, for local embeddings
- Optional OCR runtime:
  - GLM-OCR through llama.cpp for high-quality local OCR
  - Apple Vision on macOS for fast local OCR

## Quickstart

Install dependencies and create the local data layout:

```bash
uv sync --extra dev
uv run scribebase init
```

Start Weaviate:

```bash
docker compose -f docker-compose.weaviate.yml up -d
```

Start the default embedding model with llama.cpp:

```bash
llama-server \
  --model ./models/Qwen3-Embedding-4B-Q4_K_M.gguf \
  --embedding \
  --pooling last \
  --ctx-size 32768 \
  -ngl 99 \
  --port 8080
```

Check the setup:

```bash
uv run scribebase doctor
```

Optional: start the HTTP API and ingestion worker in separate terminals for
remote ingestion, search, and context clients:

```bash
uv sync --extra server
export SCRIBEBASE_API_TOKEN=change-me
uv run scribebase serve --host 0.0.0.0 --port 8765
# In another terminal with the same environment:
uv run scribebase worker
```

Ingest a PDF, Markdown file, or text file:

```bash
uv run scribebase ingest ./books/example.pdf \
  --title "Example Book" \
  --source-type book \
  --language en
```

Search it:

```bash
uv run scribebase search "explain the main idea" --top-k 10
```

## Common workflows

### Ingest a true-text PDF

```bash
uv run scribebase ingest ./books/biology.pdf \
  --title "Biology 101" \
  --source-type book \
  --language en
```

For born-digital PDFs with a good text layer, ScribeBase extracts Markdown directly. OCR is not used unless you force it.

### Ingest a scanned PDF

```bash
uv run scribebase ingest ./scans/chapter_4_scanned.pdf \
  --title "Cognitive Psychology" \
  --source-type book \
  --chapter "4" \
  --ocr auto
```

`--ocr auto` uses OCR only for pages that do not have a usable text layer.

### Ingest Markdown or plain text

```bash
uv run scribebase ingest ./articles/gitops.md \
  --title "GitOps Notes" \
  --source-type article \
  --language en \
  --tags "kubernetes,gitops" \
  --origin company_blog \
  --publisher "Example Blog" \
  --url "https://example.com/gitops" \
  --collection "infra-reading"
```

```bash
uv run scribebase ingest ./notes/scheduling.txt \
  --title "Kubernetes Scheduling Notes" \
  --source-type notes \
  --language en
```

Markdown is preserved as Markdown. Plain text is copied into the normal Markdown
extraction layout so it can be chunked, embedded, and searched like PDFs.
Optional generic metadata can be passed with fields such as `--tags`, `--origin`,
`--publisher`, `--author`, `--url`, `--external-id`, and `--collection`.
Markdown files can also provide those fields as YAML frontmatter; explicit CLI
or API fields override frontmatter values.

```markdown
---
title: "GitOps Notes"
source_type: article
language: en
tags: ["kubernetes", "gitops"]
origin: company_blog
publisher: "Example Blog"
url: "https://example.com/gitops"
collection: "infra-reading"
---

# GitOps Notes

Article body...
```

### Ingest images or handwritten notes

```bash
uv run scribebase ingest ./notes/lecture-1/ \
  --title "Lecture 1 Notes" \
  --source-type notes \
  --course "Neuroscience" \
  --ocr auto
```

Image inputs always require OCR. `--ocr auto` uses GLM-OCR. Apple Vision is an
explicit opt-in and is never used as an automatic fallback.

### Extract without indexing

```bash
uv run scribebase extract ./book.pdf \
  --title "Book" \
  --source-type book \
  --ocr auto
```

### Index an existing extracted source

```bash
uv run scribebase index --source-id SOURCE_ID
```

### Rebuild indexes

```bash
uv run scribebase rebuild-index --source-id SOURCE_ID
uv run scribebase rebuild-index --all
```

Use `rebuild-index --all` after changing embedding models, dimensions, or the
Weaviate schema. It builds and verifies a versioned physical collection before
atomically promoting the configured collection alias. The first rebuild of a
legacy physical collection briefly frees its name to create the alias; later
rebuilds switch without search downtime. Failed staged rebuilds leave the live
index unchanged. If that one-time alias creation fails after the legacy name is
freed, ScribeBase preserves the verified staged collection and reports its name
for recovery instead of deleting the only rebuilt copy.

### Inspect sources and chunks

```bash
uv run scribebase sources list
uv run scribebase sources show SOURCE_ID
uv run scribebase chunks list --source-id SOURCE_ID
uv run scribebase chunks show CHUNK_ID
```

## OCR behavior

ScribeBase chooses the extraction path per input and per PDF page.

For PDFs:

1. It checks the embedded text layer with PyMuPDF.
2. If the text layer is usable and OCR is not forced, it extracts Markdown with PyMuPDF4LLM.
3. If the text layer is missing or poor, it renders the page to an image and sends it to the configured OCR provider.

For image files and image directories, ScribeBase goes directly to OCR.

For Markdown and plain-text files, ScribeBase reads the document directly; OCR
is not used.

### OCR options

Default high-quality OCR provider:

```bash
uv run scribebase ingest ./scan.pdf --title "Scan" --source-type book --ocr auto
```

The `glm_ocr` provider runs:

```bash
./scripts/run_local_ocr.py --input {input_image} --output {output_md} \
  --base-url http://localhost:8082/v1 --model GLM-OCR
```

That adapter calls a separate local OpenAI-compatible vision endpoint on port
8082. The embedding-only server remains on port 8080. Recommended GLM-OCR
server:

```bash
llama-server \
  --model ./models/ocr/GLM-OCR-Q8_0.gguf \
  --mmproj ./models/ocr/mmproj-GLM-OCR-Q8_0.gguf \
  --alias GLM-OCR \
  --ctx-size 8192 \
  --parallel 1 \
  --cache-ram 0 \
  -ngl 0 \
  --host 127.0.0.1 \
  --port 8082
```

See [Mac mini deployment](docs/macmini-deployment.md) for exact download and
launchd setup commands.

Fast macOS OCR provider:

```bash
uv run scribebase ingest ./scan.pdf \
  --title "Scan" \
  --source-type book \
  --ocr apple_vision
```

Apple Vision uses `scripts/run_apple_vision_ocr.swift` and runs on-device.

### OCR flags

- `--ocr auto`: use OCR only when needed. This is the normal PDF mode.
- `--ocr always`: force OCR for PDF pages using the default OCR provider.
- `--ocr never`: disable OCR. PDF pages without usable text will fail; image inputs always fail.
- `--ocr glm_ocr`: explicitly use the GLM-OCR provider when OCR is needed.
- `--ocr apple_vision`: use Apple Vision when OCR is needed.

ScribeBase keeps one PyMuPDF document open for the extraction pass. If
PyMuPDF4LLM fails or returns empty output for a text-routed page, ScribeBase
uses the cached PyMuPDF text and records `pymupdf4llm_failed:*` or
`pymupdf4llm_empty` in that page's quality flags instead of hiding the fallback.
Rendered PDF pages with no meaningful dark pixels are recorded as skipped blank
pages. Empty OCR output from any nonblank page remains a hard failure.
Documents containing only blank pages fail extraction instead of publishing
page-marker-only content.

## Embeddings

The default embedding configuration expects `Qwen3-Embedding-4B-Q4_K_M.gguf` served by llama.cpp on port `8080`.

```bash
llama-server \
  --model ./models/Qwen3-Embedding-4B-Q4_K_M.gguf \
  --embedding \
  --pooling last \
  -ngl 99 \
  --port 8080
```

Notes:

- `--pooling last` is required for Qwen embedding models.
- The model name in `.scribebase/config.yaml` must match the server model name.
- ScribeBase stores embedding model metadata and rejects accidental mixed-model retrieval by default.
- The default chunking profile targets 1,200 characters with 150 characters of overlap,
  balancing passage coherence with precise local retrieval.

## Local data layout

By default, ScribeBase writes to `.scribebase/`:

```text
.scribebase/
  config.yaml
  sources/<source_id>/
    original/
    pages/
    markdown/
      page_0001.md
      document.md
      chapters/
    metadata/
      page_0001.json
      chunks.jsonl
      manifest.json
  uploads/
  jobs/
  logs/app.log
```

Originals, Markdown, and JSON metadata are the source of truth. Weaviate can be rebuilt from local files.

## Configuration

Run `scribebase init` to create `.scribebase/config.yaml`. Important sections:

```yaml
weaviate:
  url: "http://localhost:8081"
  collection: "Chunk"
  vector_name: "text_vector"

embedding:
  provider: "llamacpp"
  base_url: "http://localhost:8080/v1"
  model: "Qwen3-Embedding-4B-Q4_K_M.gguf"
  batch_size: 8

chunking:
  target_chars: 1200
  overlap_chars: 150
  min_chars: 250
  chunker_version: "v2"

ocr:
  default_provider: "glm_ocr"
  render_dpi: 300
  providers:
    glm_ocr:
      command: "./scripts/run_local_ocr.py --input {input_image} --output {output_md} --base-url {base_url} --model {model_name}"
      timeout_seconds: 900
      model_name: "GLM-OCR"
      base_url: "http://localhost:8082/v1"
      require_multimodal: true
    apple_vision:
      command: "swift ./scripts/run_apple_vision_ocr.swift --input {input_image} --output {output_md}"
      timeout_seconds: 120
      model_name: "Apple Vision"
      base_url: null
      require_multimodal: false
      render_dpi: 200

server:
  host: "127.0.0.1"
  port: 8765
  api_token_env: "SCRIBEBASE_API_TOKEN"
  max_upload_bytes: 262144000
  max_active_jobs: 20
  max_upload_storage_bytes: 1073741824
  worker_poll_seconds: 2.0
  worker_dependency_retry_seconds: 10.0
  worker_heartbeat_seconds: 2.0
  worker_stale_seconds: 15.0
  upload_reservation_timeout_seconds: 3600
  identity_orphan_job_seconds: 300
  identity_direct_reservation_seconds: 86400
  identity_reservation_heartbeat_seconds: 60.0
  failed_upload_retention_seconds: 604800
```

### Environment overrides

ScribeBase loads `.env` when present and lets deployment-specific environment variables override the YAML file:

```bash
SCRIBEBASE_DATA_DIR=/Users/yourname/scribebase-data
SCRIBEBASE_CONFIG=/Users/yourname/scribebase-data/config.yaml
SCRIBEBASE_HOST=0.0.0.0
SCRIBEBASE_PORT=8765
SCRIBEBASE_API_TOKEN=change-me
```

- `SCRIBEBASE_DATA_DIR`: local source, job, upload, and log directory.
- `SCRIBEBASE_CONFIG`: explicit config path. Defaults to `$SCRIBEBASE_DATA_DIR/config.yaml`.
- `SCRIBEBASE_HOST` and `SCRIBEBASE_PORT`: HTTP API bind address and port.
- `SCRIBEBASE_API_TOKEN`: required API bearer token, read from the environment and not written to config.

When loading an older generated configuration, ScribeBase migrates legacy
`shell` and implicit `apple_vision` defaults to `glm_ocr` in memory. A custom
provider named `shell` is rejected with an upgrade message rather than being
silently treated as GLM-OCR. Apple Vision remains available through explicit
`--ocr apple_vision` selection.

See `.env.example` for a copyable template.

## HTTP API

Install the server extra and set a bearer token before starting the API:

```bash
uv sync --extra server
export SCRIBEBASE_API_TOKEN=change-me
uv run scribebase serve
uv run scribebase worker
```

Endpoints:

- `GET /health`: readiness summary for ScribeBase, Weaviate, embeddings,
  GLM-OCR, and the worker. Top-level status is `degraded` when any required
  service is unavailable.
- `GET /sources`: list indexed source manifests.
- `POST /ingest`: upload a document and enqueue extraction/indexing.
- `POST /articles`: submit Markdown/text article content as JSON and enqueue ingestion.
- `GET /jobs/{job_id}`: inspect ingestion job status and errors.
- `POST /jobs/{job_id}/retry`: requeue a failed ingestion job.
- `POST /search`: hybrid search over chunks.
- `POST /context`: search and return a ready-to-paste context pack.

Protected endpoints require `Authorization: Bearer $SCRIBEBASE_API_TOKEN`.
Uploaded documents and articles remain `queued` until the separate worker
claims them. Run exactly one worker per data directory; interrupted `running`
jobs are returned to the queue when the worker starts. Upload size, active queue
capacity, total upload storage, retention, and polling are configured under
`server` in `config.yaml`. Failed jobs retain uploads for seven days by default
and can be requeued with `POST /jobs/{job_id}/retry`. Use a local data directory;
shared NFS/SMB queues and multiple-host workers are unsupported.

Example search:

```bash
curl -s http://127.0.0.1:8765/search \
  -H "Authorization: Bearer $SCRIBEBASE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"explain kubelet eviction","filters":{"source_type":"book"},"top_k":5}'
```

Metadata filters can target article/text metadata as well:

```bash
curl -s http://127.0.0.1:8765/search \
  -H "Authorization: Bearer $SCRIBEBASE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"progressive delivery","filters":{"source_type":"article","tags":["kubernetes","gitops"],"origin":"company_blog","collection":"infra-reading","created_at_source_after":"2026-01-01T00:00:00Z"},"top_k":5}'
```

Example remote ingestion:

```bash
curl -s http://127.0.0.1:8765/ingest \
  -H "Authorization: Bearer $SCRIBEBASE_API_TOKEN" \
  -F "file=@./paper.pdf" \
  -F "title=Paper Title" \
  -F "source_type=paper" \
  -F "language=en"
```

Markdown and text uploads use the same endpoint:

```bash
curl -s http://127.0.0.1:8765/ingest \
  -H "Authorization: Bearer $SCRIBEBASE_API_TOKEN" \
  -F "file=@./article.md" \
  -F "title=Article Title" \
  -F "source_type=article" \
  -F "language=en" \
  -F "tags=kubernetes,gitops" \
  -F "origin=company_blog" \
  -F "publisher=Example Blog" \
  -F "url=https://example.com/gitops" \
  -F "collection=infra-reading"
```

Automations can avoid multipart upload by using the article JSON endpoint:

```bash
curl -s http://127.0.0.1:8765/articles \
  -H "Authorization: Bearer $SCRIBEBASE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Article Title",
    "body": "# Article Title\n\nArticle body...",
    "language": "en",
    "tags": ["kubernetes", "gitops"],
    "origin": "company_blog",
    "publisher": "Example Blog",
    "url": "https://example.com/gitops",
    "collection": "infra-reading"
  }'
```

`POST /articles` defaults `source_type` to `article`. The `body` may include
Markdown frontmatter; explicit JSON fields override frontmatter values.
Submissions are deduplicated by external ID/origin, canonical URL, then content
SHA-256. Duplicates return `409` with the existing `source_id`. Use
`duplicate_policy=create` (multipart) or `"duplicate_policy": "create"` (JSON)
only when a separate copy is intentional.

Existing manifests are not hashed during request handling. Run the explicit
one-time migration before relying on content-hash deduplication for legacy
sources:

```bash
uv run scribebase sources backfill-identities
```

For reusable automation payload templates for company blogs, Hacker News,
newsletters/RSS, notes, snippets, and docs, see
[`docs/article-automation-contract.md`](docs/article-automation-contract.md).

The response includes a `job_id`. Poll it until `status` is `succeeded` or `failed`:

```bash
curl -s http://127.0.0.1:8765/jobs/JOB_ID \
  -H "Authorization: Bearer $SCRIBEBASE_API_TOKEN"
```

For a full Mac mini deployment with launchd examples, see
[`docs/macmini-deployment.md`](docs/macmini-deployment.md).

For user-invoked agent skill templates that upload documents, submit article
JSON, and retrieve context from a remote ScribeBase server, see
[`docs/skills/`](docs/skills/).

## Command reference

```bash
scribebase init
scribebase doctor
scribebase serve [--host HOST] [--port PORT]
scribebase worker [--once]
scribebase extract PATH --title TITLE [--ocr auto|always|never|glm_ocr|apple_vision]
scribebase ingest PATH --title TITLE [--no-index]
scribebase index --source-id SOURCE_ID
scribebase rebuild-index --source-id SOURCE_ID
scribebase rebuild-index --all
scribebase search QUERY [filters]
scribebase sources list
scribebase sources show SOURCE_ID
scribebase sources backfill-identities
scribebase chunks list --source-id SOURCE_ID
scribebase chunks show CHUNK_ID
```

Use `uv run scribebase COMMAND --help` for command-specific options.

## Development

```bash
uv run --extra dev ruff check .
uv run --extra dev pytest
```
