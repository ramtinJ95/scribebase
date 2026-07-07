# ScribeBase

ScribeBase is a local-first CLI for turning documents into cited Markdown and searchable RAG context.

It can ingest PDFs, scanned pages, images, and handwritten notes; extract or OCR them into page-level Markdown; chunk and embed the text locally; index it in Weaviate; and retrieve grounded context for search, answers, and quizzes.

Local-first means extraction, OCR, embeddings, indexing, and retrieval run on your machine. Using an LLM for final answers is optional.

## What ScribeBase does

- Extracts true-text PDFs with PyMuPDF/PyMuPDF4LLM.
- OCRs scanned PDFs and images with local providers.
- Stores originals, rendered pages, Markdown, manifests, page metadata, and chunks under a local data directory.
- Creates embeddings through a local llama.cpp-compatible `/v1/embeddings` server.
- Indexes chunks into local Weaviate using self-provided vectors.
- Searches with hybrid retrieval and metadata filters.
- Builds context packs when no LLM is configured.
- Optionally calls an OpenAI-compatible chat API for answers and quizzes.

No Ollama dependency is used.

## Requirements

- Python 3.11+
- `uv`
- Docker, for local Weaviate
- `llama-server`, for local embeddings
- Optional OCR runtime:
  - GLM-OCR through llama.cpp for high-quality local OCR
  - Apple Vision on macOS for fast local OCR
- Optional OpenAI-compatible chat API key for generated answers/quizzes

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

Ingest a PDF:

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

Ask a question:

```bash
uv run scribebase ask "Explain the main idea with page citations." --top-k 10
```

If no LLM is configured, `ask` saves a context pack that you can paste into another model.

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

### Ingest images or handwritten notes

```bash
uv run scribebase ingest ./notes/lecture-1/ \
  --title "Lecture 1 Notes" \
  --source-type notes \
  --course "Neuroscience" \
  --ocr shell
```

Image inputs always require OCR. Use `--ocr shell` for the default GLM-OCR adapter, or `--ocr apple_vision` on macOS.

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

Use `rebuild-index --all` after changing embedding models or dimensions. It recreates the Weaviate collection so the vector index matches the current model.

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

### OCR options

Default high-quality OCR provider:

```bash
uv run scribebase ingest ./scan.pdf --title "Scan" --source-type book --ocr shell
```

The `shell` provider runs:

```bash
./scripts/run_local_ocr.py --input {input_image} --output {output_md}
```

That adapter calls a local OpenAI-compatible vision endpoint, defaulting to `http://localhost:8082/v1`. Recommended GLM-OCR server:

```bash
llama-server \
  -m ./models/ocr/GLM-OCR-Q8_0.gguf \
  --mmproj ./models/ocr/mmproj-GLM-OCR-Q8_0.gguf \
  -ngl 0 \
  --port 8082
```

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
- `--ocr shell`: use the GLM-OCR shell provider when OCR is needed.
- `--ocr apple_vision`: use Apple Vision when OCR is needed.

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
- The model name in `.study_local/config.yaml` must match the server model name.
- ScribeBase stores embedding model metadata and rejects accidental mixed-model retrieval by default.
- The default chunking settings are conservative enough for small local embedding models.

## LLM configuration

LLM usage is optional. Without an LLM key, `ask` and `quiz` save context packs instead of failing.

To enable generated answers and quizzes, configure an OpenAI-compatible chat API in `.study_local/config.yaml`:

```yaml
llm:
  enabled: true
  provider: "openai_compatible"
  base_url: "https://api.openai.com/v1"
  model: "gpt-5.5-pro"
  api_key_env: "OPENAI_API_KEY"
  temperature: 0.2
```

## Local data layout

By default, ScribeBase writes to `.study_local/`:

```text
.study_local/
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
  outputs/
    context_packs/
    answers/
    quizzes/
  logs/app.log
```

Originals, Markdown, and JSON metadata are the source of truth. Weaviate can be rebuilt from local files.

## Configuration

Run `scribebase init` to create `.study_local/config.yaml`. Important sections:

```yaml
weaviate:
  url: "http://localhost:8081"
  collection: "StudyChunk"
  vector_name: "text_vector"

embedding:
  provider: "llamacpp"
  base_url: "http://localhost:8080/v1"
  model: "Qwen3-Embedding-4B-Q4_K_M.gguf"
  batch_size: 8

ocr:
  default_provider: "shell"
  render_dpi: 300
  providers:
    shell:
      command: "./scripts/run_local_ocr.py --input {input_image} --output {output_md}"
      timeout_seconds: 900
      model_name: "GLM-OCR"
    apple_vision:
      command: "swift ./scripts/run_apple_vision_ocr.swift --input {input_image} --output {output_md}"
      timeout_seconds: 120
      model_name: "Apple Vision"
      render_dpi: 200

server:
  host: "127.0.0.1"
  port: 8765
  api_token_env: "SCRIBEBASE_API_TOKEN"
```

### Environment overrides

ScribeBase loads `.env` when present and lets deployment-specific environment variables override the YAML file:

```bash
SCRIBEBASE_DATA_DIR=/Users/ramtin/scribebase-data
SCRIBEBASE_CONFIG=/Users/ramtin/scribebase-data/config.yaml
SCRIBEBASE_HOST=0.0.0.0
SCRIBEBASE_PORT=8765
SCRIBEBASE_API_TOKEN=change-me
```

- `SCRIBEBASE_DATA_DIR`: local source, output, and log directory.
- `SCRIBEBASE_CONFIG`: explicit config path. Defaults to `$SCRIBEBASE_DATA_DIR/config.yaml`.
- `SCRIBEBASE_HOST` and `SCRIBEBASE_PORT`: reserved for the upcoming HTTP server.
- `SCRIBEBASE_API_TOKEN`: shared secret read from the environment, not written to config.

See `.env.example` for a copyable template.

## Command reference

```bash
scribebase init
scribebase doctor
scribebase extract PATH --title TITLE [--ocr auto|always|never|shell|apple_vision]
scribebase ingest PATH --title TITLE [--no-index]
scribebase index --source-id SOURCE_ID
scribebase rebuild-index --source-id SOURCE_ID
scribebase rebuild-index --all
scribebase search QUERY [filters]
scribebase ask QUESTION [filters]
scribebase quiz [filters]
scribebase sources list
scribebase sources show SOURCE_ID
scribebase chunks list --source-id SOURCE_ID
scribebase chunks show CHUNK_ID
```

Use `uv run scribebase COMMAND --help` for command-specific options.

## Development

```bash
uv run --extra dev ruff check .
uv run --extra dev pytest
```
