# ScribeBase

ScribeBase is a local-first CLI study app for turning PDFs, scanned documents, images, and handwritten notes into page-cited Markdown and searchable RAG context.

It uses local tooling for extraction, OCR, embeddings, indexing, and retrieval. A frontier LLM is optional and only receives selected retrieved context.

## Features

- ingest true-text PDFs with PyMuPDF/PyMuPDF4LLM
- route scanned PDF pages and image inputs through a configurable local shell OCR provider
- save originals, page Markdown, combined Markdown, page metadata, manifests, and chunks under `.study_local/`
- create embeddings through a local llama.cpp-compatible `/v1/embeddings` endpoint
- index chunks into local Weaviate with self-provided vectors
- search with Weaviate hybrid search plus metadata filters
- generate context packs for manual GPT use when no LLM API key is configured
- optionally call an OpenAI-compatible chat API for answers and quizzes

No Ollama dependency is used.

## Install

```bash
uv sync --extra dev
uv run study init
```

Or install the package and use the `study` console script from your environment.

## Start local Weaviate

```bash
docker compose -f docker-compose.weaviate.yml up -d
```

The default app config points to `http://localhost:8081` and collection `StudyChunk`.

## Start a llama.cpp embedding server

Use any OpenAI-compatible llama.cpp embedding model. Example:

```bash
llama-server \
  --model ./models/Qwen3-Embedding-0.6B-Q8_0.gguf \
  --embedding \
  --pooling last \
  --ctx-size 32768 \
  --port 8080
```

Then verify local services:

```bash
uv run study doctor
```

## OCR provider configuration

ScribeBase v1 is model-agnostic. It shells out to your local OCR adapter.

Default config in `.study_local/config.yaml`:

```yaml
ocr:
  default_provider: "shell"
  render_dpi: 300
  providers:
    shell:
      command: "python ./scripts/run_local_ocr.py --input {input_image} --output {output_md}"
      timeout_seconds: 300
      model_name: "GLM-OCR"
```

The command template supports:

- `{input_image}`
- `{output_md}`
- `{output_json}`
- `{page_number}`
- `{source_id}`

Copy `scripts/run_local_ocr.py.example` to `scripts/run_local_ocr.py` and replace the placeholder with GLM-OCR, PaddleOCR-VL, Chandra, Surya, DeepSeek-OCR, or another local OCR command.

## Common workflows

### Ingest a true-text PDF

```bash
uv run study ingest ./books/biology.pdf \
  --title "Biology 101" \
  --source-type book \
  --language en
```

### Ingest a scanned PDF

```bash
uv run study ingest ./scans/chapter_4_scanned.pdf \
  --title "Cognitive Psychology" \
  --source-type book \
  --chapter "4" \
  --ocr auto
```

### Ingest handwritten note images

```bash
uv run study ingest ./notes/2026-07-lecture-1/ \
  --title "Lecture 1 Notes" \
  --source-type notes \
  --course "Neuroscience" \
  --ocr shell
```

### Extract without indexing

```bash
uv run study extract ./book.pdf --title "Book" --source-type book --ocr auto
```

### Index an existing extracted source

```bash
uv run study index --source-id SOURCE_ID
```

### Search

```bash
uv run study search "working memory limitations" \
  --title "Cognitive Psychology" \
  --chapter "4" \
  --top-k 12
```

### Ask

```bash
uv run study ask "Explain working memory limitations using this chapter." \
  --title "Cognitive Psychology" \
  --chapter "4" \
  --top-k 12
```

If LLM config is disabled or the API key is missing, ScribeBase saves a context pack under `.study_local/outputs/context_packs/`.

### Quiz

```bash
uv run study quiz \
  --title "Cognitive Psychology" \
  --chapter "4" \
  --questions 20 \
  --types mcq,short-answer,flashcard
```

If no LLM is configured, this saves a quiz prompt/context pack under `.study_local/outputs/quizzes/`.

### Inspect local source of truth

```bash
uv run study sources list
uv run study sources show SOURCE_ID
uv run study chunks list --source-id SOURCE_ID
uv run study chunks show CHUNK_ID
```

## Data layout

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

Weaviate is rebuildable. Originals, Markdown, and JSON metadata are the canonical source of truth.

## LLM configuration

LLM usage is optional. Enable an OpenAI-compatible chat API in `.study_local/config.yaml`:

```yaml
llm:
  enabled: true
  provider: "openai_compatible"
  base_url: "https://api.openai.com/v1"
  model: "gpt-5.5-pro"
  api_key_env: "OPENAI_API_KEY"
  temperature: 0.2
```

If `OPENAI_API_KEY` is absent, commands do not fail; they save context packs.

## Rebuild index

```bash
uv run study rebuild-index --source-id SOURCE_ID
uv run study rebuild-index --all
```

Change embedding models only with an index rebuild. Search rejects mixed embedding model results by default; pass `--allow-model-mismatch` only when you deliberately want that.

## Development

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
```
