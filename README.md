# ScribeBase

ScribeBase is a local-first CLI app for turning PDFs, scanned documents, images, and handwritten notes into page-cited Markdown and searchable RAG context.

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
uv run scribebase init
```

Or install the package and use the `scribebase` console script from your environment.

## Start local Weaviate

```bash
docker compose -f docker-compose.weaviate.yml up -d
```

The default app config points to `http://localhost:8081` and collection `StudyChunk`.

## Start a llama.cpp embedding server

Default quality target: `Qwen3-Embedding-4B-Q4_K_M.gguf`. It was slower than
smaller models in local indexing tests, but gave the best retrieval quality on
the Kubernetes book benchmark. Start it with llama.cpp:

```bash
llama-server \
  --model ./models/Qwen3-Embedding-4B-Q4_K_M.gguf \
  --embedding \
  --pooling last \
  --ctx-size 32768 \
  -ngl 99 \
  --port 8080
```

Then verify local services:

```bash
uv run scribebase doctor
```

`--pooling last` is required for Qwen embedding models. Many small embedding
models have a 512-token context window. The default chunking settings are
intentionally conservative for those models; increase `chunking.target_chars`
only after confirming your embedding server accepts larger inputs.

Make sure `embedding.model` in `.study_local/config.yaml` matches the model you
started. ScribeBase stores this name in chunk metadata and uses it to prevent
accidental mixed-model retrieval.

## OCR provider configuration

ScribeBase first reads true-text PDFs with PyMuPDF/PyMuPDF4LLM. OCR is used for
image inputs, scanned pages, or when you pass `--ocr always`.

The default high-accuracy OCR adapter is GLM-OCR Q8 through a local llama.cpp
server. In local tests, GLM-OCR was the best accuracy choice, but Metal/GPU
offload was unstable; run it on CPU with `-ngl 0`:

```bash
llama-server \
  -m ./models/ocr/GLM-OCR-Q8_0.gguf \
  --mmproj ./models/ocr/mmproj-GLM-OCR-Q8_0.gguf \
  -ngl 0 \
  --port 8082
```

macOS users can choose a much faster OCR path with Apple Vision:

```bash
uv run scribebase extract ./scan.pdf --title "Scan" --source-type book --ocr apple_vision
```

Default config in `.study_local/config.yaml`:

```yaml
ocr:
  default_provider: "shell"
  render_dpi: 300
  providers:
    shell:
      command: "./scripts/run_local_ocr.py --input {input_image} --output {output_md}"
      timeout_seconds: 900
      model_name: "GLM-OCR"
      render_dpi:
    apple_vision:
      command: "swift ./scripts/run_apple_vision_ocr.swift --input {input_image} --output {output_md}"
      timeout_seconds: 120
      model_name: "Apple Vision"
      render_dpi: 200
```

The command template supports:

- `{input_image}`
- `{output_md}`
- `{output_json}`
- `{page_number}`
- `{source_id}`

`scripts/run_local_ocr.py` is a committed GLM-OCR adapter. It calls
`$SCRIBEBASE_OCR_BASE_URL/chat/completions`, defaulting to
`http://localhost:8082/v1`. Override `SCRIBEBASE_OCR_MODEL`,
`SCRIBEBASE_OCR_PROMPT`, or `SCRIBEBASE_OCR_MAX_TOKENS` if your local OCR server
needs different values.

`scripts/run_apple_vision_ocr.swift` uses Apple's on-device Vision framework and
requires macOS with Swift available.

Benchmark notes from `The Kubernetes Book 2025.pdf`:

| OCR path | Avg sec/page | Word F1 | Decision |
|---|---:|---:|---|
| Apple Vision 200 DPI | 0.504 | 0.9777 | Fast macOS option |
| GLM-OCR Q8 CPU | 14.769 | 0.9882 | Default high-accuracy OCR |
| Qwen3-VL-2B Q8 Metal | 7.549 | 0.9782 | Good GPU option, not default |
| PaddleOCR-VL GGUF | 43.236 | 0.9183 | Too slow/lower precision here |
| DeepSeek-OCR-2 GGUF | n/a | n/a | llama.cpp path unstable |
| Unsloth DeepSeek-OCR-2 Transformers | failed | 0.6997 on 4 pages | Required CUDA-to-MPS patches and looped on page 300 |

## Common workflows

### Ingest a true-text PDF

```bash
uv run scribebase ingest ./books/biology.pdf \
  --title "Biology 101" \
  --source-type book \
  --language en
```

### Ingest a scanned PDF

```bash
uv run scribebase ingest ./scans/chapter_4_scanned.pdf \
  --title "Cognitive Psychology" \
  --source-type book \
  --chapter "4" \
  --ocr auto
```

### Ingest handwritten note images

```bash
uv run scribebase ingest ./notes/2026-07-lecture-1/ \
  --title "Lecture 1 Notes" \
  --source-type notes \
  --course "Neuroscience" \
  --ocr shell
```

### Extract without indexing

```bash
uv run scribebase extract ./book.pdf --title "Book" --source-type book --ocr auto
```

### Index an existing extracted source

```bash
uv run scribebase index --source-id SOURCE_ID
```

### Search

```bash
uv run scribebase search "working memory limitations" \
  --title "Cognitive Psychology" \
  --chapter "4" \
  --top-k 12
```

### Ask

```bash
uv run scribebase ask "Explain working memory limitations using this chapter." \
  --title "Cognitive Psychology" \
  --chapter "4" \
  --top-k 12
```

If LLM config is disabled or the API key is missing, ScribeBase saves a context pack under `.study_local/outputs/context_packs/`.

### Quiz

```bash
uv run scribebase quiz \
  --title "Cognitive Psychology" \
  --chapter "4" \
  --questions 20 \
  --types mcq,short-answer,flashcard
```

If no LLM is configured, this saves a quiz prompt/context pack under `.study_local/outputs/quizzes/`.

### Inspect local source of truth

```bash
uv run scribebase sources list
uv run scribebase sources show SOURCE_ID
uv run scribebase chunks list --source-id SOURCE_ID
uv run scribebase chunks show CHUNK_ID
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
uv run scribebase rebuild-index --source-id SOURCE_ID
uv run scribebase rebuild-index --all
```

Change embedding models only with an index rebuild. Search rejects mixed embedding model results by default; pass `--allow-model-mismatch` only when you deliberately want that.
Use `rebuild-index --all` after changing embedding dimensions; it recreates the
Weaviate collection so the named vector index matches the new model.

## Development

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
```
