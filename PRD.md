# PRD: Local OCR → Markdown → Weaviate RAG Knowledge Node v1

## 0. Product summary

Build a local-first Python app that ingests PDFs, scanned PDFs, images, and handwritten note scans; extracts/OCRs them into clean Markdown with page metadata; chunks the text; creates local embeddings using a llama.cpp-compatible embedding server; stores chunks and vectors in local Weaviate; retrieves relevant passages with hybrid search; and optionally sends the selected context to a frontier LLM such as GPT-5.5 Pro for tutoring, explanations, quiz generation, and Q&A.

The v1 app should be CLI-first, with clean internal modules that can later support a web UI.

The core design principle is:

```text
Use local tooling for extraction, OCR, embeddings, indexing, and retrieval.
Use the frontier model only for reasoning over selected retrieved context.
```

## 1. Goals

### Primary goals

1. Ingest a local PDF or image folder.
2. Detect whether PDF pages contain real/selectable text.
3. Use PyMuPDF4LLM for true text PDFs.
4. Use a local OCR provider for image-only/scanned pages.
5. Save canonical extracted content as Markdown, page-level Markdown, and metadata JSON.
6. Chunk Markdown while preserving source, chapter, section, and page metadata.
7. Generate embeddings locally through llama.cpp or a llama.cpp-compatible OpenAI-style embeddings endpoint.
8. Store chunks, metadata, and self-provided vectors in local Weaviate.
9. Retrieve relevant chunks using Weaviate hybrid search.
10. Generate a context pack for GPT-5.5 Pro / GPT-5.5.
11. Optionally call an OpenAI-compatible chat model API to answer questions, create quizzes, and produce flashcards.
12. Support complete local operation for extraction, OCR, embeddings, storage, and retrieval.

### Secondary goals

1. Make OCR models swappable.
2. Make embedding models swappable, but require index rebuild when the embedding model changes.
3. Allow filtering retrieval by source, source type, title, chapter, section, page range, and language.
4. Make Weaviate a rebuildable index, not the canonical storage layer.
5. Provide a clean README so the user can run Weaviate locally and point the app at a local llama.cpp embedding server.

## 2. Non-goals for v1

1. No full polished web app.
2. No multi-user authentication.
3. No cloud vector databases.
4. No Ollama dependency.
5. No mandatory cloud embeddings.
6. No fine-tuning.
7. No automatic purchase/download of books or copyrighted material.
8. No perfect chapter detection for every textbook.
9. No advanced multimodal chat over raw page images.
10. No complex reranking in v1, though the architecture should leave room for it.
11. No background workers/queues unless simple synchronous ingestion becomes impractical.

## 3. Target user workflows

### Workflow A: Ingest a true text PDF

```bash
scribebase ingest ./books/biology.pdf \
  --title "Biology 101" \
  --source-type book \
  --language en
```

Expected behavior:

1. App creates a source ID.
2. App copies or records the original PDF path.
3. App checks whether pages contain extractable text.
4. App uses PyMuPDF4LLM extraction for pages with real text.
5. App writes page Markdown files and a combined document Markdown file.
6. App writes `manifest.json`.
7. App chunks the document.
8. App embeds chunks using local llama.cpp embeddings.
9. App inserts chunks into Weaviate with self-provided vectors.

### Workflow B: Ingest a scanned PDF

```bash
scribebase ingest ./scans/chapter_4_scanned.pdf \
  --title "Cognitive Psychology" \
  --source-type book \
  --chapter "4" \
  --ocr auto
```

Expected behavior:

1. App detects pages with insufficient real text.
2. App renders those pages to PNG at configured DPI.
3. App calls the configured local OCR provider.
4. App stores OCR Markdown per page.
5. App merges page Markdown into source/chapter Markdown.
6. App chunks, embeds, and indexes into Weaviate.

### Workflow C: Ingest handwritten notes

```bash
scribebase ingest ./notes/2026-07-lecture-1/ \
  --title "Lecture 1 Notes" \
  --source-type notes \
  --course "Neuroscience" \
  --ocr chandra
```

Expected behavior:

1. App treats input images as pages.
2. App runs the selected OCR provider.
3. App stores page images, page Markdown, and metadata.
4. App indexes chunks with source type `notes`.
5. Later questions can filter to only notes or combine notes with books.

### Workflow D: Ask a question over a source/chapter

```bash
scribebase ask "Explain working memory limitations using this chapter." \
  --title "Cognitive Psychology" \
  --chapter "4" \
  --top-k 12
```

Expected behavior:

1. App embeds the query locally.
2. App runs Weaviate hybrid search with metadata filters.
3. App retrieves relevant chunks.
4. App builds a context pack with page citations.
5. If LLM API settings exist, app sends the prompt to the configured chat model.
6. If no LLM API settings exist, app writes a prompt pack Markdown file that can be pasted into ChatGPT/GPT-5.5 Pro manually.

### Workflow E: Generate a quiz

```bash
scribebase quiz \
  --title "Cognitive Psychology" \
  --chapter "4" \
  --questions 20 \
  --types mcq,short-answer,flashcard
```

Expected behavior:

1. If the whole chapter Markdown is available and below the configured context limit, use the full chapter.
2. Otherwise use Weaviate retrieval and/or chapter chunks.
3. Generate quiz questions with an answer key.
4. Include page references for every answer when possible.
5. Save output to `outputs/quizzes/<source_id>_chapter_4_quiz.md`.

## 4. System architecture

```text
Input files
  ↓
Source registry
  ↓
PDF/image router
  ├─ True PDF pages → PyMuPDF4LLM
  └─ Scanned/image pages → local OCR provider
  ↓
Canonical Markdown + metadata JSON
  ↓
Chunker
  ↓
llama.cpp embedding client
  ↓
Weaviate local collection with self-provided vectors
  ↓
Hybrid retrieval
  ↓
Context pack builder
  ↓
Optional frontier LLM answer/quiz generation
```

## 5. Recommended v1 technology choices

### Language and packaging

- Python 3.11+
- `uv` or standard `pip` install flow
- Typer for CLI
- Pydantic for config/data models
- PyMuPDF and PyMuPDF4LLM for PDF extraction/rendering
- Weaviate Python client v4
- Requests or httpx for llama.cpp embeddings endpoint
- python-dotenv or YAML-based config loading
- pytest for tests

### Local vector DB

- Weaviate running locally through Docker Compose.
- Use self-provided vectors.
- Do not use Weaviate’s Ollama vectorizer in v1.
- Use hybrid search: query text + externally generated query vector.

### Local embeddings

Use a llama.cpp-compatible embeddings server.

Recommended embedding models to document:

1. `Qwen3-Embedding-0.6B-GGUF`
2. `BAAI/bge-m3` GGUF variant
3. Optional heavier models:
   - Qwen3-Embedding-4B
   - Qwen3-Embedding-8B

The app should not hardcode one model, but default config can assume:

```yaml
embedding:
  provider: llamacpp
  base_url: "http://localhost:8080/v1"
  model: "Qwen3-Embedding-0.6B-GGUF"
  dimension: null
  query_instruction: "Instruct: Given a question, retrieve relevant source passages that answer it\nQuery: "
```

The embedding dimension should be detected from the first successful embedding call and stored in the index metadata.

### Local OCR providers

Implement an OCR provider abstraction. v1 should support at least a generic shell-command provider so the user can connect GLM-OCR, PaddleOCR-VL, Chandra OCR, Surya OCR, DeepSeek-OCR-2, or another local OCR tool without rewriting the app.

Preferred models to document for user benchmarking:

1. GLM-OCR
2. PaddleOCR-VL-1.6
3. Chandra OCR 2 for handwriting/forms/math-heavy notes
4. Unlimited-OCR for long scanned documents
5. DeepSeek-OCR-2 as fallback/comparison
6. Surya OCR 2 for lightweight/local/GGUF-friendly use

The v1 app does not need to implement every model-specific runtime. It must provide a stable adapter interface and at least one configurable command-line adapter.

Example config:

```yaml
ocr:
  default_provider: "shell"
  render_dpi: 300
  providers:
    shell:
      command: "python ./ocr_adapters/glm_ocr_cli.py --input {input_image} --output {output_md}"
      timeout_seconds: 300
```

## 6. Data directory layout

Default data directory:

```text
.scribebase/
  config.yaml
  sources/
    <source_id>/
      original/
        source.pdf
        # or copied image files
      pages/
        page_0001.png
        page_0002.png
      markdown/
        page_0001.md
        page_0002.md
        document.md
        chapters/
          chapter_04.md
      metadata/
        page_0001.json
        page_0002.json
        chunks.jsonl
        manifest.json
  outputs/
    context_packs/
    answers/
    quizzes/
  logs/
```

Weaviate should be treated as a rebuildable search index. The canonical source of truth is:

1. original files,
2. extracted Markdown,
3. metadata JSON/JSONL.

## 7. Source manifest schema

Each ingested source should have a `manifest.json`.

Example:

```json
{
  "schema_version": "1.0",
  "source_id": "cognitive_psychology_2026_abc123",
  "title": "Cognitive Psychology",
  "source_type": "book",
  "course": null,
  "language": "en",
  "original_path": "/absolute/or/copied/path/source.pdf",
  "data_dir": ".scribebase/sources/cognitive_psychology_2026_abc123",
  "created_at": "2026-07-07T12:00:00+02:00",
  "updated_at": "2026-07-07T12:00:00+02:00",
  "extraction_summary": {
    "pages_total": 32,
    "pages_extracted_with_pymupdf4llm": 30,
    "pages_ocr": 2,
    "ocr_provider": "shell",
    "ocr_model": "GLM-OCR"
  },
  "embedding_summary": {
    "embedding_model": "Qwen3-Embedding-0.6B-GGUF",
    "embedding_dimension": 1024,
    "embedding_base_url": "http://localhost:8080/v1",
    "indexed_in_weaviate": true,
    "weaviate_collection": "Chunk"
  }
}
```

## 8. Page metadata schema

Each page should have a metadata JSON file.

Example:

```json
{
  "source_id": "cognitive_psychology_2026_abc123",
  "page_number": 87,
  "page_index": 86,
  "input_type": "pdf_page",
  "text_layer_detected": true,
  "extraction_method": "pymupdf4llm",
  "ocr_provider": null,
  "ocr_model": null,
  "image_path": ".scribebase/sources/.../pages/page_0087.png",
  "markdown_path": ".scribebase/sources/.../markdown/page_0087.md",
  "char_count": 2422,
  "word_count": 387,
  "quality_flags": []
}
```

For OCR pages:

```json
{
  "source_id": "lecture_notes_2026_07_07_abc123",
  "page_number": 3,
  "page_index": 2,
  "input_type": "image",
  "text_layer_detected": false,
  "extraction_method": "ocr",
  "ocr_provider": "shell",
  "ocr_model": "Chandra OCR 2",
  "image_path": ".scribebase/sources/.../pages/page_0003.png",
  "markdown_path": ".scribebase/sources/.../markdown/page_0003.md",
  "char_count": 1040,
  "word_count": 168,
  "quality_flags": ["handwriting", "ocr_uncertain"]
}
```

## 9. Chunk schema

Chunks should be stored in `chunks.jsonl` and indexed into Weaviate.

Example JSONL row:

```json
{
  "chunk_id": "cognitive_psychology_2026_abc123_ch04_p087_0003",
  "source_id": "cognitive_psychology_2026_abc123",
  "source_type": "book",
  "title": "Cognitive Psychology",
  "course": null,
  "chapter": "4",
  "section": "4.2 Working Memory",
  "page_start": 87,
  "page_end": 88,
  "chunk_index": 3,
  "text": "Working memory is commonly described as...",
  "file_path": ".scribebase/sources/.../markdown/chapters/chapter_04.md",
  "extraction_method": "pymupdf4llm",
  "ocr_model": null,
  "language": "en",
  "embedding_model": "Qwen3-Embedding-0.6B-GGUF",
  "embedding_dimension": 1024,
  "chunker_version": "v1"
}
```

## 10. PDF detection logic

The app should process PDFs page by page.

For each page:

1. Try PyMuPDF text extraction.
2. Calculate basic text quality signals:
   - character count,
   - word count,
   - alphanumeric ratio,
   - repeated replacement characters,
   - average word length,
   - number of lines.
3. Mark page as true text if:
   - char count >= configurable threshold, e.g. 200 chars,
   - alphanumeric ratio is plausible,
   - extracted text is not mostly symbols/garbage.
4. If true text, use PyMuPDF4LLM or PyMuPDF-based Markdown extraction.
5. If not true text, render page to image and run OCR.

This should be page-level, not only document-level, because many PDFs contain mixed text and scanned pages.

Config:

```yaml
pdf_detection:
  min_chars_per_page: 200
  min_alpha_ratio: 0.45
  max_replacement_char_ratio: 0.02
```

## 11. OCR provider interface

Define a Python protocol/interface:

```python
class OCRProvider(Protocol):
    name: str

    def ocr_image(
        self,
        image_path: Path,
        output_md_path: Path,
        metadata: dict
    ) -> OCRResult:
        ...
```

`OCRResult`:

```python
class OCRResult(BaseModel):
    markdown_path: Path
    text: str
    provider: str
    model: str | None = None
    confidence: float | None = None
    warnings: list[str] = []
    raw_output_path: Path | None = None
```

### ShellOCRProvider

The shell provider should:

1. Accept command template from config.
2. Replace:
   - `{input_image}`
   - `{output_md}`
   - `{output_json}`
   - `{page_number}`
   - `{source_id}`
3. Run command with timeout.
4. Validate output Markdown exists and is non-empty.
5. Return result.

Example command config:

```yaml
ocr:
  providers:
    shell:
      command: "python ./scripts/run_local_ocr.py --input {input_image} --output {output_md}"
      timeout_seconds: 300
      model_name: "GLM-OCR"
```

This keeps v1 model-agnostic.

## 12. Markdown normalization requirements

The app should normalize extracted text to Markdown.

Required:

1. Add page markers:

```markdown
<!-- page: 87 -->
```

or:

```markdown
[Page 87]
```

Use a consistent format. Prefer HTML comments for machine parsing and visible page headings for user-facing context packs.

2. Preserve headings if available.
3. Remove obvious repeated headers/footers only if easy and safe.
4. Preserve tables as Markdown if extraction provides them.
5. Preserve equations as plain text/LaTeX if extraction provides them.
6. Do not silently delete uncertain OCR text.
7. For OCR uncertainty, allow markers like `[?]`.

## 13. Chunking requirements

Chunking should preserve semantics and metadata.

Rules:

1. Prefer boundaries in this order:
   - chapter heading,
   - section heading,
   - page boundary,
   - paragraph boundary,
   - fallback character/token split.
2. Default chunk size:
   - 500–1200 tokens or equivalent character approximation.
3. Default overlap:
   - 100–200 tokens or equivalent character approximation.
4. Each chunk must have:
   - `source_id`,
   - `title`,
   - `source_type`,
   - `chapter`,
   - `section`,
   - `page_start`,
   - `page_end`,
   - `chunk_index`,
   - `text`,
   - `file_path`.

Use a simple tokenizer approximation in v1 if needed. Do not make token-perfect chunking a blocker.

## 14. llama.cpp embedding client

The app should call a local OpenAI-compatible llama.cpp embeddings endpoint.

Config:

```yaml
embedding:
  provider: "llamacpp"
  base_url: "http://localhost:8080/v1"
  model: "Qwen3-Embedding-0.6B-GGUF"
  timeout_seconds: 120
  batch_size: 16
  query_instruction: "Instruct: Given a question, retrieve relevant source passages that answer it\nQuery: "
  normalize: true
```

### Embedding API request

Endpoint:

```text
POST {base_url}/embeddings
```

Payload:

```json
{
  "model": "Qwen3-Embedding-0.6B-GGUF",
  "input": ["text chunk 1", "text chunk 2"],
  "encoding_format": "float"
}
```

Expected response:

```json
{
  "data": [
    {
      "index": 0,
      "embedding": [0.1, 0.2, 0.3]
    }
  ]
}
```

### Query embedding

For document chunks:

```text
<chunk text>
```

For user queries:

```text
Instruct: Given a question, retrieve relevant source passages that answer it
Query: <user question>
```

Make this instruction configurable.

### Embedding consistency

The app must prevent accidental mixed-model querying/indexing.

When retrieving from a collection/index, check that:

1. stored `embedding_model` matches current configured model, or
2. show a warning and require `--allow-model-mismatch`.

Default behavior should reject mismatches.

## 15. Weaviate collection design

Collection name:

```text
Chunk
```

Use self-provided vectors. Prefer named vector:

```text
text_vector
```

Properties:

| Property | Type | Required | Description |
|---|---:|---:|---|
| `text` | text | yes | Chunk text |
| `chunk_id` | text | yes | Stable chunk ID |
| `source_id` | text | yes | Stable source ID |
| `source_type` | text | yes | book, notes, paper, article, other |
| `title` | text | yes | Book/note title |
| `course` | text | no | Course/project name |
| `chapter` | text | no | Chapter label |
| `section` | text | no | Section heading |
| `page_start` | int | no | First page in chunk |
| `page_end` | int | no | Last page in chunk |
| `chunk_index` | int | yes | Sequential chunk index |
| `file_path` | text | yes | Path to canonical Markdown |
| `extraction_method` | text | yes | pymupdf4llm, ocr, mixed |
| `ocr_model` | text | no | OCR model used, if any |
| `language` | text | no | en, sv, mixed, unknown |
| `embedding_model` | text | yes | Local embedding model name |
| `embedding_dimension` | int | yes | Vector dimension |
| `created_at` | date | yes | Ingestion timestamp |

Indexing behavior:

1. Create collection if missing.
2. Validate vector dimension.
3. Insert or upsert by stable `chunk_id`.
4. Support deleting all chunks for a source and rebuilding.

## 16. Retrieval behavior

Use Weaviate hybrid search with:

1. query text for BM25/keyword side,
2. local query vector for vector side,
3. metadata filters,
4. configurable `alpha`.

Defaults:

```yaml
retrieval:
  alpha: 0.65
  top_k: 12
  candidate_k: 30
  include_metadata: true
```

The app should support filters:

```bash
--source-id
--title
--source-type
--course
--chapter
--section
--page-start
--page-end
--language
```

Example command:

```bash
scribebase search "working memory limitations" \
  --title "Cognitive Psychology" \
  --chapter "4" \
  --top-k 12
```

Search output should include:

```text
1. Cognitive Psychology, chapter 4, section 4.2, pages 87–88
   score: ...
   chunk_id: ...
   snippet: ...
```

## 17. Context pack builder

For LLM usage, build a structured context pack.

Format:

```markdown
# Context Pack

User question:
Explain working memory limitations.

Instructions:
Use only the provided context. Cite sources as [Title, p. 87]. If the answer is not in the context, say so.

## Source 1
Title: Cognitive Psychology
Chapter: 4
Section: 4.2 Working Memory
Pages: 87–88
Chunk ID: cognitive_psychology_2026_abc123_ch04_p087_0003

<chunk text>

## Source 2
...
```

Save to:

```text
.scribebase/outputs/context_packs/<timestamp>_<slug>.md
```

## 18. LLM integration

The app should support optional chat completion via an OpenAI-compatible API.

Config:

```yaml
llm:
  enabled: false
  provider: "openai_compatible"
  base_url: "https://api.openai.com/v1"
  model: "gpt-5.5-pro"
  api_key_env: "OPENAI_API_KEY"
  temperature: 0.2
```

If `llm.enabled = false` or no API key exists:

1. Do not fail.
2. Generate and save a context pack.
3. Print the context pack path.

If enabled:

1. Retrieve or assemble context.
2. Send context plus task prompt to configured model.
3. Save answer Markdown.
4. Print answer path and answer summary.

### Default answer prompt

```text
You are a source-grounded assistant.

Use only the supplied context.
Cite every factual claim with page references from the context.
If the context does not contain the answer, say that the provided material does not contain enough information.
Be clear, concrete, and pedagogical.
```

### Default quiz prompt

```text
Create a quiz from the supplied context.

Requirements:
- Use only the supplied context.
- Create the requested number and types of questions.
- Include a separate answer key.
- Include supporting page references for every answer.
- Mix easy, medium, and hard questions.
- Prefer active recall over trivia.
```

## 19. CLI specification

Use Typer.

### `scribebase init`

Creates config and directory structure.

```bash
scribebase init --data-dir .scribebase
```

### `scribebase doctor`

Checks:

1. Python dependencies.
2. Weaviate connectivity.
3. llama.cpp embedding endpoint connectivity.
4. Embedding dimension.
5. OCR provider config.
6. Optional LLM API config.

```bash
scribebase doctor
```

### `scribebase ingest`

Ingests source and indexes by default.

```bash
scribebase ingest PATH \
  --title TEXT \
  --source-type book|notes|paper|article|other \
  --course TEXT \
  --chapter TEXT \
  --language en|sv|mixed|unknown \
  --ocr auto|never|always|shell|chandra|glm|paddle|deepseek \
  --no-index
```

### `scribebase extract`

Only extract/OCR to Markdown, no indexing.

```bash
scribebase extract PATH --title TEXT --ocr auto
```

### `scribebase index`

Index existing extracted source into Weaviate.

```bash
scribebase index --source-id SOURCE_ID
```

### `scribebase rebuild-index`

Delete and rebuild chunks/vectors for source or all sources.

```bash
scribebase rebuild-index --source-id SOURCE_ID
scribebase rebuild-index --all
```

### `scribebase search`

Retrieve chunks without LLM answering.

```bash
scribebase search "query text" --title "Cognitive Psychology" --chapter "4"
```

### `scribebase ask`

Retrieve and answer or produce context pack.

```bash
scribebase ask "question text" \
  --title "Cognitive Psychology" \
  --chapter "4" \
  --top-k 12 \
  --mode rag
```

Modes:

```text
rag       retrieve chunks from Weaviate
chapter   use whole chapter Markdown if available
auto      use whole chapter if it fits, otherwise RAG
```

### `scribebase quiz`

Generate quiz from chapter/source/retrieved context.

```bash
scribebase quiz \
  --title "Cognitive Psychology" \
  --chapter "4" \
  --questions 20 \
  --types mcq,short-answer,flashcard
```

### `scribebase sources`

List known sources.

```bash
scribebase sources list
scribebase sources show SOURCE_ID
```

### `scribebase chunks`

Inspect chunks.

```bash
scribebase chunks list --source-id SOURCE_ID
scribebase chunks show CHUNK_ID
```

## 20. Docker Compose for local Weaviate

Include `docker-compose.weaviate.yml`.

Basic requirement:

```yaml
services:
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:latest
    ports:
      - "8081:8080"
      - "50051:50051"
    volumes:
      - weaviate_data:/var/lib/weaviate
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"
      PERSISTENCE_DATA_PATH: "/var/lib/weaviate"
      DEFAULT_VECTORIZER_MODULE: "none"
      ENABLE_MODULES: ""
      CLUSTER_HOSTNAME: "node1"

volumes:
  weaviate_data:
```

The app config should default to:

```yaml
weaviate:
  url: "http://localhost:8081"
  collection: "Chunk"
```

## 21. Example llama.cpp embedding server docs

README should include example commands but not require a specific model file.

Example:

```bash
llama-server \
  --model ./models/Qwen3-Embedding-0.6B-Q8_0.gguf \
  --embedding \
  --pooling last \
  --ctx-size 32768 \
  --port 8080
```

Then:

```bash
scribebase doctor
```

should verify:

1. `GET /v1/models` if available, or
2. `POST /v1/embeddings` with test input.

## 22. Config file

Default config path:

```text
.scribebase/config.yaml
```

Example:

```yaml
data_dir: ".scribebase"

weaviate:
  url: "http://localhost:8081"
  collection: "Chunk"
  vector_name: "text_vector"

embedding:
  provider: "llamacpp"
  base_url: "http://localhost:8080/v1"
  model: "Qwen3-Embedding-0.6B-GGUF"
  timeout_seconds: 120
  batch_size: 16
  query_instruction: "Instruct: Given a question, retrieve relevant source passages that answer it\nQuery: "
  normalize: true

pdf_detection:
  min_chars_per_page: 200
  min_alpha_ratio: 0.45
  max_replacement_char_ratio: 0.02

ocr:
  default_provider: "shell"
  render_dpi: 300
  providers:
    shell:
      command: "python ./scripts/run_local_ocr.py --input {input_image} --output {output_md}"
      timeout_seconds: 300
      model_name: "GLM-OCR"

chunking:
  target_chars: 4000
  overlap_chars: 600
  min_chars: 800
  chunker_version: "v1"

retrieval:
  alpha: 0.65
  top_k: 12
  candidate_k: 30

llm:
  enabled: false
  provider: "openai_compatible"
  base_url: "https://api.openai.com/v1"
  model: "gpt-5.5-pro"
  api_key_env: "OPENAI_API_KEY"
  temperature: 0.2
```

## 23. Error handling

The app should fail clearly and recoverably.

Required error cases:

1. Weaviate not running:
   - show command to start Docker Compose.
2. llama.cpp embeddings endpoint unavailable:
   - show configured URL and a sample server command.
3. OCR provider command missing/failing:
   - mark page failed,
   - continue if `--continue-on-ocr-error`,
   - otherwise stop with clear error.
4. Empty extraction:
   - suggest `--ocr always`.
5. Embedding dimension mismatch:
   - reject indexing and suggest rebuilding index.
6. Collection missing:
   - create automatically unless `--no-create-collection`.
7. LLM API key missing:
   - save context pack instead of failing.

## 24. Logging

Use structured-ish logs.

Minimum:

```text
[INFO] Ingest source: ...
[INFO] Page 1: text layer detected, using PyMuPDF4LLM
[INFO] Page 2: insufficient text, rendering at 300 DPI
[INFO] Page 2: OCR with shell provider
[INFO] Created 42 chunks
[INFO] Embedding batch 1/3
[INFO] Indexed 42 chunks into Weaviate collection Chunk
```

Write logs to:

```text
.scribebase/logs/app.log
```

## 25. Tests

Use pytest.

### Unit tests

1. PDF text quality detection:
   - real text accepted,
   - empty text rejected,
   - symbol garbage rejected.
2. Source ID generation.
3. Manifest read/write.
4. Page metadata read/write.
5. Chunker preserves page metadata.
6. Embedding client parses OpenAI-style embeddings response.
7. Shell OCR provider command formatting.
8. Context pack includes citations and metadata.
9. Config loading and defaults.

### Integration tests, optional but preferred

1. Weaviate collection creation.
2. Insert sample chunks with mocked vectors.
3. Search sample chunks.
4. End-to-end ingest of a tiny text PDF fixture.
5. End-to-end context pack generation without LLM API.

Mocks are acceptable for llama.cpp and OCR in CI.

## 26. Acceptance criteria for v1

v1 is complete when:

1. User can run `scribebase init`.
2. User can start local Weaviate using provided Docker Compose.
3. User can start their own llama.cpp embedding server and confirm with `scribebase doctor`.
4. User can ingest a true text PDF and get Markdown output.
5. User can ingest a scanned/image source through the shell OCR provider.
6. User can index chunks into Weaviate using self-provided local embeddings.
7. User can run `scribebase search` and see relevant chunks with page metadata.
8. User can run `scribebase ask` and either:
   - receive an LLM answer if configured, or
   - receive a saved context pack if LLM is disabled.
9. User can run `scribebase quiz` and receive either:
   - generated quiz output if LLM is configured, or
   - a saved quiz prompt context pack if LLM is disabled.
10. No Ollama dependency exists anywhere in the app.
11. Weaviate is local by default.
12. Embeddings are local by default.
13. Original files and Markdown are kept locally as source of truth.
14. Rebuilding the Weaviate index from Markdown works.

## 27. Suggested repository structure

```text
scribebase/
  README.md
  pyproject.toml
  docker-compose.weaviate.yml
  scribebase/
    __init__.py
    cli.py
    config.py
    paths.py
    models.py
    source_registry.py
    pdf_router.py
    extractors/
      __init__.py
      pymupdf_extractor.py
      image_renderer.py
    ocr/
      __init__.py
      base.py
      shell_provider.py
    markdown/
      __init__.py
      normalize.py
      chapter_splitter.py
    chunking/
      __init__.py
      chunker.py
    embeddings/
      __init__.py
      llamacpp_client.py
    vectorstores/
      __init__.py
      weaviate_store.py
    retrieval/
      __init__.py
      search.py
      context_pack.py
    llm/
      __init__.py
      openai_compatible.py
      prompts.py
    commands/
      __init__.py
      ingest.py
      extract.py
      index.py
      search.py
      ask.py
      quiz.py
      doctor.py
  scripts/
    run_local_ocr.py.example
  tests/
    test_pdf_detection.py
    test_chunker.py
    test_embedding_client.py
    test_context_pack.py
    test_config.py
```

## 28. Implementation notes for Codex

1. Implement the CLI-first app.
2. Do not implement a large UI in v1.
3. Do not use Ollama.
4. Do not use cloud embeddings.
5. Use llama.cpp-compatible `/v1/embeddings` for embeddings.
6. Use Weaviate self-provided vectors.
7. Keep OCR model execution abstract through the shell provider.
8. Keep original files and extracted Markdown as source of truth.
9. Make the app useful even without an LLM API key by writing context packs.
10. Use clear errors and a `scribebase doctor` command.
11. Include a good README with:
    - installation,
    - Weaviate Docker Compose,
    - llama.cpp embedding server example,
    - OCR provider configuration,
    - example ingest/search/ask/quiz commands.

## 29. Future v2 ideas

Do not implement these in v1 unless trivial:

1. Local reranker:
   - BGE reranker,
   - Qwen reranker,
   - cross-encoder adapter.
2. Streamlit or local web UI.
3. Direct model-specific adapters for GLM-OCR, PaddleOCR-VL, Chandra, Surya, DeepSeek-OCR-2.
4. Chapter/TOC detection using LLM or document layout signals.
5. Handwriting correction pass.
6. Learning session tracking.
7. Spaced repetition export to Anki.
8. Multi-source comparison mode.
9. Automatic benchmark harness for OCR models.
10. OCR confidence visualization.
11. Page image viewer with text alignment.
12. Local-only chat model mode.
13. Multi-vector retrieval for long chunks.
14. Separate collections for pages, chunks, summaries, and quizzes.
15. Watch-folder ingestion.

## 30. Initial Codex prompt

Use this prompt with Codex CLI:

```text
Implement the v1 app described in PRD.md.

Priorities:
1. Build the CLI-first Python app with Typer.
2. Implement config loading, data directory layout, manifests, and source registry.
3. Implement PDF routing:
   - use PyMuPDF/PyMuPDF4LLM for pages with real text,
   - render pages and call shell OCR provider for pages without real text.
4. Implement Markdown normalization and chunking with page metadata.
5. Implement llama.cpp-compatible embedding client using /v1/embeddings.
6. Implement Weaviate v4 integration using self-provided named vectors.
7. Implement search, ask, quiz, and context-pack generation.
8. Add docker-compose.weaviate.yml and README.
9. Add tests for detection, chunking, embeddings client, config, and context pack generation.

Constraints:
- No Ollama dependency.
- No cloud embeddings.
- Weaviate is local by default.
- Embeddings are local through llama.cpp by default.
- OCR provider is configurable and local.
- The app must still be useful without an LLM API key by saving context packs.
- Keep extracted Markdown and metadata as source of truth; Weaviate is rebuildable.
```
