# ScribeBase

ScribeBase is a planned local-first study app for turning PDFs, scanned documents, images, and handwritten notes into page-cited Markdown and searchable RAG context.

The v1 direction is CLI-first:

- extract text from true PDFs with PyMuPDF/PyMuPDF4LLM
- route scanned or image-only pages through a configurable local OCR provider
- keep original files, Markdown, and metadata as the canonical source of truth
- create local embeddings through a llama.cpp-compatible `/v1/embeddings` endpoint
- index chunks into local Weaviate using self-provided vectors
- retrieve page-cited context for search, tutoring, quizzes, and flashcards
- optionally call an OpenAI-compatible chat model, while still working without one by saving context packs

See [PRD.md](PRD.md) for the implementation plan.

## Development

ScribeBase targets Python 3.14 and uses `uv` for Python package and environment
management.

```bash
uv sync
```

## Status

Repository scaffold only. Implementation has not started yet.
