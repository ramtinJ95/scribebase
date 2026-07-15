# ScribeBase product contract

## Purpose

ScribeBase is a local-first knowledge node. It turns local documents and
automation-submitted text into durable Markdown, metadata, embeddings, and
cited retrieval context for agents.

The local source tree is authoritative. Weaviate is a disposable retrieval
index that must be rebuildable from that tree.

## Product boundary

ScribeBase owns:

- source ingestion and stable identity;
- PDF text extraction and page-level OCR routing;
- image OCR through configured local providers;
- Markdown and plain-text normalization;
- source, page, chapter, section, and generic metadata;
- chunking and local embedding generation;
- Weaviate indexing, filtering, and hybrid retrieval;
- cited context-pack construction;
- a read/search/context/ingestion HTTP API;
- a durable, single-worker ingestion queue.

ScribeBase does not own:

- final answer generation;
- tutoring or quiz generation;
- chat-model provider configuration;
- web crawling or article extraction from arbitrary URLs;
- public-internet authentication or multi-tenant isolation;
- distributed queue or shared-filesystem coordination.

Consuming agents decide how to reason over returned context.

## Supported inputs

- born-digital PDFs;
- scanned and mixed PDFs;
- PNG, JPEG, TIFF, WebP, and BMP images;
- Markdown with optional YAML frontmatter;
- UTF-8 plain text;
- article/text JSON submitted to `POST /articles`.

## Source identity

Identity precedence is:

1. `origin + external_id`;
2. canonical URL, falling back to URL;
3. content SHA-256.

Duplicates are rejected by default and return the existing source ID. Creating
a separate copy requires an explicit `duplicate_policy=create` request.
Completed sources are published from isolated staging directories. Failed
extractions must not create visible source manifests.

## Local data contract

Each source is stored beneath:

```text
<data_dir>/sources/<source_id>/
  original/
  pages/
  markdown/
    page_0001.md
    document.md
    chapters/
  metadata/
    manifest.json
    page_0001.json
    chunks.jsonl
```

Source IDs are path-safe identifiers. Manifests include schema, extraction,
embedding, identity, and generic source metadata. Page Markdown contains
explicit page markers so citations survive chunking.

## PDF extraction

One caller-owned PyMuPDF document is used for each extraction pass.

For every page ScribeBase evaluates:

- extracted text quality;
- embedded-image presence;
- low-resolution visual content when text is unusable.

True-text pages use PyMuPDF4LLM with internal OCR disabled. Scanned or visual
pages route to the configured ScribeBase OCR provider. Blank pages in an
otherwise true-text PDF do not require OCR.

Page-local PyMuPDF4LLM failures may visibly fall back to cached PyMuPDF text.
API incompatibility, document ownership failures, resource failures, and
repeated layout failures are fatal rather than silently degrading a document.

## Chunking and metadata

Chunks preserve:

- source ID and source type;
- title, course, chapter, and section;
- page start/end;
- generic metadata and tags;
- extraction/OCR provenance;
- embedding model and dimension;
- deterministic chunk ID and chunker version.

Explicit source-level chapter metadata overrides inference. Otherwise chapter
inference may use supported chapter heading forms. Page prologue markers belong
to the chapter beginning on that page, not the preceding chapter.

## Embeddings and index

Embeddings come from an OpenAI-compatible local llama.cpp endpoint. Query
embeddings use the configured retrieval instruction. Mixed embedding models or
dimensions are rejected unless an explicit full rebuild is in progress.

The configured Weaviate collection name is a stable alias. Full rebuilds:

1. create a versioned physical collection;
2. stream embedding/index batches with bounded memory;
3. verify expected object counts;
4. promote the alias;
5. transactionally install matching local chunk/manifest metadata;
6. retain a durable journal until remote and local generations agree.

Single-source updates snapshot existing vectors and restore them on failed
replacement. A durable operation journal restores an interrupted mutation or
finishes local publication after the remote commit. Index mutations are
serialized per data directory.

## Retrieval

Retrieval uses Weaviate hybrid search with a named self-provided vector. It
supports source, chapter, section, page range, tags, generic metadata, and
source-date filters.

Results include chunk text, score information, page metadata, and stable chunk
IDs. Context packs instruct consumers to use supplied context and cite sources.

## HTTP API

- `GET /health`
- `GET /sources`
- `POST /search`
- `POST /context`
- `POST /ingest`
- `POST /articles`
- `GET /jobs/{job_id}`
- `POST /jobs/{job_id}/retry`

All endpoints except health require a bearer token. The service is intended for
LAN/Tailscale deployment, not direct public-internet exposure.

## Ingestion worker

The API persists queued work but does not perform OCR/indexing in request
background tasks. One separate worker per local data directory claims jobs with
filesystem locks and ownership tokens.

Jobs have durable extraction/indexing phases. Worker health requires both the
worker lock and a fresh heartbeat. Interrupted jobs resume from their durable
phase. Index work waits and retries while Weaviate or embeddings are unavailable
after reboot. Upload size, queue capacity, total upload storage, reservations,
and failed-upload retention are bounded by configuration.

Uploads and canonical source trees are synced to stable local storage before
their durable job or live source entry is published. On macOS this includes a
full-storage sync before the atomic rename.

NFS/SMB queues and multiple hosts sharing one data directory are unsupported.

## Deployment

The reference node is a Mac mini running:

- ScribeBase API;
- ScribeBase ingestion worker;
- local Weaviate;
- llama.cpp embeddings;
- configured local OCR provider.

Weaviate and embedding/OCR services remain localhost-only. The ScribeBase API
may bind to LAN/Tailscale interfaces with bearer authentication.

## Acceptance criteria

- Every successful source has a durable original, Markdown, and manifest.
- Failed extraction does not publish a source.
- Duplicate automation retries do not create duplicate sources.
- Every searchable chunk can be traced to source/page metadata.
- Existing search remains available during failed staged rebuilds.
- Worker/process crashes and sudden machine loss preserve enough state for
  automatic worker recovery on a healthy local filesystem.
- Index/schema/model changes fail visibly rather than silently mixing states.
- The complete Weaviate index can be rebuilt from local source data.
