# Changelog

All notable changes to ScribeBase are documented here.

This project has not published Git tags and has declared version `0.1.0` since
its initial implementation. The `0.1.0` section is therefore a historical
backfill from the repository history, not a claim that a tagged release exists.

## [Unreleased]

### Added

- Migrated the default embedding profile from Qwen3-Embedding-4B to
  `nvidia/Nemotron-3-Embed-1B-BF16`, using mean pooling, 2048-dimensional
  vectors, and asymmetric `query: ` / `passage: ` instructions applied only at
  embed time.
- Added a sentence-transformers parity checker for converted GGUF models and
  documented the verified conversion and Mac mini cutover.
- Added embedding-profile fingerprints covering the provider, model,
  dimension, query/document instructions, and normalization setting. Manifests
  and Weaviate chunks now persist the fingerprint.

### Changed

- Embedding searches and incremental indexing now fail closed when profile
  metadata is missing or mismatched. Existing indexes require one full rebuild
  after this guard is deployed.
- Embedding health checks now require `/v1/models` to advertise the configured
  model alias.
- Nemotron serving guidance now consistently uses mean pooling, disables the
  tokenizer BOS token, sets physical and logical batches to 4096, and limits
  context to 8192 tokens.
- Reduced the Mac mini embedding server's steady-state RSS by lowering context
  size from 32768 to 8192.

### Removed

- Removed the CLI and HTTP API escape hatch for searching with mismatched
  embedding models.

### Fixed

- The parity checker now requires the expected 2048 dimensions instead of only
  checking that server and reference dimensions match.
- Documented the parity-only `sentence-transformers` environment and corrected
  the Phase 0 server command to include the mandatory BOS override.

## [0.1.0] - 2026-07-17 (historical backfill; untagged)

### Added

- Built the local-first extraction, OCR, Markdown normalization, chunking,
  llama.cpp embedding, Weaviate indexing, hybrid retrieval, and cited context
  pipeline.
- Added the `scribebase` CLI, local configuration generation, environment
  overrides, setup diagnostics, and recovery guidance.
- Added a read-only HTTP search/context API, authenticated remote ingestion,
  article JSON ingestion, and a durable background ingestion worker.
- Added PDF, image, Markdown, plain-text, and article sources with generic
  metadata, frontmatter parsing, filters, stable source identities, and chapter
  inference.
- Added remote agent and Claude skill templates plus Mac mini deployment and
  launchd documentation.
- Added GLM-OCR and Apple Vision providers with readiness validation and
  automatic document/page routing.

### Changed

- Narrowed the project from built-in answer generation to a retrieval-focused
  knowledge node that returns cited context for consuming agents.
- Increased the default passage size and improved page-marker/chapter handling.
- Reworked full index rebuilds to stage a new collection and promote it through
  a Weaviate alias.
- Reused a single PDF document handle during extraction for lower overhead.

### Fixed

- Enforced named self-provided vectors and rejected accidental mixed embedding
  models and dimensions.
- Added transactional index snapshots, rollback, restart recovery, durable file
  publication, and safe handling of abrupt machine loss.
- Corrected true-text, mixed-scan, blank-page, and failed-OCR behavior so empty
  or degraded sources fail visibly instead of publishing misleading content.
- Added upload cleanup, queue limits, source identity conflict handling, and
  typed dependency failures for recoverable service outages.
- Made GLM-OCR the reliable automatic default while preserving neutral custom
  provider configuration and migrating recognized legacy OCR defaults.
