# Text and article ingestion handoff

## Goal

Extend ScribeBase so the Mac mini knowledge node can ingest not only PDFs/books,
but also article-style knowledge from automations such as company blog monitors,
Hacker News filters, newsletters, and manual web saves.

Keep one ScribeBase node and one Weaviate instance by default. Separate content
with explicit metadata and filters, not separate databases, unless a future corpus
needs a different embedding model, retention policy, or access boundary.

## Current state

ScribeBase currently has general source metadata:

- `source_id`
- `title`
- `source_type`
- `course`
- `chapter`
- `section`
- `page_start` / `page_end`
- `language`

This is enough for books/PDFs, but weak for articles. We should not overload
`course` and `chapter` long term for article concepts like publisher, URL,
origin, or tags.

Current ingestion supports PDFs, images, and image directories. It does not yet
support `.txt`, `.md`, or `.markdown` as first-class inputs. Add generic text
and Markdown ingestion before the richer generic metadata work so the medium
expansion is useful independently of the article schema.

## Target generic metadata

Add a generic metadata layer that works for books, articles, notes, transcripts,
snippets, copied docs, and future text sources. This should not be article-only.

Recommended optional fields for all source types:

- `source_type`: content category, e.g. `book`, `paper`, `article`, `notes`,
  `transcript`, `documentation`, `snippet`, `other`.
- `title`: human-readable source title.
- `language`: `en`, `sv`, `mixed`, or `unknown`.
- `tags`: list of topical labels, e.g. `kubernetes`, `agents`, `databases`.
- `origin`: where this came from, e.g. `manual`, `hacker_news`,
  `company_blog`, `rss`, `newsletter`, `meeting_notes`, `docs`, `copy_paste`,
  `youtube`, `other`.
- `publisher`: organization, site, vendor, or owner when known.
- `author`: author, speaker, or note owner when known.
- `created_at_source`: when the source content itself was created, if known.
- `updated_at_source`: when the source content itself was updated, if known.
- `retrieved_at`: when ScribeBase or an automation fetched/saved it.
- `url`: source URL, if any.
- `canonical_url`: canonical URL, if any.
- `external_id`: upstream ID, e.g. HN item ID, RSS GUID, ticket ID, video ID,
  or automation ID.
- `collection`: user-defined grouping, e.g. `infra-reading`,
  `company-research`, `kubernetes-study`, `personal-notes`.
- `summary`: short source-level summary when supplied by an automation or user.

Keep existing academic/book-oriented fields for compatibility:

- `course`
- `chapter`
- `section`
- `page_start` / `page_end`

Use `course` and `chapter` for actual courses/books. Do not overload them for
generic web/article concepts once the generic fields exist.

## Target article metadata

Article metadata should mostly reuse the generic fields above. Article-specific
usage conventions:

- `source_type`: `article`.
- `url`: source URL.
- `canonical_url`: canonicalized URL when known.
- `origin`: source channel, e.g. `hacker_news`, `company_blog`, `newsletter`,
  `rss`, `manual`, `other`.
- `publisher`: organization or site, e.g. `Cloudflare`, `Anthropic`, `Stripe`.
- `author`: article author when known.
- `created_at_source`: article publication date/datetime when known.
- `retrieved_at`: time the automation fetched the article.
- `tags`: list of topical labels, e.g. `kubernetes`, `agents`, `databases`.
- `external_id`: upstream ID, e.g. HN item ID, RSS GUID, or automation ID.

These fields should be optional so existing manifests remain valid. Avoid adding
HN-only or site-specific fields in the base model unless they become broadly
useful.

## Preferred automation contract

Automations should eventually send article bodies directly rather than converting
them to PDFs. Preferred interchange format is Markdown with YAML frontmatter:

```markdown
---
title: "Article Title"
source_type: article
origin: company_blog
publisher: "Cloudflare"
url: "https://example.com/article"
canonical_url: "https://example.com/article"
author: "Author Name"
created_at_source: "2026-07-08"
retrieved_at: "2026-07-08T09:00:00Z"
tags: ["kubernetes", "networking"]
external_id: "optional-upstream-id"
collection: "infra-reading"
summary: "Short optional source-level summary."
---

# Article Title

Article body...
```

For Hacker News items, use:

```yaml
source_type: article
origin: hacker_news
publisher: "Hacker News"
external_id: "41234567"
url: "https://actual-article-url.example/post"
canonical_url: "https://actual-article-url.example/post"
tags: ["databases", "infra"]
```

For generic notes or copied text that is not an article:

```markdown
---
title: "Kubernetes scheduling notes"
source_type: notes
origin: manual
collection: "kubernetes-study"
tags: ["kubernetes", "scheduling"]
created_at_source: "2026-07-08"
language: en
---

# Kubernetes scheduling notes

Notes body...
```

For `.txt` files, metadata can be supplied by CLI/API fields. If a `.txt` file
contains YAML frontmatter, treat it as plain text until we deliberately add text
frontmatter support; Markdown is the canonical frontmatter format.

If we later want HN-specific metrics like score or comment count, add them in a
separate metadata extension rather than blocking the base article schema.

## PR stack

### PR 1: Generic text and Markdown ingestion

Goal: support `.txt`, `.md`, and `.markdown` with the existing metadata model.

Scope:

- Support `.txt`, `.md`, and `.markdown` inputs in CLI and HTTP ingestion.
- Preserve Markdown files as Markdown.
- Convert plain `.txt` files into basic Markdown text without inventing
  structure.
- Write content into the existing source layout:
  - `original/`
  - `markdown/document.md`
  - `metadata/manifest.json`
  - `metadata/chunks.jsonl`
- Add page/document metadata for text inputs without pretending they are PDF
  pages.
- Chunk and index text/Markdown through the existing pipeline.
- Add tests for:
  - plain `.txt`
  - Markdown without frontmatter
  - HTTP upload of `.txt` or `.md`

Non-goals:

- Do not parse YAML frontmatter yet.
- Do not add generic metadata fields yet.

### PR 2: Generic metadata model

Goal: add generic metadata fields without changing ingestion formats.

Scope:

- Extend manifest/source models with optional generic metadata fields.
- Extend chunk model with the same fields needed during retrieval.
- Add CLI/API ingest parameters for the new fields.
- Persist fields in manifests and chunks.
- Preserve backward compatibility for existing manifests.
- Add tests for model defaults, old manifest loading, and generic metadata persistence.

Non-goals:

- Do not change Weaviate schema/filtering yet unless required by model tests.

### PR 3: Markdown frontmatter metadata

Goal: let Markdown files carry generic metadata inline.

Scope:

- Parse optional YAML frontmatter from `.md` and `.markdown` files.
- Use frontmatter as metadata defaults.
- Let explicit CLI/API fields override frontmatter when both are present.
- Validate frontmatter types for dates and tags.
- Add tests for Markdown with frontmatter.

Non-goals:

- Do not add a separate article API endpoint yet.

### PR 4: Weaviate schema and filters

Goal: make generic metadata searchable and filterable.

Scope:

- Add Weaviate properties for generic metadata.
- Extend search filters with relevant fields:
  - `url`
  - `canonical_url`
  - `origin`
  - `publisher`
  - `author`
  - `tags`
  - `external_id`
  - `collection`
  - `created_at_source_after` / `created_at_source_before`
  - `retrieved_at_after` / `retrieved_at_before`
- Update filter builder and search/context API models.
- Add tests for filter construction and API request parsing.
- Document that schema changes require index rebuild/recreation.

Non-goals:

- Do not add new ingestion file types in this PR.

### PR 5: Article API ergonomics

Goal: make remote automations easy and predictable.

Scope:

- Document multipart `POST /ingest` examples for Markdown/text sources.
- Consider adding `POST /articles` as a convenience endpoint with JSON body:
  - `title`
  - `body`
  - `url`
  - generic/article metadata
- Add examples for company blog and Hacker News automation output.
- Add error messages for missing `title`, empty body, or invalid metadata.

Decision point:

- If multipart upload is sufficient for automations, skip `POST /articles` and
  keep the API surface smaller.

### PR 6: Automation templates and skills

Goal: make agents and automations use article ingestion consistently.

Scope:

- Update `scribebase-ingest` skill with Markdown/article examples.
- Add an optional `scribebase-article-ingest` skill only if the general ingest
  skill becomes too branchy.
- Document the automation contract for:
  - company blog articles
  - Hacker News-selected articles
  - newsletters/RSS
- Add sample frontmatter templates for articles, notes, snippets, and docs.

Non-goals:

- Do not add automation-specific scrapers to ScribeBase in this PR stack.
  Existing external automations should call ScribeBase.

## Operational guidance

- Keep articles in the same Weaviate instance as books/PDFs.
- Prefer metadata filters over separate databases.
- Split into a new Weaviate collection or instance only if there is a hard reason:
  - different embedding model/dimension
  - separate access/security boundary
  - separate lifecycle/retention policy
  - very large unrelated corpus causing relevance or rebuild problems
  - experimental data that is frequently wiped

## Acceptance criteria for the full stack

By the end of the stack, an automation should be able to submit a company blog or
HN-selected article with title, URL, publisher/origin, tags, and Markdown body;
and a user should be able to submit generic notes/snippets/docs with title,
collection, origin, tags, and text/Markdown body;
ScribeBase should index it into the shared knowledge base; and an agent should be
able to retrieve it with filters such as:

```json
{
  "query": "progressive delivery with Argo CD",
  "filters": {
    "source_type": "article",
    "origin": "company_blog",
    "tags": ["kubernetes", "gitops"]
  },
  "top_k": 8
}
```
