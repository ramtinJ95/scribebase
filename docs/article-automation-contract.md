# Article automation contract

This document describes how external automations should submit web articles,
newsletter items, Hacker News selections, snippets, and copied documentation to
ScribeBase.

ScribeBase keeps these sources in the same knowledge base as PDFs/books. Use
metadata filters such as `source_type`, `origin`, `publisher`, `tags`, and
`collection` to separate corpora.

## Preferred endpoint

Use `POST /articles` for automation-generated content. It accepts JSON and avoids
multipart upload.

Required:

- `body`: Markdown or plain text content.
- `title`: required unless `body` contains Markdown frontmatter with `title`.

Recommended:

- `source_type`: defaults to `article`.
- `language`: `en`, `sv`, `mixed`, or `unknown`.
- `url`: original URL.
- `canonical_url`: canonical URL when known.
- `origin`: where the item came from, e.g. `company_blog`, `hacker_news`,
  `rss`, `newsletter`, `manual`, `docs`.
- `publisher`: site, organization, vendor, or source owner.
- `author`: author or speaker when known.
- `created_at_source`: publication or source creation time.
- `updated_at_source`: source update time when known.
- `retrieved_at`: time your automation fetched/saved the item.
- `tags`: topical labels.
- `external_id`: upstream ID such as HN item ID, RSS GUID, or automation ID.
- `collection`: user grouping such as `infra-reading`, `kubernetes-reading`, or
  `company-research`.
- `summary`: short source-level summary when available.

## Stable identity and duplicates

ScribeBase identifies a submission in this order:

1. `external_id` together with `origin`;
2. `canonical_url`, falling back to `url`;
3. SHA-256 of the submitted content.

The default `duplicate_policy` is `reject`. A duplicate request returns HTTP
`409` with the existing `source_id` and `identity_key`; automation should treat
that as an already-ingested result rather than retrying indefinitely. Set
`"duplicate_policy": "create"` only when a separate copy is intentional.

## JSON example

```bash
curl -s "$SCRIBEBASE_URL/articles" \
  -H "Authorization: Bearer $SCRIBEBASE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "How We Run Kubernetes",
    "body": "# How We Run Kubernetes\n\nArticle body...",
    "language": "en",
    "tags": ["kubernetes", "platform-engineering"],
    "origin": "company_blog",
    "publisher": "Example Engineering",
    "author": "Author Name",
    "created_at_source": "2026-07-08T00:00:00Z",
    "retrieved_at": "2026-07-08T09:00:00Z",
    "url": "https://example.com/blog/kubernetes",
    "canonical_url": "https://example.com/blog/kubernetes",
    "external_id": "example-blog-kubernetes-2026-07-08",
    "collection": "infra-reading",
    "summary": "Engineering blog post about Kubernetes operations."
  }'
```

The response includes `job_id`. Poll until the job is terminal:

```bash
curl -s "$SCRIBEBASE_URL/jobs/JOB_ID" \
  -H "Authorization: Bearer $SCRIBEBASE_API_TOKEN"
```

Do not treat the article as searchable until `status` is `succeeded`.

## Markdown frontmatter

The `body` may include YAML frontmatter. Explicit JSON fields override
frontmatter values.

```markdown
---
title: "How We Run Kubernetes"
source_type: article
language: en
origin: company_blog
publisher: "Example Engineering"
url: "https://example.com/blog/kubernetes"
created_at_source: "2026-07-08T00:00:00Z"
retrieved_at: "2026-07-08T09:00:00Z"
tags: ["kubernetes", "platform-engineering"]
collection: "infra-reading"
---

# How We Run Kubernetes

Article body...
```

## Company blog template

```json
{
  "title": "Article title",
  "body": "# Article title\n\nArticle body...",
  "source_type": "article",
  "origin": "company_blog",
  "publisher": "Company or engineering blog name",
  "author": "Author name",
  "url": "https://example.com/article",
  "canonical_url": "https://example.com/article",
  "created_at_source": "2026-07-08T00:00:00Z",
  "retrieved_at": "2026-07-08T09:00:00Z",
  "tags": ["kubernetes", "infra"],
  "collection": "infra-reading"
}
```

## Hacker News-selected article template

Use the linked article URL in `url`. Put the HN item ID in `external_id` unless
your automation has a more stable composite ID.

```json
{
  "title": "Linked article title",
  "body": "# Linked article title\n\nArticle body...",
  "source_type": "article",
  "origin": "hacker_news",
  "publisher": "Original publisher or site",
  "url": "https://actual-article.example/post",
  "canonical_url": "https://actual-article.example/post",
  "external_id": "hn:41234567",
  "retrieved_at": "2026-07-08T09:00:00Z",
  "tags": ["databases", "infra"],
  "collection": "hn-reading"
}
```

## Newsletter/RSS template

```json
{
  "title": "Newsletter item title",
  "body": "# Newsletter item title\n\nItem body...",
  "source_type": "article",
  "origin": "newsletter",
  "publisher": "Newsletter name",
  "author": "Author name",
  "url": "https://example.com/newsletter/item",
  "external_id": "rss-guid-or-message-id",
  "created_at_source": "2026-07-08T00:00:00Z",
  "retrieved_at": "2026-07-08T09:00:00Z",
  "tags": ["agents", "research"],
  "collection": "newsletter-reading"
}
```

## Generic notes/snippets/docs template

For non-article text, either use `POST /ingest` with a local `.md`/`.txt` file or
use `POST /articles` and set `source_type` explicitly.

```json
{
  "title": "Kubernetes scheduling notes",
  "body": "# Kubernetes scheduling notes\n\nNotes body...",
  "source_type": "notes",
  "origin": "manual",
  "language": "en",
  "tags": ["kubernetes", "scheduling"],
  "collection": "kubernetes-reading"
}
```

## Retrieval filters

Example query for article context:

```json
{
  "query": "progressive delivery with Argo CD",
  "top_k": 8,
  "filters": {
    "source_type": "article",
    "origin": "company_blog",
    "tags": ["kubernetes", "gitops"],
    "collection": "infra-reading",
    "created_at_source_after": "2026-01-01T00:00:00Z"
  }
}
```

Supported metadata filters include:

- `tags`
- `origin`
- `publisher`
- `author`
- `url`
- `canonical_url`
- `external_id`
- `collection`
- `created_at_source_after` / `created_at_source_before`
- `updated_at_source_after` / `updated_at_source_before`
- `retrieved_at_after` / `retrieved_at_before`
