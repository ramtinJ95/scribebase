---
name: scribebase-context
description: Retrieve cited context from a remote ScribeBase server for the current task.
---

# ScribeBase context

Use this skill when the user asks to search ScribeBase, fetch relevant notes,
retrieve documents into context, or ground the current task in stored sources.

## Requirements

The current shell must have:

```bash
SCRIBEBASE_URL=http://macmini.local:8765
SCRIBEBASE_API_TOKEN=...
```

If either variable is missing, ask the user for the missing value before trying
to search.

## Retrieval modes

Prefer `/context` when the user needs material to use in the current answer or
task. It returns a ready-to-paste context pack with citations.

Use `/search` when the user only wants ranked snippets, source IDs, chunk IDs,
or a quick inventory of matching material.

## Filters

Use filters when the user provides them:

- `source_id`
- `title`
- `source_type`
- `course`
- `chapter`
- `section`
- `page_start`
- `page_end`
- `language`

Default retrieval settings:

- `top_k=8` for normal context gathering
- `top_k=12` for broad or ambiguous questions
- omit `alpha` unless the user asks to tune retrieval
- keep `allow_model_mismatch=false` unless the user explicitly accepts stale mixed-model results

## Commands

Health:

```bash
curl -s "$SCRIBEBASE_URL/health"
```

List sources:

```bash
curl -s "$SCRIBEBASE_URL/sources" \
  -H "Authorization: Bearer $SCRIBEBASE_API_TOKEN"
```

Get a context pack:

```bash
curl -s "$SCRIBEBASE_URL/context" \
  -H "Authorization: Bearer $SCRIBEBASE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "question or task",
    "task": "answer",
    "top_k": 8,
    "filters": {
      "source_type": "paper",
      "language": "en"
    }
  }'
```

Search snippets:

```bash
curl -s "$SCRIBEBASE_URL/search" \
  -H "Authorization: Bearer $SCRIBEBASE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "question or topic",
    "top_k": 8,
    "filters": {}
  }'
```

## Response handling

For `/context`, use `context_pack` as the retrieved source material. Preserve
citations and chunk IDs when answering.

For `/search`, inspect `results[].chunk`:

- `title`
- `source_id`
- `chapter`
- `page_start` / `page_end`
- `chunk_id`
- `text`
- `score`

If no results come back, say that ScribeBase returned no matching context and
ask whether to broaden filters or try different wording.

If the server returns `502`, the API is up but Weaviate or embeddings failed.
Suggest running `uv run scribebase doctor` on the Mac mini.
