---
name: scribebase-ingest
description: Upload a local document to a remote ScribeBase server and poll ingestion until it finishes.
---

# ScribeBase ingest

Use this skill when the user asks to add, ingest, upload, index, or store a PDF
or document in the remote ScribeBase knowledge base.

## Requirements

The current shell must have:

```bash
SCRIBEBASE_URL=http://macmini.local:8765
SCRIBEBASE_API_TOKEN=...
```

If either variable is missing, ask the user for the missing value before trying
to upload.

## Inputs to collect

Required:

- local file path
- title

Optional:

- `source_type`: `book`, `paper`, `article`, `notes`, or `other`
- `course`
- `chapter`
- `language`: `en`, `sv`, `mixed`, or `unknown`
- `ocr`: `auto`, `always`, `never`, `shell`, or `apple_vision`
- `no_index`: set only when the user wants extraction without Weaviate indexing

Use these safe defaults unless the user says otherwise:

- `source_type=${SCRIBEBASE_DEFAULT_SOURCE_TYPE:-paper}`
- `language=${SCRIBEBASE_DEFAULT_LANGUAGE:-en}`
- `ocr=auto`
- `no_index=false`

## Procedure

1. Verify the file exists locally.
2. Check server health.
3. Upload the file with `POST /ingest`.
4. Extract `job_id` from the response.
5. Poll `GET /jobs/{job_id}` until status is `succeeded` or `failed`.
6. Report `source_id` on success. Report the job `error` on failure.

## Commands

Health:

```bash
curl -s "$SCRIBEBASE_URL/health"
```

Upload:

```bash
curl -s "$SCRIBEBASE_URL/ingest" \
  -H "Authorization: Bearer $SCRIBEBASE_API_TOKEN" \
  -F "file=@/path/to/document.pdf" \
  -F "title=Document Title" \
  -F "source_type=paper" \
  -F "language=en" \
  -F "ocr=auto"
```

Poll:

```bash
curl -s "$SCRIBEBASE_URL/jobs/JOB_ID" \
  -H "Authorization: Bearer $SCRIBEBASE_API_TOKEN"
```

## Response handling

A queued job looks like:

```json
{
  "job_id": "...",
  "status": "queued",
  "title": "Document Title",
  "source_id": null,
  "error": null
}
```

Terminal statuses:

- `succeeded`: ingestion completed. Use `source_id` in the response.
- `failed`: ingestion failed. Show the `error` field and suggest checking the Mac mini logs.

Do not claim the document is searchable until the job status is `succeeded`.
