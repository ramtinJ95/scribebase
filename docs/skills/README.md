# Remote ScribeBase skill templates

These templates are starting points for agent skills that talk to a ScribeBase
server running on the Mac mini.

Templates:

- `scribebase-ingest`: upload a local PDF or document and poll the ingestion job.
- `scribebase-context`: retrieve cited search results or a context pack for the current task.

## Install

Copy one or both template directories into the skill directory used by your
agent harness, then edit only the examples and defaults you want to specialize.

Example:

```bash
mkdir -p ~/.agents/skills
cp -R docs/skills/scribebase-ingest ~/.agents/skills/
cp -R docs/skills/scribebase-context ~/.agents/skills/
```

## Required environment

Each session that uses these skills needs:

```bash
export SCRIBEBASE_URL=http://macmini.local:8765
export SCRIBEBASE_API_TOKEN=replace-with-the-server-token
```

Optional defaults:

```bash
export SCRIBEBASE_DEFAULT_SOURCE_TYPE=paper
export SCRIBEBASE_DEFAULT_LANGUAGE=en
```

## Smoke test

```bash
curl -s "$SCRIBEBASE_URL/health"

curl -s "$SCRIBEBASE_URL/sources" \
  -H "Authorization: Bearer $SCRIBEBASE_API_TOKEN"
```

If `macmini.local` does not resolve, use the Mac mini IP address.
