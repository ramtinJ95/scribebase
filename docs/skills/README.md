# Remote ScribeBase skill templates

These templates are starting points for manually invoked agent skills that talk
to a ScribeBase server running on the Mac mini.

Templates:

- `scribebase-ingest`: upload a local PDF or document and poll the ingestion job.
- `scribebase-context`: retrieve cited search results or a context pack for the current task.

Each template exists in two install layouts:

- `docs/skills/agents/`: copy into `~/.agents/skills/`.
- `docs/skills/claude/`: copy into `~/.claude/skills/`.

The skill text is intentionally explicit that the skill is manual-invocation
only. Do not configure these as automatically triggered skills.

## Install for `.agents`

```bash
mkdir -p ~/.agents/skills
cp -R docs/skills/agents/scribebase-ingest ~/.agents/skills/
cp -R docs/skills/agents/scribebase-context ~/.agents/skills/
```

## Install for Claude

```bash
mkdir -p ~/.claude/skills
cp -R docs/skills/claude/scribebase-ingest ~/.claude/skills/
cp -R docs/skills/claude/scribebase-context ~/.claude/skills/
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
