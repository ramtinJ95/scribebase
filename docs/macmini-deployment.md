# Mac mini deployment

This guide turns a Mac mini into the ScribeBase node for remote PDF ingestion,
search, and context retrieval.

## Target shape

```text
Other agent sessions
  -> HTTP API on the Mac mini
  -> ScribeBase local data dir
  -> local Weaviate
  -> local llama.cpp embeddings
  -> optional local OCR runtime
```

Canonical files stay on the Mac mini under `SCRIBEBASE_DATA_DIR`. Weaviate is a
rebuildable index, not the source of truth.

## Prerequisites

Install these on the Mac mini:

- Python 3.11+
- `uv`
- Docker Desktop or another Docker runtime
- `llama-server` from llama.cpp
- the embedding model GGUF file
- optional OCR runtime:
  - Apple Vision needs no daemon
  - GLM-OCR or another vision model needs its own local server/adapter

The examples below assume:

```bash
REPO=/Users/ramtin/personal/scribebase
DATA=/Users/ramtin/scribebase-data
HOST=0.0.0.0
PORT=8765
```

Change paths to match the Mac mini.

## 1. Clone and install

```bash
git clone git@github.com:ramtinJ95/scribebase.git "$REPO"
cd "$REPO"
uv sync --extra server --extra dev
```

Initialize the data directory:

```bash
SCRIBEBASE_DATA_DIR="$DATA" uv run scribebase init --data-dir "$DATA"
```

## 2. Configure environment

Create `$REPO/.env`:

```bash
SCRIBEBASE_DATA_DIR=/Users/ramtin/scribebase-data
SCRIBEBASE_HOST=0.0.0.0
SCRIBEBASE_PORT=8765
SCRIBEBASE_API_TOKEN=replace-with-a-long-random-token
```

Generate a token with:

```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Do not commit `.env`.

## 3. Start Weaviate

```bash
cd "$REPO"
docker compose -f docker-compose.weaviate.yml up -d
```

Verify:

```bash
docker compose -f docker-compose.weaviate.yml ps
curl -s http://127.0.0.1:8081/v1/.well-known/ready
```

Docker Desktop can be set to start on login. Keep the compose project in the
repo so the named `weaviate_data` volume is reused across restarts.

## 4. Start embeddings

Default ScribeBase config expects llama.cpp embeddings at
`http://localhost:8080/v1` with model name `Qwen3-Embedding-4B-Q4_K_M.gguf`.

Example manual start:

```bash
llama-server \
  --model "$REPO/models/Qwen3-Embedding-4B-Q4_K_M.gguf" \
  --embedding \
  --pooling last \
  --ctx-size 32768 \
  -ngl 99 \
  --port 8080
```

Check:

```bash
curl -s http://127.0.0.1:8080/v1/models
```

If you change the embedding model or dimension, run:

```bash
cd "$REPO"
uv run scribebase rebuild-index --all
```

## 5. Configure OCR

For true-text PDFs, no OCR server is needed.

Fast macOS OCR:

```bash
uv run scribebase ingest ./scan.pdf \
  --title "Scan" \
  --source-type book \
  --ocr apple_vision
```

GLM-OCR style local vision server:

```bash
llama-server \
  -m "$REPO/models/ocr/GLM-OCR-Q8_0.gguf" \
  --mmproj "$REPO/models/ocr/mmproj-GLM-OCR-Q8_0.gguf" \
  -ngl 0 \
  --port 8082
```

The default `shell` OCR provider calls:

```bash
./scripts/run_local_ocr.py --input {input_image} --output {output_md}
```

That adapter defaults to `http://localhost:8082/v1`.

## 6. Start the ScribeBase API

Manual start:

```bash
cd "$REPO"
uv run scribebase serve
```

Health check from the Mac mini:

```bash
curl -s http://127.0.0.1:8765/health
```

Health is intentionally unauthenticated. Source, search, context, ingest, and
job endpoints require:

```bash
Authorization: Bearer $SCRIBEBASE_API_TOKEN
```

## 7. Optional launchd autostart

Template plists live in `docs/launchd/`:

- `com.scribebase.server.plist.example`
- `com.scribebase.worker.plist.example`
- `com.scribebase.embedding.plist.example`

Copy each template to `~/Library/LaunchAgents/`, remove `.example`, and edit:

- repository path
- data path
- token
- model path
- log paths

Then load them:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.scribebase.embedding.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.scribebase.server.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.scribebase.worker.plist
```

Restart after edits:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.scribebase.server.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.scribebase.server.plist
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.scribebase.worker.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.scribebase.worker.plist
```

Inspect logs:

```bash
tail -f /Users/ramtin/scribebase-data/logs/scribebase-server.out.log
tail -f /Users/ramtin/scribebase-data/logs/scribebase-server.err.log
tail -f /Users/ramtin/scribebase-data/logs/scribebase-worker.err.log
```

## 8. Remote smoke tests

From another machine on the same network:

```bash
export SCRIBEBASE_URL=http://macmini.local:8765
export SCRIBEBASE_API_TOKEN=replace-with-the-same-token

curl -s "$SCRIBEBASE_URL/health"

curl -s "$SCRIBEBASE_URL/sources" \
  -H "Authorization: Bearer $SCRIBEBASE_API_TOKEN"
```

Upload a PDF:

```bash
curl -s "$SCRIBEBASE_URL/ingest" \
  -H "Authorization: Bearer $SCRIBEBASE_API_TOKEN" \
  -F "file=@./paper.pdf" \
  -F "title=Paper Title" \
  -F "source_type=paper" \
  -F "language=en"
```

Poll the returned job:

```bash
curl -s "$SCRIBEBASE_URL/jobs/JOB_ID" \
  -H "Authorization: Bearer $SCRIBEBASE_API_TOKEN"
```

Search:

```bash
curl -s "$SCRIBEBASE_URL/search" \
  -H "Authorization: Bearer $SCRIBEBASE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"what does this source say about the topic?","top_k":5}'
```

## 9. Agent skill templates

Skill templates for other sessions live in `docs/skills/`:

- `scribebase-ingest`: upload a file to this Mac mini and poll the job.
- `scribebase-context`: retrieve cited context packs or search snippets.

Both `.agents` and Claude-compatible layouts are included:

- `docs/skills/agents/` -> `~/.agents/skills/`
- `docs/skills/claude/` -> `~/.claude/skills/`

The templates are manual-invocation only. They should run only when the user
explicitly asks for ScribeBase ingestion or retrieval.

Set these variables in sessions that use them:

```bash
export SCRIBEBASE_URL=http://macmini.local:8765
export SCRIBEBASE_API_TOKEN=replace-with-the-same-token
```

## Operations checklist

Run this after reboot or before debugging a remote client:

```bash
cd "$REPO"
docker compose -f docker-compose.weaviate.yml ps
curl -s http://127.0.0.1:8080/v1/models
uv run scribebase doctor
curl -s http://127.0.0.1:8765/health
```

Useful recovery commands:

```bash
# Weaviate is down
docker compose -f docker-compose.weaviate.yml up -d

# Embedding model changed
uv run scribebase rebuild-index --all

# Inspect ScribeBase data
uv run scribebase sources list
uv run scribebase sources show SOURCE_ID

# Check recent server logs
tail -n 200 "$DATA/logs/scribebase-server.err.log"
```

## Troubleshooting

### `401 Invalid bearer token`

The client token does not match `SCRIBEBASE_API_TOKEN` on the Mac mini.
Restart the launchd service after changing `.env` or plist environment values.

### `502` from `/search` or `/context`

The API is running, but Weaviate or embeddings failed. Run:

```bash
uv run scribebase doctor
```

### Job stays `failed`

Inspect the job response `error` field, then check:

```bash
tail -n 200 "$DATA/logs/app.log"
```

Common causes are missing OCR runtime, unsupported file type, Weaviate down, or
embedding server down.

### Mac mini is unreachable from another machine

Check:

- the ScribeBase server is bound to `0.0.0.0`, not `127.0.0.1`
- both machines are on the same network
- macOS firewall allows incoming connections for the Python/uv process
- `macmini.local` resolves; otherwise use the Mac mini IP address
