from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

import typer

from scribebase.config import (
    AppConfig,
    load_config,
    read_api_token,
    resolve_config_path,
    resolve_data_dir,
    write_default_config,
)
from scribebase.extraction import extract_source
from scribebase.indexing import index_source, load_chunks, rebuild_index
from scribebase.logging_utils import setup_logging
from scribebase.models import SearchFilters
from scribebase.ocr.health import check_ocr_provider_health
from scribebase.paths import ensure_data_layout
from scribebase.retrieval.search import format_search_results, search_chunks
from scribebase.source_registry import backfill_source_identities, find_source, list_manifests
from scribebase.server_jobs import run_worker

app = typer.Typer(help="Local OCR, indexing, and cited retrieval knowledge node.")
sources_app = typer.Typer(help="List and inspect sources.")
chunks_app = typer.Typer(help="Inspect chunks.")
app.add_typer(sources_app, name="sources")
app.add_typer(chunks_app, name="chunks")


def _fail(exc: Exception) -> None:
    message = str(exc).strip() or exc.__class__.__name__
    typer.echo(f"[ERROR] {message}", err=True)
    if "Weaviate" in message or "Connection refused" in message:
        typer.echo(
            "Start Weaviate with: docker compose -f docker-compose.weaviate.yml up -d", err=True
        )
    if "embedding" in message.lower() or "localhost:8080" in message:
        typer.echo(
            "Start llama.cpp embeddings, e.g.: "
            "llama-server --model ./models/Qwen3-Embedding-4B-Q4_K_M.gguf "
            "--embedding --pooling last -ngl 99 --port 8080",
            err=True,
        )
    raise typer.Exit(code=1)


def _config() -> AppConfig:
    config_path = resolve_config_path()
    if not config_path.exists():
        write_default_config(resolve_data_dir(), config_path=config_path)
    config = load_config(config_path)
    ensure_data_layout(config.data_dir)
    return config


def _logger(config: AppConfig):
    return setup_logging(config.data_dir)


@app.command()
def init(data_dir: Path = typer.Option(Path(".scribebase"), help="Local data directory.")) -> None:
    ensure_data_layout(data_dir)
    path = write_default_config(data_dir)
    typer.echo(f"Created data layout under {data_dir}")
    typer.echo(f"Config: {path}")


@app.command()
def doctor() -> None:
    config = _config()
    healthy = True
    typer.echo("ScribeBase doctor")
    for dep in ["typer", "pydantic", "yaml", "fitz", "pymupdf4llm", "httpx", "weaviate"]:
        ok = importlib.util.find_spec(dep) is not None
        healthy = healthy and ok
        typer.echo(f"[{'OK' if ok else 'MISSING'}] dependency: {dep}")

    try:
        from scribebase.vectorstores.weaviate_store import WeaviateStore

        store = WeaviateStore(config.weaviate)
        ready = store.is_ready()
        healthy = healthy and ready
        typer.echo(f"[{'OK' if ready else 'FAIL'}] Weaviate: {config.weaviate.url}")
        store.close()
    except Exception as exc:
        healthy = False
        typer.echo(f"[FAIL] Weaviate: {exc}")
        typer.echo("Start it with: docker compose -f docker-compose.weaviate.yml up -d")

    try:
        from scribebase.embeddings.llamacpp_client import LlamaCppEmbeddingClient

        ok, msg = LlamaCppEmbeddingClient(config.embedding).check_health()
        healthy = healthy and ok
        typer.echo(f"[{'OK' if ok else 'FAIL'}] embeddings: {msg}")
    except Exception as exc:
        healthy = False
        typer.echo(f"[FAIL] embeddings: {exc}")
        typer.echo(
            "Example: llama-server --model ./models/Qwen3-Embedding-4B-Q4_K_M.gguf "
            "--embedding --pooling last -ngl 99 --port 8080"
        )

    provider = config.ocr.providers.get(config.ocr.default_provider)
    ocr_ok, ocr_msg = check_ocr_provider_health(config.ocr.default_provider, provider)
    healthy = healthy and ocr_ok
    typer.echo(
        f"[{'OK' if ocr_ok else 'FAIL'}] OCR provider: {config.ocr.default_provider}"
        + (f" ({provider.command})" if provider else "")
        + (f"; {ocr_msg}" if ocr_msg else "")
    )
    if not healthy:
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, help="Bind host. Defaults to config.server.host."),
    port: Optional[int] = typer.Option(None, help="Bind port. Defaults to config.server.port."),
) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        typer.echo("Install server dependencies with: uv sync --extra server", err=True)
        raise typer.Exit(code=1) from exc

    config = _config()
    api_token = read_api_token(config)
    if not api_token:
        typer.echo(f"Set {config.server.api_token_env} before starting the server.", err=True)
        raise typer.Exit(code=1)

    from scribebase.server import create_app

    uvicorn.run(
        create_app(config, api_token),
        host=host or config.server.host,
        port=port or config.server.port,
    )


@app.command()
def extract(
    path: Path,
    title: Optional[str] = typer.Option(None),
    source_type: Optional[str] = None,
    course: Optional[str] = None,
    chapter: Optional[str] = None,
    language: Optional[str] = None,
    tags: Optional[str] = typer.Option(None, help="Comma-separated tags."),
    origin: Optional[str] = None,
    publisher: Optional[str] = None,
    author: Optional[str] = None,
    created_at_source: Optional[str] = None,
    updated_at_source: Optional[str] = None,
    retrieved_at: Optional[str] = None,
    url: Optional[str] = None,
    canonical_url: Optional[str] = None,
    external_id: Optional[str] = None,
    collection: Optional[str] = None,
    summary: Optional[str] = None,
    ocr: str = typer.Option("auto"),
    duplicate_policy: str = typer.Option("reject", help="reject or create"),
    continue_on_ocr_error: bool = False,
) -> None:
    try:
        config = _config()
        manifest = extract_source(
            path,
            title,
            source_type,
            course,
            chapter,
            language,
            ocr,
            config,
            _logger(config),
            continue_on_ocr_error,
            tags=tags,
            origin=origin,
            publisher=publisher,
            author=author,
            created_at_source=created_at_source,
            updated_at_source=updated_at_source,
            retrieved_at=retrieved_at,
            url=url,
            canonical_url=canonical_url,
            external_id=external_id,
            collection=collection,
            summary=summary,
            duplicate_policy=duplicate_policy,
        )
        typer.echo(f"Extracted source_id={manifest.source_id}")
    except Exception as exc:
        _fail(exc)


@app.command()
def worker(
    once: bool = typer.Option(False, help="Process at most one queued job."),
    poll_seconds: Optional[float] = typer.Option(None, min=0.1, help="Queue polling interval."),
) -> None:
    """Run the durable ingestion queue worker."""
    try:
        config = _config()
        run_worker(config, once=once, poll_seconds=poll_seconds)
    except Exception as exc:
        _fail(exc)


@app.command()
def ingest(
    path: Path,
    title: Optional[str] = typer.Option(None),
    source_type: Optional[str] = None,
    course: Optional[str] = None,
    chapter: Optional[str] = None,
    language: Optional[str] = None,
    tags: Optional[str] = typer.Option(None, help="Comma-separated tags."),
    origin: Optional[str] = None,
    publisher: Optional[str] = None,
    author: Optional[str] = None,
    created_at_source: Optional[str] = None,
    updated_at_source: Optional[str] = None,
    retrieved_at: Optional[str] = None,
    url: Optional[str] = None,
    canonical_url: Optional[str] = None,
    external_id: Optional[str] = None,
    collection: Optional[str] = None,
    summary: Optional[str] = None,
    ocr: str = typer.Option("auto"),
    duplicate_policy: str = typer.Option("reject", help="reject or create"),
    no_index: bool = typer.Option(False, help="Extract only; do not index into Weaviate."),
    continue_on_ocr_error: bool = False,
) -> None:
    try:
        config = _config()
        logger = _logger(config)
        manifest = extract_source(
            path,
            title,
            source_type,
            course,
            chapter,
            language,
            ocr,
            config,
            logger,
            continue_on_ocr_error,
            tags=tags,
            origin=origin,
            publisher=publisher,
            author=author,
            created_at_source=created_at_source,
            updated_at_source=updated_at_source,
            retrieved_at=retrieved_at,
            url=url,
            canonical_url=canonical_url,
            external_id=external_id,
            collection=collection,
            summary=summary,
            duplicate_policy=duplicate_policy,
        )
        if not no_index:
            index_source(manifest.source_id, config, logger)
        typer.echo(f"Ingested source_id={manifest.source_id}")
    except Exception as exc:
        _fail(exc)


@app.command(name="index")
def index_cmd(source_id: str = typer.Option(..., "--source-id")) -> None:
    try:
        config = _config()
        index_source(source_id, config, _logger(config))
    except Exception as exc:
        _fail(exc)


@app.command(name="rebuild-index")
def rebuild_index_cmd(
    source_id: Optional[str] = typer.Option(None, "--source-id"),
    all_sources: bool = typer.Option(False, "--all"),
) -> None:
    try:
        config = _config()
        rebuild_index(source_id, all_sources, config, _logger(config))
    except Exception as exc:
        _fail(exc)


@app.command()
def search(
    query: str,
    source_id: Optional[str] = None,
    title: Optional[str] = None,
    source_type: Optional[str] = None,
    course: Optional[str] = None,
    chapter: Optional[str] = None,
    section: Optional[str] = None,
    tags: Optional[str] = typer.Option(None, help="Comma-separated tags."),
    origin: Optional[str] = None,
    publisher: Optional[str] = None,
    author: Optional[str] = None,
    url: Optional[str] = None,
    canonical_url: Optional[str] = None,
    external_id: Optional[str] = None,
    collection: Optional[str] = None,
    created_at_source_after: Optional[str] = None,
    created_at_source_before: Optional[str] = None,
    updated_at_source_after: Optional[str] = None,
    updated_at_source_before: Optional[str] = None,
    retrieved_at_after: Optional[str] = None,
    retrieved_at_before: Optional[str] = None,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    language: Optional[str] = None,
    top_k: Optional[int] = None,
    alpha: Optional[float] = None,
    allow_model_mismatch: bool = False,
) -> None:
    try:
        config = _config()
        results = search_chunks(
            query,
            SearchFilters(
                source_id=source_id,
                title=title,
                source_type=source_type,
                course=course,
                chapter=chapter,
                section=section,
                tags=tags,
                origin=origin,
                publisher=publisher,
                author=author,
                url=url,
                canonical_url=canonical_url,
                external_id=external_id,
                collection=collection,
                created_at_source_after=created_at_source_after,
                created_at_source_before=created_at_source_before,
                updated_at_source_after=updated_at_source_after,
                updated_at_source_before=updated_at_source_before,
                retrieved_at_after=retrieved_at_after,
                retrieved_at_before=retrieved_at_before,
                page_start=page_start,
                page_end=page_end,
                language=language,
            ),
            config,
            top_k,
            alpha,
            allow_model_mismatch,
        )
        typer.echo(format_search_results(results))
    except Exception as exc:
        _fail(exc)


@sources_app.command("list")
def sources_list() -> None:
    config = _config()
    for manifest in list_manifests(config.data_dir):
        typer.echo(
            f"{manifest.source_id}\t{manifest.title}\t{manifest.source_type}\t{manifest.chapter or ''}"
        )


@sources_app.command("show")
def sources_show(source_id: str) -> None:
    config = _config()
    typer.echo(find_source(config.data_dir, source_id).model_dump_json(indent=2))


@sources_app.command("backfill-identities")
def sources_backfill_identities() -> None:
    config = _config()
    count = backfill_source_identities(config.data_dir)
    typer.echo(f"Backfilled identity metadata for {count} sources")


@chunks_app.command("list")
def chunks_list(source_id: Optional[str] = typer.Option(None, "--source-id")) -> None:
    config = _config()
    for chunk in load_chunks(config.data_dir, source_id):
        typer.echo(f"{chunk.chunk_id}\t{chunk.title}\tp{chunk.page_start}-{chunk.page_end}")


@chunks_app.command("show")
def chunks_show(chunk_id: str) -> None:
    config = _config()
    for chunk in load_chunks(config.data_dir):
        if chunk.chunk_id == chunk_id:
            typer.echo(chunk.model_dump_json(indent=2))
            return
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
