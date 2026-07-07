from __future__ import annotations

import importlib.util
import os
import shlex
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
from scribebase.llm.openai_compatible import OpenAICompatibleChatClient, save_markdown
from scribebase.llm.prompts import ANSWER_SYSTEM_PROMPT, QUIZ_SYSTEM_PROMPT
from scribebase.logging_utils import setup_logging
from scribebase.models import Chunk, SearchFilters, SearchResult
from scribebase.paths import ensure_data_layout
from scribebase.paths import chapter_file_name
from scribebase.retrieval.context_pack import build_context_pack, save_context_pack
from scribebase.retrieval.search import format_search_results, search_chunks
from scribebase.source_registry import find_source, list_manifests

app = typer.Typer(help="Local OCR → Markdown → Weaviate RAG app.")
sources_app = typer.Typer(help="List and inspect sources.")
chunks_app = typer.Typer(help="Inspect chunks.")
app.add_typer(sources_app, name="sources")
app.add_typer(chunks_app, name="chunks")


def _fail(exc: Exception) -> None:
    message = str(exc).strip() or exc.__class__.__name__
    typer.echo(f"[ERROR] {message}", err=True)
    if "Weaviate" in message or "Connection refused" in message:
        typer.echo("Start Weaviate with: docker compose -f docker-compose.weaviate.yml up -d", err=True)
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
def init(data_dir: Path = typer.Option(Path(".study_local"), help="Local data directory.")) -> None:
    ensure_data_layout(data_dir)
    path = write_default_config(data_dir)
    typer.echo(f"Created data layout under {data_dir}")
    typer.echo(f"Config: {path}")


@app.command()
def doctor() -> None:
    config = _config()
    typer.echo("ScribeBase doctor")
    for dep in ["typer", "pydantic", "yaml", "fitz", "pymupdf4llm", "httpx", "weaviate"]:
        ok = importlib.util.find_spec(dep) is not None
        typer.echo(f"[{'OK' if ok else 'MISSING'}] dependency: {dep}")

    try:
        from scribebase.vectorstores.weaviate_store import WeaviateStore

        store = WeaviateStore(config.weaviate)
        ready = store.is_ready()
        typer.echo(f"[{'OK' if ready else 'FAIL'}] Weaviate: {config.weaviate.url}")
        store.close()
    except Exception as exc:
        typer.echo(f"[FAIL] Weaviate: {exc}")
        typer.echo("Start it with: docker compose -f docker-compose.weaviate.yml up -d")

    try:
        from scribebase.embeddings.llamacpp_client import LlamaCppEmbeddingClient

        ok, msg = LlamaCppEmbeddingClient(config.embedding).check_health()
        typer.echo(f"[{'OK' if ok else 'FAIL'}] embeddings: {msg}")
    except Exception as exc:
        typer.echo(f"[FAIL] embeddings: {exc}")
        typer.echo(
            "Example: llama-server --model ./models/Qwen3-Embedding-4B-Q4_K_M.gguf "
            "--embedding --pooling last -ngl 99 --port 8080"
        )

    provider = config.ocr.providers.get(config.ocr.default_provider)
    ocr_ok, ocr_msg = _ocr_doctor_message(provider.command if provider else None)
    typer.echo(
        f"[{'OK' if ocr_ok else 'WARN'}] OCR provider: {config.ocr.default_provider}"
        + (f" ({provider.command})" if provider else "")
        + (f"; {ocr_msg}" if ocr_msg else "")
    )
    llm_key = os.getenv(config.llm.api_key_env)
    llm_ok = (not config.llm.enabled) or bool(llm_key)
    typer.echo(f"[{'OK' if llm_ok else 'WARN'}] LLM config: enabled={config.llm.enabled}")


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
    title: str = typer.Option(...),
    source_type: str = typer.Option("other"),
    course: Optional[str] = None,
    chapter: Optional[str] = None,
    language: str = "unknown",
    ocr: str = typer.Option("auto"),
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
        )
        typer.echo(f"Extracted source_id={manifest.source_id}")
    except Exception as exc:
        _fail(exc)


@app.command()
def ingest(
    path: Path,
    title: str = typer.Option(...),
    source_type: str = typer.Option("other"),
    course: Optional[str] = None,
    chapter: Optional[str] = None,
    language: str = "unknown",
    ocr: str = typer.Option("auto"),
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


@app.command()
def ask(
    question: str,
    source_id: Optional[str] = None,
    title: Optional[str] = None,
    source_type: Optional[str] = None,
    course: Optional[str] = None,
    chapter: Optional[str] = None,
    section: Optional[str] = None,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    language: Optional[str] = None,
    top_k: Optional[int] = None,
    mode: str = typer.Option("rag", help="rag, chapter, or auto"),
    allow_model_mismatch: bool = False,
) -> None:
    try:
        config = _config()
        filters = SearchFilters(
            source_id=source_id,
            title=title,
            source_type=source_type,
            course=course,
            chapter=chapter,
            section=section,
            page_start=page_start,
            page_end=page_end,
            language=language,
        )
        results = _context_results(question, filters, config, top_k, mode, allow_model_mismatch)
        pack = build_context_pack(question, results, "answer")
        client = OpenAICompatibleChatClient(config.llm)
        if not client.available():
            path = save_context_pack(config.data_dir / "outputs" / "context_packs", question, pack)
            typer.echo(f"LLM disabled or API key missing. Context pack saved: {path}")
            return
        answer = client.complete(ANSWER_SYSTEM_PROMPT, pack)
        path = save_markdown(config.data_dir / "outputs" / "answers", question, answer)
        typer.echo(answer)
        typer.echo(f"Answer saved: {path}")
    except Exception as exc:
        _fail(exc)


@app.command()
def quiz(
    title: Optional[str] = None,
    source_id: Optional[str] = None,
    chapter: Optional[str] = None,
    questions: int = 20,
    types: str = typer.Option("mcq,short-answer,flashcard"),
    top_k: Optional[int] = None,
    allow_model_mismatch: bool = False,
) -> None:
    try:
        config = _config()
        prompt = f"Create {questions} quiz questions. Types: {types}."
        filters = SearchFilters(source_id=source_id, title=title, chapter=chapter)
        results = _context_results(prompt, filters, config, top_k, "auto", allow_model_mismatch)
        pack = build_context_pack(prompt, results, "quiz")
        client = OpenAICompatibleChatClient(config.llm)
        if not client.available():
            path = save_context_pack(config.data_dir / "outputs" / "quizzes", prompt, pack)
            typer.echo(f"LLM disabled or API key missing. Quiz prompt saved: {path}")
            return
        quiz_md = client.complete(QUIZ_SYSTEM_PROMPT, pack)
        path = save_markdown(config.data_dir / "outputs" / "quizzes", title or source_id or "quiz", quiz_md)
        typer.echo(f"Quiz saved: {path}")
    except Exception as exc:
        _fail(exc)


@sources_app.command("list")
def sources_list() -> None:
    config = _config()
    for manifest in list_manifests(config.data_dir):
        typer.echo(f"{manifest.source_id}\t{manifest.title}\t{manifest.source_type}\t{manifest.chapter or ''}")


@sources_app.command("show")
def sources_show(source_id: str) -> None:
    config = _config()
    typer.echo(find_source(config.data_dir, source_id).model_dump_json(indent=2))


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


def _context_results(
    query: str,
    filters: SearchFilters,
    config: AppConfig,
    top_k: int | None,
    mode: str,
    allow_model_mismatch: bool,
) -> list[SearchResult]:
    if mode not in {"rag", "chapter", "auto"}:
        raise typer.BadParameter("mode must be rag, chapter, or auto")
    if mode in {"chapter", "auto"}:
        local = _local_filtered_chunks(config, filters)
        if local and (mode == "chapter" or len("\n".join(c.text for c in local)) <= 120_000):
            return [SearchResult(chunk=chunk) for chunk in local[: top_k or config.retrieval.top_k]]
    return search_chunks(query, filters, config, top_k, None, allow_model_mismatch)


def _local_filtered_chunks(config: AppConfig, filters: SearchFilters) -> list[Chunk]:
    chunks = load_chunks(config.data_dir, filters.source_id)
    out: list[Chunk] = []
    for chunk in chunks:
        if filters.title and chunk.title != filters.title:
            continue
        if filters.chapter and chunk.chapter != filters.chapter:
            continue
        if filters.source_type and chunk.source_type != filters.source_type:
            continue
        if filters.course and chunk.course != filters.course:
            continue
        if filters.language and chunk.language != filters.language:
            continue
        out.append(chunk)
    if out:
        return out
    return _markdown_context_chunks(config, filters)


def _markdown_context_chunks(config: AppConfig, filters: SearchFilters) -> list[Chunk]:
    out: list[Chunk] = []
    for manifest in list_manifests(config.data_dir):
        if filters.source_id and manifest.source_id != filters.source_id:
            continue
        if filters.title and manifest.title != filters.title:
            continue
        if filters.chapter and manifest.chapter != filters.chapter:
            continue
        if filters.source_type and manifest.source_type != filters.source_type:
            continue
        if filters.course and manifest.course != filters.course:
            continue
        if filters.language and manifest.language != filters.language:
            continue
        root = Path(manifest.data_dir)
        md_path = root / "markdown" / "document.md"
        if filters.chapter or manifest.chapter:
            chapter = filters.chapter or manifest.chapter
            chapter_path = root / "markdown" / "chapters" / chapter_file_name(chapter or "")
            if chapter_path.exists():
                md_path = chapter_path
        if not md_path.exists():
            continue
        text = md_path.read_text().strip()
        if not text:
            continue
        out.append(
            Chunk(
                chunk_id=f"{manifest.source_id}_markdown_context",
                source_id=manifest.source_id,
                source_type=manifest.source_type,
                title=manifest.title,
                course=manifest.course,
                chapter=filters.chapter or manifest.chapter,
                section=None,
                page_start=None,
                page_end=None,
                chunk_index=0,
                text=text,
                file_path=str(md_path),
                extraction_method="mixed",
                language=manifest.language,
            )
        )
    return out


def _ocr_doctor_message(command: str | None) -> tuple[bool, str]:
    if not command:
        return False, "no command configured"
    try:
        parts = shlex.split(command.format(input_image="x", output_md="y", output_json="z", page_number=1, source_id="s"))
    except Exception as exc:
        return False, f"invalid command template: {exc}"
    for part in parts[1:]:
        if part.endswith(".py") or part.startswith("./"):
            path = Path(part)
            if not path.exists():
                return False, f"missing adapter path: {part}"
            break
    return True, "configured"


if __name__ == "__main__":
    app()
