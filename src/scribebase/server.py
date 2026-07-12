from __future__ import annotations

import secrets
from datetime import datetime
from io import BytesIO
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from scribebase.config import AppConfig, load_config, read_api_token
from scribebase.embeddings.llamacpp_client import LlamaCppEmbeddingClient
from scribebase.models import (
    GenericMetadata,
    Language,
    SearchFilters,
    SearchResult,
    SourceManifest,
    SourceType,
)
from scribebase.paths import ensure_data_layout
from scribebase.retrieval.context_pack import build_context_pack
from scribebase.retrieval.search import search_chunks
from scribebase.server_jobs import (
    IngestJobResponse,
    QueueFullError,
    UnsupportedUploadError,
    UploadTooLargeError,
    create_ingest_job,
    public_job,
    read_job,
    retry_job,
    worker_status,
)
from scribebase.source_registry import list_manifests, slugify
from scribebase.vectorstores.weaviate_store import WeaviateStore


class ServiceHealth(BaseModel):
    ok: bool
    message: str


class HealthResponse(BaseModel):
    status: str
    data_dir: str
    sources_count: int
    weaviate: ServiceHealth
    embeddings: ServiceHealth
    worker: ServiceHealth
    auth_required: bool


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int | None = Field(default=None, ge=1, le=100)
    alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    allow_model_mismatch: bool = False


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class ContextRequest(SearchRequest):
    task: str = "answer"


class ContextResponse(BaseModel):
    query: str
    task: str
    context_pack: str
    results: list[SearchResult]


class ArticleIngestRequest(GenericMetadata):
    body: str = Field(min_length=1)
    title: str | None = None
    source_type: SourceType = "article"
    course: str | None = None
    chapter: str | None = None
    language: Language | None = None
    no_index: bool = False


_bearer = HTTPBearer(auto_error=False)


def create_app(config: AppConfig | None = None, api_token: str | None = None) -> FastAPI:
    config = config or load_config()
    ensure_data_layout(config.data_dir)
    api_token = api_token if api_token is not None else read_api_token(config)

    app = FastAPI(
        title="ScribeBase API",
        version="0.1.0",
        description="API for searching and ingesting a local ScribeBase knowledge base.",
    )
    app.state.config = config
    app.state.api_token = api_token

    @app.middleware("http")
    async def reject_known_oversized_requests(request: Request, call_next):  # noqa: ANN001, ANN202
        if request.url.path in {"/ingest", "/articles"}:
            content_length = request.headers.get("content-length")
            request_limit = config.server.max_upload_bytes + 1024 * 1024
            if content_length and int(content_length) > request_limit:
                return JSONResponse(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    content={"detail": f"Request exceeds {request_limit} bytes"},
                )
        return await call_next(request)

    require_auth = _auth_dependency(api_token)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            data_dir=str(config.data_dir),
            sources_count=len(list_manifests(config.data_dir)),
            weaviate=_weaviate_health(config),
            embeddings=_embedding_health(config),
            worker=_worker_health(config),
            auth_required=bool(api_token),
        )

    @app.get("/sources", response_model=list[SourceManifest], dependencies=[Depends(require_auth)])
    def sources() -> list[SourceManifest]:
        return list_manifests(config.data_dir)

    @app.post("/search", response_model=SearchResponse, dependencies=[Depends(require_auth)])
    def search(request: SearchRequest) -> SearchResponse:
        try:
            results = search_chunks(
                request.query,
                request.filters,
                config,
                request.top_k,
                request.alpha,
                request.allow_model_mismatch,
            )
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        return SearchResponse(query=request.query, results=results)

    @app.post("/context", response_model=ContextResponse, dependencies=[Depends(require_auth)])
    def context(request: ContextRequest) -> ContextResponse:
        try:
            results = search_chunks(
                request.query,
                request.filters,
                config,
                request.top_k,
                request.alpha,
                request.allow_model_mismatch,
            )
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        context_pack = build_context_pack(request.query, results, request.task)
        return ContextResponse(
            query=request.query,
            task=request.task,
            context_pack=context_pack,
            results=results,
        )

    @app.post("/ingest", response_model=IngestJobResponse, dependencies=[Depends(require_auth)])
    def ingest(
        file: UploadFile = File(...),
        title: str | None = Form(None),
        source_type: SourceType | None = Form(None),
        course: str | None = Form(None),
        chapter: str | None = Form(None),
        language: Language | None = Form(None),
        tags: str | None = Form(None),
        origin: str | None = Form(None),
        publisher: str | None = Form(None),
        author: str | None = Form(None),
        created_at_source: datetime | None = Form(None),
        updated_at_source: datetime | None = Form(None),
        retrieved_at: datetime | None = Form(None),
        url: str | None = Form(None),
        canonical_url: str | None = Form(None),
        external_id: str | None = Form(None),
        collection: str | None = Form(None),
        summary: str | None = Form(None),
        ocr: str = Form("auto"),
        no_index: bool = Form(False),
        continue_on_ocr_error: bool = Form(False),
    ) -> IngestJobResponse:
        try:
            job = create_ingest_job(
                config,
                file.filename or "upload",
                file.file,
                title,
                source_type,
                course,
                chapter,
                language,
                ocr,
                no_index,
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
                expected_size=file.size,
            )
        except UploadTooLargeError as exc:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)
            ) from exc
        except UnsupportedUploadError as exc:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
            ) from exc
        except QueueFullError as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return public_job(job)

    @app.post("/articles", response_model=IngestJobResponse, dependencies=[Depends(require_auth)])
    def ingest_article(
        request: ArticleIngestRequest,
    ) -> IngestJobResponse:
        if not request.body.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="body must not be empty"
            )
        filename = f"{slugify(request.title or 'article')[:80]}.md"
        try:
            job = create_ingest_job(
                config,
                filename,
                BytesIO(request.body.encode("utf-8")),
                request.title,
                request.source_type,
                request.course,
                request.chapter,
                request.language,
                "auto",
                request.no_index,
                False,
                tags=request.tags if "tags" in request.model_fields_set else None,
                origin=request.origin,
                publisher=request.publisher,
                author=request.author,
                created_at_source=request.created_at_source,
                updated_at_source=request.updated_at_source,
                retrieved_at=request.retrieved_at,
                url=request.url,
                canonical_url=request.canonical_url,
                external_id=request.external_id,
                collection=request.collection,
                summary=request.summary,
                expected_size=len(request.body.encode("utf-8")),
            )
        except UploadTooLargeError as exc:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)
            ) from exc
        except QueueFullError as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return public_job(job)

    @app.get(
        "/jobs/{job_id}", response_model=IngestJobResponse, dependencies=[Depends(require_auth)]
    )
    def job_status(job_id: str) -> IngestJobResponse:
        try:
            return public_job(read_job(config.data_dir, job_id))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post(
        "/jobs/{job_id}/retry",
        response_model=IngestJobResponse,
        dependencies=[Depends(require_auth)],
    )
    def retry_failed_job(job_id: str) -> IngestJobResponse:
        try:
            return public_job(retry_job(config, job_id))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except QueueFullError as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return app


def _auth_dependency(api_token: str | None):
    async def require_auth(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    ) -> None:
        if not api_token:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Set SCRIBEBASE_API_TOKEN before starting the server.",
            )
        if credentials is None or not secrets.compare_digest(credentials.credentials, api_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return require_auth


def _weaviate_health(config: AppConfig) -> ServiceHealth:
    store = WeaviateStore(config.weaviate)
    try:
        ready = store.is_ready()
        return ServiceHealth(ok=ready, message=config.weaviate.url)
    except Exception as exc:
        return ServiceHealth(ok=False, message=str(exc))
    finally:
        store.close()


def _embedding_health(config: AppConfig) -> ServiceHealth:
    ok, message = LlamaCppEmbeddingClient(config.embedding).check_health()
    return ServiceHealth(ok=ok, message=message)


def _worker_health(config: AppConfig) -> ServiceHealth:
    ok, message = worker_status(config)
    return ServiceHealth(ok=ok, message=message)
