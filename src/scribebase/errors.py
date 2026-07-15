from __future__ import annotations

import httpx
import weaviate.exceptions as weaviate_exceptions
from weaviate.exceptions import (
    UnexpectedStatusCodeError,
    WeaviateConnectionError,
    WeaviateRetryError,
    WeaviateTimeoutError,
)


_OPTIONAL_TRANSIENT_WEAVIATE_ERRORS = tuple(
    error_type
    for name in (
        "WeaviateBatchError",
        "WeaviateBatchFailedToReestablishStreamError",
        "WeaviateBatchSendError",
        "WeaviateBatchStreamError",
        "WeaviateGRPCUnavailableError",
        "WeaviateStartUpError",
    )
    if isinstance(error_type := getattr(weaviate_exceptions, name, None), type)
)


class DependencyUnavailableError(RuntimeError):
    """A local service failed in a way that is safe to retry later."""


def as_dependency_unavailable(exc: Exception) -> DependencyUnavailableError | None:
    if isinstance(exc, DependencyUnavailableError):
        return exc
    if isinstance(
        exc,
        (
            httpx.TransportError,
            WeaviateConnectionError,
            WeaviateRetryError,
            WeaviateTimeoutError,
        )
        + _OPTIONAL_TRANSIENT_WEAVIATE_ERRORS,
    ):
        return DependencyUnavailableError(str(exc).strip() or exc.__class__.__name__)
    if isinstance(exc, UnexpectedStatusCodeError) and (
        exc.status_code >= 500 or exc.status_code in {408, 425, 429}
    ):
        return DependencyUnavailableError(str(exc))
    return None
