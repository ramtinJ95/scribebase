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
        "WeaviateBatchFailedToReestablishStreamError",
        "WeaviateGRPCUnavailableError",
        "WeaviateStartUpError",
    )
    if isinstance(error_type := getattr(weaviate_exceptions, name, None), type)
)

_OPTIONAL_CONDITIONAL_BATCH_ERRORS = tuple(
    error_type
    for name in (
        "WeaviateBatchError",
        "WeaviateBatchSendError",
        "WeaviateBatchStreamError",
    )
    if isinstance(error_type := getattr(weaviate_exceptions, name, None), type)
)

_TRANSIENT_DEPENDENCY_ERRORS = (
    httpx.TransportError,
    WeaviateConnectionError,
    WeaviateRetryError,
    WeaviateTimeoutError,
) + _OPTIONAL_TRANSIENT_WEAVIATE_ERRORS

_TRANSIENT_BATCH_MESSAGE_MARKERS = (
    "StatusCode.UNAVAILABLE",
    "StatusCode.DEADLINE_EXCEEDED",
    "Connection refused",
    "connection lost",
    "connection reset",
    "failed to connect",
    "service unavailable",
    "transport is closing",
)


class DependencyUnavailableError(RuntimeError):
    """A local service failed in a way that is safe to retry later."""


def as_dependency_unavailable(exc: Exception) -> DependencyUnavailableError | None:
    if isinstance(exc, DependencyUnavailableError):
        return exc
    if isinstance(exc, _TRANSIENT_DEPENDENCY_ERRORS):
        return DependencyUnavailableError(str(exc).strip() or exc.__class__.__name__)
    if isinstance(exc, _OPTIONAL_CONDITIONAL_BATCH_ERRORS):
        return dependency_unavailable_from_messages([str(exc)])
    if isinstance(exc, UnexpectedStatusCodeError) and (
        exc.status_code >= 500 or exc.status_code in {408, 425, 429}
    ):
        return DependencyUnavailableError(str(exc))
    return None


def dependency_unavailable_from_messages(
    messages: list[str],
) -> DependencyUnavailableError | None:
    """Recover typed retryability after the Weaviate batch API stringifies errors."""
    detail = "; ".join(message for message in messages if message)
    transient_type_names = tuple(error_type.__name__ for error_type in _TRANSIENT_DEPENDENCY_ERRORS)
    normalized_detail = detail.casefold()
    if any(
        marker.casefold() in normalized_detail
        for marker in transient_type_names + _TRANSIENT_BATCH_MESSAGE_MARKERS
    ):
        return DependencyUnavailableError(detail or "Weaviate batch dependency unavailable")
    return None
