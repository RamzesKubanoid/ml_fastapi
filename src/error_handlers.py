"""
error_handlers.py — common error response format and global exception handlers.

All handlers are registered via register_error_handlers(app) so main.py stays
clean. Every error the service returns — regardless of origin — is serialised
into the same ErrorResponseChurn shape, making client error handling trivial.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.utils.log_control import get_logger

log = get_logger(__name__)


# ── Common error schema ──────────────────────────────────────────────────────

class ErrorResponseChurn(BaseModel):
    """
    Uniform error envelope returned by every error handler in the service.

    Fields:
        code    — short machine-readable identifier (e.g. "validation_error").
        message — human-readable summary of what went wrong.
        details — optional structured context (field path, received value, …).
    """
    code:    str
    message: str
    details: list[dict] | dict | None = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _json(status: int, code: str, message: str,
          details: list[dict] | dict | None = None) -> JSONResponse:
    """Serialises an ErrorResponseChurn to a JSONResponse."""
    body = ErrorResponseChurn(code=code, message=message, details=details)
    return JSONResponse(status_code=status, content=body.model_dump())


# ── Handlers ─────────────────────────────────────────────────────────────────

async def _handle_http_exception(
    _request: Request, exc: HTTPException
) -> JSONResponse:
    """
    Catches all HTTPException instances raised explicitly in endpoint code
    (e.g. 404 dataset not found, 409 model not trained) and wraps them in the
    common error envelope.

    The 'code' is derived from the status code so clients can switch on it
    without parsing the message string.
    """
    code_map = {
        400: "bad_request",
        404: "not_found",
        409: "model_not_ready",
        422: "unprocessable_entity",
        500: "internal_server_error",
    }
    code = code_map.get(exc.status_code, f"http_{exc.status_code}")
    log.warning("HTTP %d [%s]: %s", exc.status_code, code, exc.detail)
    return _json(exc.status_code, code, str(exc.detail))


async def _handle_validation_error(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Catches Pydantic / FastAPI validation failures (wrong types, missing
    fields, failed constraints) and maps every error to a structured detail
    entry containing the field path and what was received.

    Covers:
      - Missing required field       → "field_required"
      - Wrong value type             → "type_error"
      - Constraint violation (ge, …) → "value_error"
      - Extra / unknown fields       → "extra_field"
    """
    details = [
        {
            "field": " → ".join(str(p) for p in err["loc"]),
            "issue": err["type"],
            "message": err["msg"],
        }
        for err in exc.errors()
    ]
    return _json(
        422,
        "validation_error",
        f"{len(details)} validation error(s) in the request body.",
        details,
    )


async def _handle_value_error(
    _request: Request, exc: ValueError
) -> JSONResponse:
    """
    Catches ValueError raised inside service logic (e.g. empty dataset after
    dropping missing-target rows, unsupported model_type in model_factory).
    Returns 422 because the request was structurally valid but the data is
    semantically unusable.
    """
    return _json(422, "data_error", str(exc))


async def _handle_unhandled_exception(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    Catch-all for any exception that was not handled by a more specific
    handler. Returns a 500 with a generic message — the technical traceback
    is intentionally withheld from the client to avoid leaking internals.
    The full exception is printed server-side for debugging.
    """
    log.error("Unhandled %s on %s %s: %s",
              type(exc).__name__, request.method, request.url, exc)
    return _json(
        500,
        "internal_server_error",
        "An unexpected error occurred. Please try again later.",
    )


# ── Registration ─────────────────────────────────────────────────────────────

def register_error_handlers(app: FastAPI) -> None:
    """
    Attaches all global error handlers to the FastAPI application.

    Handler priority (FastAPI resolves from most to least specific):
        RequestValidationError → _handle_validation_error (422)
        HTTPException → _handle_http_exception (4xx)
        ValueError → _handle_value_error (422)
        Exception → _handle_unhandled_exception (500)
    """
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(HTTPException, _handle_http_exception)
    app.add_exception_handler(ValueError, _handle_value_error)
    app.add_exception_handler(Exception, _handle_unhandled_exception)
