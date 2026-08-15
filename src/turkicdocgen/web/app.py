"""
TurkicDocGen Web Panel application.
FastAPI dashboard, generation control, gallery, and sample detail views.
"""

from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from turkicdocgen import __version__

from .routers import api, pages

_HERE = Path(__file__).parent

app = FastAPI(
    title="TurkicDocGen Web Panel",
    description="Local dev panel for dataset generation, QA, and gallery review.",
    version=__version__,
)

logger = logging.getLogger("uvicorn.error")
_WEB_TOKEN: str | None = None


def get_web_token() -> str:
    global _WEB_TOKEN
    if _WEB_TOKEN is None:
        _WEB_TOKEN = os.environ.get("TURKICDOCGEN_WEB_TOKEN") or secrets.token_hex(16)
        if "TURKICDOCGEN_WEB_TOKEN" not in os.environ:
            logger.warning("=" * 80)
            logger.warning("WARNING: TURKICDOCGEN_WEB_TOKEN is not set in environment.")
            logger.warning(
                "Generated an ephemeral token without logging it. Set "
                "TURKICDOCGEN_WEB_TOKEN before allowing remote access."
            )
            logger.warning("=" * 80)
    return _WEB_TOKEN


@app.middleware("http")
async def check_auth_middleware(request: Request, call_next):
    client = request.client
    is_forwarded = (
        "x-forwarded-for" in request.headers or "x-real-ip" in request.headers
    )
    is_loopback = False
    if client and not is_forwarded:
        is_loopback = client.host in ("127.0.0.1", "localhost", "::1", "testclient")
    else:
        is_loopback = False if is_forwarded else True
    if not is_loopback:
        auth_header = request.headers.get("Authorization")
        req_token = None
        if auth_header and auth_header.startswith("Bearer "):
            req_token = auth_header.split(" ", 1)[1]
        if req_token is None or not secrets.compare_digest(req_token, get_web_token()):
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Unauthorized: Token verification failed for non-localhost request."
                },
            )
    return await call_next(request)


@app.middleware("http")
async def set_default_lang_middleware(request: Request, call_next):
    lang_qp = request.query_params.get("lang")
    response = await call_next(request)
    is_streaming = response.__class__.__name__ in (
        "StreamingResponse",
        "FileResponse",
    ) or hasattr(response, "streaming_content")
    if not is_streaming:
        try:
            if lang_qp in ("ru", "en"):
                response.set_cookie("lang", lang_qp, max_age=31536000, path="/")
            elif "lang" not in request.cookies:
                response.set_cookie("lang", "ru", max_age=31536000, path="/")
        except (RuntimeError, ValueError):
            pass
    return response


# Mount static assets
_static_dir = _HERE / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# Include routers
app.include_router(pages.router)
app.include_router(api.router)
