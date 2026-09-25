# Derived from Laya (Apache-2.0); see NOTICE. Modified for laya-mlx.
"""HTTP surface: TypeSafe Jev ``/v1/systemone`` wire protocol on native MLX.

``Agent.predict`` already returns the Jev answer schema (``choice`` / ``score`` /
``noul`` plus an ``input_tokens``/``output_tokens`` usage block), so this module
adds only the HTTP layer: one route, a bearer check, and health probes.

    POST /v1/systemone   {"state": ..., "questions": {...}, "model": "multilingual"}
    GET  /health/live    process is up
    GET  /health/ready   a checkpoint is resident (503 otherwise)

``state`` accepts a string or any JSON object/array, matching upstream. Binds to
127.0.0.1 by default: this runtime is meant to run next to the caller, and an
unauthenticated GPU inference endpoint should not face a network.

Run with ``laya-mlx-http`` or ``python -m laya_mlx_server http``.
"""

import hmac
import json
import logging
import os
from typing import Any, Dict, Optional

from .runtime import MAX_BODY_BYTES, Runtime

log = logging.getLogger("laya_mlx_server")


def _authorized(header: Optional[str], api_key: Optional[str]) -> bool:
    if not api_key:
        return True
    if not header or not header.lower().startswith("bearer "):
        return False
    return hmac.compare_digest(header[7:].strip(), api_key)


def create_app(runtime: Optional[Runtime] = None):
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "laya-mlx-server requires fastapi and uvicorn: pip install 'laya-mlx-server'"
        ) from exc

    rt = runtime if runtime is not None else Runtime()
    api_key = os.environ.get("LAYA_API_KEY") or None
    if not api_key:
        log.warning("LAYA_API_KEY is unset; /v1/systemone accepts unauthenticated requests")

    from . import __version__

    app = FastAPI(
        title="laya-mlx",
        version=__version__,
        description="Local Laya typed decisions on Apple silicon.",
    )

    @app.get("/health/live")
    def live() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready() -> JSONResponse:
        loaded = rt.status()["loaded"]
        if not loaded:
            return JSONResponse({"status": "loading", "loaded": []}, status_code=503)
        return JSONResponse({"status": "ready", "loaded": loaded})

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> JSONResponse:
        if not _authorized(request.headers.get("authorization"), api_key):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return JSONResponse({"error": "request body too large"}, status_code=413)
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            return JSONResponse({"error": "request body must be JSON"}, status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse({"error": "request body must be a JSON object"}, status_code=400)

        state: Any = payload.get("state")
        if state is None:
            return JSONResponse({"error": "'state' is required"}, status_code=400)
        questions = payload.get("questions")
        if not isinstance(questions, dict) or not questions:
            return JSONResponse({"error": "'questions' must be a non-empty object"}, status_code=400)

        try:
            result = rt.predict(state, questions, payload.get("model"))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:  # checkpoint/IO faults are the caller's to see
            log.exception("inference failed")
            return JSONResponse({"error": f"inference failed: {exc}"}, status_code=500)
        return JSONResponse(result)

    return app


def main(argv: Optional[list] = None) -> None:
    import uvicorn

    logging.basicConfig(level=os.environ.get("LAYA_LOG_LEVEL", "info").upper())
    host = os.environ.get("LAYA_HOST", "127.0.0.1")
    port = int(os.environ.get("LAYA_PORT", "8080"))
    runtime = Runtime()
    if os.environ.get("LAYA_PRELOAD", "1").strip().lower() not in ("0", "false", "no"):
        for repo in runtime.preload():
            log.info("preloaded %s", repo)
    uvicorn.run(create_app(runtime), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
