# Derived from Laya (Apache-2.0); see NOTICE. Modified for laya-mlx.
"""HTTP and MCP surfaces for the laya-mlx runtime, shipped as ``laya-mlx-server``.

Both surfaces share one :class:`~laya_mlx_server.runtime.Runtime`, so a checkpoint
is built once per process no matter how many times a client names it. The model
itself comes from ``laya-mlx``; this package adds only the wire layers.

    laya-mlx-http          FastAPI, TypeSafe Jev wire compatible (POST /v1/systemone)
    laya-mlx-mcp           MCP stdio server for agent clients
    python -m laya_mlx_server {http,mcp}   same two entry points
"""

from .runtime import (
    MAX_BODY_BYTES,
    MAX_QUESTIONS,
    MAX_STATE_CHARS,
    MLX_CHECKPOINTS,
    Runtime,
    env_bool,
    flatten_state,
    resolve_checkpoint,
)

__version__ = "0.1.0"

__all__ = [
    "Runtime",
    "MLX_CHECKPOINTS",
    "resolve_checkpoint",
    "env_bool",
    "flatten_state",
    "MAX_QUESTIONS",
    "MAX_STATE_CHARS",
    "MAX_BODY_BYTES",
]
