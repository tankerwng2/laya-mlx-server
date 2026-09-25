# Derived from Laya (Apache-2.0); see NOTICE. Modified for laya-mlx.
"""HTTP and MCP surfaces for the MLX runtime.

Both surfaces share one :class:`~laya_mlx.serve.runtime.Runtime`, so a checkpoint
is built once per process no matter how many times it is named by a client.

    laya-mlx-http          FastAPI, TypeSafe Jev wire compatible (POST /v1/systemone)
    laya-mlx-mcp           MCP stdio server for agent clients
    python -m laya_mlx.serve {http,mcp}   same two entry points
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
