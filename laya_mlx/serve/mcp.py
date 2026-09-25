# Derived from Laya (Apache-2.0); see NOTICE. Modified for laya-mlx.
"""MCP stdio server exposing the MLX typed-decision tools.

Speaks MCP over stdio, so it plugs into any MCP client without a port, a token,
or a long-lived daemon:

    laya-mlx-mcp
    python -m laya_mlx.serve mcp

Environment: the ``LAYA_*`` names documented in :mod:`laya_mlx.serve.runtime`.
Standard I/O is the transport, so startup logging goes to stderr.

Tool choice mirrors upstream ``laya.mcp`` (predict / preset / status) so a client
configured for either server sees the same three names. Auto-routing between
checkpoints is deliberately absent: each resident checkpoint costs ~600 MiB of GPU
memory on a laptop, so this process serves one checkpoint per request instead of
keeping all three warm.
"""

import logging
import sys
from importlib import metadata as _metadata
from typing import Any, Dict, List, Optional, Union

try:
    from mcp.server.mcpserver import MCPServer
except ImportError as exc:  # pragma: no cover - optional extra
    raise ImportError(
        "the mcp extra is required to run the MCP server: pip install 'laya-mlx[mcp]'"
    ) from exc

from .runtime import Runtime, env_bool

# Jev sends `state` as either plain text or a JSON object; the runtime flattens the
# latter (LAYA_STATE_MODE). Declaring it `str` here made the SDK reject object states
# that the HTTP surface happily accepts.
State = Union[str, Dict[str, Any], List[Any]]

try:
    _VERSION = _metadata.version("laya-mlx")
except Exception:  # running from a source checkout without install metadata
    import laya_mlx as _laya

    _VERSION = getattr(_laya, "__version__", "")

_INSTRUCTIONS = (
    "Local Laya typed decisions on Apple silicon (MLX). One forward pass, ~4ms per "
    "question batch, no text generation. Use for structured decisions only: choice "
    "(finite labels), score (ordinal rubric), noul (calibrated P(true)). One question "
    "per entry; keep each entry a single AND-free proposition. Do NOT use for open "
    "Q&A, summarization, rewriting, code, or multi-hop reasoning. Confidence for "
    "choice questions with 11 or more options is clamped by the calibration and must "
    "not be used as a threshold."
)

server = MCPServer("laya-mlx", version=_VERSION, instructions=_INSTRUCTIONS)
_RUNTIME: Optional[Runtime] = None


def _runtime() -> Runtime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = Runtime()
    return _RUNTIME


@server.tool(
    name="laya_predict",
    title="Typed decisions",
    description=(
        "Answer typed questions about a state in one forward pass. questions maps a "
        "question id to {type: choice|score|noul, instructions, criteria?}. Returns "
        "per-question probabilities plus usage. Pass checkpoint 'multilingual' for "
        "non-English input."
    ),
)
def laya_predict(state: State, questions: Dict[str, Any], checkpoint: str = "") -> Dict[str, Any]:
    return _runtime().predict(state, questions, checkpoint or None)


@server.tool(
    name="laya_preset",
    title="Preset question set",
    description=(
        "Run a built-in question set against a state. preset is one of: triage, "
        "email, guard, moderation, router."
    ),
)
def laya_preset(preset: str, state: State, checkpoint: str = "") -> Dict[str, Any]:
    from .. import presets

    try:
        builder = getattr(presets, f"{str(preset).strip().lower()}_questions")
    except AttributeError:
        available = [n[: -len("_questions")] for n in dir(presets) if n.endswith("_questions")]
        raise ValueError(f"unknown preset {preset!r}; expected one of {available}") from None
    questions = builder()
    return _runtime().predict(state, questions, checkpoint or None)


@server.tool(
    name="laya_status",
    title="Runtime status",
    description="Report device, dtype, resident checkpoints and GPU memory.",
)
def laya_status() -> Dict[str, Any]:
    return _runtime().status()


def main(argv: Optional[list] = None) -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    runtime = _runtime()
    if env_bool("LAYA_PRELOAD", True):
        for repo in runtime.preload():
            logging.info("preloaded %s", repo)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
