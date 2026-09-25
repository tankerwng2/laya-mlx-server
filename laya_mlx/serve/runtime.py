# Derived from Laya (Apache-2.0); see NOTICE. Modified for laya-mlx.
"""Checkpoint cache and env contract shared by the HTTP and MCP surfaces.

Weights are the MLX conversions published under ``aac6fef``; clients name a
checkpoint by short alias, and anything that is not an alias (a local directory
or another Hugging Face repo id) is passed straight through to ``laya.load``.

Configuration mirrors upstream ``laya.serve`` wherever the name already exists, so
the same unit file drives the torch server and this one. Two names are MLX-only,
and ``LAYA_MODELS`` means something slightly different here -- see the note.

====================  ==========================================================  =========
env var               meaning                                                      default
====================  ==========================================================  =========
``LAYA_MODELS``       comma list to preload; empty = load on first request      ``multilingual``
``LAYA_DEVICE``       ``gpu`` / ``metal`` / ``cpu``                                 auto
``LAYA_DTYPE``        ``float16`` / ``bfloat16`` / ``float32``                    ``float16``
``LAYA_BATCH_SIZE``   questions per forward pass                                       ``16``
``LAYA_COMPILE``      ``1`` to compile the forward path                               ``0``
``LAYA_MAX_LOADED``   checkpoints kept resident before the oldest is dropped        ``2``
``LAYA_HOST``         bind address for the HTTP surface                        ``127.0.0.1``
``LAYA_PORT``         bind port for the HTTP surface                                 ``8080``
``LAYA_API_KEY``      if set, require ``Authorization: Bearer <it>``                none
``LAYA_STATE_MODE``   ``flatten`` (keys+values as text) or ``json``            ``flatten``
====================  ==========================================================  =========

Note on ``LAYA_MODELS``: upstream preloads *every* checkpoint when the value is
empty. Here an empty value loads nothing, because a laptop keeps 614 MiB of
weights per checkpoint resident on the GPU and preloading all three is a
guaranteed memory cliff. Set it explicitly to warm more than one.

Note on ``LAYA_STATE_MODE``: Jev clients canonically send ``state`` as an object,
and ``laya_mlx.common.serialize_state`` turns that into literal JSON before
tokenizing. On short input the JSON punctuation flips the answer -- measured on
the multilingual checkpoint, ``{"message": "I was charged twice"}`` scores
``noul`` 0.16 while the same text as ``message: I was charged twice`` scores
0.69 and keeps the field names the questions refer to. ``flatten`` renders
``key: value`` lines instead; set ``json`` for byte-identical upstream parity.
"""

import json
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

# Guardrails carried over from upstream laya.serve: the state is tokenized once
# per question, so an unbounded body can OOM a single-worker process.
MAX_QUESTIONS = 64
MAX_STATE_CHARS = 50_000
MAX_BODY_BYTES = 2 * 1024 * 1024

MLX_CHECKPOINTS = {
    "english": "aac6fef/laya-mlx",
    "multilingual": "aac6fef/laya-multilingual-mlx",
    "typed-decisions": "aac6fef/laya-typed-decisions-mlx",
}

# A client may send a published HF id (or this project's own repo id) instead of
# an alias; map the ones that name a specific MLX checkpoint.
_PUBLISHED_IDS = {
    "aac6fef/laya-mlx": "english",
    "aac6fef/laya-multilingual-mlx": "multilingual",
    "aac6fef/laya-typed-decisions-mlx": "typed-decisions",
    # The torch bundle ids resolve to the same conversion, so a client written
    # against upstream keeps working.
    "convaiinnovations/laya": "english",
    "convaiinnovations/laya-multilingual": "multilingual",
    "convaiinnovations/laya-typed-decisions": "typed-decisions",
}


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def resolve_checkpoint(name: Optional[str]) -> str:
    """Map a client-supplied model name onto a loadable MLX checkpoint repo.

    Unknown names fall through untouched so a local path or a private conversion
    works without an env change; ``laya.load`` raises on anything incomplete.
    """
    if not name:
        return MLX_CHECKPOINTS["multilingual"]
    value = str(name).strip()
    key = value.lower()
    alias = _PUBLISHED_IDS.get(key, key)
    # Unknown names keep their original case: a local path may be case-sensitive.
    return MLX_CHECKPOINTS.get(alias, value)


class Runtime:
    """Builds each checkpoint once and answers typed-decision requests from it."""

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        batch_size: Optional[int] = None,
        compile: Optional[bool] = None,
        max_loaded: Optional[int] = None,
        state_mode: Optional[str] = None,
    ) -> None:
        self.device = device or os.environ.get("LAYA_DEVICE") or None
        self.dtype = dtype or os.environ.get("LAYA_DTYPE") or "float16"
        self.compile = env_bool("LAYA_COMPILE", False) if compile is None else compile
        raw_batch = batch_size if batch_size is not None else os.environ.get("LAYA_BATCH_SIZE")
        # `int(raw_batch or 16)` would silently turn an explicit 0 into 16.
        self.batch_size = 16 if raw_batch in (None, "") else int(raw_batch)
        if self.batch_size < 1:
            raise ValueError("LAYA_BATCH_SIZE must be a positive integer")
        raw_max = (
            max_loaded if max_loaded is not None else os.environ.get("LAYA_MAX_LOADED")
        )
        self.max_loaded = 2 if raw_max in (None, "") else int(raw_max)
        if self.max_loaded < 1:
            raise ValueError("LAYA_MAX_LOADED must be a positive integer")
        self.state_mode = (
            state_mode or os.environ.get("LAYA_STATE_MODE") or "flatten"
        ).strip().lower()
        if self.state_mode not in ("flatten", "json"):
            raise ValueError("LAYA_STATE_MODE must be 'flatten' or 'json'")
        self._agents: "OrderedDict[str, Any]" = OrderedDict()

    def preload(self, names: Optional[List[str]] = None) -> List[str]:
        """Warm the named checkpoints, defaulting to LAYA_MODELS then multilingual."""
        if names is None:
            raw = os.environ.get("LAYA_MODELS", "").strip()
            names = [n.strip() for n in raw.split(",") if n.strip()] or ["multilingual"]
        return [str(self.agent(n).model_id) for n in names if n]

    def agent(self, name: Optional[str] = None):
        """Return a cached Agent for ``name``, building (and evicting) as needed."""
        repo = resolve_checkpoint(name)
        cached = self._agents.get(repo)
        if cached is not None:
            self._agents.move_to_end(repo)
            return cached
        # Import here so the runtime module stays importable without mlx on path.
        from ..agent import load

        agent = load(
            repo,
            device=self.device,
            dtype=self.dtype,
            batch_size=self.batch_size,
            compile=self.compile,
        )
        self._agents[repo] = agent
        while len(self._agents) > self.max_loaded:
            import mlx.core as mx

            victim, _ = self._agents.popitem(last=False)
            mx.clear_cache()
            del victim
        return agent

    def predict(self, state: Any, questions: Dict, checkpoint: Optional[str] = None) -> Dict:
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a non-empty object keyed by question id")
        if len(questions) > MAX_QUESTIONS:
            raise ValueError(f"at most {MAX_QUESTIONS} questions per request")
        # Validate question shapes before building a checkpoint, so a typo in a
        # client request cannot cost 600 MiB of weight loading before it fails.
        from ..common import QTYPES

        for qid, definition in questions.items():
            if not isinstance(definition, dict) or definition.get("type") not in QTYPES:
                raise ValueError(f"question {qid!r} must declare type choice, score, or noul")
        text = flatten_state(state) if self.state_mode == "flatten" else state
        if len(_as_text(text)) > MAX_STATE_CHARS:
            raise ValueError(f"state is longer than {MAX_STATE_CHARS} characters")
        return self.agent(checkpoint).predict(text, questions)

    def status(self) -> Dict:
        import mlx.core as mx

        return {
            "runtime": "laya-mlx",
            "device": str(self.device or mx.default_device()),
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "compile": self.compile,
            "max_loaded": self.max_loaded,
            "loaded": list(self._agents),
            "checkpoints": MLX_CHECKPOINTS,
            "metal": mx.metal.is_available(),
            "active_memory_mib": round(mx.get_active_memory() / 1_048_576, 1),
        }


def flatten_state(state: Any) -> str:
    """Render a structured state as plain ``key: value`` text.

    Keeps field names (preset questions refer to them in their instructions)
    while dropping the JSON punctuation that skews short inputs.
    """
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return "\n".join(f"{k}: {_as_text(v)}" for k, v in state.items())
    if isinstance(state, (list, tuple)):
        return "\n".join(f"- {_as_text(v)}" for v in state)
    return _as_text(state)


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)
