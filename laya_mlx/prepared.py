"""Bounded tokenized-prefix reuse. Encoder states and predictions are never cached."""

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

from .common import (
    QTYPES,
    build_prefix,
    finish_sequence,
    parallel_layout,
    render_options,
    serialize_state,
    uses_parallel_layout,
)


@dataclass(frozen=True)
class PreparedQuestion:
    ids: tuple
    markers: tuple
    option_stats: tuple


class PrefixCache:
    def __init__(self, capacity=128):
        self.capacity = capacity
        self.entries = OrderedDict()
        self._lock = RLock()

    def prepare(self, agent, state, questions):
        if not isinstance(questions, dict):
            raise ValueError("questions must be a dictionary keyed by question id")
        if not questions:
            return [], []
        tok = agent.tok
        max_len, head_len = agent.cfg.get("max_len", 512), agent.cfg.get("head_max_len", 192)
        state_ids = tok(
            serialize_state(state).replace(tok.mask_token, " "), add_special_tokens=False
        )["input_ids"]
        items, internal = [], []
        for qid, definition in questions.items():
            q = agent._question(qid, definition)
            options = render_options(q)
            key = (
                id(tok),
                tok.cls_token_id,
                tok.sep_token_id,
                tok.mask_token_id,
                tok.mask_token,
                head_len,
                q["t"],
                q["ins"],
                tuple(options),
            )
            with self._lock:
                if key not in self.entries:
                    ids, markers, stats = build_prefix(tok, q, head_len, return_stats=True)
                    self.entries[key] = PreparedQuestion(
                        tuple(ids), tuple(markers), tuple(stats.items())
                    )
                    if len(self.entries) > self.capacity:
                        self.entries.popitem(last=False)
                self.entries.move_to_end(key)
                prefix = self.entries[key]
            ids, markers, state_stats = finish_sequence(
                tok, prefix.ids, prefix.markers, state_ids, max_len, isinstance(state, list)
            )
            if len(markers) != len(options):
                raise ValueError(f"Question {qid!r} has too many options for the token budget")
            items.append(
                {
                    "ids": ids,
                    "markers": markers,
                    "qtype": QTYPES[q["t"]],
                    "options": dict(prefix.option_stats),
                    "state_stats": state_stats,
                }
            )
            if uses_parallel_layout(agent.cfg):
                layout = parallel_layout(
                    list(prefix.markers), len(prefix.ids), max(len(prefix.ids), len(ids))
                )
                items[-1]["layout"] = {k: v[: len(ids)] for k, v in layout.items()}
            internal.append(q)
        return items, internal
