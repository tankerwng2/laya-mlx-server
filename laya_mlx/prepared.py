"""Bounded, thread-safe question-prefix cache; no encoder/output caching."""

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

from .common import build_head, finish_sequence, parallel_layout, render_options


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

    def build_sequence(
        self,
        tok,
        state,
        q,
        max_len,
        head_max_len,
        *,
        option_order=None,
        truncate_left=False,
        state_ids=None,
        return_stats=False,
        return_truncation_stats=False,
        return_layout=False,
    ):
        # Tokenizer identity, rendered options, order and budget are all part of
        # the key. Retain the tokenizer so its identity cannot be reused.
        key = (
            tok,
            tok.cls_token_id,
            tok.sep_token_id,
            tok.mask_token_id,
            tok.mask_token,
            head_max_len,
            q["t"],
            q["ins"],
            tuple(render_options(q)),
            tuple(option_order) if option_order is not None else None,
        )
        with self._lock:
            prefix = self.entries.get(key)
            if prefix is None:
                ids, markers, stats = build_head(tok, q, head_max_len, option_order)
                prefix = PreparedQuestion(tuple(ids), tuple(markers), tuple(stats.items()))
                self.entries[key] = prefix
            self.entries.move_to_end(key)
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        if state_ids is None:
            from .common import encode_text, serialize_state

            state_ids = encode_text(
                tok, serialize_state(state).replace(tok.mask_token, " "), add_special_tokens=False
            )["input_ids"]
        ids, markers, state_stats = finish_sequence(
            tok, prefix.ids, prefix.markers, state_ids, max_len, truncate_left
        )
        result = (ids, markers)
        if return_stats:
            result += (dict(prefix.option_stats),)
        if return_truncation_stats:
            result += (state_stats,)
        if return_layout:
            layout = parallel_layout(
                list(prefix.markers), len(prefix.ids), max(len(prefix.ids), len(ids))
            )
            result += ({k: v[: len(ids)] for k, v in layout.items()},)
        return result

    def prepare(self, agent, state, questions):
        from copy import copy

        scoped = copy(agent)
        scoped._prefix_cache = self
        return scoped.prepare(state, questions)
