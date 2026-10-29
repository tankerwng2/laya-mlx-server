"""Exercise the actual optional framework (installed explicitly in CI)."""

import asyncio

import pytest

pytest.importorskip("llama_index.core")
from llama_index.core.selectors import BaseSelector, SelectorResult  # noqa: E402
from llama_index.core.tools import ToolMetadata  # noqa: E402

from laya_mlx.integrations.llamaindex import (  # noqa: E402
    _LLAMA_INDEX_AVAILABLE,
    LayaMultiSelector,
    LayaSingleSelector,
)


class Runner:
    def predict(self, state, questions, **kwargs):
        return {
            "answers": {
                "selector": {
                    "choice": "choice_0",
                    "confidence": 0.95,
                    "probabilities": {"choice_0": 0.95, "choice_1": 0.05},
                }
            }
        }


@pytest.mark.parametrize("selector_type", [LayaSingleSelector, LayaMultiSelector])
def test_real_selector_types_sync_async_and_prompts(selector_type):
    assert _LLAMA_INDEX_AVAILABLE
    selector = selector_type(agent=Runner())
    assert isinstance(selector, BaseSelector)
    assert selector.get_prompts() == {}
    selector.update_prompts({})
    with pytest.raises(ValueError, match="instructions"):
        selector.update_prompts({"unknown": object()})
    choices = [
        ToolMetadata(name="billing", description="billing"),
        ToolMetadata(name="other", description="other"),
    ]
    sync = selector.select(choices, "refund")
    asynchronous = asyncio.run(selector.aselect(choices, "refund"))
    assert isinstance(sync, SelectorResult)
    assert isinstance(asynchronous, SelectorResult)
    assert sync.inds == asynchronous.inds == [0]
