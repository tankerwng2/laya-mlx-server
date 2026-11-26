"""Offline smoke for both installed native packages and actual optional frameworks.

Run in a dedicated environment containing both wheels plus agno, langchain,
langgraph, llamaindex and crewai extras. JSON goes to stdout. Framework/model
network calls are forbidden; routing runs with a deterministic local decision
runner so this verifies integration types rather than model quality.
"""

import asyncio
import importlib
import importlib.metadata
import json
import os
import socket
import traceback

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_DISABLED"] = "true"


class Runner:
    def predict(self, state, questions, **kwargs):
        answers = {}
        for key, q in questions.items():
            choices = list(q.get("criteria", {"yes": None}))
            answers[key] = {
                "choice": choices[0],
                "confidence": 0.95,
                "answer_confidence": 0.95,
                "probabilities": {
                    choice: 1.0 if i == 0 else 0.0 for i, choice in enumerate(choices)
                },
            }
        return {"answers": answers, "usage": {"input_tokens": 1, "output_tokens": 0}}

    def predict_batch(self, states, questions, **kwargs):
        return [self.predict(s, questions, **kwargs) for s in states]


report = {
    "versions": {},
    "checks": [],
    "scope": "Real framework types; deterministic native decision runner; no external model/tool execution",
}
for name in (
    "laya-coreml",
    "laya-mlx",
    "agno",
    "langchain-core",
    "langgraph",
    "llama-index-core",
    "crewai",
):
    report["versions"][name] = importlib.metadata.version(name)


def blocked(*args, **kwargs):
    raise RuntimeError("Network forbidden in integration smoke")


socket.socket.connect = blocked
report["package_paths"] = {}
for package in ("laya_mlx", "laya_coreml"):
    report["package_paths"][package] = importlib.import_module(package).__file__
    for kind in ("langchain", "llamaindex", "crewai", "agno"):
        try:
            m = importlib.import_module(package + ".integrations." + kind)
            if kind == "langchain":
                from langchain_core.runnables import Runnable
                from langgraph.graph import END, START, StateGraph

                router = m.LayaRouter({"billing": "billing", "other": "other"}, agent=Runner())
                assert isinstance(router, Runnable)
                assert router.invoke("refund") == "billing"
                assert router.batch(["a", "b"]) == ["billing", "billing"]
                assert asyncio.run(router.ainvoke("refund")) == "billing"
                graph = StateGraph(dict)
                graph.add_node("route", lambda state: {"route": router.invoke(state)})
                graph.add_edge(START, "route")
                graph.add_edge("route", END)
                assert graph.compile().invoke({"text": "refund"}) == {"route": "billing"}
            elif kind == "llamaindex":
                from llama_index.core.schema import QueryBundle
                from llama_index.core.selectors import BaseSelector
                from llama_index.core.tools import ToolMetadata

                assert m._LLAMA_INDEX_AVAILABLE
                selector = m.LayaSingleSelector(agent=Runner())
                assert isinstance(selector, BaseSelector)
                choices = [
                    ToolMetadata(name="billing", description="billing"),
                    ToolMetadata(name="other", description="other"),
                ]
                assert selector.get_prompts() == {}
                selector.update_prompts({})
                assert selector.select(choices, QueryBundle("refund")).ind == 0
                multi = m.LayaMultiSelector(agent=Runner())
                assert isinstance(multi, BaseSelector)
                assert multi.get_prompts() == {}
                assert multi.select(choices, QueryBundle("refund")).inds == [0]
                assert asyncio.run(multi.aselect(choices, QueryBundle("refund"))).inds == [0]
                assert asyncio.run(selector.aselect(choices, QueryBundle("refund"))).ind == 0
            elif kind == "crewai":
                from crewai import Agent, Task

                assert m._CREWAI_AVAILABLE
                agents = [Agent.model_construct(role="Billing", goal="Help", backstory="billing")]
                task = Task.model_construct(description="Refund", expected_output="routing")
                assert m.LayaCrewRouter(agent=Runner()).route(task, agents).agent_index == 0
            else:
                from agno.agent import Agent

                agents = [Agent(name="Billing", role="billing")]
                assert m.LayaAgnoRouter(agent=Runner()).route("refund", agents).agent_index == 0
            report["checks"].append({"package": package, "integration": kind, "passed": True})
        except Exception:
            report["checks"].append(
                {
                    "package": package,
                    "integration": kind,
                    "passed": False,
                    "error": traceback.format_exc(),
                }
            )
print(json.dumps(report, indent=2))
raise SystemExit(not all(x["passed"] for x in report["checks"]))
