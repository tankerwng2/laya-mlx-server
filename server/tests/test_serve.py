"""Contract tests for the serve layer: state flattening, checkpoint aliases, wire guards.

The model is stubbed where a GPU would be needed; what is under test is the wire
contract and the guards. The guard tests use a real ``Runtime`` but never reach
``agent()``, because every guard runs before the checkpoint is built.
"""

import pytest
from fastapi.testclient import TestClient

from laya_mlx_server.http import create_app
from laya_mlx_server.runtime import (
    MAX_QUESTIONS,
    MAX_STATE_CHARS,
    Runtime,
    flatten_state,
    resolve_checkpoint,
)

ONE = {"a": {"type": "noul", "instructions": "q"}}


class StubRuntime:
    """Stands in for Runtime so these tests never load 600 MiB of weights."""

    def __init__(self, loaded=("aac6fef/laya-multilingual-mlx",)):
        self.loaded = list(loaded)
        self.seen = []

    def predict(self, state, questions, checkpoint=None):
        self.seen.append((state, checkpoint))
        return {
            "model": "laya-rl-agent",
            "answers": {qid: {"type": "noul", "noul": 0.5} for qid in questions},
            "usage": {"input_tokens": 8, "output_tokens": 0},
        }

    def status(self):
        return {"loaded": self.loaded, "device": "cpu", "dtype": "float16"}


def test_flatten_state_keeps_field_names_and_drops_json_punctuation():
    assert flatten_state("plain") == "plain"
    assert flatten_state({"message": "hi", "channel": "email"}) == "message: hi\nchannel: email"
    assert flatten_state(["a", "b"]) == "- a\n- b"
    # A nested value falls back to JSON, and the key a question refers to survives.
    assert flatten_state({"meta": {"retry": 2}}) == 'meta: {"retry": 2}'


def test_resolve_checkpoint_maps_aliases_and_passes_unknown_through():
    multilingual = "aac6fef/laya-multilingual-mlx"
    assert resolve_checkpoint("multilingual") == multilingual
    assert resolve_checkpoint("MULTILINGUAL") == multilingual
    assert resolve_checkpoint("convaiinnovations/laya-multilingual") == multilingual
    assert resolve_checkpoint(None) == multilingual  # default for a language-agnostic server
    # A local path or a private conversion must not be rewritten.
    assert resolve_checkpoint("/tmp/my-conversion") == "/tmp/my-conversion"


def test_runtime_guards_reject_before_any_checkpoint_is_built():
    runtime = Runtime()
    assert runtime._agents == {}
    with pytest.raises(ValueError):
        runtime.predict("x", {})
    with pytest.raises(ValueError):
        runtime.predict("x", {"a": {"type": "summarize", "instructions": "q"}})
    with pytest.raises(ValueError):
        runtime.predict("x" * (MAX_STATE_CHARS + 1), ONE)
    with pytest.raises(ValueError):
        runtime.predict("x", {str(i): {"type": "noul", "instructions": "q"} for i in range(MAX_QUESTIONS + 1)})
    assert runtime._agents == {}  # nothing was loaded while rejecting


def test_runtime_rejects_bad_configuration():
    with pytest.raises(ValueError):
        Runtime(state_mode="yaml")
    with pytest.raises(ValueError):
        Runtime(batch_size=0)


@pytest.fixture
def stub():
    return StubRuntime()


@pytest.fixture
def client(monkeypatch, stub):
    monkeypatch.setenv("LAYA_API_KEY", "sekret")
    return TestClient(create_app(stub))


def test_http_requires_bearer(client):
    body = {"state": "x", "questions": ONE}
    assert client.post("/v1/systemone", json=body).status_code == 401
    assert client.post(
        "/v1/systemone", json=body, headers={"Authorization": "Bearer wrong"}
    ).status_code == 401
    ok = client.post("/v1/systemone", json=body, headers={"Authorization": "Bearer sekret"})
    assert ok.status_code == 200
    assert ok.json()["answers"]["a"]["noul"] == 0.5


def test_http_validates_body_before_inference(monkeypatch, stub):
    monkeypatch.setenv("LAYA_API_KEY", "sekret")
    c = TestClient(create_app(stub))
    auth = {"Authorization": "Bearer sekret", "Content-Type": "application/json"}

    def code(payload):
        return c.post("/v1/systemone", content=payload, headers=auth).status_code

    assert code(b"not json") == 400
    assert code(b"[1,2]") == 400
    assert code(b'{"questions": {"a": {"type":"noul","instructions":"q"}}}') == 400
    assert code(b'{"state": "x"}') == 400
    assert code(b'{"state": "x", "questions": {}}') == 400
    assert stub.seen == []  # no rejected request reached the model


class FakeAgent:
    """Real Runtime, fake weights: exercises flattening and checkpoint resolution."""

    def __init__(self):
        self.seen = []
        self.repos = []

    def predict(self, state, questions):
        self.seen.append(state)
        return {
            "model": "fake",
            "answers": {qid: {"type": "noul", "noul": 0.5} for qid in questions},
            "usage": {"input_tokens": 1, "output_tokens": 0},
        }


@pytest.fixture
def fake_agent(monkeypatch):
    fake = FakeAgent()

    def fake_load(repo, *args, **kwargs):
        fake.repos.append(repo)
        return fake

    monkeypatch.setattr("laya_mlx.agent.load", fake_load)
    monkeypatch.delenv("LAYA_STATE_MODE", raising=False)
    return fake


def test_http_flattens_object_state_and_honours_json_mode(monkeypatch, fake_agent):
    monkeypatch.setenv("LAYA_API_KEY", "sekret")
    auth = {"Authorization": "Bearer sekret"}
    payload = {"state": {"message": "charged twice"}, "questions": ONE}

    TestClient(create_app(Runtime())).post("/v1/systemone", json=payload, headers=auth)
    assert fake_agent.seen[-1] == "message: charged twice"

    # LAYA_STATE_MODE is read once at construction, so json mode needs a new Runtime.
    monkeypatch.setenv("LAYA_STATE_MODE", "json")
    TestClient(create_app(Runtime())).post("/v1/systemone", json=payload, headers=auth)
    assert fake_agent.seen[-1] == payload["state"]


def test_http_resolves_the_named_checkpoint_to_its_mlx_repo(monkeypatch, fake_agent):
    monkeypatch.setenv("LAYA_API_KEY", "sekret")
    client = TestClient(create_app(Runtime()))
    auth = {"Authorization": "Bearer sekret"}

    for name in ("multilingual", "MULTILINGUAL", "convaiinnovations/laya-multilingual"):
        client.post(
            "/v1/systemone",
            json={"state": "x", "questions": ONE, "model": name},
            headers=auth,
        )
    # Every alias resolves to the same MLX repo, so only one checkpoint is ever built.
    assert set(fake_agent.repos) == {"aac6fef/laya-multilingual-mlx"}



def test_health_endpoints_report_residency(client):
    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").status_code == 200
    empty = TestClient(create_app(StubRuntime(loaded=())))
    assert empty.get("/health/ready").status_code == 503
