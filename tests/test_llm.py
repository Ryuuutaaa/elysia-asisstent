import time
from unittest.mock import MagicMock

import agent.llm as llm


def _install_fake_client(monkeypatch, probed):
    class FakeModels:
        def generate_content(self, model, contents):
            probed.append(model)
            return object()

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setattr(llm.genai, "Client", FakeClient)
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setattr(llm, "_resolved_model", None)
    monkeypatch.setattr(llm, "_resolved_for", None)


def test_resolve_model_reresolves_when_preferred_changes(monkeypatch):
    probed = []
    _install_fake_client(monkeypatch, probed)

    assert llm.resolve_model("model-a") == "model-a"
    assert llm.resolve_model("model-b") == "model-b"

    assert probed == ["model-a", "model-b"]


def test_resolve_model_memoizes_same_preferred(monkeypatch):
    probed = []
    _install_fake_client(monkeypatch, probed)

    assert llm.resolve_model("model-a") == "model-a"
    assert llm.resolve_model("model-a") == "model-a"

    assert probed == ["model-a"]


def test_resolve_model_falls_back_when_preferred_fails(monkeypatch):
    probed = []

    class FakeModels:
        def generate_content(self, model, contents):
            probed.append(model)
            if model == "missing-model":
                raise RuntimeError("404 model not found")
            return object()

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setattr(llm.genai, "Client", FakeClient)
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setattr(llm, "_resolved_model", None)
    monkeypatch.setattr(llm, "_resolved_for", None)

    resolved = llm.resolve_model("missing-model")

    assert resolved == "gemini-flash-latest"
    assert probed[:2] == ["missing-model", "gemini-flash-latest"]


def test_resolve_model_reuses_a_single_client(monkeypatch):
    created = []

    class FakeModels:
        def generate_content(self, model, contents):
            return object()

    class FakeClient:
        def __init__(self, api_key):
            created.append(api_key)
            self.models = FakeModels()

    monkeypatch.setattr(llm.genai, "Client", FakeClient)
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setattr(llm, "_resolved_model", None)
    monkeypatch.setattr(llm, "_resolved_for", None)

    assert llm.resolve_model("model-a") == "model-a"
    assert llm.resolve_model("model-b") == "model-b"

    assert len(created) == 1


def test_scan_available_models_strips_prefix(monkeypatch):
    class FakeModel:
        def __init__(self, name):
            self.name = name

    class FakeModels:
        def list(self):
            return [FakeModel("models/gemini-a"), FakeModel("models/gemini-b")]

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setattr(llm.genai, "Client", FakeClient)
    monkeypatch.setattr(llm, "_client", None)

    assert llm.scan_available_models() == ["gemini-a", "gemini-b"]


def test_scan_available_models_returns_empty_on_timeout(monkeypatch):
    monkeypatch.setattr(llm, "get_client", lambda: MagicMock())

    def fake_run(fn, timeout, budget="llm"):
        raise TimeoutError("slow scan")

    monkeypatch.setattr(llm, "run_with_timeout", fake_run)

    assert llm.scan_available_models() == []


def test_scan_available_models_returns_empty_when_client_fails(monkeypatch):
    def boom():
        raise RuntimeError("invalid key")

    monkeypatch.setattr(llm, "get_client", boom)

    assert llm.scan_available_models() == []


class _FakeTime:
    def __init__(self, start: float = 1000.0):
        self.value = start

    def monotonic(self) -> float:
        self.value += 1.0
        return self.value


def test_resolve_model_stops_when_total_budget_is_exhausted(monkeypatch):
    class FakeModels:
        def generate_content(self, model, contents):
            return object()

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    attempts = []
    fake_time = _FakeTime()

    def fake_run(fn, timeout, budget="llm"):
        attempts.append((timeout, budget))
        fake_time.value += 4.0
        raise TimeoutError("probe too slow")

    monkeypatch.setattr(llm.genai, "Client", FakeClient)
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setattr(llm, "_resolved_model", None)
    monkeypatch.setattr(llm, "_resolved_for", None)
    monkeypatch.setattr(llm, "time", fake_time)
    monkeypatch.setattr(llm, "run_with_timeout", fake_run)

    resolved = llm.resolve_model("m1")

    assert resolved == "m1"
    assert len(attempts) == 2


def test_turn_deadline_is_in_the_future():
    assert llm.turn_deadline() > time.monotonic()


def test_choose_app_stops_when_shared_deadline_is_exhausted(monkeypatch):
    generated = []
    monkeypatch.setattr(llm, "_plain_generate", lambda prompt: generated.append(prompt))

    status, candidate = llm.choose_app("breif", deadline=0.0)

    assert (status, candidate) == ("error", None)
    assert generated == []
