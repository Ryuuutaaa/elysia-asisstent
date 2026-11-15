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
    monkeypatch.setattr(llm, "_resolved_model", None)
    monkeypatch.setattr(llm, "_resolved_for", None)

    resolved = llm.resolve_model("missing-model")

    assert resolved == "gemini-flash-latest"
    assert probed[:2] == ["missing-model", "gemini-flash-latest"]
