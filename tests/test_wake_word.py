import openwakeword
import openwakeword.model
import audio.wake_word as ww

PATHS = [
    "/models/alexa_v0.1.onnx",
    "/models/hey_jarvis_v0.1.onnx",
    "/models/weather_v0.1.onnx",
]


def test_loads_only_configured_model(monkeypatch):
    captured = {}

    class FakeModel:
        def __init__(self, wakeword_model_paths=None, **kwargs):
            captured["paths"] = wakeword_model_paths

    monkeypatch.setattr(openwakeword, "get_pretrained_model_paths", lambda: PATHS)
    monkeypatch.setattr(openwakeword.model, "Model", FakeModel)
    monkeypatch.setattr(ww.settings, "OPENWAKEWORD_MODEL", "hey_jarvis")

    detector = ww.PorcupineWakeWord()
    detector._engine = "openwakeword"
    detector.load()

    assert captured["paths"] == ["/models/hey_jarvis_v0.1.onnx"]
    assert detector._openwakeword_model is not None


def test_unknown_model_falls_back_to_porcupine(monkeypatch):
    monkeypatch.setattr(openwakeword, "get_pretrained_model_paths", lambda: PATHS)
    monkeypatch.setattr(ww.settings, "OPENWAKEWORD_MODEL", "does_not_exist")
    monkeypatch.setattr(ww.settings, "PICOVOICE_ACCESS_KEY", "")

    detector = ww.PorcupineWakeWord()
    detector._engine = "openwakeword"
    detector.load()

    assert detector._openwakeword_model is None
    assert detector._engine == "porcupine"
