import pathlib
import pytest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

def test_tts_edge_failure_returns_none():
    import audio.tts as tts_mod
    with patch.object(tts_mod, "_edge_tts_synthesize", side_effect=Exception("network error")):
        result = tts_mod.synthesize_edge("halo")
        assert result is None

def test_tts_piper_failure_returns_none():
    import audio.tts as tts_mod
    with patch("shutil.which", return_value="/usr/bin/piper"):
        with patch("subprocess.run", side_effect=Exception("piper not found")):
            result = tts_mod.synthesize_piper("halo")
            assert result is None

def test_piper_nonzero_exit():
    import audio.tts as tts_mod
    with patch("shutil.which", return_value="/usr/bin/piper"):
        with patch("subprocess.run", return_value=MagicMock(returncode=1, stdout=b"", stderr=b"error")):
            result = tts_mod.synthesize_piper("halo")
            assert result is None

def test_speak_falls_through_to_console():
    import audio.tts as tts_mod
    with patch.object(tts_mod, "synthesize_edge", return_value=None):
        with patch.object(tts_mod, "synthesize_piper", return_value=None):
            tts_mod.speak("test fallback message")

def test_no_shell_true_in_source():
    source_files = [PROJECT_ROOT / "main.py"]
    for package in ("core", "audio", "agent", "execution"):
        source_files.extend((PROJECT_ROOT / package).rglob("*.py"))
    for path in source_files:
        content = path.read_text()
        assert "shell=True" not in content, f"Found shell=True in {path.relative_to(PROJECT_ROOT)}"


def test_safe_execute_rejects_non_allowlisted(monkeypatch):
    import execution.linux as linux_mod
    spawned = []
    monkeypatch.setattr(linux_mod.subprocess, "Popen", lambda *a, **k: spawned.append((a, k)))

    result = linux_mod.safe_execute(["rm", "-rf", "/"])

    assert result.success is False
    assert "tidak diizinkan" in result.message
    assert spawned == []


def test_all_registry_argv_are_allowlisted():
    from execution.apps import APP_REGISTRY, DESTRUCTIVE_ACTIONS
    from execution.linux import is_allowlisted

    for argv in (*APP_REGISTRY.values(), *DESTRUCTIVE_ACTIONS.values()):
        assert is_allowlisted(argv) is True


def test_child_env_strips_secrets(monkeypatch):
    import execution.linux as linux_mod
    monkeypatch.setenv("GOOGLE_API_KEY", "secret-google")
    monkeypatch.setenv("PICOVOICE_ACCESS_KEY", "secret-pv")
    monkeypatch.setenv("ELYSA_TEST_MARKER", "kept")

    env = linux_mod._child_env()

    assert "GOOGLE_API_KEY" not in env
    assert "PICOVOICE_ACCESS_KEY" not in env
    assert env["ELYSA_TEST_MARKER"] == "kept"


class _FakeTime:
    def __init__(self, start: float = 1000.0, step: float = 1.0):
        self.value = start
        self.step = step

    def monotonic(self) -> float:
        self.value += self.step
        return self.value


def test_chat_stops_when_total_budget_is_exhausted(monkeypatch):
    import agent.llm as llm
    monkeypatch.setattr(llm, "time", _FakeTime())
    monkeypatch.setattr(llm.settings, "LLM_TOTAL_BUDGET_MS", 2500)
    monkeypatch.setattr(llm.settings, "LLM_MAX_ATTEMPTS", 5)
    monkeypatch.setattr(llm.settings, "LLM_REQUEST_TIMEOUT_MS", 2500)

    calls = []

    def fake_run(fn, timeout, budget="llm"):
        calls.append((timeout, budget))
        raise TimeoutError("boom")

    monkeypatch.setattr(llm, "run_with_timeout", fake_run)

    with pytest.raises(TimeoutError):
        llm.chat("halo", "prompt")

    assert len(calls) == 2
    assert all(budget == "llm" for _, budget in calls)


def test_chat_skips_attempt_when_budget_too_small(monkeypatch):
    import agent.llm as llm
    monkeypatch.setattr(llm, "time", _FakeTime())
    monkeypatch.setattr(llm.settings, "LLM_TOTAL_BUDGET_MS", 1000)
    monkeypatch.setattr(llm.settings, "LLM_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(llm.settings, "LLM_REQUEST_TIMEOUT_MS", 1000)

    calls = []
    monkeypatch.setattr(llm, "run_with_timeout", lambda *a, **k: calls.append(1))

    with pytest.raises(TimeoutError):
        llm.chat("halo", "prompt")

    assert calls == []


def test_chat_does_not_retry_client_errors(monkeypatch):
    import agent.llm as llm
    from google.genai import errors

    calls = []

    def fake_run(fn, timeout, budget="llm"):
        calls.append(1)
        raise errors.ClientError(400, {"error": {"message": "bad"}})

    monkeypatch.setattr(llm, "run_with_timeout", fake_run)

    with pytest.raises(errors.ClientError):
        llm.chat("halo", "prompt")

    assert len(calls) == 1


def test_is_retryable_skips_4xx_but_allows_429_and_5xx():
    from agent.llm import _is_retryable
    from google.genai import errors
    assert _is_retryable(errors.ClientError(400, {"error": {"message": "bad"}})) is False
    assert _is_retryable(errors.ClientError(429, {"error": {"message": "rate"}})) is True
    assert _is_retryable(errors.ServerError(503, {"error": {"message": "busy"}})) is True
    assert _is_retryable(TimeoutError("t")) is True


def test_piper_sample_rate_reads_model_config(tmp_path):
    import json
    from audio.tts import _piper_sample_rate
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"stub")
    (tmp_path / "voice.onnx.json").write_text(json.dumps({"audio": {"sample_rate": 16000}}))
    assert _piper_sample_rate(str(model)) == 16000


def test_piper_sample_rate_falls_back_when_config_missing(tmp_path):
    from audio.tts import _piper_sample_rate
    assert _piper_sample_rate(str(tmp_path / "missing.onnx")) == 22050
