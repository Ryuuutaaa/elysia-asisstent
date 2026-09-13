import pathlib
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

def test_tts_edge_failure_returns_none():
    import audio.tts as tts_mod
    with patch.object(tts_mod, "_edge_tts_synthesize", side_effect=Exception("network error")):
        result = tts_mod.synthesize_edge("halo")
        assert result is None

def test_tts_piper_failure_returns_none():
    import audio.tts as tts_mod
    with patch("subprocess.run", side_effect=Exception("piper not found")):
        result = tts_mod.synthesize_piper("halo")
        assert result is None

def test_piper_nonzero_exit():
    import audio.tts as tts_mod
    with patch("subprocess.run", return_value=MagicMock(returncode=1, stdout=b"", stderr=b"error")):
        result = tts_mod.synthesize_piper("halo")
        assert result is None

def test_speak_falls_through_to_console():
    import audio.tts as tts_mod
    with patch.object(tts_mod, "synthesize_edge", return_value=None):
        with patch.object(tts_mod, "synthesize_piper", return_value=None):
            tts_mod.speak("test fallback message")

def test_no_shell_true_in_source():
    project_files = [
        "core/config.py", "core/logger.py", "core/state.py",
        "audio/recorder.py", "audio/vad.py", "audio/stt.py",
        "audio/tts.py", "audio/wake_word.py",
        "agent/llm.py", "agent/tools.py", "agent/prompt.py",
        "execution/apps.py", "execution/linux.py", "main.py",
    ]
    for fname in project_files:
        path = PROJECT_ROOT / fname
        assert path.exists(), f"expected source file is missing: {fname}"
        content = path.read_text()
        assert "shell=True" not in content, f"Found shell=True in {fname}"


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
