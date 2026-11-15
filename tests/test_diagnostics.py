import sys
import types

import numpy as np
from unittest.mock import MagicMock

import audio.diagnostics as diag


def _install_fake_sounddevice(monkeypatch, stream_factory=None, play=None):
    fake = types.SimpleNamespace(
        InputStream=stream_factory or (lambda **kwargs: None),
        play=play or (lambda *args, **kwargs: None),
        query_devices=lambda *args, **kwargs: [],
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    return fake


def test_bar_bounds():
    assert diag._bar(0.0, width=4) == "░░░░"
    assert diag._bar(1.0, width=4) == "████"
    assert diag._bar(0.5, width=4) == "██░░"


def test_run_returns_stdout_on_success(monkeypatch):
    completed = MagicMock(returncode=0, stdout=b"hello\n")
    monkeypatch.setattr(diag.subprocess, "run", lambda *a, **k: completed)

    assert diag._run(["pactl"]) == "hello"


def test_run_returns_empty_on_failure(monkeypatch):
    completed = MagicMock(returncode=1, stdout=b"nope")
    monkeypatch.setattr(diag.subprocess, "run", lambda *a, **k: completed)

    assert diag._run(["pactl"]) == ""


def test_run_returns_empty_when_command_raises(monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("pactl")

    monkeypatch.setattr(diag.subprocess, "run", boom)

    assert diag._run(["pactl"]) == ""


def test_diagnose_mics_marks_default_and_monitor(monkeypatch):
    monkeypatch.setattr(
        diag,
        "enumerate_audio_sources",
        lambda: [
            {"id": "1", "name": "alsa_input.usb"},
            {"id": "2", "name": "alsa_output.a.monitor"},
        ],
    )

    def fake_run(cmd, timeout=3.0):
        if cmd == ["pactl", "get-default-source"]:
            return "alsa_input.usb"
        if cmd == ["pactl", "list", "sources"]:
            return "Name: alsa_input.usb\n\tMute: no\n\tVolume: mono: 100%\n"
        return ""

    monkeypatch.setattr(diag, "_run", fake_run)

    mics = diag.diagnose_mics()

    assert mics[0]["default"] is True
    assert mics[0]["is_monitor"] is False
    assert mics[1]["default"] is False
    assert mics[1]["is_monitor"] is True


class _FakeStream:
    def __init__(self, callback, audio):
        self._callback = callback
        self._audio = audio

    def start(self):
        self._callback(self._audio, len(self._audio), None, None)

    def stop(self):
        return None

    def close(self):
        return None


def test_mic_quality_returns_measurements(monkeypatch):
    audio = np.ones((1600, 1), dtype=np.int16) * 3000
    _install_fake_sounddevice(
        monkeypatch,
        stream_factory=lambda **kwargs: _FakeStream(kwargs["callback"], audio),
    )
    monkeypatch.setattr(diag.time, "sleep", lambda seconds: None)

    result = diag.test_mic_quality(None, duration=0.0)

    assert result["ok"] is True
    assert result["samples"] == 1600
    assert result["verdict"]
    assert result["rms"] > 0


def test_mic_quality_reports_stream_failure(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("no device")

    _install_fake_sounddevice(monkeypatch, stream_factory=boom)

    result = diag.test_mic_quality(None, duration=0.0)

    assert result["ok"] is False
    assert "no device" in result["error"]
