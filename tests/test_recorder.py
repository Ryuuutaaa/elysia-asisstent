from unittest.mock import MagicMock

import audio.recorder as rec
from audio.recorder import AudioRecorder


def _fake_run(calls, returncode=0):
    def run(argv, **kwargs):
        calls.append(argv)
        return MagicMock(returncode=returncode)

    return run


def test_mute_sends_pactl_mute_command(monkeypatch):
    calls = []
    monkeypatch.setattr(rec, "resolve_pulse_source", lambda source: "test-source")
    monkeypatch.setattr(rec.subprocess, "run", _fake_run(calls))

    AudioRecorder().mute()

    assert ["pactl", "set-source-mute", "test-source", "1"] in calls


def test_unmute_sends_pactl_unmute_command(monkeypatch):
    calls = []
    monkeypatch.setattr(rec, "resolve_pulse_source", lambda source: "test-source")
    monkeypatch.setattr(rec.subprocess, "run", _fake_run(calls))

    AudioRecorder().unmute()

    assert ["pactl", "set-source-mute", "test-source", "0"] in calls


def test_mute_survives_pactl_failure(monkeypatch):
    monkeypatch.setattr(rec, "resolve_pulse_source", lambda source: "test-source")

    def boom(*args, **kwargs):
        raise OSError("pactl missing")

    monkeypatch.setattr(rec.subprocess, "run", boom)

    AudioRecorder().mute()


def test_unmute_survives_pactl_failure(monkeypatch):
    monkeypatch.setattr(rec, "resolve_pulse_source", lambda source: "test-source")

    def boom(*args, **kwargs):
        raise OSError("pactl missing")

    monkeypatch.setattr(rec.subprocess, "run", boom)

    AudioRecorder().unmute()
