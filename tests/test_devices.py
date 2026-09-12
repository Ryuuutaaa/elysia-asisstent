from unittest.mock import patch
from audio.recorder import resolve_pulse_source

SOURCES = [
    {"id": "54", "name": "alsa_output.usb-mic.analog-stereo.monitor"},
    {"id": "55", "name": "alsa_input.usb-mic.mono-fallback"},
    {"id": "56", "name": "alsa_output.hdmi.monitor"},
]


def _pactl_with_sources(lines):
    def fake(args):
        if args == ["list", "short", "sources"]:
            return "\n".join(f"{i}\t{name}\tPipeWire\ts16le 2ch 48000Hz\tIDLE" for i, name in enumerate(lines, 1))
        return None
    return fake


def _fake_pactl(args):
    if args == ["get-default-sink"]:
        return "alsa_output.usb-mic.analog-stereo"
    if args == ["get-default-source"]:
        return "alsa_input.usb-mic.mono-fallback"
    return _pactl_with_sources([s["name"] for s in SOURCES])(args)


def test_speaker_resolves_default_sink_monitor():
    with patch("audio.recorder._run_pactl", side_effect=_fake_pactl):
        assert resolve_pulse_source("speaker") == "alsa_output.usb-mic.analog-stereo.monitor"


def test_speaker_falls_back_to_first_monitor():
    fake = _pactl_with_sources(["alsa_output.a.monitor", "alsa_output.b.monitor"])
    with patch("audio.recorder._run_pactl", side_effect=fake):
        assert resolve_pulse_source("speaker") == "alsa_output.a.monitor"


def test_speaker_none_when_no_monitor():
    fake = _pactl_with_sources(["alsa_input.usb-mic.mono-fallback"])
    with patch("audio.recorder._run_pactl", side_effect=fake):
        assert resolve_pulse_source("speaker") is None


def test_mic_resolves_default_source():
    with patch("audio.recorder._run_pactl", side_effect=_fake_pactl):
        assert resolve_pulse_source("mic") == "alsa_input.usb-mic.mono-fallback"


def test_headset_matches_keyword_and_skips_monitor():
    fake = _pactl_with_sources(["alsa_output.usb-x.monitor", "alsa_input.usb-headset"])
    with patch("audio.recorder._run_pactl", side_effect=fake):
        assert resolve_pulse_source("headset") == "alsa_input.usb-headset"


def test_headset_none_when_no_match():
    fake = _pactl_with_sources(["alsa_output.usb-x.monitor", "alsa_input.builtin-mic"])
    with patch("audio.recorder._run_pactl", side_effect=fake):
        assert resolve_pulse_source("headset") is None
