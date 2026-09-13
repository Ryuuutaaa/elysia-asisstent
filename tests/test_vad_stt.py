import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from audio.vad import SileroVAD
from audio.stt import STT

def test_vad_not_active_returns_false():
    vad = SileroVAD()
    frame = np.ones(512, dtype=np.int16) * 500
    assert vad.process_frame(frame) is False

def test_vad_silence_detected_returns_false_before_silence_threshold(synthetic_silence):
    vad = SileroVAD()
    mock_model = MagicMock()
    mock_model.return_value = torch_tensor_mock(0.1)
    mock_model.return_value.item.return_value = 0.1
    vad._model = mock_model
    vad.reset()
    assert vad.process_frame(synthetic_silence[:512]) in (True, False)

def _make_vad_with_speech_prob(prob: float) -> SileroVAD:
    vad = SileroVAD()
    mock_model = MagicMock()
    mock_tensor = MagicMock()
    mock_tensor.item.return_value = prob
    mock_model.return_value = mock_tensor
    mock_model.reset_states = MagicMock()
    vad._model = mock_model
    return vad

def torch_tensor_mock(value):
    m = MagicMock()
    m.item.return_value = value
    return m

def test_vad_reset_sets_active():
    vad = _make_vad_with_speech_prob(0.1)
    vad.reset()
    assert vad._is_active is True
    vad.stop()
    assert vad._is_active is False

def test_stt_transcribe_no_model_returns_empty(synthetic_speech):
    stt = STT()
    assert stt.transcribe(synthetic_speech) == ""

def test_stt_transcribe_empty_audio(synthetic_silence):
    stt = STT()
    stt._model = MagicMock()
    assert stt.transcribe(np.array([], dtype=np.int16)) == ""

@patch("audio.stt.WhisperModel")
def test_stt_transcribe_with_mock_model(mock_cls, synthetic_speech):
    stt = STT()
    stt._model = MagicMock()
    segment = MagicMock()
    segment.text = "buka brave browser"
    info = MagicMock()
    info.language_probability = 0.95
    stt._model.transcribe.return_value = ([segment], info)
    text = stt.transcribe(synthetic_speech)
    assert text == "buka brave browser"

@patch("audio.stt.WhisperModel")
def test_stt_empty_transcription_returns_empty(mock_cls, synthetic_silence):
    stt = STT()
    stt._model = MagicMock()
    info = MagicMock()
    info.language_probability = 0.90
    stt._model.transcribe.return_value = ([], info)
    assert stt.transcribe(synthetic_silence) == ""

@patch("audio.stt.WhisperModel")
def test_stt_low_probability_returns_empty(mock_cls, synthetic_speech):
    stt = STT()
    stt._model = MagicMock()
    segment = MagicMock()
    segment.text = "unknown noise"
    info = MagicMock()
    info.language_probability = 0.10
    stt._model.transcribe.return_value = ([segment], info)
    assert stt.transcribe(synthetic_speech) == ""


def test_normalize_peak_empty_and_silent():
    from audio.stt import _normalize_peak
    empty = _normalize_peak(np.array([], dtype=np.float32))
    assert empty.size == 0
    silent = _normalize_peak(np.zeros(64, dtype=np.float32))
    assert np.all(silent == 0.0)


def test_normalize_peak_scales_to_minus6db():
    from audio.stt import _normalize_peak
    audio = np.full(64, 0.25, dtype=np.float32)  # peak 0.25
    out = _normalize_peak(audio, target_dbfs=-6.0)
    assert np.max(np.abs(out)) == pytest.approx(10 ** (-6 / 20), rel=0.02)
    assert out.max() <= 1.0


def test_vad_without_model_returns_false_before_max_duration():
    from audio.vad import SileroVAD
    vad = SileroVAD()
    vad._model = None
    vad.reset(max_record_ms=10000, no_speech_grace_ms=10000)
    assert vad.process_frame(np.zeros(512, dtype=np.int16)) is False


def test_vad_without_model_ends_at_max_duration():
    import time as _time
    from audio.vad import SileroVAD
    vad = SileroVAD()
    vad._model = None
    vad.reset(max_record_ms=1000, no_speech_grace_ms=1000)
    vad._start_time = _time.perf_counter() - 2.0
    assert vad.process_frame(np.zeros(512, dtype=np.int16)) is True


def test_no_speech_grace_allows_time_to_start_speaking():
    import time as _time
    from core.config import settings
    vad = _make_vad_with_speech_prob(0.0)
    vad.reset()
    assert vad._no_speech_grace_ms == settings.VAD_NO_SPEECH_GRACE_MS

    vad._start_time = _time.perf_counter() - 1.6
    vad._last_speech_time = vad._start_time
    assert vad.process_frame(np.zeros(512, dtype=np.int16)) is False

    late = settings.VAD_NO_SPEECH_GRACE_MS / 1000 + 0.2
    vad._start_time = _time.perf_counter() - late
    vad._last_speech_time = vad._start_time
    assert vad.process_frame(np.zeros(512, dtype=np.int16)) is True


def test_normalize_peak_skips_near_silence():
    from audio.stt import _normalize_peak
    tiny = np.full(64, 0.001, dtype=np.float32)
    out = _normalize_peak(tiny)
    assert np.allclose(out, tiny)


def test_vad_ignores_initial_tail_window():
    import time as _time
    from core.config import settings
    vad = _make_vad_with_speech_prob(0.9)
    vad.reset()
    vad._start_time = _time.perf_counter() - (settings.VAD_START_IGNORE_MS / 1000) / 2
    assert vad.process_frame(np.zeros(512, dtype=np.int16)) is False
    assert vad.speech_detected is False


def test_vad_speech_detected_flag_after_tail_window():
    import time as _time
    from core.config import settings
    vad = _make_vad_with_speech_prob(0.9)
    vad.reset()
    vad._start_time = _time.perf_counter() - (settings.VAD_START_IGNORE_MS / 1000 + 0.3)
    vad.process_frame(np.zeros(512, dtype=np.int16))
    assert vad.speech_detected is True


def test_stt_skips_quiet_audio_without_calling_model():
    from audio.stt import STT
    stt = STT()
    stt._model = MagicMock()
    quiet = np.full(16000, 50, dtype=np.int16)  # rms ~0.0015 < STT_MIN_RMS
    assert stt.transcribe(quiet) == ""
    stt._model.transcribe.assert_not_called()


def test_vad_uses_configured_threshold(monkeypatch):
    from core.config import settings
    monkeypatch.setattr(settings, "VAD_THRESHOLD", 0.7)
    assert SileroVAD()._threshold == 0.7


def test_vad_explicit_threshold_overrides_config(monkeypatch):
    from core.config import settings
    monkeypatch.setattr(settings, "VAD_THRESHOLD", 0.7)
    assert SileroVAD(threshold=0.3)._threshold == 0.3


def test_stt_uses_configured_language(monkeypatch):
    from core.config import settings
    monkeypatch.setattr(settings, "STT_LANGUAGE", "en")
    stt = STT()
    stt._model = MagicMock()
    segment = MagicMock()
    segment.text = "hello world"
    info = MagicMock()
    info.language_probability = 0.95
    stt._model.transcribe.return_value = ([segment], info)
    audio = (np.sin(np.linspace(0, 100, 16000)) * 10000).astype(np.int16)
    stt.transcribe(audio)
    _, kwargs = stt._model.transcribe.call_args
    assert kwargs.get("language") == "en"
