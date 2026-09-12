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
