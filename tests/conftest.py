import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from core.state import StateMachine

@pytest.fixture(autouse=True)
def _clear_suggestion_cache():
    from execution.apps import clear_suggestion_cache

    clear_suggestion_cache()
    yield
    clear_suggestion_cache()


@pytest.fixture
def state_machine():
    return StateMachine(cooldown_sec=0.05)

@pytest.fixture
def synthetic_silence():
    return np.zeros(4096, dtype=np.int16)

@pytest.fixture
def synthetic_speech():
    t = np.linspace(0, 0.5, 8000, False)
    wave = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    return wave

@pytest.fixture
def mock_sounddevice():
    mock = MagicMock()
    with patch("audio.recorder.sd") as sd_mock:
        sd_mock.query_devices.return_value = {"name": "mock device", "index": 0, "max_input_channels": 2}
        sd_mock.query_devices.side_effect = [
            {"name": "mock device", "index": 0, "max_input_channels": 2},
            [{"name": "mock device", "max_input_channels": 2}],
        ]
        yield sd_mock

@pytest.fixture
def mock_whisper():
    with patch("audio.stt.WhisperModel") as mock_model:
        mock_instance = MagicMock()
        mock_model.return_value = mock_instance
        yield mock_model
