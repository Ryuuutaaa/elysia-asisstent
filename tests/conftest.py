import numpy as np
import pytest
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
