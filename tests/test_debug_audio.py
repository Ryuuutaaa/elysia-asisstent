import os
import numpy as np
from audio.recorder import save_debug_audio
from core.config import settings


def test_disabled_flag_returns_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "DEBUG_RECORD_AUDIO", False)
    assert save_debug_audio("abc123", np.zeros(16, dtype=np.int16)) is None
    assert not (tmp_path / ".debug_audio").exists()


def test_enabled_flag_writes_wav(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "DEBUG_RECORD_AUDIO", True)
    audio = (np.sin(np.linspace(0, 6.28, 1600)) * 8000).astype(np.int16)
    path = save_debug_audio("abc123", audio)
    assert path is not None
    assert (tmp_path / ".debug_audio" / "req-abc123.wav").exists()


def test_rotation_keeps_newest_50(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "DEBUG_RECORD_AUDIO", True)
    debug_dir = tmp_path / ".debug_audio"
    debug_dir.mkdir()
    for i in range(55):
        p = debug_dir / f"req-old{i:03d}.wav"
        p.write_bytes(b"\x00" * 4)
        os.utime(p, (1_000_000 + i, 1_000_000 + i))

    save_debug_audio("newest", np.zeros(16, dtype=np.int16))

    remaining = sorted(p.name for p in debug_dir.glob("req-*.wav"))
    assert len(remaining) == 50
    assert "req-newest.wav" in remaining
