import os
import subprocess
import time
import threading
import numpy as np
import sounddevice as sd
from pathlib import Path
from typing import Optional, Callable
from core.config import settings
from core.logger import get_logger

log = get_logger("recorder")

def _run_pactl(args: list[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            ["pactl", *args],
            capture_output=True,
            timeout=3.0,
            shell=False,
        )
        if result.returncode == 0:
            return result.stdout.decode(errors="replace").strip()
    except Exception as e:
        log.warning("pactl_command_failed", args=args, error=str(e))
    return None


def enumerate_audio_sources() -> list[dict]:
    sources = []
    output = _run_pactl(["list", "short", "sources"])
    if not output:
        return sources
    for line in output.split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2:
            sources.append({"id": parts[0], "name": parts[1]})
    return sources


def resolve_pulse_source(source_type: str) -> Optional[str]:
    """Resolve the PipeWire/PulseAudio source name to capture from for the requested source type."""
    names = [s["name"] for s in enumerate_audio_sources()]

    if source_type == "speaker":
        default_sink = _run_pactl(["get-default-sink"])
        if default_sink:
            monitor = f"{default_sink}.monitor"
            if monitor in names:
                return monitor
        monitors = [n for n in names if "monitor" in n.lower()]
        return monitors[0] if monitors else None

    if source_type == "headset":
        keywords = ("usb", "bluetooth", "bt", "headset", "headphone")
        for name in names:
            lowered = name.lower()
            if "monitor" in lowered:
                continue
            if any(k in lowered for k in keywords):
                return name
        return None

    default_source = _run_pactl(["get-default-source"])
    if default_source and default_source in names:
        return default_source
    return next((n for n in names if "monitor" not in n.lower()), None)


def _find_virtual_input_index() -> Optional[int]:
    devices = sd.query_devices()
    for preferred in ["pulse", "pipewire", "default"]:
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0 and preferred in dev["name"].lower():
                return i
    return None


def find_device_index(source_type: str) -> Optional[int]:
    for attempt in range(3):
        try:
            source_name = resolve_pulse_source(source_type)
            if source_type == "speaker" and not source_name:
                log.warning("speaker_monitor_not_found", attempt=attempt + 1)
                time.sleep(1.0)
                continue

            index = _find_virtual_input_index()
            if index is None:
                log.warning("audio_input_device_not_found", attempt=attempt + 1)
                time.sleep(1.0)
                continue

            if source_name:
                os.environ["PULSE_SOURCE"] = source_name
            else:
                os.environ.pop("PULSE_SOURCE", None)

            log.info(
                "audio_device_selected",
                source_type=source_type,
                device=source_name or "system-default",
                index=index,
            )
            return index
        except Exception as e:
            log.warning("device_enumerate_retry", attempt=attempt + 1, error=str(e))
            time.sleep(1.0)

    log.critical("audio_device_not_found", error_code="ERR_AUDIO_INPUT", source_type=source_type)
    try:
        default = sd.query_devices(kind="input")
        if default:
            os.environ.pop("PULSE_SOURCE", None)
            log.warning("audio_device_fallback_default", device=default["name"])
            return default["index"]
    except Exception:
        pass
    return None


def save_debug_audio(session_id: str, audio: np.ndarray, suffix: str = "", max_files: int = 50) -> Optional[str]:
    """Persist raw audio buffer for VAD/STT debugging (opt-in via DEBUG_RECORD_AUDIO).
    Keeps at most `max_files` newest WAV files in .debug_audio/."""
    if not settings.DEBUG_RECORD_AUDIO:
        return None

    debug_dir = Path(".debug_audio")
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"req-{session_id}{suffix}.wav"
    try:
        import wave

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(settings.SAMPLE_RATE)
            wf.writeframes(audio.astype(np.int16).tobytes())

        files = sorted(debug_dir.glob("req-*.wav"), key=lambda p: p.stat().st_mtime)
        for old in files[:-max_files]:
            old.unlink(missing_ok=True)

        log.info("debug_audio_saved", path=str(path), samples=len(audio))
        return str(path)
    except Exception as e:
        log.warning("debug_audio_save_failed", error=str(e))
        return None


class AudioRecorder:
    def __init__(self, device_index: Optional[int] = None):
        self._device_index = device_index
        self._sample_rate = settings.SAMPLE_RATE
        self._buffer: list[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._recording = False
        self._lock = threading.Lock()
        self._on_frame: Optional[Callable[[np.ndarray], None]] = None

    def set_on_frame(self, callback: Callable[[np.ndarray], None]):
        self._on_frame = callback

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            log.warning("audio_callback_status", status=str(status))
        frame = indata[:, 0].copy()
        if self._recording:
            with self._lock:
                self._buffer.append(frame)
        if self._on_frame:
            try:
                self._on_frame(frame)
            except Exception as e:
                log.error("on_frame_callback_error", error=str(e))

    def start_stream(self):
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            blocksize=512,
            channels=1,
            dtype="int16",
            device=self._device_index,
            callback=self._audio_callback,
        )
        self._stream.start()
        log.info("audio_stream_started", device=self._device_index, sample_rate=self._sample_rate)

    def stop_stream(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            log.info("audio_stream_stopped")

    def start_recording(self):
        with self._lock:
            self._buffer.clear()
        self._recording = True
        log.info("recording_started")

    def stop_recording(self) -> np.ndarray:
        self._recording = False
        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.int16)
            audio = np.concatenate(self._buffer)
            self._buffer.clear()
        duration_ms = len(audio) / self._sample_rate * 1000
        log.info("recording_stopped", audio_duration_ms=round(duration_ms, 1), samples=len(audio))
        return audio

    def mute(self):
        if self._stream and self._stream.active:
            self._stream.stop()
            log.info("mic_muted")

    def unmute(self):
        if self._stream and not self._stream.active:
            self._stream.start()
            log.info("mic_unmuted")
