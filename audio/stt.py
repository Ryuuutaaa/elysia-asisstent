import time
import numpy as np
from typing import Optional
from faster_whisper import WhisperModel
from core.config import settings
from core.logger import get_logger
from core.utils import run_with_timeout

log = get_logger("stt")


def _normalize_peak(audio_float32: np.ndarray, target_dbfs: float = -6.0) -> np.ndarray:
    """Peak-normalize audio to a target dBFS level (default -6 dBFS) before STT.
    No-op on silence or empty input."""
    if audio_float32.size == 0:
        return audio_float32
    peak = float(np.max(np.abs(audio_float32)))
    if peak <= 0.0:
        return audio_float32
    target_linear = 10 ** (target_dbfs / 20.0)
    return np.clip(audio_float32 * (target_linear / peak), -1.0, 1.0)

class STT:
    def __init__(self):
        self._model: Optional[WhisperModel] = None

    def load(self):
        if self._model is not None:
            return
        log.info("whisper_loading", model=settings.WHISPER_MODEL_SIZE, device=settings.WHISPER_DEVICE)
        start = time.perf_counter()
        try:
            self._model = WhisperModel(
                settings.WHISPER_MODEL_SIZE,
                device=settings.WHISPER_DEVICE,
                compute_type=settings.WHISPER_COMPUTE_TYPE,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            log.info("whisper_loaded", load_ms=round(latency_ms, 1))
        except Exception as e:
            log.error("whisper_load_failed", error=str(e))
            raise e

    def transcribe(self, audio: np.ndarray) -> str:
        if self._model is None:
            return ""

        if len(audio) == 0:
            return ""

        audio_float32 = audio.astype(np.float32) / 32768.0
        audio_float32 = _normalize_peak(audio_float32)

        def _do_transcribe():
            segments, info = self._model.transcribe(
                audio_float32,
                language="id",
                vad_filter=False,
            )
            return " ".join([segment.text for segment in segments]).strip(), info

        start = time.perf_counter()
        try:
            text, info = run_with_timeout(_do_transcribe, settings.STT_TIMEOUT_MS / 1000)
            latency_ms = (time.perf_counter() - start) * 1000

            if not text or info.language_probability < 0.3:
                log.warning("stt_empty_or_low_prob", text=text, prob=round(info.language_probability, 2))
                return ""

            log.info("stt_completed", stt_latency_ms=round(latency_ms, 1), text=text, prob=round(info.language_probability, 2))
            return text
        except TimeoutError:
            log.error("stt_timeout", error_code="ERR_STT_TIMEOUT", timeout_ms=settings.STT_TIMEOUT_MS)
            return ""
        except Exception as e:
            log.error("stt_failed", error=str(e))
            return ""

_instance = STT()

def get_stt() -> STT:
    return _instance
