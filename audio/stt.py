import threading
import time
import numpy as np
from typing import Optional
from faster_whisper import WhisperModel
from core.config import settings
from core.logger import get_logger
from core.utils import run_with_timeout

log = get_logger("stt")


class STTBusyError(RuntimeError):
    """A prior transcription is still running on the shared model."""


def _normalize_peak(audio_float32: np.ndarray, target_dbfs: float = -6.0, min_peak: float = 0.01) -> np.ndarray:
    """Peak-normalize audio to a target dBFS level (default -6 dBFS) before STT.

    No-op for empty input, true silence, or signals whose peak is below `min_peak`:
    amplifying near-silence only feeds noise to Whisper and causes hallucinations."""
    if audio_float32.size == 0:
        return audio_float32
    peak = float(np.max(np.abs(audio_float32)))
    if peak < min_peak:
        return audio_float32
    target_linear = 10 ** (target_dbfs / 20.0)
    return np.clip(audio_float32 * (target_linear / peak), -1.0, 1.0)

class STT:
    def __init__(self):
        self._model: Optional[WhisperModel] = None
        self._run_lock = threading.Lock()

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

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> str:
        if self._model is None:
            log.warning("stt_model_not_loaded", error_code="ERR_STT_EMPTY")
            return ""

        if len(audio) == 0:
            return ""

        language = language or settings.STT_LANGUAGE

        audio_float32 = audio.astype(np.float32) / 32768.0

        rms = float(np.sqrt(np.mean(audio_float32 ** 2))) if audio_float32.size else 0.0
        if rms < settings.STT_MIN_RMS:
            log.warning("stt_no_speech", rms=round(rms, 5), error_code="ERR_STT_EMPTY")
            return ""

        audio_float32 = _normalize_peak(audio_float32)

        def _do_transcribe():
            # CTranslate2 is not thread-safe. A previous transcription whose
            # timeout fired may still be running, so refuse to touch the model
            # concurrently instead of corrupting its state.
            if not self._run_lock.acquire(blocking=False):
                raise STTBusyError("transcription already in progress")
            try:
                segments, info = self._model.transcribe(
                    audio_float32,
                    language=language,
                    vad_filter=False,
                )
                seg_list = list(segments)
                text = " ".join([segment.text for segment in seg_list]).strip()
                max_no_speech = 0.0
                for segment in seg_list:
                    value = getattr(segment, "no_speech_prob", None)
                    if isinstance(value, (int, float)):
                        max_no_speech = max(max_no_speech, float(value))
                return text, info, max_no_speech
            finally:
                self._run_lock.release()

        start = time.perf_counter()
        try:
            text, info, max_no_speech = run_with_timeout(_do_transcribe, settings.STT_TIMEOUT_MS / 1000)
            latency_ms = (time.perf_counter() - start) * 1000

            if max_no_speech > 0.8:
                log.warning("stt_no_speech_prob", no_speech_prob=round(max_no_speech, 2), text=text)
                return ""

            lang_prob = getattr(info, "language_probability", 1.0)
            if not text or lang_prob < 0.3:
                log.warning("stt_empty_or_low_prob", text=text, prob=round(float(lang_prob), 2))
                return ""

            log.info("stt_completed", stt_latency_ms=round(latency_ms, 1), text=text, prob=round(float(lang_prob), 2))
            return text
        except STTBusyError:
            log.warning("stt_busy_skipped", error_code="ERR_STT_EMPTY")
            return ""
        except TimeoutError:
            log.error("stt_timeout", error_code="ERR_STT_TIMEOUT", timeout_ms=settings.STT_TIMEOUT_MS)
            return ""
        except Exception as e:
            log.error("stt_failed", error=str(e))
            return ""

_instance = STT()

def get_stt() -> STT:
    return _instance
