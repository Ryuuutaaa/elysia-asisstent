import time
import torch
import numpy as np
from typing import Optional
from core.config import settings
from core.logger import get_logger

log = get_logger("vad")

class SileroVAD:
    def __init__(self, threshold: Optional[float] = None):
        self._model = None
        self._sample_rate = settings.SAMPLE_RATE
        self._threshold = threshold if threshold is not None else settings.VAD_THRESHOLD
        self._silence_threshold_ms = settings.VAD_SILENCE_THRESHOLD_MS
        self._max_record_ms = settings.VAD_MAX_RECORD_MS
        self._no_speech_grace_ms = settings.VAD_NO_SPEECH_GRACE_MS
        self._start_ignore_ms = settings.VAD_START_IGNORE_MS
        self._last_speech_time = 0.0
        self._start_time = 0.0
        self._has_speech_started = False
        self._is_active = False

    def load(self):
        if self._model is not None:
            return
        log.info("silero_vad_loading")
        # torch.hub downloads and executes the repo's hubconf.py on first run;
        # trust_repo=True skips the confirmation prompt. Set SILERO_VAD_HUB_REF
        # (e.g. a release tag) to stop tracking `master`.
        repo = "snakers4/silero-vad"
        if settings.SILERO_VAD_HUB_REF:
            repo = f"{repo}:{settings.SILERO_VAD_HUB_REF}"
        try:
            self._model, utils = torch.hub.load(
                repo_or_dir=repo,
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
                verbose=False,
            )
            self._model.eval()
            log.info("silero_vad_loaded", repo=repo)
        except Exception as e:
            log.error("silero_vad_load_failed", error=str(e))
            raise e

    def reset(self, max_record_ms: Optional[int] = None, no_speech_grace_ms: Optional[int] = None, threshold: Optional[float] = None):
        self._is_active = True
        self._has_speech_started = False
        self._last_speech_time = time.perf_counter()
        self._start_time = time.perf_counter()
        self._max_record_ms = max_record_ms if max_record_ms is not None else settings.VAD_MAX_RECORD_MS
        self._no_speech_grace_ms = no_speech_grace_ms if no_speech_grace_ms is not None else settings.VAD_NO_SPEECH_GRACE_MS
        self._start_ignore_ms = settings.VAD_START_IGNORE_MS
        if threshold is not None:
            self._threshold = threshold
        if self._model is not None:
            try:
                self._model.reset_states()
            except Exception as e:
                log.warning("vad_reset_states_failed", error=str(e))

    @property
    def speech_detected(self) -> bool:
        return self._has_speech_started

    def is_loaded(self) -> bool:
        return self._model is not None

    def stop(self):
        self._is_active = False

    def process_frame(self, frame: np.ndarray) -> bool:
        """Returns True if endpoint reached (silence detected or max duration)."""
        if not self._is_active:
            return False

        now = time.perf_counter()
        elapsed_ms = (now - self._start_time) * 1000

        if elapsed_ms > self._max_record_ms:
            log.info("vad_max_duration_reached", max_ms=self._max_record_ms)
            return True

        if elapsed_ms < self._start_ignore_ms:
            # Ignore the tail of the wake word: otherwise it marks speech as started
            # and the very next silence (800ms) ends the recording before the user
            # has had a chance to say the command.
            return False

        if self._model is None:
            # VAD model unavailable: cannot endpoint on silence, but the max-duration
            # guard above still terminates the recording instead of hanging forever.
            return False

        tensor = torch.from_numpy(frame.astype(np.float32) / 32768.0).unsqueeze(0)
        try:
            with torch.no_grad():
                prob = self._model(tensor, self._sample_rate).item()
        except Exception as e:
            log.error("vad_process_error", error=str(e))
            return False

        if prob >= self._threshold:
            self._has_speech_started = True
            self._last_speech_time = now

        # Don't cutoff silence until user has actually started speaking, or grace window passed
        if not self._has_speech_started and elapsed_ms < self._no_speech_grace_ms:
            return False

        silence_duration_ms = (now - self._last_speech_time) * 1000
        if silence_duration_ms >= self._silence_threshold_ms:
            log.info("vad_silence_detected", silence_ms=round(silence_duration_ms, 1))
            return True

        return False
