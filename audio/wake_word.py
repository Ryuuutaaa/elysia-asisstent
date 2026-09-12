import os
import time
import numpy as np
from pathlib import Path
from typing import Optional, Callable
from core.config import settings
from core.logger import get_logger

log = get_logger("wake_word")

class PorcupineWakeWord:
    """Wake word detector supporting both OpenWakeWord (offline/no-key) and Porcupine."""

    def __init__(self, keyword_path: Optional[str] = None):
        self._porcupine = None
        self._openwakeword_model = None
        self._engine = settings.WAKE_WORD_ENGINE
        self._keyword_path = keyword_path
        self._paused = False
        self._on_wake: Optional[Callable[[], None]] = None
        self._rolling_buffer = np.zeros(0, dtype=np.int16)
        self._target_frame_size = 1280

    def set_on_wake(self, callback: Callable[[], None]):
        self._on_wake = callback

    def _resolve_openwakeword_paths(self) -> list[str]:
        custom_path = settings.OPENWAKEWORD_MODEL_PATH
        if custom_path:
            if not os.path.exists(custom_path):
                log.warning("openwakeword_custom_model_missing", path=custom_path)
                return []
            return [custom_path]

        import openwakeword

        requested = settings.OPENWAKEWORD_MODEL
        all_paths = openwakeword.get_pretrained_model_paths()
        paths = [p for p in all_paths if requested in p]
        if not paths:
            log.warning(
                "openwakeword_model_not_found",
                requested=requested,
                available=[Path(p).stem for p in all_paths],
            )
        return paths

    def load(self):
        log.info("wake_word_loading", engine=self._engine)

        if self._engine == "openwakeword":
            try:
                from openwakeword.model import Model

                # Load only the configured model: loading all bundled models
                # (alexa, hey_mycroft, ...) would wake on unrelated phrases.
                model_paths = self._resolve_openwakeword_paths()
                if not model_paths:
                    self._engine = "porcupine"
                else:
                    self._openwakeword_model = Model(wakeword_model_paths=model_paths)
                    log.info("openwakeword_loaded", paths=model_paths)
                    return
            except Exception as e:
                log.warning("openwakeword_load_failed", error=str(e), note="falling back to porcupine if key available")
                self._engine = "porcupine"

        if self._engine == "porcupine":
            import pvporcupine
            access_key = settings.PICOVOICE_ACCESS_KEY
            if not access_key:
                log.warning("porcupine_key_missing", note="PICOVOICE_ACCESS_KEY is empty")
                return
            try:
                if self._keyword_path and os.path.exists(self._keyword_path):
                    self._porcupine = pvporcupine.create(
                        access_key=access_key,
                        keyword_paths=[self._keyword_path],
                    )
                    log.info("porcupine_loaded_custom", keyword_path=self._keyword_path)
                else:
                    self._porcupine = pvporcupine.create(
                        access_key=access_key,
                        keywords=["jarvis"],
                    )
                    log.info("porcupine_loaded_built_in", keyword="jarvis")
            except Exception as e:
                log.error("porcupine_load_failed", error=str(e))
                raise e

    def pause(self):
        self._paused = True
        log.info("wake_word_paused")

    def resume(self):
        self._paused = False
        log.info("wake_word_resumed")

    def is_paused(self) -> bool:
        return self._paused

    def process_frame(self, frame: np.ndarray) -> bool:
        if self._paused:
            return False

        start = time.perf_counter()

        if self._engine == "openwakeword" and self._openwakeword_model is not None:
            try:
                self._rolling_buffer = np.concatenate([self._rolling_buffer, frame.astype(np.int16)])
                if len(self._rolling_buffer) < self._target_frame_size:
                    return False

                # Take the latest 1280 samples and slide by frame length
                input_chunk = self._rolling_buffer[-self._target_frame_size:]
                self._rolling_buffer = self._rolling_buffer[-self._target_frame_size:]

                prediction = self._openwakeword_model.predict(input_chunk)
                for model_name, score in prediction.items():
                    if score >= 0.35:
                        latency_ms = (time.perf_counter() - start) * 1000
                        log.info("wake_word_detected", engine="openwakeword", model=model_name, score=round(float(score), 2), latency_ms=round(latency_ms, 1))
                        self._rolling_buffer = np.zeros(0, dtype=np.int16)
                        if self._on_wake:
                            try:
                                self._on_wake()
                            except Exception as e:
                                log.error("on_wake_callback_error", error=str(e))
                        return True
            except Exception as e:
                log.error("openwakeword_process_error", error=str(e))
            return False

        if self._engine == "porcupine" and self._porcupine is not None:
            if len(frame) != self._porcupine.frame_length:
                return False
            try:
                keyword_index = self._porcupine.process(frame.astype(np.int16))
                latency_ms = (time.perf_counter() - start) * 1000
                if keyword_index >= 0:
                    log.info("wake_word_detected", engine="porcupine", keyword_index=keyword_index, latency_ms=round(latency_ms, 1))
                    if self._on_wake:
                        try:
                            self._on_wake()
                        except Exception as e:
                            log.error("on_wake_callback_error", error=str(e))
                    return True
            except Exception as e:
                log.error("porcupine_process_error", error=str(e))

        return False

    def delete(self):
        if self._porcupine is not None:
            self._porcupine.delete()
            self._porcupine = None
        self._openwakeword_model = None
        log.info("wake_word_deleted")
