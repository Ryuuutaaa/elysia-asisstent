import sys
import time
import signal
import threading
import numpy as np
from typing import Optional

from core.config import settings
from core.logger import configure_logging, get_logger, new_session_id
from core.state import StateMachine, AssistantState
from audio.recorder import AudioRecorder, find_device_index, save_debug_audio
from audio.vad import SileroVAD
from audio.stt import get_stt
from audio.tts import speak
from audio.wake_word import PorcupineWakeWord
from agent.prompt import build_system_prompt
from agent.llm import chat, extract_function_call, extract_text
from agent.tools import handle_function_call, execute_confirmed_action, cancel_action, is_affirmation, ToolResponse

configure_logging()
log = get_logger("main")

class ElysiaAssistant:
    def __init__(self):
        self._running = False
        self._fsm = StateMachine()
        self._recorder: AudioRecorder = None
        self._vad = SileroVAD()
        self._stt = get_stt()
        self._wake_word = PorcupineWakeWord()
        self._system_prompt = build_system_prompt()
        self._pending_confirmation = None
        self._session_id = ""

    def initialize(self):
        log.info("startup_begin")
        start = time.perf_counter()

        device_index = find_device_index(settings.AUDIO_INPUT_SOURCE)
        self._recorder = AudioRecorder(device_index=device_index)

        try:
            self._wake_word.load()
        except Exception as e:
            log.warning("wake_word_init_skipped", error=str(e), note="Check PICOVOICE_ACCESS_KEY in .env")

        try:
            self._vad.load()
        except Exception as e:
            log.error("vad_init_failed", error=str(e))

        try:
            self._stt.load()
        except Exception as e:
            log.error("stt_init_failed", error=str(e))

        self._wake_word.set_on_wake(self._on_wake_detected)
        self._recorder.set_on_frame(self._on_audio_frame)

        startup_ms = (time.perf_counter() - start) * 1000
        log.info("startup_complete", startup_ms=round(startup_ms, 1))

    def _on_wake_detected(self):
        if self._fsm.current_state != AssistantState.IDLE:
            return

        if not self._fsm.can_trigger_wake_word():
            return

        self._session_id = new_session_id()
        log.info("wake_word_triggered", session_id=self._session_id)

        if self._fsm.transition_to(AssistantState.LISTENING):
            self._wake_word.pause()
            self._start_listening()

    def _start_listening(self, max_record_ms: Optional[int] = None, no_speech_grace_ms: Optional[int] = None):
        self._vad.reset(max_record_ms=max_record_ms, no_speech_grace_ms=no_speech_grace_ms)
        self._recorder.start_recording()

    def _on_audio_frame(self, frame: np.ndarray):
        state = self._fsm.current_state

        if state == AssistantState.IDLE:
            self._wake_word.process_frame(frame)
        elif state == AssistantState.LISTENING:
            is_endpoint = self._vad.process_frame(frame)
            if is_endpoint:
                self._handle_recording_complete()

    def _handle_recording_complete(self):
        self._vad.stop()
        audio = self._recorder.stop_recording()

        if not self._fsm.transition_to(AssistantState.PROCESSING):
            return

        threading.Thread(target=self._process_pipeline, args=(audio,), daemon=True).start()

    def _process_pipeline(self, audio: np.ndarray):
        start_e2e = time.perf_counter()
        try:
            suffix = "-confirm" if self._pending_confirmation else ""
            save_debug_audio(self._session_id, audio, suffix=suffix)
            text = self._stt.transcribe(audio) if len(audio) > 0 else ""

            if self._pending_confirmation:
                self._handle_confirmation_response(text)
                return

            if not text:
                self._respond_error("Maaf, Elysia tidak mendengar perintahmu.")
                return

            log.info("processing_user_text", text=text, session_id=self._session_id)

            try:
                response = chat(text, self._system_prompt)
            except Exception as e:
                log.error("llm_call_failed", error=str(e), error_code="ERR_LLM_TIMEOUT")
                self._respond_error("Maaf, koneksi ke otak Elysia sedang terganggu.")
                return

            fn_call = extract_function_call(response)
            if fn_call:
                tool_res: ToolResponse = handle_function_call(fn_call)
                if tool_res.needs_confirmation:
                    self._pending_confirmation = {
                        "action": tool_res.pending_action,
                        "argv": tool_res.pending_argv,
                    }
                    self._speak_and_listen(tool_res.text)
                else:
                    self._speak_and_idle(tool_res.text, start_e2e)
            else:
                reply = extract_text(response)
                if not reply:
                    reply = "Maaf, saya tidak mengerti."
                self._speak_and_idle(reply, start_e2e)

        except Exception as e:
            log.error("pipeline_error", error=str(e))
            self._respond_error("Terjadi kesalahan teknis.")
            if self._fsm.current_state != AssistantState.IDLE:
                self._fsm.reset_to_idle("pipeline_error")
                self._wake_word.resume()

    def _handle_confirmation_response(self, text: str):
        pending = self._pending_confirmation
        self._pending_confirmation = None

        if not pending:
            self._speak_and_idle("Tidak ada perintah yang menunggu konfirmasi.")
            return

        if not text.strip():
            log.warning("destructive_timeout", action=pending["action"])
            self._speak_and_idle("Tidak ada konfirmasi. Perintah dibatalkan.")
            return

        if is_affirmation(text):
            tool_res = execute_confirmed_action(pending["argv"], pending["action"])
        else:
            tool_res = cancel_action(pending["action"])

        self._speak_and_idle(tool_res.text)

    def _speak_and_listen(self, text: str):
        if not self._fsm.transition_to(AssistantState.SPEAKING):
            return

        self._recorder.mute()
        self._wake_word.pause()

        try:
            speak(text)
        finally:
            self._recorder.unmute()

        timeout_ms = int(settings.CONFIRMATION_TIMEOUT_SEC * 1000)
        if self._fsm.transition_to(AssistantState.LISTENING):
            self._start_listening(max_record_ms=timeout_ms, no_speech_grace_ms=timeout_ms)
        else:
            self._pending_confirmation = None
            self._fsm.transition_to(AssistantState.IDLE)
            self._wake_word.resume()

    def _speak_and_idle(self, text: str, start_e2e: float = 0.0):
        if not self._fsm.transition_to(AssistantState.SPEAKING):
            return

        self._recorder.mute()
        self._wake_word.pause()

        try:
            speak(text)
        finally:
            if start_e2e > 0:
                e2e_ms = (time.perf_counter() - start_e2e) * 1000
                log.info("cycle_complete", total_e2e_latency_ms=round(e2e_ms, 1), session_id=self._session_id)

            # Cooldown delay before unmuting mic & resuming wake word to avoid acoustic feedback/reverb
            time.sleep(settings.COOLDOWN_SEC)
            self._recorder.unmute()
            self._fsm.transition_to(AssistantState.IDLE)
            self._wake_word.resume()

    def _respond_error(self, message: str):
        self._speak_and_idle(message)

    def run(self):
        self._running = True
        try:
            self._recorder.start_stream()
        except Exception as e:
            log.critical("audio_stream_start_failed", error_code="ERR_AUDIO_INPUT", error=str(e))
            print(
                "[Elysia] Gagal membuka perangkat audio. Periksa mikrofon dan AUDIO_INPUT_SOURCE di .env.",
                file=sys.stderr,
            )
            self.shutdown()
            sys.exit(1)

        log.info("elysia_running", message="Elysia mendengarkan wake word...")

        while self._running:
            try:
                time.sleep(0.1)
            except KeyboardInterrupt:
                break

        self.shutdown()

    def shutdown(self):
        if not self._running:
            return
        log.info("shutdown_initiated")
        self._running = False
        if self._recorder:
            self._recorder.stop_stream()
        if self._wake_word:
            self._wake_word.delete()
        log.info("shutdown_complete")

def main():
    assistant = ElysiaAssistant()

    def sig_handler(sig, frame):
        assistant.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    assistant.initialize()
    assistant.run()

if __name__ == "__main__":
    main()
