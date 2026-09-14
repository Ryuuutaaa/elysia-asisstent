import signal
import sys
import threading
import time
from typing import Optional

import numpy as np

from agent.llm import chat, extract_function_call, extract_text
from agent.prompt import build_system_prompt
from agent.tools import (
    ToolResponse,
    cancel_action,
    classify_followup,
    execute_confirmed_action,
    handle_function_call,
    is_affirmation,
)
from audio.recorder import AudioRecorder, find_device_index, save_debug_audio
from audio.stt import get_stt
from audio.tts import speak
from audio.vad import SileroVAD
from audio.wake_word import PorcupineWakeWord
from core.config import settings
from core.logger import configure_logging, get_logger, new_session_id
from core.state import AssistantState, StateMachine
from execution.apps import extract_app_keys, resolve_app
from execution.linux import safe_execute

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
        self._pending = None
        self._pending_lock = threading.Lock()
        self._session_id = ""
        self._await_mode: Optional[str] = None
        self._suggest_retries = 0
        self._vad_errors = 0
        self._pipeline_thread: Optional[threading.Thread] = None
        self._shutdown_done = False

    def initialize(self, skip_diagnostics: bool = False):
        log.info("startup_begin")
        start = time.perf_counter()

        device_index = find_device_index(settings.AUDIO_INPUT_SOURCE)

        if not skip_diagnostics:
            try:
                from audio.diagnostics import run_preflight_diagnostics

                run_preflight_diagnostics(device_index, interactive=True)
            except Exception as e:
                log.warning("diagnostics_failed", error=str(e))
        if not settings.GOOGLE_API_KEY or settings.GOOGLE_API_KEY.strip() in (
            "",
            "AIzaSy...",
            "AIzaSy_REPLACE_WITH_YOUR_KEY",
        ):
            log.error("gemini_api_key_missing")
            print("\n  ✗ GEMINI API KEY kosong/placeholder — isi .env dulu")
            sys.exit(1)
        else:
            try:
                from agent.llm import resolve_model, scan_available_models

                t0 = time.perf_counter()
                available = scan_available_models()
                scan_ms = (time.perf_counter() - t0) * 1000
                print(
                    f"\n  ── Gemini model scan ({len(available)} model, {scan_ms:.0f}ms) ──"
                )
                for m in available[:12]:
                    print(f"    • {m}")
                wanted = settings.GEMINI_MODEL
                print(f"\n  ── Resolve model: {wanted} ──")
                t1 = time.perf_counter()
                resolved = resolve_model(wanted, available)
                probe_ms = (time.perf_counter() - t1) * 1000
                if resolved != wanted:
                    print(
                        f"  ⚠ FALLBACK  {wanted} tidak tersedia → pakai {resolved}  ({probe_ms:.0f}ms)"
                    )
                    log.warning(
                        "gemini_model_fallback", requested=wanted, resolved=resolved
                    )
                else:
                    print(f"  ✓ ACTIVE  {resolved}  ({probe_ms:.0f}ms)")
            except Exception as e:
                log.error("gemini_model_resolve_failed", error=str(e)[:200])
                print(f"  ! Resolusi model gagal: {str(e)[:200]}")
                print(
                    "    Elysia tetap jalan; perintah suara akan error sampai GEMINI_MODEL valid."
                )

        self._recorder = AudioRecorder(device_index=device_index)

        try:
            self._wake_word.load()
        except Exception as e:
            log.critical("wake_word_init_failed", error=str(e))
            print(
                "\n  ✗ Wake word engine gagal dimuat.\n"
                "    Opsi: (a) pastikan paket openwakeword terpasang & OPENWAKEWORD_MODEL valid,\n"
                "          (b) set WAKE_WORD_ENGINE=porcupine + PICOVOICE_ACCESS_KEY di .env.\n"
                f"    Detail: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

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
        self._await_mode = None
        with self._pending_lock:
            self._suggest_retries = 0
        log.info("wake_word_triggered", session_id=self._session_id)

        if self._fsm.transition_to(AssistantState.LISTENING):
            self._wake_word.pause()
            self._start_listening()

    def _start_listening(
        self,
        max_record_ms: Optional[int] = None,
        no_speech_grace_ms: Optional[int] = None,
    ):
        self._vad.reset(
            max_record_ms=max_record_ms, no_speech_grace_ms=no_speech_grace_ms
        )
        self._recorder.start_recording()

    def _on_audio_frame(self, frame: np.ndarray):
        if self._recorder is None:
            return
        try:
            state = self._fsm.current_state
        except Exception:
            return
        if state == AssistantState.IDLE:
            try:
                self._wake_word.process_frame(frame)
            except Exception as e:
                log.error("wake_process_error", error=str(e))
        elif state == AssistantState.LISTENING:
            try:
                is_endpoint = self._vad.process_frame(frame)
            except Exception as e:
                self._vad_errors += 1
                log.error(
                    "vad_callback_error", error=str(e), consecutive=self._vad_errors
                )
                if self._vad_errors >= 5:
                    self._vad_errors = 0
                    self._recorder.stop_recording()
                    self._fsm.reset_to_idle("vad_repeated_error")
                    self._wake_word.resume()
                return
            self._vad_errors = 0
            if is_endpoint:
                try:
                    self._handle_recording_complete()
                except Exception as e:
                    log.error("recording_complete_error", error=str(e))
                    self._fsm.reset_to_idle("callback_error")
                    self._wake_word.resume()

    def _handle_recording_complete(self):
        had_speech = self._vad.speech_detected
        self._vad.stop()
        audio = self._recorder.stop_recording()
        if not self._fsm.transition_to(AssistantState.PROCESSING):
            return
        self._pipeline_thread = threading.Thread(
            target=self._process_pipeline, args=(audio, had_speech), daemon=True
        )
        self._pipeline_thread.start()

    def _process_pipeline(self, audio: np.ndarray, had_speech: bool = True):
        start_e2e = time.perf_counter()
        try:
            with self._pending_lock:
                has_pending = self._pending is not None
            suffix = "-confirm" if has_pending else ""
            save_debug_audio(self._session_id, audio, suffix=suffix)
            if had_speech:
                text = self._stt.transcribe(audio) if len(audio) > 0 else ""
            else:
                log.info("no_speech_recorded", session_id=self._session_id)
                text = ""
            with self._pending_lock:
                has_pending = self._pending is not None
            if has_pending:
                self._handle_confirmation_response(text, start_e2e)
                return
            if self._await_mode == "followup_answer":
                self._handle_followup_answer(text)
                return
            if not text:
                self._respond_error("Maaf, Elysia tidak mendengar perintahmu.")
                return
            self._run_command(text, start_e2e)

        except Exception as e:
            log.error("pipeline_error", error=str(e))
            self._respond_error("Terjadi kesalahan teknis.")
            if self._fsm.current_state != AssistantState.IDLE:
                self._fsm.reset_to_idle("pipeline_error")
                self._wake_word.resume()

    def _run_command(self, text: str, start_e2e: float = 0.0):
        log.info("processing_user_text", text=text, session_id=self._session_id)
        try:
            response = chat(text, self._system_prompt)
        except Exception as e:
            log.error("llm_call_failed", error=str(e), error_code="ERR_LLM_TIMEOUT")
            self._respond_error("Maaf, koneksi ke otak Elysia sedang terganggu.")
            return
        fn_call = extract_function_call(response)
        if fn_call:
            tool_res: ToolResponse = handle_function_call(fn_call, source_text=text)
            if tool_res.needs_confirmation:
                with self._pending_lock:
                    self._pending = {
                        "kind": tool_res.kind or "destructive",
                        "action": tool_res.pending_action,
                        "argv": tool_res.pending_argv,
                        "raw": tool_res.raw,
                        "source": tool_res.source,
                    }
                self._speak_and_listen(tool_res.text, start_e2e)
            elif tool_res.kind == "app_error":
                self._speak_and_listen_for_command(tool_res.text, start_e2e)
            else:
                self._speak_and_ask_followup(tool_res.text, start_e2e)
        else:
            reply = extract_text(response)
            if not reply:
                reply = "Maaf, saya tidak mengerti."
            self._speak_and_ask_followup(reply, start_e2e)

    def _handle_confirmation_response(self, text: str, start_e2e: float = 0.0):
        with self._pending_lock:
            pending = self._pending
            if pending:
                self._pending = None

        if not pending:
            self._speak_and_idle("Tidak ada perintah yang menunggu konfirmasi.")
            return

        if pending["kind"] == "app_suggestion":
            self._handle_app_suggestion_response(pending, text, start_e2e)
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

    def _handle_app_suggestion_response(self, pending: dict, text: str, start_e2e: float = 0.0):
        rejected_argv = pending.get("argv")

        # 1-3) A concrete app named in the reply (≠ the rejected candidate) wins,
        # even alongside 'iya' — e.g. "iya, firefox" / "bukan brave, firefox".
        if text.strip():
            for key in extract_app_keys(text):
                argv = resolve_app(key)
                if argv is None or argv == rejected_argv:
                    continue
                result = safe_execute(argv)
                log.info("app_suggestion_alt", raw=pending.get("raw"), chosen=key)
                self._speak_and_ask_followup(
                    f"{key.title()} sudah dibuka."
                    if result.success
                    else f"Gagal membuka {key}: {result.message}",
                    start_e2e,
                )
                return

        # 4) Plain affirmation -> open the offered candidate.
        if is_affirmation(text):
            result = safe_execute(rejected_argv)
            log.info("app_suggestion_confirmed", raw=pending.get("raw"), chosen=pending["action"])
            self._speak_and_ask_followup(
                f"{pending['action'].title()} sudah dibuka."
                if result.success
                else f"Gagal membuka {pending['action']}: {result.message}",
                start_e2e,
            )
            return

        # 5) Nothing usable -> ask again, capped per session.
        with self._pending_lock:
            self._suggest_retries += 1
            retries = self._suggest_retries
        if retries > 2:
            log.warning("app_suggestion_give_up", raw=pending.get("raw"))
            self._speak_and_idle(
                "Maaf, Elysia belum bisa mengerti. Silakan coba lagi nanti."
            )
            return
        log.info("app_suggestion_retry", raw=pending.get("raw"), attempt=retries)
        self._speak_and_listen_for_command("Baik, silakan sebut ulang.", start_e2e)

    def _speak_and_listen(self, text: str, start_e2e: float = 0.0):
        if not self._fsm.transition_to(AssistantState.SPEAKING):
            with self._pending_lock:
                self._pending = None
            return

        self._recorder.mute()
        self._wake_word.pause()

        try:
            speak(text)
        finally:
            if start_e2e > 0:
                e2e_ms = (time.perf_counter() - start_e2e) * 1000
                log.info(
                    "cycle_complete",
                    total_e2e_latency_ms=round(e2e_ms, 1),
                    session_id=self._session_id,
                )
            # Keep the mic muted through the cooldown: the prompt itself says
            # "Jawab 'ya' untuk konfirmasi", so its echo/reverb must not be
            # captured as a positive confirmation of a destructive action.
            self._settle_after_speech()
            try:
                self._recorder.unmute()
            except Exception as e:
                log.warning("unmute_after_confirm_prompt_failed", error=str(e))

        timeout_ms = int(settings.CONFIRMATION_TIMEOUT_SEC * 1000)
        if self._fsm.transition_to(AssistantState.LISTENING):
            self._start_listening(
                max_record_ms=timeout_ms, no_speech_grace_ms=timeout_ms
            )
            try:
                self._wake_word.reset()
            except Exception:
                pass
        else:
            with self._pending_lock:
                self._pending = None
            if not self._fsm.transition_to(AssistantState.IDLE):
                self._fsm.reset_to_idle("speak_and_listen_fallback")
            self._wake_word.resume()

    def _speak_and_ask_followup(self, text: str, start_e2e: float = 0.0):
        """Speak the response plus "Ada perintah lain?", then listen for the answer."""
        self._speak_and_await(f"{text} Ada perintah lain?", "followup_answer", start_e2e)

    def _speak_and_await(self, prompt: str, mode: str, start_e2e: float = 0.0):
        """Speak `prompt`, then listen for the user's reply without a wake word.
        `mode` tells _process_pipeline how to interpret that reply."""
        if not self._fsm.transition_to(AssistantState.SPEAKING):
            return

        self._recorder.mute()
        self._wake_word.pause()

        try:
            speak(prompt)
        finally:
            if start_e2e > 0:
                e2e_ms = (time.perf_counter() - start_e2e) * 1000
                log.info(
                    "cycle_complete",
                    total_e2e_latency_ms=round(e2e_ms, 1),
                    session_id=self._session_id,
                )
            self._settle_after_speech()
            try:
                self._recorder.unmute()
            except Exception as e:
                log.warning("unmute_after_followup_ask_failed", error=str(e))

        answer_ms = int(settings.FOLLOWUP_TIMEOUT_SEC * 1000)
        self._await_mode = mode
        if self._fsm.transition_to(AssistantState.LISTENING):
            # `answer_ms` bounds how long we wait for the user to start; the extra
            # window lets them finish a reply that begins near the deadline.
            self._start_listening(
                max_record_ms=answer_ms + 5000, no_speech_grace_ms=answer_ms
            )
            try:
                self._wake_word.reset()
            except Exception:
                pass
        else:
            self._await_mode = None
            if not self._fsm.transition_to(AssistantState.IDLE):
                self._fsm.reset_to_idle("followup_ask_fallback")
            self._wake_word.resume()

    def _handle_followup_answer(self, text: str):
        self._await_mode = None

        intent = classify_followup(text)

        if intent == "no":
            log.info("session_ended", reason="no_followup")
            self._speak_and_idle(self._end_session_message())
            return

        if intent == "yes":
            log.info("followup_accepted")
            self._speak_and_listen_for_command()
            return

        # User menjawab langsung dengan perintah berikutnya.
        log.info("followup_direct_command", text=text)
        self._run_command(text, time.perf_counter())

    def _speak_and_listen_for_command(self, prompt: str = "Silakan.", start_e2e: float = 0.0):
        """Short ack, then listen for the next command without a wake word."""
        if not self._fsm.transition_to(AssistantState.SPEAKING):
            if not self._fsm.transition_to(AssistantState.IDLE):
                self._fsm.reset_to_idle("followup_command_fallback")
            self._wake_word.resume()
            return

        self._recorder.mute()
        self._wake_word.pause()
        try:
            speak(prompt)
        finally:
            if start_e2e > 0:
                e2e_ms = (time.perf_counter() - start_e2e) * 1000
                log.info(
                    "cycle_complete",
                    total_e2e_latency_ms=round(e2e_ms, 1),
                    session_id=self._session_id,
                )
            self._settle_after_speech()
            try:
                self._recorder.unmute()
            except Exception as e:
                log.warning("unmute_after_followup_ack_failed", error=str(e))

        if self._fsm.transition_to(AssistantState.LISTENING):
            self._start_listening()
            try:
                self._wake_word.reset()
            except Exception:
                pass
        else:
            if not self._fsm.transition_to(AssistantState.IDLE):
                self._fsm.reset_to_idle("followup_command_fallback")
            self._wake_word.resume()

    def _settle_after_speech(self):
        """Keep the mic muted briefly after TTS so echo/reverb is not captured."""
        try:
            time.sleep(settings.SPEECH_COOLDOWN_SEC)
        except Exception:
            pass

    def _wake_word_name(self) -> str:
        if settings.WAKE_WORD_ENGINE == "porcupine":
            return "jarvis"
        name = settings.OPENWAKEWORD_MODEL_PATH or settings.OPENWAKEWORD_MODEL
        if not name:
            return ""
        stem = name.rsplit("/", 1)[-1].split(".", 1)[0]
        return stem.replace("_", " ").strip()

    def _end_session_message(self) -> str:
        name = self._wake_word_name()
        if name:
            return f"Baik, panggil '{name}' kalau butuh lagi."
        return "Baik, panggil aku kalau butuh lagi."

    def _speak_and_idle(self, text: str, start_e2e: float = 0.0):
        self._await_mode = None
        with self._pending_lock:
            self._pending = None
        if not self._fsm.transition_to(AssistantState.SPEAKING):
            return

        self._recorder.mute()
        self._wake_word.pause()

        try:
            speak(text)
        finally:
            if start_e2e > 0:
                e2e_ms = (time.perf_counter() - start_e2e) * 1000
                log.info(
                    "cycle_complete",
                    total_e2e_latency_ms=round(e2e_ms, 1),
                    session_id=self._session_id,
                )

            self._settle_after_speech()
            try:
                self._recorder.unmute()
            except Exception as e:
                log.warning("unmute_after_speak_failed", error=str(e))
            try:
                self._wake_word.reset()
            except Exception:
                pass
            try:
                if not self._fsm.transition_to(AssistantState.IDLE):
                    self._fsm.reset_to_idle("speak_and_idle_fallback")
            except Exception as e:
                log.warning("fsm_idle_after_speak_failed", error=str(e))
                self._fsm.reset_to_idle("speak_and_idle_fallback")
            try:
                self._wake_word.resume()
            except Exception as e:
                log.warning("wake_resume_after_speak_failed", error=str(e))

    def _respond_error(self, message: str):
        self._speak_and_idle(message)

    def run(self):
        self._running = True
        try:
            self._recorder.start_stream()
        except Exception as e:
            log.critical(
                "audio_stream_start_failed", error_code="ERR_AUDIO_INPUT", error=str(e)
            )
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
        if self._shutdown_done:
            return
        self._shutdown_done = True
        log.info("shutdown_initiated")
        self._running = False

        with self._pending_lock:
            self._pending = None
        self._await_mode = None

        if self._pipeline_thread is not None and self._pipeline_thread.is_alive():
            self._pipeline_thread.join(timeout=3.0)

        if self._recorder is not None:
            try:
                self._recorder.stop_stream()
            except Exception as e:
                log.warning("recorder_stop_stream_failed", error=str(e))
        if self._wake_word is not None:
            try:
                self._wake_word.delete()
            except Exception as e:
                log.warning("wake_word_delete_failed", error=str(e))
        log.info("shutdown_complete")


def main():
    import argparse

    p = argparse.ArgumentParser(description="Elysia assistant")
    p.add_argument(
        "--skip-diagnostics", action="store_true", help="skip speaker/mic quality check"
    )
    p.add_argument(
        "--diagnostics-only", action="store_true", help="only run diagnostics then exit"
    )
    args = p.parse_args()

    if args.diagnostics_only:
        from audio.diagnostics import run_preflight_diagnostics
        from audio.recorder import find_device_index as _fdi

        idx = _fdi(settings.AUDIO_INPUT_SOURCE)
        run_preflight_diagnostics(idx, interactive=True)
        sys.exit(0)

    assistant = ElysiaAssistant()

    def sig_handler(sig, frame):
        assistant.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    assistant.initialize(skip_diagnostics=args.skip_diagnostics)
    assistant.run()


if __name__ == "__main__":
    main()
