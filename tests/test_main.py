import numpy as np
from unittest.mock import MagicMock

from agent.tools import ToolResponse
from core.config import settings
from core.state import AssistantState
from main import ElysiaAssistant


def _stub_llm_text(monkeypatch, text: str):
    monkeypatch.setattr("main.chat", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr("main.extract_function_call", lambda response: None)
    monkeypatch.setattr("main.extract_text", lambda response: text)


def test_last_suggestion_is_initialized():
    assistant = ElysiaAssistant()
    assert assistant._last_suggestion is None


def test_affirmative_first_command_does_not_crash(monkeypatch):
    monkeypatch.setattr(settings, "DEBUG_RECORD_AUDIO", False)

    assistant = ElysiaAssistant()
    assistant._stt = MagicMock()
    assistant._stt.transcribe.return_value = "ya"
    assistant._fsm.transition_to(AssistantState.LISTENING)
    assistant._fsm.transition_to(AssistantState.PROCESSING)

    spoken = []
    monkeypatch.setattr(assistant, "_speak_and_ask_followup", lambda text, start_e2e=0.0: spoken.append(text))
    monkeypatch.setattr(assistant, "_speak_and_idle", lambda text, start_e2e=0.0: spoken.append(text))
    monkeypatch.setattr(assistant, "_respond_error", lambda text: spoken.append(("ERR", text)))
    _stub_llm_text(monkeypatch, "Halo, ada yang bisa dibantu?")

    assistant._process_pipeline(np.zeros(1600, dtype=np.int16), had_speech=True)

    assert spoken == ["Halo, ada yang bisa dibantu?"]


def test_confirmation_affirm_executes(monkeypatch):
    assistant = ElysiaAssistant()
    monkeypatch.setattr(assistant, "_speak_and_idle", lambda *args, **kwargs: None)
    assistant._pending_confirmation = {"action": "shutdown", "argv": ["systemctl", "poweroff"]}

    executed = []
    monkeypatch.setattr(
        "main.execute_confirmed_action",
        lambda argv, action: executed.append((argv, action)) or ToolResponse(text="ok"),
    )

    assistant._handle_confirmation_response("ya")

    assert executed == [(["systemctl", "poweroff"], "shutdown")]
    assert assistant._pending_confirmation is None


def test_confirmation_negative_cancels(monkeypatch):
    assistant = ElysiaAssistant()
    monkeypatch.setattr(assistant, "_speak_and_idle", lambda *args, **kwargs: None)
    assistant._pending_confirmation = {"action": "shutdown", "argv": ["systemctl", "poweroff"]}

    executed = []
    cancelled = []
    monkeypatch.setattr("main.execute_confirmed_action", lambda *args: executed.append(args))
    monkeypatch.setattr(
        "main.cancel_action",
        lambda action: cancelled.append(action) or ToolResponse(text="batal"),
    )

    assistant._handle_confirmation_response("tidak")

    assert executed == []
    assert cancelled == ["shutdown"]


def test_confirmation_timeout_cancels(monkeypatch):
    assistant = ElysiaAssistant()
    spoken = []
    monkeypatch.setattr(assistant, "_speak_and_idle", lambda text, *args, **kwargs: spoken.append(text))
    assistant._pending_confirmation = {"action": "shutdown", "argv": ["systemctl", "poweroff"]}

    executed = []
    monkeypatch.setattr("main.execute_confirmed_action", lambda *args: executed.append(args))

    assistant._handle_confirmation_response("   ")

    assert executed == []
    assert "dibatalkan" in spoken[-1].lower()


def test_speak_and_listen_cools_down_before_unmute(monkeypatch):
    assistant = ElysiaAssistant()
    assistant._recorder = MagicMock()
    assistant._wake_word = MagicMock()
    assistant._start_listening = MagicMock()

    events = []
    assistant._recorder.mute.side_effect = lambda: events.append("mute")
    assistant._recorder.unmute.side_effect = lambda: events.append("unmute")
    monkeypatch.setattr("main.speak", lambda text: events.append("speak"))
    monkeypatch.setattr("main.time.sleep", lambda seconds: events.append(("sleep", seconds)))

    assistant._fsm.transition_to(AssistantState.LISTENING)
    assistant._fsm.transition_to(AssistantState.PROCESSING)

    assistant._speak_and_listen("Kamu yakin? Jawab 'ya' untuk konfirmasi.")

    sleep_idx = next(i for i, e in enumerate(events) if isinstance(e, tuple) and e[0] == "sleep")
    assert sleep_idx < events.index("unmute"), "mic must stay muted during the cooldown"
    assert events.index("speak") < events.index("unmute")


def test_speak_and_idle_resets_when_idle_transition_fails(monkeypatch):
    assistant = ElysiaAssistant()
    assistant._recorder = MagicMock()
    assistant._wake_word = MagicMock()

    monkeypatch.setattr("main.speak", lambda text: None)
    monkeypatch.setattr("main.time.sleep", lambda seconds: None)

    real_transition = assistant._fsm.transition_to

    def fake_transition(new_state):
        if new_state == AssistantState.IDLE:
            return False
        return real_transition(new_state)

    monkeypatch.setattr(assistant._fsm, "transition_to", fake_transition)
    resets = []
    monkeypatch.setattr(assistant._fsm, "reset_to_idle", lambda reason="": resets.append(reason))

    assistant._fsm.transition_to(AssistantState.LISTENING)
    assistant._fsm.transition_to(AssistantState.PROCESSING)

    assistant._speak_and_idle("halo")

    assert resets == ["speak_and_idle_fallback"]


def test_shutdown_is_idempotent_and_clears_pending():
    assistant = ElysiaAssistant()
    assistant._recorder = MagicMock()
    assistant._wake_word = MagicMock()
    assistant._pending_confirmation = {"action": "shutdown", "argv": ["systemctl", "poweroff"]}

    assistant.shutdown()
    assistant.shutdown()

    assistant._recorder.stop_stream.assert_called_once()
    assistant._wake_word.delete.assert_called_once()
    assert assistant._pending_confirmation is None


def test_on_audio_frame_recovers_after_repeated_vad_errors():
    assistant = ElysiaAssistant()
    assistant._recorder = MagicMock()
    assistant._wake_word = MagicMock()
    assistant._vad = MagicMock()
    assistant._vad.process_frame.side_effect = RuntimeError("boom")
    assistant._fsm.transition_to(AssistantState.LISTENING)

    for _ in range(5):
        assistant._on_audio_frame(np.zeros(512, dtype=np.int16))

    assert assistant._fsm.current_state == AssistantState.IDLE
    assistant._recorder.stop_recording.assert_called_once()
    assistant._wake_word.resume.assert_called()


def test_followup_negative_ends_session(monkeypatch):
    assistant = ElysiaAssistant()
    assistant._await_mode = "followup_answer"
    spoken = []
    monkeypatch.setattr(assistant, "_speak_and_idle", lambda text, *args, **kwargs: spoken.append(text))

    assistant._handle_followup_answer("tidak")

    assert assistant._await_mode is None
    assert spoken and "panggil aku" in spoken[0]


def test_followup_empty_ends_session(monkeypatch):
    assistant = ElysiaAssistant()
    assistant._await_mode = "followup_answer"
    spoken = []
    monkeypatch.setattr(assistant, "_speak_and_idle", lambda text, *args, **kwargs: spoken.append(text))

    assistant._handle_followup_answer("   ")

    assert spoken


def test_followup_affirmative_listens_for_command(monkeypatch):
    assistant = ElysiaAssistant()
    assistant._await_mode = "followup_answer"
    called = []
    monkeypatch.setattr(assistant, "_speak_and_listen_for_command", lambda: called.append(True))

    assistant._handle_followup_answer("iya")

    assert called == [True]
    assert assistant._await_mode is None


def test_followup_direct_command_runs(monkeypatch):
    assistant = ElysiaAssistant()
    assistant._await_mode = "followup_answer"
    commands = []
    monkeypatch.setattr(assistant, "_run_command", lambda text, start_e2e=0.0: commands.append(text))

    assistant._handle_followup_answer("buka spotify")

    assert commands == ["buka spotify"]


def test_pipeline_routes_to_followup_handler(monkeypatch):
    monkeypatch.setattr(settings, "DEBUG_RECORD_AUDIO", False)
    assistant = ElysiaAssistant()
    assistant._stt = MagicMock()
    assistant._stt.transcribe.return_value = "tidak"
    assistant._await_mode = "followup_answer"
    assistant._fsm.transition_to(AssistantState.LISTENING)
    assistant._fsm.transition_to(AssistantState.PROCESSING)

    handled = []
    monkeypatch.setattr(assistant, "_handle_followup_answer", lambda text: handled.append(text))

    assistant._process_pipeline(np.zeros(1600, dtype=np.int16), had_speech=True)

    assert handled == ["tidak"]


def test_speak_and_ask_followup_sets_mode_and_listens(monkeypatch):
    assistant = ElysiaAssistant()
    assistant._recorder = MagicMock()
    assistant._wake_word = MagicMock()
    assistant._start_listening = MagicMock()
    monkeypatch.setattr("main.speak", lambda text: None)
    monkeypatch.setattr("main.time.sleep", lambda seconds: None)

    assistant._fsm.transition_to(AssistantState.LISTENING)
    assistant._fsm.transition_to(AssistantState.PROCESSING)

    assistant._speak_and_ask_followup("Brave browser sudah dibuka.", 0.0)

    assert assistant._await_mode == "followup_answer"
    assert assistant._fsm.current_state == AssistantState.LISTENING
    assistant._start_listening.assert_called_once()


def test_speak_and_idle_clears_followup_mode(monkeypatch):
    assistant = ElysiaAssistant()
    assistant._recorder = MagicMock()
    assistant._wake_word = MagicMock()
    assistant._await_mode = "followup_answer"
    monkeypatch.setattr("main.speak", lambda text: None)
    monkeypatch.setattr("main.time.sleep", lambda seconds: None)

    assistant._fsm.transition_to(AssistantState.LISTENING)
    assistant._fsm.transition_to(AssistantState.PROCESSING)

    assistant._speak_and_idle("selesai")

    assert assistant._await_mode is None
    assert assistant._fsm.current_state == AssistantState.IDLE
