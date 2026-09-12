import time
import pytest
from core.state import StateMachine, AssistantState

def test_full_cycle_idle_to_idle(state_machine):
    fsm = state_machine
    assert fsm.current_state == AssistantState.IDLE
    assert fsm.transition_to(AssistantState.LISTENING) is True
    assert fsm.transition_to(AssistantState.PROCESSING) is True
    assert fsm.transition_to(AssistantState.SPEAKING) is True
    assert fsm.transition_to(AssistantState.IDLE) is True

def test_invalid_transition_rejected(state_machine):
    fsm = state_machine
    assert fsm.transition_to(AssistantState.PROCESSING) is False
    assert fsm.transition_to(AssistantState.SPEAKING) is False
    assert fsm.current_state == AssistantState.IDLE

def test_cooldown_blocks_immediate_retrigger(state_machine):
    fsm = state_machine
    fsm.transition_to(AssistantState.LISTENING)
    fsm.transition_to(AssistantState.PROCESSING)
    fsm.transition_to(AssistantState.SPEAKING)
    fsm.transition_to(AssistantState.IDLE)
    assert fsm.can_trigger_wake_word() is False
    assert fsm.transition_to(AssistantState.LISTENING) is False
    time.sleep(0.06)
    assert fsm.can_trigger_wake_word() is True
    assert fsm.transition_to(AssistantState.LISTENING) is True

def test_process_to_idle_recovery(state_machine):
    fsm = state_machine
    fsm.transition_to(AssistantState.LISTENING)
    fsm.transition_to(AssistantState.PROCESSING)
    assert fsm.transition_to(AssistantState.IDLE) is True

def test_wake_word_rejected_during_speaking(state_machine):
    fsm = state_machine
    fsm.transition_to(AssistantState.LISTENING)
    fsm.transition_to(AssistantState.PROCESSING)
    fsm.transition_to(AssistantState.SPEAKING)
    assert fsm.can_trigger_wake_word() is False

def test_reset_to_idle_from_any_state(state_machine):
    fsm = state_machine
    fsm.transition_to(AssistantState.LISTENING)
    fsm.reset_to_idle("test_error")
    assert fsm.current_state == AssistantState.IDLE

def test_speaking_to_listening_allowed(state_machine):
    fsm = state_machine
    fsm.transition_to(AssistantState.LISTENING)
    fsm.transition_to(AssistantState.PROCESSING)
    fsm.transition_to(AssistantState.SPEAKING)
    assert fsm.transition_to(AssistantState.LISTENING) is True

def test_state_change_callback_called(state_machine):
    fsm = state_machine
    called = []
    fsm.set_on_state_change(lambda old, new: called.append((old, new)))
    fsm.transition_to(AssistantState.LISTENING)
    assert len(called) == 1
    assert called[0] == (AssistantState.IDLE, AssistantState.LISTENING)

def test_listening_to_idle_valid(state_machine):
    fsm = state_machine
    fsm.transition_to(AssistantState.LISTENING)
    assert fsm.transition_to(AssistantState.IDLE) is True
