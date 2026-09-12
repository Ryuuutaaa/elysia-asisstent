import time
from enum import Enum
from typing import Optional, Callable
from core.config import settings
from core.logger import get_logger

log = get_logger("state_machine")

class AssistantState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"

class StateMachine:
    def __init__(self, cooldown_sec: Optional[float] = None):
        self._state = AssistantState.IDLE
        self._last_speaking_end_time: float = 0.0
        self._cooldown_sec = cooldown_sec if cooldown_sec is not None else settings.COOLDOWN_SEC
        self._on_state_change: Optional[Callable[[AssistantState, AssistantState], None]] = None

    @property
    def current_state(self) -> AssistantState:
        return self._state

    def set_on_state_change(self, callback: Callable[[AssistantState, AssistantState], None]):
        self._on_state_change = callback

    def can_trigger_wake_word(self) -> bool:
        if self._state != AssistantState.IDLE:
            return False
        elapsed = time.time() - self._last_speaking_end_time
        return elapsed >= self._cooldown_sec

    def transition_to(self, new_state: AssistantState) -> bool:
        old_state = self._state

        valid_transitions = {
            AssistantState.IDLE: [AssistantState.LISTENING],
            AssistantState.LISTENING: [AssistantState.PROCESSING, AssistantState.IDLE],
            AssistantState.PROCESSING: [AssistantState.SPEAKING, AssistantState.IDLE],
            AssistantState.SPEAKING: [AssistantState.IDLE, AssistantState.LISTENING],
        }

        if new_state not in valid_transitions.get(old_state, []):
            log.warning(
                "invalid_state_transition",
                from_state=old_state.value,
                to_state=new_state.value,
            )
            return False

        if old_state == AssistantState.IDLE and new_state == AssistantState.LISTENING:
            if not self.can_trigger_wake_word():
                log.warning(
                    "cooldown_active_trigger_dropped",
                    cooldown_remaining=self._cooldown_sec - (time.time() - self._last_speaking_end_time),
                )
                return False

        if old_state == AssistantState.SPEAKING and new_state == AssistantState.IDLE:
            self._last_speaking_end_time = time.time()

        self._state = new_state
        log.info("state_changed", from_state=old_state.value, to_state=new_state.value)

        if self._on_state_change:
            try:
                self._on_state_change(old_state, new_state)
            except Exception as e:
                log.error("state_change_callback_error", error=str(e))

        return True

    def reset_to_idle(self, reason: str = "error_recovery"):
        old_state = self._state
        self._state = AssistantState.IDLE
        self._last_speaking_end_time = time.time()
        log.warning("fsm_reset_to_idle", from_state=old_state.value, reason=reason)
        if self._on_state_change:
            try:
                self._on_state_change(old_state, AssistantState.IDLE)
            except Exception:
                pass
