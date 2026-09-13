import re
from dataclasses import dataclass
from typing import Optional
from google.genai import types
from execution.apps import resolve_app, is_destructive_action, resolve_destructive_action
from execution.linux import safe_execute
from core.logger import get_logger

log = get_logger("tools")

CANCEL_WORDS = {
    "tidak", "tdk", "enggak", "engga", "nggak", "ngga", "gak", "ga", "gk",
    "gausah", "batal", "batalkan", "jangan", "no", "stop", "cancel", "urungkan",
}
CONFIRM_WORDS = {
    "ya", "yah", "yak", "yoi", "iya", "benar", "bener", "betul", "lanjut",
    "lanjutkan", "oke", "ok", "okay", "okelah", "yes", "sip", "siap", "baik",
    "baiklah", "boleh", "gas", "ayo", "setuju",
}
DENY_WORDS = {
    "tidak", "tdk", "enggak", "engga", "nggak", "ngga", "gak", "ga", "gk",
    "gausah", "udah", "sudah", "cukup", "selesai", "beres", "no", "stop",
    "cancel", "batal",
}


def is_affirmation(text: str) -> bool:
    """Positive verbal confirmation for a destructive action. Anything else (including
    negation like 'jangan ya') is treated as no-confirmation. Words are matched exactly
    after stripping punctuation so STT variants don't require fragile substring hacks."""
    words = set(re.findall(r"\w+", text.lower()))
    if not words:
        return False
    if words & CANCEL_WORDS:
        return False
    return bool(words & CONFIRM_WORDS)


def is_denial(text: str) -> bool:
    """Negative answer to a yes/no question (e.g. 'tidak ada', 'sudah cukup').
    An explicit confirmation word wins over a denial word ('ya, tidak' -> not denial)."""
    words = set(re.findall(r"\w+", text.lower()))
    if not words:
        return False
    return bool(words & DENY_WORDS) and not bool(words & CONFIRM_WORDS)


def classify_followup(text: str) -> str:
    """Classify a reply to 'Ada perintah lain?'. Returns 'yes', 'no', or 'command'.

    Only a bare answer (at most two tokens, e.g. 'iya', 'tidak usah') counts as
    yes/no. A longer utterance is treated as the next command, so 'oke buka
    spotify' runs the command instead of being swallowed as a plain 'yes'."""
    words = re.findall(r"\w+", text.lower())
    if not words:
        return "no"
    word_set = set(words)
    if len(words) <= 2:
        if word_set & CONFIRM_WORDS:
            return "yes"
        if word_set & DENY_WORDS:
            return "no"
    return "command"

@dataclass
class ToolResponse:
    text: str
    needs_confirmation: bool = False
    pending_action: Optional[str] = None
    pending_argv: Optional[list[str]] = None

def handle_function_call(fn_call: types.FunctionCall) -> ToolResponse:
    name = fn_call.name
    args = dict(fn_call.args) if fn_call.args else {}

    if name == "open_application":
        return _handle_open_application(args)
    elif name == "system_action":
        return _handle_system_action(args)
    else:
        log.warning("unknown_tool_called", tool_name=name, args=args)
        return ToolResponse(text=f"Maaf, tool {name} tidak dikenali.")

def _handle_open_application(args: dict) -> ToolResponse:
    app_name = args.get("app_name", "")
    if not app_name:
        return ToolResponse(text="Maaf, nama aplikasi tidak diberikan.")

    argv = resolve_app(app_name)
    if argv is None:
        log.warning("tool_rejected", error_code="ERR_TOOL_NOT_ALLOWED", requested=app_name)
        return ToolResponse(
            text=f"Maaf, aplikasi {app_name} tidak ada dalam daftar yang diizinkan."
        )

    result = safe_execute(argv)
    if result.success:
        return ToolResponse(text=f"{app_name.title()} sudah dibuka.")
    else:
        return ToolResponse(text=f"Gagal membuka {app_name}: {result.message}")

def _handle_system_action(args: dict) -> ToolResponse:
    action = args.get("action", "")
    if not action:
        return ToolResponse(text="Maaf, aksi sistem tidak diberikan.")

    if not is_destructive_action(action):
        log.warning("tool_rejected", error_code="ERR_TOOL_NOT_ALLOWED", requested=action)
        return ToolResponse(
            text=f"Maaf, aksi {action} tidak ada dalam daftar yang diizinkan."
        )

    argv = resolve_destructive_action(action)
    if argv is None:
        return ToolResponse(text=f"Maaf, aksi {action} tidak dapat dijalankan.")

    action_labels = {
        "shutdown": "mematikan komputer",
        "matikan komputer": "mematikan komputer",
        "reboot": "me-restart komputer",
        "restart komputer": "me-restart komputer",
        "lock screen": "mengunci layar",
        "kunci layar": "mengunci layar",
    }
    label = action_labels.get(action.strip().lower(), action)

    log.info("destructive_confirmation_required", action=action)
    return ToolResponse(
        text=f"Kamu yakin ingin {label}? Jawab 'ya' untuk konfirmasi.",
        needs_confirmation=True,
        pending_action=action,
        pending_argv=argv,
    )

def execute_confirmed_action(argv: list[str], action: str) -> ToolResponse:
    log.info("destructive_confirmed", action=action, command=argv)
    # Non-blocking on purpose: `lock screen` runs hyprlock in the foreground for
    # as long as the screen stays locked, so a blocking wait with a timeout would
    # kill the process and unlock the screen again.
    result = safe_execute(argv)
    if result.success:
        return ToolResponse(text=f"Perintah {action} sedang dijalankan.")
    else:
        return ToolResponse(text=f"Gagal menjalankan {action}: {result.message}")

def cancel_action(action: str) -> ToolResponse:
    log.info("destructive_cancelled", action=action)
    return ToolResponse(text="Oke, perintah dibatalkan.")
