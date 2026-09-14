import re
from dataclasses import dataclass
from typing import Optional
from google.genai import types
from execution.apps import (
    APP_REGISTRY,
    get_cached_suggestion,
    is_destructive_action,
    norm_for_match,
    normalize_app_name,
    resolve_app,
    resolve_destructive_action,
    set_cached_suggestion,
    suggest_app,
)
from execution.linux import safe_execute
from agent.llm import choose_app
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
    "cancel", "batal", "bukan",
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


FOLLOWUP_FILLER = {
    "tolong", "silakan", "mohon", "coba", "saja", "aja", "dong", "deh", "lah",
    "nih", "sih", "kok", "terima", "kasih", "makasih", "thanks", "thank",
    "please", "ada", "lagi", "usah", "perlu", "dulu", "sekarang",
}


def classify_followup(text: str) -> str:
    """Classify a reply to 'Ada perintah lain?'. Returns 'yes', 'no', or 'command'.

    A reply counts as yes/no only when nothing is left after removing the yes/no
    words and filler ('iya', 'tidak, terima kasih'). Any leftover content word
    means it is the next command ('oke buka spotify', 'tidak, putar musik')."""
    words = set(re.findall(r"\w+", text.lower()))
    if not words:
        return "no"
    if words - CONFIRM_WORDS - DENY_WORDS - FOLLOWUP_FILLER:
        return "command"
    if words & CONFIRM_WORDS:
        return "yes"
    if words & DENY_WORDS:
        return "no"
    return "no"

@dataclass
class ToolResponse:
    text: str
    needs_confirmation: bool = False
    pending_action: Optional[str] = None
    pending_argv: Optional[list[str]] = None
    kind: Optional[str] = None
    raw: str = ""
    source: str = ""

def handle_function_call(fn_call: types.FunctionCall, source_text: str = "") -> ToolResponse:
    name = fn_call.name
    args = dict(fn_call.args) if fn_call.args else {}

    if name == "open_application":
        return _handle_open_application(args, source_text)
    elif name == "system_action":
        return _handle_system_action(args)
    else:
        log.warning("unknown_tool_called", tool_name=name, args=args)
        return ToolResponse(text=f"Maaf, tool {name} tidak dikenali.")


def _app_name_present(app_name: str, source_text: str) -> bool:
    """True when `app_name` is a registry key AND actually appears in what the
    user said. Both sides are normalized the same way as `suggest_app`. Prevents
    an LLM-side 'correction' from being treated as exact."""
    key = normalize_app_name(app_name)
    if key not in APP_REGISTRY:
        return False
    text = norm_for_match(source_text)
    norm_key = norm_for_match(key)
    return re.search(rf"(?<!\w){re.escape(norm_key)}(?!\w)", text) is not None


def _handle_open_application(args: dict, source_text: str = "") -> ToolResponse:
    app_name = args.get("app_name", "")
    if not isinstance(app_name, str) or not app_name.strip():
        return ToolResponse(text="Maaf, nama aplikasi tidak diberikan.")

    # Exact match — but only trust it when the user really said that name.
    if not source_text or _app_name_present(app_name, source_text):
        argv = resolve_app(app_name)
        if argv is not None:
            result = safe_execute(argv)
            if result.success:
                return ToolResponse(text=f"{app_name.title()} sudah dibuka.")
            return ToolResponse(text=f"Gagal membuka {app_name}: {result.message}")

    # Otherwise treat it as a misheard/typo'd name: cached -> fuzzy -> LLM.
    found, candidate = get_cached_suggestion(app_name)
    source = "cache"
    if not found:
        candidate = suggest_app(app_name)
        source = "fuzzy"
        if candidate is not None:
            set_cached_suggestion(app_name, candidate)  # fuzzy hit is definitive
        else:
            status, candidate = choose_app(app_name, source_text)
            if status == "found":
                source = "llm"
                set_cached_suggestion(app_name, candidate)
            elif status == "none":
                source = "llm-none"
                set_cached_suggestion(app_name, None)  # definitive -> cache
            else:  # "error" is transient -> leave uncached so we retry next time
                source = "llm-error"

    if candidate is None:
        log.info("app_suggestion_none", raw=app_name, source=source)
        if source == "llm-error":
            # Transient LLM/network failure — a system error, not "app unknown".
            return ToolResponse(
                text="Maaf, koneksi ke otak Elysia sedang terganggu. Coba sebut lagi ya.",
                kind="app_error",
            )
        return ToolResponse(text=f"Maaf, Elysia belum mengenali '{app_name}'. Coba sebut ulang.")

    argv = resolve_app(candidate)
    if argv is None:
        log.warning("tool_rejected", error_code="ERR_TOOL_NOT_ALLOWED", requested=app_name)
        return ToolResponse(text=f"Maaf, aplikasi {app_name} tidak ada dalam daftar yang diizinkan.")

    log.info("debug_suggestion", raw=app_name, candidate=candidate, source=source)
    return ToolResponse(
        text=f"Maksudmu '{candidate}'?",
        needs_confirmation=True,
        pending_action=candidate,
        pending_argv=argv,
        kind="app_suggestion",
        raw=app_name,
        source=source,
    )

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
        kind="destructive",
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
