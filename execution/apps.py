import difflib
import re
from collections import OrderedDict
from typing import Optional
from core.logger import get_logger

log = get_logger("apps_registry")

APP_REGISTRY: dict[str, list[str]] = {
    "brave browser": ["brave-browser"],
    "brave": ["brave-browser"],
    "browser": ["brave-browser"],
    "chrome": ["google-chrome"],
    "firefox": ["firefox"],
    "file manager": ["nautilus"],
    "file explorer": ["nautilus"],
    "terminal": ["kitty"],
    "vscode": ["code"],
    "visual studio code": ["code"],
    "spotify": ["spotify"],
    "discord": ["discord"],
    "telegram": ["telegram-desktop"],
    "whatsapp": ["whatsapp-for-linux"],
    "obsidian": ["obsidian"],
    "kalkulator": ["gnome-calculator"],
    "calculator": ["gnome-calculator"],
    "text editor": ["gedit"],
    "pengaturan": ["gnome-control-center"],
    "settings": ["gnome-control-center"],
}

DESTRUCTIVE_ACTIONS: dict[str, list[str]] = {
    "shutdown": ["systemctl", "poweroff"],
    "matikan komputer": ["systemctl", "poweroff"],
    "reboot": ["systemctl", "reboot"],
    "restart komputer": ["systemctl", "reboot"],
    "lock screen": ["hyprlock"],
    "kunci layar": ["hyprlock"],
}

def normalize_app_name(raw_name: str) -> str:
    return raw_name.strip().lower()

def resolve_app(app_name: str) -> Optional[list[str]]:
    normalized = normalize_app_name(app_name)
    argv = APP_REGISTRY.get(normalized)
    if argv is None:
        log.warning("app_not_in_allowlist", requested=normalized)
        return None
    return argv

def is_destructive_action(action: str) -> bool:
    return normalize_app_name(action) in DESTRUCTIVE_ACTIONS

def resolve_destructive_action(action: str) -> Optional[list[str]]:
    normalized = normalize_app_name(action)
    argv = DESTRUCTIVE_ACTIONS.get(normalized)
    if argv is None:
        log.warning("destructive_action_not_found", requested=normalized)
        return None
    return argv

def list_allowed_apps() -> list[str]:
    return sorted(APP_REGISTRY.keys())


# --- Fuzzy matching for misheard / typo'd app names ---------------------------

_DEFAULT_CUTOFF = 0.6
_TOP_ACCEPT = 0.75
_MARGIN_ACCEPT = 0.10


def _norm_for_match(text: Optional[str]) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def suggest_app(raw_name: str, cutoff: float = _DEFAULT_CUTOFF) -> Optional[str]:
    """Closest allowlist key for a misheard/typo'd name, or None when unsure.

    Accepts the top candidate only when it is clearly the best: either a high
    absolute ratio (>= 0.75) or a clear margin (>= 0.10) over the runner-up.
    Otherwise returns None so the caller can ask the LLM instead of guessing."""
    norm = _norm_for_match(raw_name)
    if not norm:
        return None
    matches = difflib.get_close_matches(norm, list(APP_REGISTRY.keys()), n=2, cutoff=cutoff)
    if not matches:
        return None
    scored = sorted(
        ((difflib.SequenceMatcher(None, norm, key).ratio(), key) for key in matches),
        reverse=True,
    )
    top_score, top_key = scored[0]
    if top_score >= _TOP_ACCEPT:
        return top_key
    if len(scored) > 1 and (top_score - scored[1][0]) >= _MARGIN_ACCEPT:
        return top_key
    return None


def extract_app_keys(text: str) -> list[str]:
    """Registry keys mentioned in `text` as whole phrases (longest first, then
    by position). Used to catch 'bukan brave, firefox' → ['firefox', ...]."""
    norm = _norm_for_match(text)
    if not norm:
        return []
    found = [
        key
        for key in APP_REGISTRY
        if re.search(rf"(?<!\w){re.escape(_norm_for_match(key))}(?!\w)", norm)
    ]
    found.sort(key=lambda k: (-len(k), norm.find(_norm_for_match(k))))
    return found


# --- Session cache: raw name -> chosen candidate (or None = "none found") -----

_SUGGEST_CACHE: "OrderedDict[str, Optional[str]]" = OrderedDict()
_SUGGEST_CACHE_MAX = 100


def get_cached_suggestion(raw_name: str) -> tuple[bool, Optional[str]]:
    key = _norm_for_match(raw_name)
    if key and key in _SUGGEST_CACHE:
        _SUGGEST_CACHE.move_to_end(key)
        return True, _SUGGEST_CACHE[key]
    return False, None


def set_cached_suggestion(raw_name: str, candidate: Optional[str]) -> None:
    key = _norm_for_match(raw_name)
    if not key:
        return
    _SUGGEST_CACHE[key] = candidate
    _SUGGEST_CACHE.move_to_end(key)
    while len(_SUGGEST_CACHE) > _SUGGEST_CACHE_MAX:
        _SUGGEST_CACHE.popitem(last=False)


def clear_suggestion_cache() -> None:
    _SUGGEST_CACHE.clear()
