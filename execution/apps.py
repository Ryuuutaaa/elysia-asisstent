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
