import pytest
from execution.apps import (
    resolve_app, resolve_destructive_action, is_destructive_action,
    list_allowed_apps, normalize_app_name, APP_REGISTRY
)

def test_resolve_known_app():
    assert resolve_app("brave browser") == ["brave-browser"]
    assert resolve_app("terminal") == ["kitty"]
    assert resolve_app("file manager") == ["nautilus"]

def test_resolve_case_insensitive():
    assert resolve_app("BRAVE BROWSER") == ["brave-browser"]
    assert resolve_app("  brave browser  ") == ["brave-browser"]

def test_unknown_app_returns_none():
    assert resolve_app("hacker tool xyz") is None
    assert resolve_app("") is None
    assert resolve_app("rm -rf /") is None

def test_no_shell_injection_in_registry():
    for name, argv in APP_REGISTRY.items():
        joined = " ".join(argv)
        assert ";" not in joined
        assert "|" not in joined
        assert "$" not in joined
        assert "`" not in joined
        assert "&&" not in joined

def test_destructive_actions_detected():
    assert is_destructive_action("shutdown") is True
    assert is_destructive_action("reboot") is True
    assert is_destructive_action("lock screen") is True
    assert is_destructive_action("brave browser") is False

def test_destructive_resolve():
    assert resolve_destructive_action("shutdown") == ["systemctl", "poweroff"]
    assert resolve_destructive_action("brave browser") is None

def test_list_allowed_apps_sorted():
    apps = list_allowed_apps()
    assert apps == sorted(apps)
    assert len(apps) > 10
    assert "brave browser" in apps

def test_normalize_app_name():
    assert normalize_app_name("  BRAVE BROWSER  ") == "brave browser"
    assert normalize_app_name("Spotify") == "spotify"
