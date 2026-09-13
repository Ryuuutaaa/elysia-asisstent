import pytest
from unittest.mock import patch, MagicMock
from google.genai import types
from agent.tools import handle_function_call, cancel_action, execute_confirmed_action, is_affirmation, is_denial

@patch("agent.tools.safe_execute", return_value=MagicMock(success=True, message="opened"))
def test_open_valid_app(mock_exec):
    fc = types.FunctionCall(name="open_application", args={"app_name": "brave browser"})
    res = handle_function_call(fc)
    assert "dibuka" in res.text.lower()
    assert res.needs_confirmation is False

def test_open_app_not_in_allowlist():
    fc = types.FunctionCall(name="open_application", args={"app_name": "hacker xyz"})
    res = handle_function_call(fc)
    assert "tidak ada dalam daftar" in res.text
    assert res.needs_confirmation is False

def test_open_empty_app_name():
    fc = types.FunctionCall(name="open_application", args={"app_name": ""})
    res = handle_function_call(fc)
    assert "tidak diberikan" in res.text

@patch("agent.tools.safe_execute", return_value=MagicMock(success=False, message="not found"))
def test_open_app_execution_failed(mock_exec):
    fc = types.FunctionCall(name="open_application", args={"app_name": "brave browser"})
    res = handle_function_call(fc)
    assert "Gagal" in res.text or "tidak ditemukan" in res.text

def test_unknown_tool():
    fc = types.FunctionCall(name="unknown_tool", args={})
    res = handle_function_call(fc)
    assert "tidak dikenali" in res.text

def test_system_action_requires_confirmation():
    fc = types.FunctionCall(name="system_action", args={"action": "shutdown"})
    res = handle_function_call(fc)
    assert res.needs_confirmation is True
    assert res.pending_argv is not None
    assert "konfirmasi" in res.text.lower() or "yakin" in res.text.lower()

def test_system_action_unknown():
    fc = types.FunctionCall(name="system_action", args={"action": "destroy planet"})
    res = handle_function_call(fc)
    assert "tidak ada dalam daftar" in res.text

def test_system_action_empty():
    fc = types.FunctionCall(name="system_action", args={"action": ""})
    res = handle_function_call(fc)
    assert "tidak diberikan" in res.text

def test_cancel_action():
    res = cancel_action("shutdown")
    assert "dibatalkan" in res.text.lower()

def test_execute_confirmed_action_success():
    with patch("agent.tools.safe_execute", return_value=MagicMock(success=True, message="ok")):
        res = execute_confirmed_action(["systemctl", "poweroff"], "shutdown")
        assert "dijalankan" in res.text.lower() or "berhasil" in res.text.lower()

def test_execute_confirmed_action_failed():
    with patch("agent.tools.safe_execute", return_value=MagicMock(success=False, message="permission denied")):
        res = execute_confirmed_action(["systemctl", "poweroff"], "shutdown")
        assert "Gagal" in res.text or "permission" in res.text.lower()

def test_execute_confirmed_action_is_non_blocking():
    # `lock screen` runs hyprlock foreground; it must use the non-blocking
    # executor so no timeout can kill it and unlock the screen again.
    with patch("agent.tools.safe_execute", return_value=MagicMock(success=True, message="ok")) as se:
        execute_confirmed_action(["hyprlock"], "lock screen")
        se.assert_called_once_with(["hyprlock"])


@pytest.mark.parametrize("text", [
    "ya", "Ya", "iya", "oke", "lanjutkan", "benar", "yes", "ya, silakan",
    "yah", "sip", "okelah", "baiklah", "gas", "setuju", "betul.",
])
def test_is_affirmation_positive(text):
    assert is_affirmation(text) is True


@pytest.mark.parametrize("text", [
    "tidak", "batal", "jangan ya", "nggak ya", "stop", "", "   ",
    "gaya rambut", "halo", "gausah", "gak usah", "baik tapi jangan",
])
def test_is_affirmation_negative(text):
    assert is_affirmation(text) is False


@pytest.mark.parametrize("text", ["tidak", "gak", "nggak", "cukup", "selesai", "gausah", "stop", "sudah"])
def test_is_denial_positive(text):
    assert is_denial(text) is True


@pytest.mark.parametrize("text", ["ya", "iya, buka spotify", "buka terminal", "", "   ", "ya tidak"])
def test_is_denial_negative(text):
    assert is_denial(text) is False
