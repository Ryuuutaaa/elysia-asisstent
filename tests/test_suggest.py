import pytest

from agent.llm import choose_app
from execution.apps import (
    clear_suggestion_cache,
    extract_app_keys,
    get_cached_suggestion,
    set_cached_suggestion,
    suggest_app,
)


@pytest.mark.parametrize("raw,expected", [
    ("brave brower", "brave browser"),
    ("brav", "brave"),
    ("termnal", "terminal"),
    ("spotfy", "spotify"),
    ("file managr", "file manager"),
])
def test_suggest_app_fuzzy(raw, expected):
    assert suggest_app(raw) == expected


@pytest.mark.parametrize("raw", ["breif", "hacker xyz", ""])
def test_suggest_app_unsure_or_empty(raw):
    assert suggest_app(raw) is None


def test_extract_app_keys_longest_first():
    assert extract_app_keys("bukan brave, firefox")[0] == "firefox"


def test_extract_app_keys_phrase():
    assert extract_app_keys("buka file manager dong") == ["file manager"]


def test_extract_app_keys_none():
    assert extract_app_keys("halo apa kabar") == []


def test_cache_roundtrip():
    clear_suggestion_cache()
    assert get_cached_suggestion("breif") == (False, None)
    set_cached_suggestion("breif", "brave browser")
    assert get_cached_suggestion("breif") == (True, "brave browser")


def test_cache_remembers_none():
    clear_suggestion_cache()
    set_cached_suggestion("zzz", None)
    assert get_cached_suggestion("zzz") == (True, None)


def test_choose_app_returns_validated_candidate(monkeypatch):
    monkeypatch.setattr("agent.llm._plain_generate", lambda prompt: object())
    monkeypatch.setattr("agent.llm.extract_text", lambda resp: "brave browser")
    assert choose_app("breif") == "brave browser"


def test_choose_app_rejects_unknown(monkeypatch):
    monkeypatch.setattr("agent.llm._plain_generate", lambda prompt: object())
    monkeypatch.setattr("agent.llm.extract_text", lambda resp: "some random thing")
    assert choose_app("breif") is None


def test_choose_app_none_answer(monkeypatch):
    monkeypatch.setattr("agent.llm._plain_generate", lambda prompt: object())
    monkeypatch.setattr("agent.llm.extract_text", lambda resp: "NONE")
    assert choose_app("breif") is None


def test_choose_app_empty_raw():
    assert choose_app("") is None
