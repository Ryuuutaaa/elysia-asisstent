import threading
import time

import pytest

import core.utils as utils
from core.utils import run_with_timeout, text_fingerprint


def test_run_with_timeout_returns_value():
    assert run_with_timeout(lambda: 42, 1.0) == 42


def test_run_with_timeout_reraises_captured_exception():
    def boom():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        run_with_timeout(boom, 1.0)


def test_run_with_timeout_enforces_budget():
    release = threading.Event()

    def slow():
        release.wait(1.0)
        return "late"

    start = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            run_with_timeout(slow, 0.05)
        assert time.monotonic() - start < 0.5
    finally:
        release.set()


def test_run_with_timeout_rejects_when_budget_saturated():
    limiter = utils._inflight["stt"]
    assert limiter.acquire(blocking=False)
    assert limiter.acquire(blocking=False)
    try:
        with pytest.raises(TimeoutError):
            utils.run_with_timeout(lambda: 1, 0.1, budget="stt")
    finally:
        limiter.release()
        limiter.release()


def test_text_fingerprint_is_stable_and_short():
    digest = text_fingerprint("buka spotify")

    assert digest == text_fingerprint("buka spotify")
    assert digest != text_fingerprint("buka brave")
    assert len(digest) == 8
