import hashlib
import threading

# A Python thread that exceeds its budget cannot be killed, so it keeps running
# until it returns. Cap how many can be outstanding at once: without this,
# repeated timeouts on a slow network would pile up threads indefinitely.
# LLM and STT have separate budgets so a hung LLM call cannot starve transcription.
_LIMITS = {
    "llm": 2,
    "stt": 2,
}
_inflight = {name: threading.Semaphore(limit) for name, limit in _LIMITS.items()}


def run_with_timeout(fn, timeout_sec: float, budget: str = "llm"):
    """Run `fn` in a daemon thread and enforce a wall-clock budget.

    Used where the transport/SDK cannot enforce a short enough deadline
    (e.g. the Gemini API rejects deadlines below 10s)."""
    limiter = _inflight[budget]
    if not limiter.acquire(blocking=False):
        raise TimeoutError(
            f"terlalu banyak operasi {budget} yang belum selesai (max {_LIMITS[budget]}); coba lagi"
        )

    result: dict = {}

    def target():
        try:
            result["value"] = fn()
        except Exception as e:
            result["error"] = e
        finally:
            limiter.release()

    thread = threading.Thread(target=target, daemon=True)
    try:
        thread.start()
    except Exception:
        limiter.release()
        raise

    thread.join(timeout_sec)
    if thread.is_alive():
        raise TimeoutError(f"operation exceeded {timeout_sec:.1f}s budget")
    if "error" in result:
        raise result["error"]
    return result["value"]


def text_fingerprint(text: str) -> str:
    """Short stable hash usable in INFO logs instead of the raw transcript."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
