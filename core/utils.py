import threading

# A Python thread that exceeds its budget cannot be killed, so it keeps running
# until it returns. Cap how many can be outstanding at once: without this,
# repeated timeouts on a slow network would pile up threads indefinitely.
_MAX_INFLIGHT = 4
_inflight = threading.Semaphore(_MAX_INFLIGHT)


def run_with_timeout(fn, timeout_sec: float):
    """Run `fn` in a daemon thread and enforce a wall-clock budget.

    Used where the transport/SDK cannot enforce a short enough deadline
    (e.g. the Gemini API rejects deadlines below 10s)."""
    if not _inflight.acquire(blocking=False):
        raise TimeoutError(
            f"terlalu banyak operasi yang belum selesai (max {_MAX_INFLIGHT}); coba lagi"
        )

    result: dict = {}

    def target():
        try:
            result["value"] = fn()
        except Exception as e:
            result["error"] = e
        finally:
            _inflight.release()

    thread = threading.Thread(target=target, daemon=True)
    try:
        thread.start()
    except Exception:
        _inflight.release()
        raise

    thread.join(timeout_sec)
    if thread.is_alive():
        raise TimeoutError(f"operation exceeded {timeout_sec:.1f}s budget")
    if "error" in result:
        raise result["error"]
    return result["value"]
