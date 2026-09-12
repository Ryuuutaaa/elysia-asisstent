import threading


def run_with_timeout(fn, timeout_sec: float):
    """Run `fn` in a daemon thread and enforce a wall-clock budget.

    Used where the transport/SDK cannot enforce a short enough deadline
    (e.g. the Gemini API rejects deadlines below 10s)."""
    result: dict = {}

    def target():
        try:
            result["value"] = fn()
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_sec)
    if thread.is_alive():
        raise TimeoutError(f"operation exceeded {timeout_sec:.1f}s budget")
    if "error" in result:
        raise result["error"]
    return result["value"]
