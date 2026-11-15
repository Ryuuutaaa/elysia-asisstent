import os
import subprocess
import shutil
import time

from core.logger import get_logger
from execution.apps import APP_REGISTRY, DESTRUCTIVE_ACTIONS

log = get_logger("linux_executor")

_SECRET_ENV_KEYS = ("GOOGLE_API_KEY", "PICOVOICE_ACCESS_KEY")

# Only argv shapes that appear in the registries may ever be spawned. Keeping
# this next to Popen makes the boundary structural: a new caller cannot bypass
# the allowlist by calling safe_execute directly.
_ALLOWED_ARGV: frozenset[tuple[str, ...]] = frozenset(
    tuple(argv) for argv in (*APP_REGISTRY.values(), *DESTRUCTIVE_ACTIONS.values())
)


class ExecutionResult:
    def __init__(self, success: bool, message: str, latency_ms: float):
        self.success = success
        self.message = message
        self.latency_ms = latency_ms


def is_allowlisted(argv: list[str]) -> bool:
    return tuple(argv) in _ALLOWED_ARGV


def _child_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in _SECRET_ENV_KEYS}


def safe_execute(argv: list[str], timeout: float = 5.0) -> ExecutionResult:
    if not argv:
        return ExecutionResult(False, "Perintah kosong.", 0.0)

    if not is_allowlisted(argv):
        log.error("argv_not_allowlisted", command=argv)
        return ExecutionResult(False, "Perintah tidak diizinkan.", 0.0)

    binary = argv[0]
    binary_path = shutil.which(binary)
    if binary_path is None:
        log.error("binary_not_found", binary=binary)
        return ExecutionResult(False, f"Program {binary} tidak ditemukan di sistem.", 0.0)

    start = time.perf_counter()
    try:
        process = subprocess.Popen(
            argv,
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=_child_env(),
        )
        try:
            probe_timeout = min(timeout, 0.5) if timeout else 0.5
            ret = process.wait(timeout=probe_timeout)
            latency_ms = (time.perf_counter() - start) * 1000
            if ret != 0:
                log.error("tool_immediate_exit", command=argv, returncode=ret)
                return ExecutionResult(False, f"{binary} keluar dengan kode {ret}.", latency_ms)
        except subprocess.TimeoutExpired:
            latency_ms = (time.perf_counter() - start) * 1000
            log.info("tool_executed", command=argv, pid=process.pid, exec_latency_ms=round(latency_ms, 1), status="ok")
            return ExecutionResult(True, f"{binary} berhasil dibuka.", latency_ms)

        log.info("tool_executed", command=argv, pid=process.pid, exec_latency_ms=round(latency_ms, 1), status="ok")
        return ExecutionResult(True, f"{binary} berhasil dibuka.", latency_ms)
    except PermissionError:
        latency_ms = (time.perf_counter() - start) * 1000
        log.error("permission_denied", command=argv)
        return ExecutionResult(False, f"Tidak ada izin untuk menjalankan {binary}.", latency_ms)
    except OSError as e:
        latency_ms = (time.perf_counter() - start) * 1000
        log.error("execution_os_error", command=argv, error=str(e))
        return ExecutionResult(False, f"Gagal menjalankan {binary}: {e}", latency_ms)
