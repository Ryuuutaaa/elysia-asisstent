import subprocess
import shutil
import time
from typing import Optional
from core.logger import get_logger

log = get_logger("linux_executor")

class ExecutionResult:
    def __init__(self, success: bool, message: str, latency_ms: float):
        self.success = success
        self.message = message
        self.latency_ms = latency_ms

def safe_execute(argv: list[str], timeout: float = 5.0) -> ExecutionResult:
    if not argv:
        return ExecutionResult(False, "Perintah kosong.", 0.0)

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
        )
        latency_ms = (time.perf_counter() - start) * 1000

        log.info(
            "tool_executed",
            command=argv,
            pid=process.pid,
            exec_latency_ms=round(latency_ms, 1),
            status="ok",
        )
        return ExecutionResult(True, f"{binary} berhasil dibuka.", latency_ms)
    except PermissionError:
        latency_ms = (time.perf_counter() - start) * 1000
        log.error("permission_denied", command=argv)
        return ExecutionResult(False, f"Tidak ada izin untuk menjalankan {binary}.", latency_ms)
    except OSError as e:
        latency_ms = (time.perf_counter() - start) * 1000
        log.error("execution_os_error", command=argv, error=str(e))
        return ExecutionResult(False, f"Gagal menjalankan {binary}: {e}", latency_ms)

def safe_execute_blocking(argv: list[str], timeout: float = 10.0) -> ExecutionResult:
    if not argv:
        return ExecutionResult(False, "Perintah kosong.", 0.0)

    start = time.perf_counter()
    try:
        result = subprocess.run(
            argv,
            shell=False,
            capture_output=True,
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        if result.returncode == 0:
            log.info("blocking_tool_executed", command=argv, exec_latency_ms=round(latency_ms, 1))
            return ExecutionResult(True, "Perintah berhasil dijalankan.", latency_ms)
        else:
            stderr = result.stderr.decode(errors="replace").strip()[:200]
            log.error("blocking_tool_failed", command=argv, returncode=result.returncode, stderr=stderr)
            return ExecutionResult(False, f"Perintah gagal: {stderr}", latency_ms)
    except subprocess.TimeoutExpired:
        latency_ms = (time.perf_counter() - start) * 1000
        log.error("blocking_tool_timeout", command=argv, timeout=timeout)
        return ExecutionResult(False, "Perintah timeout.", latency_ms)
    except OSError as e:
        latency_ms = (time.perf_counter() - start) * 1000
        log.error("blocking_tool_os_error", command=argv, error=str(e))
        return ExecutionResult(False, f"Gagal: {e}", latency_ms)

def hyprctl_dispatch(action: str, args: str = "") -> ExecutionResult:
    argv = ["hyprctl", "dispatch", action]
    if args:
        argv.append(args)
    return safe_execute_blocking(argv, timeout=3.0)
