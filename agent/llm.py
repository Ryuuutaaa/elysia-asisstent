import time
from typing import Optional

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from core.config import settings
from core.logger import get_logger
from core.utils import run_with_timeout

log = get_logger("llm")

_client: Optional[genai.Client] = None


_resolved_model: Optional[str] = None
_resolved_for: Optional[str] = None


def scan_available_models(api_key: Optional[str] = None) -> list[str]:
    key = api_key or settings.GOOGLE_API_KEY
    try:
        client = get_client() if key == settings.GOOGLE_API_KEY else genai.Client(api_key=key)
        models = run_with_timeout(
            lambda: [m.name for m in client.models.list()],
            settings.MODEL_SCAN_TIMEOUT_SEC,
            budget="llm",
        )
        return [name.removeprefix("models/") for name in models]
    except Exception as e:
        log.warning("model_scan_failed", error=str(e))
        return []


def resolve_model(preferred: Optional[str] = None) -> str:
    global _resolved_model, _resolved_for
    want = preferred or settings.GEMINI_MODEL
    if _resolved_model and _resolved_for == want:
        return _resolved_model

    candidates = [want]
    for fallback in ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-2.5-flash-lite"]:
        if fallback not in candidates:
            candidates.append(fallback)

    client = get_client()
    last_err = None
    for m in candidates:
        try:
            run_with_timeout(
                lambda model=m: client.models.generate_content(model=model, contents="ping"),
                settings.MODEL_PROBE_TIMEOUT_SEC,
                budget="llm",
            )
            _resolved_model = m
            _resolved_for = want
            if m != want:
                log.warning("model_fallback", requested=want, resolved=m)
            else:
                log.info("model_resolved", model=m)
            return m
        except Exception as e:
            last_err = e
            continue

    _resolved_model = want
    _resolved_for = want
    log.warning("model_resolve_failed", requested=want, error=str(last_err)[:200] if last_err else "")
    return want


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        log.info("gemini_client_initialized")
    return _client


OPEN_APP_TOOL = types.FunctionDeclaration(
    name="open_application",
    description="Membuka aplikasi di komputer berdasarkan nama yang diberikan user. Gunakan nama aplikasi dari daftar yang diizinkan.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "app_name": types.Schema(
                type=types.Type.STRING,
                description="Nama aplikasi yang ingin dibuka (contoh: brave browser, file manager, terminal, spotify)",
            ),
        },
        required=["app_name"],
    ),
)

SYSTEM_ACTION_TOOL = types.FunctionDeclaration(
    name="system_action",
    description="Menjalankan aksi sistem seperti shutdown, reboot, atau lock screen. Aksi ini memerlukan konfirmasi user.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "action": types.Schema(
                type=types.Type.STRING,
                description="Nama aksi sistem (contoh: shutdown, reboot, lock screen)",
            ),
        },
        required=["action"],
    ),
)

TOOLS = types.Tool(function_declarations=[OPEN_APP_TOOL, SYSTEM_ACTION_TOOL])


def _request(user_text: str, system_prompt: str) -> types.GenerateContentResponse:
    client = get_client()
    model = resolve_model()
    start = time.perf_counter()

    response = client.models.generate_content(
        model=model,
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[TOOLS],
            temperature=0.3,
        ),
    )

    latency_ms = (time.perf_counter() - start) * 1000
    log.info("llm_response", llm_latency_ms=round(latency_ms, 1))
    return response


def _plain_generate(prompt: str) -> types.GenerateContentResponse:
    client = get_client()
    model = resolve_model()
    return client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )


def _is_retryable(exc: BaseException) -> bool:
    """Retry transient failures (5xx, timeouts, network). Skip 4xx client errors
    such as 400 INVALID_ARGUMENT, except 429 rate limiting which is retryable."""
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "code", None) == 429
    return True


_MIN_ATTEMPT_SEC = 0.3


def turn_deadline() -> float:
    """One wall-clock budget shared by every LLM call in a single turn."""
    return time.monotonic() + settings.LLM_TOTAL_BUDGET_MS / 1000


def chat(
    user_text: str,
    system_prompt: str,
    deadline: Optional[float] = None,
) -> types.GenerateContentResponse:
    if deadline is None:
        deadline = turn_deadline()
    last: Optional[BaseException] = None
    for attempt in range(settings.LLM_MAX_ATTEMPTS):
        remaining = deadline - time.monotonic()
        if remaining <= _MIN_ATTEMPT_SEC:
            break
        try:
            return run_with_timeout(
                lambda: _request(user_text, system_prompt),
                min(remaining, settings.LLM_REQUEST_TIMEOUT_MS / 1000),
                budget="llm",
            )
        except Exception as e:
            if not _is_retryable(e):
                raise
            last = e
            log.warning(
                "llm_retry",
                attempt=attempt + 1,
                remaining_ms=round(remaining * 1000),
                error=str(e)[:200],
            )
    if last is not None:
        raise last
    raise TimeoutError("LLM budget habis sebelum percobaan pertama")


def extract_function_call(
    response: types.GenerateContentResponse,
) -> Optional[types.FunctionCall]:
    for candidate in response.candidates or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in content.parts or []:
            if part.function_call:
                return part.function_call
    return None


def extract_text(response: types.GenerateContentResponse) -> str:
    parts = []
    for candidate in response.candidates or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in content.parts or []:
            if part.text:
                parts.append(part.text)
    return " ".join(parts).strip()


_CHOOSE_APP_ATTEMPTS = 2


def choose_app(
    raw_name: str,
    source_text: str = "",
    deadline: Optional[float] = None,
) -> tuple[str, Optional[str]]:
    """Ask the LLM to pick the most likely allowlist entry for a misheard name.

    Fallback for when deterministic fuzzy matching is unsure. Returns
    ``(status, candidate)`` where status is:

    - ``"found"`` -> candidate is a valid registry key
    - ``"none"``  -> definitive "no match" (safe to cache)
    - ``"error"`` -> transient failure (timeout/network); must NOT be cached

    The candidate is always validated against the registry, never trusted blindly."""
    from execution.apps import list_allowed_apps, resolve_app

    raw = (raw_name or "").strip()
    if not raw:
        return "none", None
    names = list_allowed_apps()
    heard = (source_text or "").strip() or raw
    if deadline is None:
        deadline = turn_deadline()
    prompt = (
        f"User mengucapkan: {heard!r}.\n"
        f"Kata aplikasi yang tertangkap: {raw!r} (mungkin salah dengar atau typo).\n"
        f"Daftar aplikasi yang tersedia: {', '.join(names)}.\n"
        "Pilih SATU nama dari daftar yang paling mungkin maksudnya. "
        "Jawab HANYA nama tersebut, atau NONE bila tidak ada yang cocok."
    )
    for attempt in range(_CHOOSE_APP_ATTEMPTS):
        remaining = deadline - time.monotonic()
        if remaining <= _MIN_ATTEMPT_SEC:
            break
        try:
            response = run_with_timeout(
                lambda: _plain_generate(prompt),
                min(remaining, settings.LLM_REQUEST_TIMEOUT_MS / 1000),
                budget="llm",
            )
        except Exception as e:
            log.warning("choose_app_failed", error=str(e), attempt=attempt + 1)
            continue

        answer = extract_text(response).strip().strip("\"'").lower()
        if not answer or answer == "none":
            return "none", None
        if resolve_app(answer) is not None:
            return "found", answer
        return "none", None

    return "error", None
