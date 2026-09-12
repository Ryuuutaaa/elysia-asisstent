import time
from typing import Optional

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from core.config import settings
from core.logger import get_logger
from core.utils import run_with_timeout

log = get_logger("llm")

_client: Optional[genai.Client] = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        log.info("gemini_client_initialized", model=settings.GEMINI_MODEL)
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
    start = time.perf_counter()

    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
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


def _is_retryable(exc: BaseException) -> bool:
    """Retry transient failures (5xx, timeouts, network). Skip 4xx client errors
    such as 400 INVALID_ARGUMENT, except 429 rate limiting which is retryable."""
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "code", None) == 429
    return True


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=3),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
def chat(user_text: str, system_prompt: str) -> types.GenerateContentResponse:
    timeout_sec = settings.LLM_REQUEST_TIMEOUT_MS / 1000
    return run_with_timeout(lambda: _request(user_text, system_prompt), timeout_sec)


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
