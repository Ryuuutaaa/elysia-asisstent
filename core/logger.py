import logging
import structlog
import uuid
from core.config import settings

def configure_logging():
    raw_level = settings.LOG_LEVEL.upper() if isinstance(settings.LOG_LEVEL, str) else "INFO"
    level = getattr(logging, raw_level, logging.INFO)
    use_json = raw_level != "DEBUG"
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer() if use_json else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str = ""):
    return structlog.get_logger(name)

def new_session_id() -> str:
    return f"req-{uuid.uuid4().hex[:8]}"
