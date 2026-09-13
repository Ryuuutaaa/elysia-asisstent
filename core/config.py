from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal

class Settings(BaseSettings):
    GOOGLE_API_KEY: str = Field(default="", description="Google AI Studio API key")
    PICOVOICE_ACCESS_KEY: str = Field(default="", description="Picovoice access key for Porcupine")
    WAKE_WORD_ENGINE: Literal["openwakeword", "porcupine"] = Field(default="openwakeword")
    OPENWAKEWORD_MODEL: str = Field(default="hey_jarvis")
    OPENWAKEWORD_MODEL_PATH: str = Field(default="", description="Path ke file model openwakeword custom (mis. hey_elysia.onnx)")
    AUDIO_INPUT_SOURCE: Literal["mic", "speaker", "headset"] = Field(default="mic")
    SAMPLE_RATE: int = Field(default=16000, ge=8000, le=48000)
    VAD_SILENCE_THRESHOLD_MS: int = Field(default=800, ge=300, le=3000)
    VAD_THRESHOLD: float = Field(default=0.5, ge=0.1, le=0.9)
    SILERO_VAD_HUB_REF: str = Field(default="", description="Pin ref torch.hub Silero VAD (mis. tag rilis); kosong = master")
    VAD_MAX_RECORD_MS: int = Field(default=15000, ge=5000, le=60000)
    VAD_NO_SPEECH_GRACE_MS: int = Field(default=5000, ge=1000, le=15000)
    VAD_START_IGNORE_MS: int = Field(default=500, ge=0, le=2000)
    COOLDOWN_SEC: float = Field(default=2.0, ge=0.5, le=10.0)
    CONFIRMATION_TIMEOUT_SEC: float = Field(default=5.0, ge=1.0, le=30.0)
    FOLLOWUP_TIMEOUT_SEC: float = Field(default=5.0, ge=1.0, le=30.0)
    SPEECH_COOLDOWN_SEC: float = Field(default=3.0, ge=0.5, le=10.0, description="Jeda setelah TTS sebelum mic di-unmute (anti-echo)")
    WHISPER_MODEL_SIZE: Literal["tiny", "base", "small", "medium", "large-v3"] = Field(default="small")
    WHISPER_DEVICE: Literal["cpu", "cuda"] = Field(default="cpu")
    WHISPER_COMPUTE_TYPE: Literal["int8", "int8_float16", "float16", "float32"] = Field(default="int8")
    STT_LANGUAGE: str = Field(default="id")
    STT_TIMEOUT_MS: int = Field(default=2500, ge=500, le=30000)
    STT_MIN_RMS: float = Field(default=0.01, ge=0.0, le=0.5)
    TTS_PROVIDER_DEFAULT: Literal["edge", "piper"] = Field(default="edge")
    TTS_VOICE_EDGE: str = Field(default="id-ID-GadisNeural")
    TTS_VOICE_PIPER: str = Field(default="id_ID-news_tts-medium")
    TTS_EDGE_TIMEOUT_SEC: float = Field(default=5.0, ge=1.0, le=30.0)
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    DEBUG_RECORD_AUDIO: bool = Field(default=False)
    GEMINI_MODEL: str = Field(default="gemini-flash-latest", description="Model Gemini; pakai nama yang tersedia untuk API key Anda (fallback otomatis dicoba)")
    LLM_REQUEST_TIMEOUT_MS: int = Field(default=3500, ge=500, le=30000)

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}

settings = Settings()

_PLACEHOLDER_KEYS = {"", "AIzaSy...", "AIzaSy_REPLACE_WITH_YOUR_KEY"}
if not settings.GOOGLE_API_KEY or settings.GOOGLE_API_KEY.strip() in _PLACEHOLDER_KEYS:
    import logging

    logging.getLogger("config").warning(
        "GOOGLE_API_KEY kosong/placeholder — LLM akan gagal. Isi .env dari .env.example"
    )
