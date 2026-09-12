import asyncio
import io
import tempfile
import time
import sounddevice as sd
import numpy as np
from typing import Optional
from core.config import settings
from core.logger import get_logger

log = get_logger("tts")

async def _edge_tts_synthesize(text: str, voice: str) -> Optional[bytes]:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data if audio_data else None

def synthesize_edge(text: str) -> Optional[bytes]:
    try:
        start = time.perf_counter()
        loop = asyncio.new_event_loop()
        try:
            mp3_bytes = loop.run_until_complete(
                _edge_tts_synthesize(text, settings.TTS_VOICE_EDGE)
            )
        finally:
            loop.close()
        latency_ms = (time.perf_counter() - start) * 1000
        if mp3_bytes:
            log.info("tts_edge_synthesized", tts_latency_ms=round(latency_ms, 1), size_bytes=len(mp3_bytes))
        return mp3_bytes
    except Exception as e:
        log.error("tts_edge_failed", error=str(e))
        return None

def synthesize_piper(text: str) -> Optional[bytes]:
    import subprocess
    try:
        start = time.perf_counter()
        result = subprocess.run(
            ["piper", "--model", settings.TTS_VOICE_PIPER, "--output-raw"],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=4.0,
            shell=False,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        if result.returncode == 0 and result.stdout:
            log.info("tts_piper_synthesized", tts_latency_ms=round(latency_ms, 1), size_bytes=len(result.stdout))
            return result.stdout
        log.error("tts_piper_failed", returncode=result.returncode)
        return None
    except Exception as e:
        log.error("tts_piper_failed", error=str(e))
        return None

def play_audio_bytes(audio_bytes: bytes, is_raw_pcm: bool = False):
    try:
        if is_raw_pcm:
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            sd.play(samples, samplerate=22050, blocking=True)
        else:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
                f.write(audio_bytes)
                f.flush()
                import subprocess
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", f.name],
                    shell=False,
                    timeout=15.0,
                )
    except Exception as e:
        log.error("audio_playback_failed", error=str(e))

def speak(text: str):
    start = time.perf_counter()

    if settings.TTS_PROVIDER_DEFAULT == "edge":
        audio = synthesize_edge(text)
        if audio:
            play_audio_bytes(audio, is_raw_pcm=False)
            latency_ms = (time.perf_counter() - start) * 1000
            log.info("tts_playback_finished", provider="edge", total_tts_ms=round(latency_ms, 1))
            return

        log.warning("tts_fallback", from_provider="edge", to_provider="piper")
        audio = synthesize_piper(text)
        if audio:
            play_audio_bytes(audio, is_raw_pcm=True)
            latency_ms = (time.perf_counter() - start) * 1000
            log.info("tts_playback_finished", provider="piper", total_tts_ms=round(latency_ms, 1))
            return
    else:
        audio = synthesize_piper(text)
        if audio:
            play_audio_bytes(audio, is_raw_pcm=True)
            latency_ms = (time.perf_counter() - start) * 1000
            log.info("tts_playback_finished", provider="piper", total_tts_ms=round(latency_ms, 1))
            return

        log.warning("tts_fallback", from_provider="piper", to_provider="edge")
        audio = synthesize_edge(text)
        if audio:
            play_audio_bytes(audio, is_raw_pcm=False)
            latency_ms = (time.perf_counter() - start) * 1000
            log.info("tts_playback_finished", provider="edge", total_tts_ms=round(latency_ms, 1))
            return

    log.error("tts_all_failed", error_code="ERR_TTS_SYNTHESIS", text=text[:100])
    print(f"[Elysia] {text}")
