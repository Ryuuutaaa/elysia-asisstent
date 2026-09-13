import subprocess
import time
import numpy as np
import sounddevice as sd
from typing import Optional
from core.config import settings
from core.logger import get_logger
from audio.recorder import enumerate_audio_sources, resolve_pulse_source

log = get_logger("diagnostics")

def _run(cmd: list[str], timeout: float = 3.0) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, shell=False)
        if r.returncode == 0:
            return r.stdout.decode(errors="replace").strip()
    except Exception:
        pass
    return ""

def _bar(val: float, width: int = 20) -> str:
    filled = int(max(0, min(1, val)) * width)
    return "█" * filled + "░" * (width - filled)

def diagnose_speakers() -> list[dict]:
    sinks = []
    out = _run(["pactl", "list", "short", "sinks"])
    default = _run(["pactl", "get-default-sink"])
    if out:
        for line in out.split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                sinks.append({"name": parts[1], "default": parts[1] == default, "state": parts[-1] if len(parts) > 4 else ""})
    try:
        for i, d in enumerate(sd.query_devices()):
            if d["max_output_channels"] > 0:
                sinks.append({"sd_index": i, "sd_name": d["name"], "channels": d["max_output_channels"]})
                break
    except Exception:
        pass
    return sinks

def diagnose_mics() -> list[dict]:
    sources = enumerate_audio_sources()
    default = _run(["pactl", "get-default-source"])
    enriched = []
    for s in sources:
        enriched.append({**s, "default": s["name"] == default, "is_monitor": "monitor" in s["name"]})
    try:
        vols = _run(["pactl", "list", "sources"])
        for e in enriched:
            if e["name"] in vols:
                idx = vols.find(e["name"])
                chunk = vols[max(0, idx - 500): idx + 1500]
                for line in chunk.split("\n"):
                    if "Volume:" in line and "mono" in line.lower():
                        e["volume_line"] = line.strip()
                    if "Mute:" in line:
                        e["mute"] = "yes" in line.lower()
                    if "State:" in line:
                        e["state"] = line.split(":")[-1].strip()
    except Exception:
        pass
    return enriched

def test_mic_quality(device_index: Optional[int], duration: float = 2.5) -> dict:
    buf: list[np.ndarray] = []
    done = [False]
    def cb(indata, frames, t, status):
        if not done[0]:
            buf.append(indata[:, 0].copy())
    try:
        stream = sd.InputStream(samplerate=settings.SAMPLE_RATE, blocksize=512, channels=1, dtype="int16", device=device_index, callback=cb)
        stream.start()
        time.sleep(duration)
        done[0] = True
        stream.stop()
        stream.close()
    except Exception as e:
        return {"ok": False, "error": str(e)}
    if not buf:
        return {"ok": False, "error": "no audio captured"}
    audio = np.concatenate(buf).astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    peak_dbfs = 20 * np.log10(max(peak, 1e-6))
    rms_dbfs = 20 * np.log10(max(rms, 1e-6))
    clipped = int(np.sum(np.abs(audio) >= 0.99))
    clip_pct = clipped / len(audio) * 100 if len(audio) else 0
    if rms < 0.005:
        verdict = "TERLALU PELAN — mic jauh / volume kecil / STT akan kosong"
        level = "warn"
    elif rms < 0.015:
        verdict = "PELAN — masih bisa STT tapi threshold STT_MIN_RMS mungkin drop"
        level = "warn"
    elif clip_pct > 1.0:
        verdict = "CLIPPING — mic terlalu keras / dekat, turunkan volume"
        level = "warn"
    elif rms < 0.12:
        verdict = "BAGUS — level ideal untuk Whisper"
        level = "ok"
    else:
        verdict = "KERAS — agak loud tapi masih ok"
        level = "ok"
    return {"ok": True, "rms": rms, "peak": peak, "rms_dbfs": rms_dbfs, "peak_dbfs": peak_dbfs, "clip_pct": clip_pct, "samples": len(audio), "duration": duration, "verdict": verdict, "level": level, "raw": audio}

def try_speaker_beep(duration: float = 0.4, freq: float = 880.0) -> dict:
    try:
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration), False)
        tone = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
        sd.play(tone, samplerate=sr, blocking=True)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def run_preflight_diagnostics(device_index: Optional[int], interactive: bool = True) -> bool:
    print("\n" + "=" * 58)
    print("  ELYSIA PRE-FLIGHT DIAGNOSTICS")
    print("=" * 58)

    sel_source = resolve_pulse_source(settings.AUDIO_INPUT_SOURCE)
    sel_sink = _run(["pactl", "get-default-sink"])
    print(f"\n  Config AUDIO_INPUT_SOURCE = {settings.AUDIO_INPUT_SOURCE}")
    print(f"  Pulse source terpilih      : {sel_source or '- (system default)'}")
    print(f"  Pulse sink  terpilih      : {sel_sink or '-'}")
    print(f"  sounddevice index          : {device_index}")
    try:
        if device_index is not None:
            info = sd.query_devices(device_index)
            print(f"  Device info                : {info['name']}  ({info['max_input_channels']} in / {info['max_output_channels']} out)")
    except Exception as e:
        print(f"  Device info                : (query gagal: {e})")

    print("\n  ── Speaker (sink) ──")
    sinks = diagnose_speakers()
    if not sinks:
        print("    (!) tidak ada sink terdeteksi via pactl")
    for s in sinks[:6]:
        mark = "● default" if s.get("default") else " "
        print(f"    {mark}  {s.get('name') or s.get('sd_name')}")

    print("\n  ── Mic (source) ──")
    mics = diagnose_mics()
    for m in mics[:8]:
        mark = "●" if m.get("default") else " "
        sel = " ← TERPILIH" if m["name"] == sel_source else ""
        mon = " [monitor]" if m.get("is_monitor") else ""
        vol = m.get("volume_line", "")
        mute = " MUTED!" if m.get("mute") else ""
        print(f"    {mark} {m['name']}{mon}{sel}{mute}")
        if vol:
            print(f"      {vol}")

    if interactive:
        print("\n  ── Tes kualitas mic (2.5 detik) ──")
        print("    Jangan bicara dulu — ukur noise lantai...")
        time.sleep(0.6)
        r_silence = test_mic_quality(device_index, duration=2.0)
        if not r_silence["ok"]:
            print(f"    GAGAL: {r_silence['error']}")
            print("    Cek: mic terpasang? pactl list sources, wpctl status")
            return False
        print(f"    Silence  RMS={r_silence['rms']:.4f} ({r_silence['rms_dbfs']:.1f} dBFS)  Peak={r_silence['peak']:.3f}  {_bar(r_silence['rms']*8)}")
        print(f"    → {r_silence['verdict']}")
        if r_silence["rms"] > 0.02:
            print("    (!) Noise lantai tinggi — ruangan bising / mic sensitif, VAD bisa false-trigger")

        print("\n    Sekarang BICARA normal 2.5 detik: 'Hey Jarvis buka brave browser'...")
        for i in range(3, 0, -1):
            print(f"      {i}...", flush=True)
            time.sleep(0.7)
        print("    ● REKAM...")
        r_speech = test_mic_quality(device_index, duration=2.5)
        print(f"    Speech   RMS={r_speech['rms']:.4f} ({r_speech['rms_dbfs']:.1f} dBFS)  Peak={r_speech['peak']:.3f}  {_bar(r_speech['rms']*5)}")
        print(f"    Clip={r_speech['clip_pct']:.2f}%  → {r_speech['verdict']}")
        snr = r_speech["rms"] / max(r_silence["rms"], 1e-6)
        print(f"    SNR≈ {snr:.1f}x  ({20*np.log10(max(snr,1e-6)):.1f} dB)")
        if r_speech["rms"] < 0.015:
            print("    (!) SUARA PELAN — STT akan 'tidak mendengar'. Dekatkan mic / naikkan wpctl/pavucontrol.")
            print(f"        STT_MIN_RMS saat ini {settings.STT_MIN_RMS} — turunkan ke 0.005 jika tetap pelan.")
        if r_speech["level"] == "ok" and r_silence["rms"] < 0.02 and snr > 4:
            print("    ✓ Mic OK — lanjut ke Elysia.")
        else:
            print("    ! Mic marginal — coba lagi atau cek pavucontrol / wpctl set-volume @DEFAULT_SOURCE@ 1.0")

        print("\n  ── Tes speaker (beep 880Hz 0.4s) ──")
        rb = try_speaker_beep()
        if rb["ok"]:
            print("    ✓ Beep keluar — speaker OK. Dengar beep? Jika tidak, cek sink default.")
        else:
            print(f"    GAGAL beep: {rb['error']}")
            print("    Cek: pactl get-default-sink, wpctl status")

    print("\n" + "=" * 58)
    print("  Diagnostics selesai — masuk Elysia...\n")
    log.info("diagnostics_complete", source=sel_source, sink=sel_sink, device_index=device_index)
    return True
