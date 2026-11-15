# Flow: Elysia Personal Voice Assistant

Dokumen ini menjelaskan alur kerja sistem Elysia secara detail — dari startup, state machine, data flow antar modul, error handling, hingga shutdown. Disusun berdasarkan `PRD.md`, `note.md`, dan `requirements.txt`.

---

## 1. Startup Flow (Inisialisasi Sistem)

```
main.py
  │
  ├─ 1. Load Configuration
  │     core/config.py → Settings(BaseSettings)
  │     ├─ Baca .env (GOOGLE_API_KEY, PICOVOICE_ACCESS_KEY, AUDIO_INPUT_SOURCE, ...)
  │     ├─ Validasi schema via pydantic-settings
  │     │   ├─ GOOGLE_API_KEY: str (wajib, gagal = exit)
  │     │   ├─ PICOVOICE_ACCESS_KEY: str (default "")
  │     │   ├─ AUDIO_INPUT_SOURCE: "mic" | "speaker" | "headset" (default "mic")
  │     │   ├─ WHISPER_MODEL_SIZE: "small" | "medium" (default "small")
  │     │   ├─ TTS_PROVIDER_DEFAULT: "edge" | "piper" (default "edge")
  │     │   ├─ LOG_LEVEL: "DEBUG" | "INFO" | "WARNING" | "ERROR" (default "INFO")
  │     │   └─ DEBUG_RECORD_AUDIO: bool (default False)
  │     └─ Jika validasi gagal → log CRITICAL + exit(1)
  │
  ├─ 2. Init Structured Logger
  │     core/logger.py → structlog
  │     ├─ Format: JSON (production) / colorful console (development)
  │     ├─ Bind session_id unik per lifecycle interaksi
  │     └─ Log: {"level": "info", "event": "startup_begin", ...}
  │
  ├─ 3. Enumerate Audio Devices
  │     audio/recorder.py
  │     ├─ Jalankan: pactl list short sources
  │     ├─ Cocokkan AUDIO_INPUT_SOURCE dari config
  │     │   ├─ "mic"     → default input device
  │     │   ├─ "speaker" → cari alsa_output...monitor (PipeWire loopback)
  │     │   └─ "headset" → cari USB/BT input device
  │     ├─ Jika device tidak ditemukan:
  │     │   ├─ Retry re-enumerate 3x (interval 1s)
  │     │   ├─ Jika tetap gagal → log CRITICAL ERR_AUDIO_INPUT
  │     │   └─ Fallback ke system default source
  │     └─ Log: {"event": "audio_device_selected", "device": "...", "source_type": "mic"}
  │
  ├─ 4. Load Whisper Model (Singleton, Cold Start)
  │     audio/stt.py → faster-whisper (CTranslate2)
  │     ├─ WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type="int8")
  │     ├─ Budget: ≤ 8 detik load time untuk model "small"
  │     ├─ RAM: ~500MB (small) / ~1.5GB (medium)
  │     └─ Model di-load SEKALI, tidak per-request (singleton pattern)
  │
  ├─ 5. Init Wake Word (Singleton, Cold Start)
  │     audio/wake_word.py
  │     ├─ Engine default: openWakeWord (offline) → model dari OPENWAKEWORD_MODEL
  │     │   (bawaan "hey_jarvis" → frasa "Hey Jarvis"; custom via OPENWAKEWORD_MODEL_PATH)
  │     ├─ Engine alternatif: Porcupine → pvporcupine.create(access_key, keyword_paths=["hei_elysia.ppn"])
  │     ├─ Budget: ≤ 1 detik load time
  │     ├─ RAM: < 50MB
  │     └─ Singleton — tidak di-recreate per cycle
  │
  ├─ 6. Load Silero VAD Model
  │     audio/vad.py → torch.hub
  │     ├─ torch.hub.load("snakers4/silero-vad", "silero_vad")
  │     └─ Cache: model di-download sekali, cache di ~/.cache/torch/hub/
  │
  ├─ 7. Init TTS Providers
  │     audio/tts.py
  │     ├─ Primary: Edge TTS (id-ID-ArdiNeural / id-ID-GadisNeural) — online
  │     ├─ Fallback: Piper TTS (id_ID news_tts) — offline
  │     └─ Tentukan provider default dari TTS_PROVIDER_DEFAULT config
  │
  ├─ 8. Init Gemini LLM Client
  │     agent/llm.py → google-genai SDK
  │     ├─ genai.configure(api_key=GOOGLE_API_KEY)
  │     ├─ Model: gemini-2.0-flash
  │     └─ System prompt di-load dari agent/prompt.py
  │
  ├─ 9. Allowlist Registry (dibaca saat build_system_prompt)
  │     execution/apps.py
  │     ├─ Dictionary: {"brave browser": ["brave-browser"], "file manager": ["nautilus"], ...}
  │     └─ Value = list[str] (argv subprocess, shell=False); di-resolve saat tool call
  │
  ├─ 10. Init State Machine
  │      core/state.py → FSM
  │      ├─ States: IDLE, LISTENING, PROCESSING, SPEAKING
  │      ├─ Initial state: IDLE
  │      └─ Log: {"event": "startup_complete", "state": "IDLE", "startup_ms": ...}
  │
  └─ ★ Sistem siap. Masuk IDLE loop.
```

**Total startup budget: ≤ 12 detik** (Whisper ≤8s + Porcupine ≤1s + lainnya ≤3s).

---

## 2. State Machine (FSM)

```
                    ┌──────────────────────────────────┐
                    │                                  │
                    ▼                                  │
              ┌──────────┐    wake word detected  ┌────┴─────┐
              │          │ ──────────────────────► │          │
              │   IDLE   │                        │ LISTENING │
              │          │ ◄─── error recovery ── │          │
              └──────────┘                        └────┬─────┘
                    ▲                                  │
                    │                           VAD silence
                    │                           detected
                    │                                  │
                    │                                  ▼
              ┌─────┴────┐                      ┌──────────┐
              │          │ ◄──── processing ─── │          │
              │ SPEAKING │      complete        │PROCESSING│
              │          │                      │          │
              └──────────┘                      └──────────┘
                    │
                    │ playback selesai
                    │ + cooldown 2s
                    │
                    ▼
              ┌──────────┐
              │   IDLE   │  (kembali listen wake word)
              └──────────┘
```

### Transisi Valid

| From | Event | To | Guard |
|---|---|---|---|
| `IDLE` | Wake word detected | `LISTENING` | Cooldown timer expired (≥2s sejak terakhir SPEAKING→IDLE) |
| `LISTENING` | VAD silence ≥ 700–900ms ATAU max duration 15s | `PROCESSING` | Audio buffer tidak kosong |
| `PROCESSING` | STT + LLM + Tool selesai | `SPEAKING` | Response text tersedia |
| `SPEAKING` | Audio playback selesai | `IDLE` | Cooldown timer di-start (2–3s) |
| `SPEAKING` | Aksi destruktif menunggu konfirmasi | `LISTENING` | Tanpa wake word; timeout konfirmasi 5s (`CONFIRMATION_TIMEOUT_SEC`) |
| `*` (any) | Unhandled exception | `IDLE` | Log error + verbal error jika memungkinkan; recovery ≤ 2s |

### Transisi DITOLAK (Invalid)

| From | Event | Aksi |
|---|---|---|
| `LISTENING` | Wake word detected | Drop / ignore (sudah dalam recording) |
| `PROCESSING` | Wake word detected | Drop / ignore (sedang proses) |
| `SPEAKING` | Wake word detected | Drop / ignore (mic muted + cooldown aktif) |
| `SPEAKING` | VAD event | Drop / ignore (mic muted) |

---

## 3. Happy Path: Alur Normal Perintah "Buka Brave Browser"

```
User berkata: "Hei Elysia, buka brave browser"

Timeline (target E2E ≤ 3.0s dari user selesai bicara):
─────────────────────────────────────────────────────────

[t=0.0s]  IDLE
          │
          │  Porcupine mendeteksi "Hei Elysia" (≤200ms processing)
          │  Log: {"event": "wake_word_detected", "latency_ms": 180}
          ▼
[t=0.2s]  IDLE → LISTENING
          │
          │  sounddevice mulai buffer audio stream
          │  Format: 16kHz, mono, 16-bit PCM
          │  User melanjutkan: "...buka brave browser"
          │
          │  Silero VAD memantau frame-by-frame:
          │    - speech_prob > 0.5 → frame = speech
          │    - speech_prob < 0.3 selama ≥ 800ms → silence detected
          │
          │  Log: {"event": "vad_silence_detected", "vad_latency_ms": 280,
          │        "audio_duration_ms": 2100}
          ▼
[t=2.5s]  LISTENING → PROCESSING
          │
          │  ┌─ Stage 1: STT (≤1.0s budget) ─────────────────────────┐
          │  │  audio buffer → faster-whisper.transcribe()            │
          │  │  Input: numpy array float32, 16kHz mono                │
          │  │  Output: "buka brave browser" (text)                   │
          │  │  Log: {"event": "stt_completed", "stt_latency_ms": 650,│
          │  │        "text": "buka brave browser",                   │
          │  │        "language": "id", "confidence": 0.94}           │
          │  └────────────────────────────────────────────────────────┘
          │
          │  ┌─ Stage 2: LLM Intent + Tool Call (≤1.2s budget) ──────┐
          │  │  text → Gemini 3.8 Flash (google-genai SDK)            │
          │  │                                                        │
          │  │  System Prompt (agent/prompt.py):                      │
          │  │    "Kamu adalah Elysia, asisten suara. Kamu hanya      │
          │  │     boleh memanggil tool dari daftar yang tersedia.     │
          │  │     Jangan pernah membuat perintah shell sendiri."      │
          │  │                                                        │
          │  │  Gemini response (function calling):                   │
          │  │    tool_call: open_application(app_name="brave browser")│
          │  │                                                        │
          │  │  Log: {"event": "llm_tool_call", "llm_latency_ms": 980,│
          │  │        "tool": "open_application",                     │
          │  │        "args": {"app_name": "brave browser"}}          │
          │  └────────────────────────────────────────────────────────┘
          │
          │  ┌─ Stage 3: Tool Execution (≤300ms budget) ─────────────┐
          │  │  agent/tools.py → execution/apps.py                    │
          │  │                                                        │
          │  │  Allowlist lookup:                                     │
          │  │    "brave browser" → ["brave-browser"]  ✓ FOUND       │
          │  │                                                        │
          │  │  execution/linux.py:                                   │
          │  │    subprocess.Popen(["brave-browser"], shell=False)    │
          │  │                                                        │
          │  │  Log: {"event": "tool_executed", "exec_latency_ms": 45,│
          │  │        "command": ["brave-browser"], "status": "ok"}   │
          │  └────────────────────────────────────────────────────────┘
          │
          │  Gemini menyusun respons: "Brave browser sudah dibuka."
          ▼
[t=4.7s]  PROCESSING → SPEAKING
          │
          │  ★ Mic MUTED + Porcupine PAUSED (anti-echo)
          │
          │  ┌─ Stage 4: TTS Synthesis + Playback (≤500ms first-byte)┐
          │  │  text → Edge TTS (id-ID-GadisNeural)                  │
          │  │  Output: audio stream → sounddevice playback           │
          │  │                                                        │
          │  │  Elysia berkata: "Brave browser sudah dibuka."        │
          │  │                                                        │
          │  │  Log: {"event": "tts_playback_started",               │
          │  │        "tts_latency_ms": 380, "provider": "edge"}     │
          │  └────────────────────────────────────────────────────────┘
          │
          │  Playback selesai
          │  ★ Mic UNMUTED + Porcupine RESUMED (setelah cooldown 2s)
          ▼
[t=7.5s]  SPEAKING → IDLE
          │
          │  Log: {"event": "cycle_complete",
          │        "total_e2e_latency_ms": 2850,
          │        "session_id": "req-abc123"}
          │
          └─ Kembali mendengarkan wake word...
```

### Latency Budget Breakdown (p95 Target)

```
┌────────────────────┬───────────┬──────────────────────────┐
│ Stage              │ Budget    │ Modul                    │
├────────────────────┼───────────┼──────────────────────────┤
│ Wake word detect   │ ≤ 200ms   │ audio/wake_word.py       │
│ VAD endpointing    │ ≤ 300ms   │ audio/vad.py             │
│ STT transcription  │ ≤ 1000ms  │ audio/stt.py             │
│ LLM + tool call    │ ≤ 1200ms  │ agent/llm.py + tools.py  │
│ Tool execution     │ ≤ 300ms   │ execution/linux.py       │
│ TTS first-byte     │ ≤ 500ms   │ audio/tts.py             │
├────────────────────┼───────────┼──────────────────────────┤
│ TOTAL E2E          │ ≤ 3000ms  │ (user diam → audio out)  │
└────────────────────┴───────────┴──────────────────────────┘
```

---

## 4. Data Flow Antar Modul (Tipe Data)

```
┌──────────────┐   numpy.ndarray   ┌──────────────┐   str (text)    ┌──────────────┐
│              │   float32, 16kHz  │              │   "buka brave  │              │
│  recorder.py │ ────────────────► │   stt.py     │ ──────────────►│   llm.py     │
│  (sounddevice│   mono, PCM       │(faster-whisper│   browser"     │ (Gemini SDK) │
│   buffer)    │                   │  .transcribe) │               │              │
└──────────────┘                   └──────────────┘               └──────┬───────┘
       ▲                                  ▲                              │
       │                                  │                    FunctionCall
       │ int16 frames                     │ numpy.ndarray       {"tool": "open_application",
       │ (512 samples/frame)              │ (same buffer)        "args": {"app_name": "..."}}
       │                                  │                              │
┌──────┴───────┐                   ┌──────┴───────┐               ┌──────▼───────┐
│              │  keyword_index    │              │               │              │
│ wake_word.py │ ────────────────► │   vad.py     │               │  tools.py    │
│ (Porcupine   │  (0 = detected)  │ (Silero VAD  │               │ (function    │
│  .process())  │                   │  .predict()) │               │  registry)   │
└──────────────┘                   └──────────────┘               └──────┬───────┘
                                                                         │
                                                          app_name: str  │
                                                          ┌──────────────▼──┐
                                                          │                 │
                                                          │    apps.py      │
                                                          │  (allowlist     │
                                                          │   lookup)       │
                                                          └────────┬────────┘
                                                                   │
                                                        argv: list[str]
                                                        ["brave-browser"]
                                                                   │
                                                          ┌────────▼────────┐
                                                          │                 │
                                                          │   linux.py      │
                                                          │ subprocess.Popen│
                                                          │ (shell=False)   │
                                                          └────────┬────────┘
                                                                   │
                                                          result: str
                                                          "Brave browser
                                                           sudah dibuka."
                                                                   │
                                                          ┌────────▼────────┐
                                                          │                 │
                                                          │    tts.py       │
                                                          │  Edge TTS /     │
                                                          │  Piper fallback │
                                                          │  → audio stream │
                                                          │  → playback     │
                                                          └─────────────────┘
```

---

## 5. Error & Fallback Flows

### 5.1 Network Failure — Gemini API

```
PROCESSING state: STT selesai, kirim teks ke Gemini
          │
          ├─ Request ke Gemini API
          │   ├─ Timeout: 3.5 detik per request
          │   ├─ Retry policy (tenacity):
          │   │   @retry(stop=stop_after_attempt(3),
          │   │          wait=wait_exponential(multiplier=1, min=1, max=3))
          │   │
          │   ├─ Attempt 1: timeout/5xx → wait 1s
          │   ├─ Attempt 2: timeout/5xx → wait 2s
          │   └─ Attempt 3: timeout/5xx → GAGAL
          │
          ├─ Setelah 3x gagal:
          │   ├─ Log: {"event": "llm_failed", "error_code": "ERR_LLM_TIMEOUT",
          │   │        "attempts": 3, "total_retry_ms": 9500}
          │   ├─ Response text = "Maaf, koneksi ke otak Elysia sedang terganggu."
          │   └─ Transisi tetap ke SPEAKING (verbal error)
          │
          └─ SPEAKING → TTS verbal error → IDLE
             (daemon TIDAK crash, kembali mendengarkan)
```

### 5.2 Network Failure — Edge TTS

```
SPEAKING state: Teks respons siap, kirim ke TTS
          │
          ├─ Primary: Edge TTS
          │   ├─ Timeout: 5.0 detik (WebSocket connection)
          │   ├─ Retry: 1x
          │   │
          │   ├─ Attempt 1: timeout/WebSocket error → retry
          │   └─ Attempt 2: gagal
          │
          ├─ Fallback: Piper TTS (offline, < 800ms switch)
          │   ├─ piper-tts --model id_ID-news_tts --output-raw
          │   ├─ Log: {"event": "tts_fallback", "from": "edge", "to": "piper",
          │   │        "switch_latency_ms": 450}
          │   └─ Audio stream → playback
          │
          ├─ Jika Piper juga gagal (sangat jarang):
          │   ├─ Log: {"event": "tts_all_failed", "error_code": "ERR_TTS_SYNTHESIS"}
          │   ├─ Print ke terminal: "[Elysia] Brave browser sudah dibuka."
          │   └─ Transisi ke IDLE (tidak crash)
          │
          └─ IDLE (daemon tetap hidup)
```

### 5.3 Audio Device Failure

```
Startup ATAU Runtime: device audio hilang / unplugged
          │
          ├─ sounddevice.PortAudioError / DeviceNotFound
          │
          ├─ Retry re-enumerate audio devices:
          │   ├─ Attempt 1: pactl list short sources → wait 1s
          │   ├─ Attempt 2: pactl list short sources → wait 1s
          │   └─ Attempt 3: pactl list short sources → wait 1s
          │
          ├─ Jika device ditemukan kembali:
          │   ├─ Log: {"event": "audio_device_recovered", "attempt": 2}
          │   └─ Lanjutkan normal
          │
          ├─ Jika tetap gagal setelah 3x:
          │   ├─ Log: {"event": "audio_device_failed",
          │   │        "error_code": "ERR_AUDIO_INPUT", "level": "critical"}
          │   ├─ Fallback ke system default source/sink
          │   └─ Jika default juga gagal → exit(1) dengan pesan jelas
          │
          └─ Daemon tetap hidup jika ada fallback device
```

### 5.4 STT Menghasilkan Teks Kosong / Noise

```
PROCESSING state: faster-whisper.transcribe() selesai
          │
          ├─ Output teks = "" atau hanya noise/batuk
          │
          ├─ Deteksi: len(text.strip()) == 0 ATAU
          │            no_speech_probability > 0.8
          │
          ├─ Log: {"event": "stt_empty", "error_code": "ERR_STT_EMPTY",
          │        "no_speech_prob": 0.92}
          │
          ├─ TIDAK dikirim ke Gemini
          │
          ├─ Response = "Maaf, Elysia tidak mendengar perintahmu. Coba ulangi."
          │
          └─ Transisi ke SPEAKING (verbal) → IDLE
```

### 5.5 Perintah di Luar Allowlist

```
PROCESSING state: Gemini mengembalikan tool call
          │
          ├─ tool_call: open_application(app_name="terminal hack xyz")
          │
          ├─ Allowlist lookup di execution/apps.py:
          │   "terminal hack xyz" → NOT FOUND
          │
          ├─ Log: {"event": "tool_rejected",
          │        "error_code": "ERR_TOOL_NOT_ALLOWED",
          │        "requested": "terminal hack xyz"}
          │
          ├─ Response = "Maaf, aplikasi terminal hack xyz tidak ada dalam daftar
          │              aplikasi yang diizinkan."
          │
          └─ Transisi ke SPEAKING (verbal rejection) → IDLE
```

### 5.6 Aksi Destruktif — Konfirmasi Verbal Dua Langkah

```
PROCESSING state: Gemini mengembalikan tool call yang berisiko
          │
          ├─ tool_call: system_action(action="shutdown")
          │
          ├─ Cek apakah aksi termasuk DESTRUCTIVE_ACTIONS:
          │   ["rm", "shutdown", "reboot", "apt", "pip install --system"]
          │   "shutdown" → ★ DESTRUCTIVE
          │
          ├─ Transisi ke SPEAKING:
          │   Elysia: "Kamu yakin ingin mematikan sistem? Jawab 'ya' untuk konfirmasi."
          │
          ├─ Transisi ke LISTENING (menunggu konfirmasi):
          │   ├─ Timeout konfirmasi: 5 detik
          │   ├─ VAD + STT mendengarkan jawaban
          │   │
          │   ├─ Jawaban = "ya" / "iya" / "lanjutkan":
          │   │   ├─ Eksekusi: subprocess.run(["shutdown", "now"], shell=False)
          │   │   ├─ Log: {"event": "destructive_confirmed", "action": "shutdown"}
          │   │   └─ Verbal: "Sistem akan dimatikan sekarang."
          │   │
          │   ├─ Jawaban = "tidak" / "batal" / (apapun selain konfirmasi):
          │   │   ├─ Log: {"event": "destructive_cancelled", "action": "shutdown"}
          │   │   └─ Verbal: "Oke, perintah dibatalkan." → IDLE
          │   │
          │   └─ Timeout 5 detik (user diam):
          │       ├─ Log: {"event": "destructive_timeout", "action": "shutdown"}
          │       └─ Verbal: "Tidak ada konfirmasi. Perintah dibatalkan." → IDLE
```

---

## 6. Anti-Echo / Acoustic Feedback Prevention

```
SPEAKING state aktif:
          │
          ├─ 1. Mic stream: MUTED (sounddevice stream.stop() / volume = 0)
          ├─ 2. Porcupine: PAUSED (tidak memproses frame)
          ├─ 3. TTS audio playback berjalan
          │
          │  Suara Elysia keluar dari speaker
          │  Mic TIDAK menangkap suara Elysia
          │  Porcupine TIDAK mendeteksi false wake word
          │
          ├─ Playback selesai
          │
          ├─ 4. Cooldown timer START: 2–3 detik
          │     (mencegah sisa echo/reverb memicu wake word)
          │
          ├─ 5. Setelah cooldown selesai:
          │     ├─ Mic stream: UNMUTED (stream.start())
          │     └─ Porcupine: RESUMED
          │
          └─ Transisi ke IDLE
```

---

## 7. Logging & Observability Flow

Setiap interaksi user menghasilkan rangkaian log entry terstruktur (JSON):

```
# Session lifecycle — satu siklus IDLE → IDLE
{"ts": "...", "level": "info",  "sid": "req-001", "event": "wake_word_detected",     "latency_ms": 180}
{"ts": "...", "level": "info",  "sid": "req-001", "event": "recording_started",      "device": "mic", "sample_rate": 16000}
{"ts": "...", "level": "info",  "sid": "req-001", "event": "vad_silence_detected",   "vad_latency_ms": 280,  "audio_duration_ms": 2100}
{"ts": "...", "level": "info",  "sid": "req-001", "event": "stt_completed",          "stt_latency_ms": 650,  "text": "buka brave browser", "confidence": 0.94}
{"ts": "...", "level": "info",  "sid": "req-001", "event": "llm_tool_call",          "llm_latency_ms": 980,  "tool": "open_application", "args": {"app_name": "brave browser"}}
{"ts": "...", "level": "info",  "sid": "req-001", "event": "tool_executed",          "exec_latency_ms": 45,  "command": ["brave-browser"], "status": "ok"}
{"ts": "...", "level": "info",  "sid": "req-001", "event": "tts_playback_started",   "tts_latency_ms": 380,  "provider": "edge", "voice": "id-ID-GadisNeural"}
{"ts": "...", "level": "info",  "sid": "req-001", "event": "tts_playback_finished",  "playback_ms": 1200}
{"ts": "...", "level": "info",  "sid": "req-001", "event": "cycle_complete",         "total_e2e_latency_ms": 2850}

# Error events (contoh)
{"ts": "...", "level": "error", "sid": "req-002", "event": "llm_failed",             "error_code": "ERR_LLM_TIMEOUT", "attempts": 3}
{"ts": "...", "level": "warn",  "sid": "req-003", "event": "tts_fallback",           "from": "edge", "to": "piper", "switch_latency_ms": 450}
{"ts": "...", "level": "crit",  "sid": "---",     "event": "audio_device_failed",    "error_code": "ERR_AUDIO_INPUT"}
```

### Debug Audio Recording (Opt-in)

```
Jika DEBUG_RECORD_AUDIO=true di .env:
          │
          ├─ Setiap siklus LISTENING → PROCESSING:
          │   ├─ Simpan audio buffer ke .debug_audio/req-{session_id}.wav
          │   ├─ Format: WAV 16kHz mono 16-bit PCM
          │   └─ Rotasi otomatis: simpan maksimal 50 file terakhir
          │
          └─ Jika DEBUG_RECORD_AUDIO=false (default):
              └─ Audio buffer HANYA di memory, dibuang setelah STT selesai
```

---

## 8. Shutdown Flow

```
Signal: SIGINT (Ctrl+C) / SIGTERM
          │
          ├─ 1. Log: {"event": "shutdown_initiated", "signal": "SIGINT"}
          │
          ├─ 2. Stop audio stream (sounddevice)
          │     └─ stream.stop() → stream.close()
          │
          ├─ 3. Release Porcupine
          │     └─ porcupine.delete()
          │
          ├─ 4. Cleanup TTS temp files (jika ada)
          │
          ├─ 5. Log: {"event": "shutdown_complete", "uptime_s": ...}
          │
          └─ exit(0)

Catatan: Semua resource di-cleanup via try/finally atau atexit handler
         agar tidak ada orphan process / leaked file descriptor.
```

---

## 9. Dependency Graph (Modul → Library)

```
┌─────────────────────────────────────────────────────────────┐
│ main.py                                                     │
│   └─ core/config.py ─────── pydantic-settings               │
│   └─ core/logger.py ─────── structlog                        │
│   └─ core/state.py ──────── (stdlib: enum, dataclass)        │
│                                                             │
│   └─ audio/wake_word.py ──── pvporcupine                     │
│   └─ audio/recorder.py ──── sounddevice, numpy               │
│   └─ audio/vad.py ────────── torch (Silero VAD via torch.hub)│
│   └─ audio/stt.py ────────── faster-whisper                  │
│   └─ audio/tts.py ────────── edge-tts, piper-tts             │
│                                                             │
│   └─ agent/llm.py ────────── google-genai, tenacity          │
│   └─ agent/tools.py ─────── (internal: execution/apps.py)    │
│   └─ agent/prompt.py ────── (plain text, no deps)            │
│                                                             │
│   └─ execution/apps.py ──── (stdlib: dict registry)          │
│   └─ execution/linux.py ─── (stdlib: subprocess)             │
│                                                             │
│   └─ tests/ ──────────────── pytest, pytest-asyncio, httpx   │
│                               (mock: unittest.mock)          │
└─────────────────────────────────────────────────────────────┘

System deps (OS-level, bukan pip):
  ffmpeg          → decode audio untuk faster-whisper
  portaudio19-dev → backend C library untuk sounddevice
```

---

## 10. Ringkasan Error Codes

| Code | Stage | Penyebab | Respons Verbal |
|---|---|---|---|
| `ERR_AUDIO_INPUT` | Startup / IDLE | Device mic tidak ditemukan / unplugged | "Maaf, Elysia tidak bisa mengakses mikrofon." |
| `ERR_STT_EMPTY` | PROCESSING | Audio hanya noise / batuk / terlalu pendek | "Maaf, Elysia tidak mendengar perintahmu. Coba ulangi." |
| `ERR_LLM_TIMEOUT` | PROCESSING | Gemini API timeout setelah 3x retry | "Maaf, koneksi ke otak Elysia sedang terganggu." |
| `ERR_TOOL_NOT_ALLOWED` | PROCESSING | App tidak ada di allowlist | "Maaf, aplikasi X tidak ada dalam daftar yang diizinkan." |
| `ERR_TTS_SYNTHESIS` | SPEAKING | Edge TTS + Piper keduanya gagal | Terminal log fallback (tidak ada audio) |
