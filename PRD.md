# PRD: Elysia Personal Voice Assistant 🎙️

## 1. Tujuan Produk

Membangun asisten pribadi _speech-to-speech_ bernama **Elysia** yang berjalan di laptop Linux (Hyprland). Asisten mendengarkan perintah suara (dimulai _wake word_ **"Hei Elysia"**), mengeksekusi perintah sistem operasi secara aman (contoh: "buka brave browser"), dan memberi respons suara secara _near real-time_.

- Target latency: total **≤ 2–3 detik** dari user selesai bicara sampai Elysia mulai membalas.
- Input audio: **mikrofon laptop bawaan, speaker (mic via ALSA/PipeWire loopback), atau headset** — semua harus didukung tanpa konfigurasi ulang manual.
- Arsitektur pipeline modular (§9), bukan true streaming speech-to-speech.

## 2. Spesifikasi Audio

- **Format**: WAV, 16kHz, mono, 16-bit PCM — wajib untuk faster-whisper dan Porcupine.
- **Input sources** (pilih 1 via config `.env` / `pydantic-settings`):
  - `mic` — mikrofon laptop/headset (default).
  - `speaker` — audio mic dari speaker via PipeWire loopback (`alsa_output...monitor`), di-enumerate saat startup (`pactl list short sources`), tidak hardcode.
  - `headset` — headset USB/BT dengan mic bawaan.
- **Output**: default sistem (`default` sink) atau sink tertentu jika ditentukan.
- **Volume**: normalize input ke −6 dBFS sebelum Whisper; normalize TTS output agar konsisten.
- **Sampling**: force 16kHz, mono, 16-bit PCM di recorder (resample otomatis jika device mengeluarkan 48kHz).

## 3. Komponen & Tech Stack

1. **Audio I/O**: `sounddevice` + PipeWire/ALSA.
2. **Wake Word Engine**: Porcupine (Picovoice)
   - Custom keyword `.ppn` ("Hei Elysia"), butuh `PICOVOICE_ACCESS_KEY`.
   - Alternatif offline tanpa API key: `openWakeWord` (default). Model bawaan: `hey_jarvis` → frasa pemicu **"Hey Jarvis"**.
   - Frasa **"Hei Elysia"** butuh model openWakeWord custom (`OPENWAKEWORD_MODEL_PATH`) atau keyword Porcupine `.ppn` (`PICOVOICE_ACCESS_KEY`).
    - **Cooldown**: `COOLDOWN_SEC = 2–3` setelah trigger untuk mencegah trigger berulang saat TTS aktif.
3. **VAD / Endpointing**: Silero VAD
   - Mendeteksi silence (threshold 700–900ms) untuk stop recording.
4. **Speech-to-Text (STT)**: faster-whisper (CTranslate2)
   - Model `small` (int8, CPU, ~500MB RAM) atau `medium`.
5. **LLM & Orkestrasi**: Gemini 3.8 Flash via SDK `google-genai`
   - Direct function calling loop tanpa over-engineering (LangGraph opsional hanya bila alur tool multi-step).
6. **Execution Layer**: Python `subprocess` + allowlist (`shell=False`).
   - Adapter platform: `hyprctl` untuk workspace/window management di Hyprland.
7. **Text-to-Speech (TTS)**: Edge TTS (`id-ID-ArdiNeural` / `id-ID-GadisNeural`) dengan automatic fallback ke Piper TTS offline (`id_ID news_tts`).

## 4. Keamanan (Wajib)

- **Input Untrusted**: Output LLM dianggap untrusted input.
- **Pattern Perintah**: Format terstruktur `"buka <nama_app>"`, ditolak jika di luar allowlist.
- **Trigger Konfirmasi Verbal**: Aksi berisiko/destruktif (`rm`, `shutdown`, `reboot`, `apt`, `pip install --system`) wajib konfirmasi verbal dua langkah ("Yakin ingin mematikan sistem?").
- **Strict Allowlist**: Registry mapping nama ramah → executable konkret. Tidak ada eksekusi free-form bash string.
- **Execution Guard**: Wajib `shell=False` dan validasi argumen list pada semua pemanggilan `subprocess.run`.

## 5. Alur Sistem (System Flow)

State machine utama: `IDLE → LISTENING → PROCESSING → SPEAKING → IDLE`

1. **IDLE**: Porcupine mendengarkan stream mic pasif (low CPU/RAM).
2. **LISTENING**: Porcupine mendeteksi wake word → stream audio diarahkan ke buffer → Silero VAD mendeteksi silence (user selesai bicara) → stop buffer.
3. **PROCESSING**: Audio diterjemahkan oleh faster-whisper → teks dikirim ke Gemini → Gemini memanggil function tool via allowlist → tool dieksekusi.
4. **SPEAKING**: Gemini menyusun kalimat balasan → TTS mensintesis audio → playback audio. **Mic dan Porcupine di-mute/pause** agar tidak terjadi acoustic feedback loop.
5. **Kembali ke IDLE**: Cooldown aktif selama 2 detik sebelum Porcupine kembali menerima trigger baru.

---

## 6. Fitur Utama & Kriteria Penerimaan (Acceptance Criteria)

| Fitur | Deskripsi | Kriteria Accept (Detail & Terukur) |
|---|---|---|
| Wake word "Hei Elysia" | Porcupine keyword custom `.ppn` | False positive < 5% dalam 100 jam standby; false negative < 3% pada jarak ≤1.5m (SNR >15 dB); re-trigger diblokir selama cooldown 2–3s setelah transisi SPEAKING |
| Rekam & VAD Endpointing | Mic → Silero VAD (silence threshold 700–900ms, max duration 15s) | VAD silence detection latency ≤ 300ms setelah user diam; 0 kasus early cut-off pada jeda bicara wajar (uji 50 sampel kalimat beruntun) |
| STT Transkripsi | faster-whisper `small` int8 | Latency STT ≤ 1.0 detik untuk rekaman ≤ 5 detik; Word Error Rate (WER) < 10% untuk pola perintah Indonesia standar |
| Latency Budget E2E | Total waktu dari user selesai bicara hingga Elysia mulai bersuara | **E2E Latency ≤ 2.5–3.0 detik (p95)** diukur via metric `structlog` (`latency_ms`). Breakdown budget: VAD stop (≤300ms) + STT (≤1000ms) + Gemini (≤1200ms) + TTS buffer first-byte (≤500ms) |
| Intent & Tool Execution | Gemini function calling → allowlist → `subprocess(shell=False)` | Perintah valid dieksekusi < 300ms pasca LLM response; perintah di luar allowlist ditolak dengan verbal; zero audit findings untuk `shell=True` |
| Konfirmasi Aksi Berisiko | Guard dua langkah untuk perintah destruktif (`shutdown`, `rm`, `reboot`, dll.) | 0 eksekusi perintah destruktif tanpa konfirmasi verbal positif ("ya" / "lanjutkan"); membatalkan aksi jika jawaban negatif atau timeout 5 detik |
| Multi-source Audio | Switch `mic`, `speaker` (PipeWire loopback), `headset` | Startup otomatis meng-enumerate `pactl list short sources`; switch source via `.env` tanpa reload source code |
| Anti-echo Loop | Mic mute + Porcupine paused selama playback TTS | 0 self-trigger loop dalam 100 siklus respons berulang pada volume speaker 75% |
| Network Outage: Gemini | Kegagalan jaringan / 5xx / timeout pada Google AI Studio | Retry 2x (exponential backoff 1s, 2s via `tenacity`, timeout 3.0s per request); fallback verbal: "Maaf, koneksi ke otak Elysia sedang terganggu" tanpa crash daemon |
| Network Outage: Edge TTS | Edge TTS timeout (5.0s) atau koneksi WebSocket putus | Retry 1x; otomatis fallback ke Piper TTS (offline) dalam waktu < 800ms; respons suara tetap keluar |
| Audio Hardware Failure | Device audio hilang/unplugged saat runtime | Error di-catch graceful; retry re-enumeration 3x (interval 1s); jika tetap gagal log `CRITICAL` dan fallback ke system default source |
| State Recovery | Error tak terduga pada stage manapun (STT/LLM/TTS) | State machine selalu kembali ke `IDLE` dalam waktu ≤ 2.0 detik; mengeluarkan verbal error message jika memungkinkan |

---

## 7. Struktur Folder

```text
elysia-assistant/
│
├── .env.example            # Template konfigurasi environment & API keys
├── requirements.txt        # Dependensi utama terpindang
├── README.md               # Setup, system packages (ffmpeg, portaudio), panduan uji
├── main.py                 # Entry point: inisialisasi state machine & runner
├── PRD.md                  # Single Source of Truth spesifikasi produk & teknis
├── note.md                 # Catatan historis arsitektur
│
├── core/
│   ├── __init__.py
│   ├── config.py           # Pydantic-settings BaseSettings (validasi .env & schema)
│   ├── state.py            # FSM (IDLE/LISTENING/PROCESSING/SPEAKING) + Event bus
│   └── logger.py           # Structured logging via structlog + trace ID per session
│
├── audio/
│   ├── __init__.py
│   ├── wake_word.py        # Porcupine listener & pause/resume controller
│   ├── recorder.py         # sounddevice audio stream buffer & resampler
│   ├── vad.py              # Silero VAD stateful endpointing
│   ├── stt.py              # faster-whisper runner + CTranslate2 thread pool
│   └── tts.py              # Dual TTS provider (Edge TTS online + Piper offline fallback)
│
├── agent/
│   ├── __init__.py
│   ├── llm.py              # Gemini 3.8 Flash SDK client + tenacity retry wrapper
│   ├── tools.py            # Function registry allowlist
│   └── prompt.py           # System prompt Elysia & injection guardrail
│
├── execution/
│   ├── __init__.py
│   ├── apps.py             # Strict allowlist: friendly name -> executable binary
│   └── linux.py            # Hyprland hyprctl adapter & safe subprocess executor
│
└── tests/
    ├── conftest.py         # Pytest fixtures, mock audio devices & synthetic WAVs
    ├── test_state.py       # FSM transitions, cooldown guard, & invalid event handling
    ├── test_apps.py        # Validasi allowlist & injection rejection
    ├── test_tools.py       # Mock execution test (subprocess check)
    ├── test_vad_stt.py     # VAD endpointing & faster-whisper mocking
    └── test_resilience.py  # Network failure, retry policy, & offline TTS fallback
```

---

## 8. Catatan Teknis

- **System Dependencies**: Wajib install `ffmpeg` (audio decode) dan `portaudio19-dev` (sounddevice/PipeWire backend) pada OS level sebelum `pip install`.
- **Cold Start Optimization**: Inisialisasi model faster-whisper dan instance Porcupine dilakukan secara singleton saat startup aplikasi, bukan lazy load saat ada request.
- **Audio Device Enumeration**: Loopback sink PipeWire (`alsa_output...monitor`) dideteksi dinamis via command `pactl list short sources`, dilarang melakukan hardcoding nama card/device ALSA.

## 9. Alternatif Arsitektur

| Arsitektur | Kelebihan | Kekurangan | Status |
|---|---|---|---|
| **Pipeline Modular (Pilihan)** | Komponen decoupled, bisa offline fallback (Piper), observabilitas per-stage tinggi | Akumulasi latency antar-komponen | **Dipakai** |
| **Gemini Live API (Bidirectional)** | Latency audio-to-audio sangat rendah, natural interruption | Full dependency ke cloud & internet stabil, biaya per-menit tinggi, kontrol OS lokal terbatas | Evaluasi Masa Depan |

## 10. Urutan Implementasi (Milestones)

1. **Milestone 1 (Core & Safe Execution)**: Config pydantic-settings, State Machine FSM, Allowlist apps, dan test suite dasar.
2. **Milestone 2 (Agent & LLM Loop)**: Integrasi Gemini 3.8 Flash + function calling allowlist via text terminal.
3. **Milestone 3 (TTS & Audio Feedback)**: Integrasi Edge TTS + Piper offline fallback + audio playback controller.
4. **Milestone 4 (Audio Capture, VAD & STT)**: Buffer sounddevice + Silero VAD endpointing + faster-whisper transcription.
5. **Milestone 5 (Wake Word & Full E2E)**: Porcupine integration, mic-mute anti-echo, latency benchmarking, dan error recovery polish.

## 11. Non-Functional Requirements (NFR)

- **Platform**: Linux (Hyprland / Wayland) dengan PipeWire audio server.
- **Memory Footprint**: Standby IDLE < 150MB RAM; Saat aktif transkripsi (Whisper `small`) peak memory ≤ 800MB RAM.
- **CPU Footprint**: Standby IDLE (Porcupine wake word) ≤ 2–4% CPU single-core di laptop modern.
- **Privasi**: Audio buffer hanya berada di memory (RAM) dan langsung dibuang setelah transkripsi selesai. Dilarang menulis raw audio ke disk kecuali flag `DEBUG_RECORD_AUDIO=true` diaktifkan secara eksplisit.

---

## 12. Testing Strategy (Strategi Pengujian)

Pengujian pipeline voice assistant dirancang agar 100% dapat dijalankan pada lingkungan CI/CD tanpa mikrofon fisik:

1. **Unit Testing & Mocking**:
   - Audio Hardware: `sounddevice.InputStream` dimock menggunakan synthetic sine-wave atau generator silence/audio array numpy (`conftest.py`).
   - Wake Word: Mocking callback `pvporcupine` untuk menguji transisi state tanpa audio sungguhan.
   - Execution Layer: Mocking `subprocess.run` pada `execution/linux.py` untuk memvalidasi pemanggilan binary tanpa membuka aplikasi sebenarnya.

2. **State Machine FSM Verification (`tests/test_state.py`)**:
   - Memverifikasi transisi valid: `IDLE → LISTENING → PROCESSING → SPEAKING → IDLE`.
   - Menguji invalid state transition (contoh: trigger wake word saat masih dalam state `SPEAKING` harus di-drop karena cooldown).
   - Pengujian pemulihan error: memicu exception di `PROCESSING` dan memverifikasi FSM kembali ke `IDLE` secara aman.

3. **Audio Fixture Testing (`tests/test_vad_stt.py`)**:
   - Menyediakan fixture 3 file WAV sintetis: `silence.wav`, `command_buka_brave.wav`, dan `long_speech.wav`.
   - Menguji Silero VAD untuk mendeteksi cutoff point dengan margin akurasi ±100ms.
   - Menguji faster-whisper dengan mock CTranslate2 model untuk memverifikasi parser teks dan output encoding UTF-8.

4. **Resilience & Fault Injection (`tests/test_resilience.py`)**:
   - Simulasi `httpx.TimeoutException` dan HTTP 503 pada panggilan Gemini untuk memverifikasi retry policy `tenacity`.
   - Simulasi Edge TTS failure untuk memastikan fallback instan ke Piper TTS terpanggil.

---

## 13. Resilience, Error Handling & Fallback Specifications

1. **Timeout Policy per Tahapan**:
   - Wake Word check: Real-time non-blocking frame process.
   - VAD Max Duration: 15 detik (mencegah rekaman infinite jika ruangan bising).
   - STT Timeout: Maksimal 2.5 detik.
   - Gemini LLM Request Timeout: Maksimal 3.5 detik.
   - TTS Synthesis Timeout: Maksimal 4.0 detik per kalimat respons.

2. **Retry Mechanism (`tenacity`)**:
   - Panggilan API Gemini: `@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=3), reraise=True)`.
   - Edge TTS: 1x retry. Jika gagal dalam 2 detik, langsung switch ke Piper TTS.

3. **Fallback Hierarchy**:
   - **TTS**: `Edge TTS (Online)` → `Piper TTS (Offline Local)` → `Terminal Text-Only Log (Fallback Terakhir)`.
   - **Wake Word**: Porcupine (`.ppn`) → openWakeWord (jika license key Porcupine expired).

4. **Error Logging Codes**:
   - `ERR_AUDIO_INPUT`: Gagal capture stream audio.
   - `ERR_STT_EMPTY`: Audio tidak menghasilkan transkripsi (noise/batuk).
   - `ERR_LLM_TIMEOUT`: Gemini tidak merespons dalam threshold waktu.
   - `ERR_TOOL_NOT_ALLOWED`: Perintah tidak ada di registry allowlist.
   - `ERR_TTS_SYNTHESIS`: Semua provider suara gagal.

---

## 14. Observability, Logging & Configuration Management

1. **Structured Logging (`structlog`)**:
   - Log diformat dalam bentuk JSON (production) atau colorful console text (development).
   - Setiap interaksi pengguna diberikan `session_id` unik untuk melacak alur lifecycle:
     ```json
     {"timestamp": "...", "level": "info", "session_id": "req-123", "state": "PROCESSING", "stage": "stt", "latency_ms": 780, "event": "transcription_completed"}
     ```
   - Metric `latency_ms` wajib dicatat di setiap akhir stage: `vad_latency_ms`, `stt_latency_ms`, `llm_latency_ms`, `tts_latency_ms`, dan `total_e2e_latency_ms`.

2. **Configuration Schema (`pydantic-settings`)**:
   Konfigurasi divalidasi ketat melalui class `core/config.py`:
   ```python
   from pydantic_settings import BaseSettings
   from typing import Literal

   class Settings(BaseSettings):
       # API Keys
       GOOGLE_API_KEY: str
       PICOVOICE_ACCESS_KEY: str = ""

       # Audio Settings
       AUDIO_INPUT_SOURCE: Literal["mic", "speaker", "headset"] = "mic"
       SAMPLE_RATE: int = 16000
       VAD_SILENCE_THRESHOLD_MS: int = 800
       CONFIRMATION_TIMEOUT_SEC: float = 5.0

       # Model Settings
       WHISPER_MODEL_SIZE: Literal["small", "medium"] = "small"
       WHISPER_DEVICE: Literal["cpu", "cuda"] = "cpu"
       WHISPER_COMPUTE_TYPE: str = "int8"
       TTS_PROVIDER_DEFAULT: Literal["edge", "piper"] = "edge"
       GEMINI_MODEL: str = "gemini-2.0-flash"
       LLM_REQUEST_TIMEOUT_MS: int = 3500

       # Observability & Debug
       LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
       DEBUG_RECORD_AUDIO: bool = False  # Opt-in debug privacy guard

       class Config:
           env_file = ".env"
           case_sensitive = True
   ```

3. **Privacy Enforcement**:
   - Default setting `DEBUG_RECORD_AUDIO=False`.
   - Saat `DEBUG_RECORD_AUDIO=True`, audio disimpan di folder temporer `.debug_audio/` dengan rotasi otomatis maksimal 50 file terakhir untuk analisis kegagalan VAD/STT.
