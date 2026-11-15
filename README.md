# Elysia Personal Voice Assistant

![CI](https://github.com/Ryuuutaaa/elysia-asisstent/actions/workflows/ci.yml/badge.svg)

Elysia adalah asisten pribadi *speech-to-speech* modular berbasis Python yang berjalan lokal di laptop Linux (Hyprland / Wayland). Asisten ini mendengarkan kata pemicu (*wake word*), memproses perintah suara secara aman lewat *allowlist*, dan memberikan respons suara secara *near real-time* (latency E2E <= 2-3 detik).

---

## Tech Stack

- **Wake Word Engine**: openWakeWord (default, offline) / Picovoice Porcupine (opsional, butuh API key)
- **VAD / Endpointing**: Silero VAD (paket `silero-vad`, model `.jit` bundled) dengan chunk 512 sampel @16kHz / 256 @8kHz
- **STT (Speech-to-Text)**: faster-whisper (CTranslate2, model small int8)
- **LLM & Orkestrasi**: Gemini via SDK google-genai dengan Function Calling + auto model resolution
- **Execution Layer**: Python subprocess + strict allowlist (shell=False)
- **TTS (Text-to-Speech)**: Edge TTS (id-ID-GadisNeural) + Piper TTS (offline fallback)
- **Audio Server**: PipeWire / PulseAudio via sounddevice

---

## System Dependencies (Prerequisites)

Sebelum menginstall paket Python, install dependensi level OS berikut:

### Ubuntu / Debian:
```bash
sudo apt update
sudo apt install -y ffmpeg portaudio19-dev python3.12-venv
```

### Arch Linux / Manjaro:
```bash
sudo pacman -S ffmpeg portaudio python
```

---

## Setup & Instalasi

1. **Clone repository & masuk folder proyek**:
   ```bash
   cd elysia-assistant
   ```
2. **Buat & aktifkan virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install paket Python**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Konfigurasi Environment**:
   Salin `.env.example` ke `.env` lalu isi API Key Anda:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` dan masukkan `GOOGLE_API_KEY` (dapatkan gratis dari [Google AI Studio](https://aistudio.google.com/)).

---

## Cara Menjalankan

Dalam keadaan virtual environment aktif (`source .venv/bin/activate`):

```bash
python main.py                    # diagnostics + Elysia
python main.py --skip-diagnostics # langsung Elysia
python main.py --diagnostics-only # hanya diagnostics
```

Elysia akan menginisialisasi model dan mulai mendengarkan wake word. Secara default (`openWakeWord`) frasa pemicunya adalah **"Hey Jarvis"**. Untuk memakai **"Hei Elysia"**, set `OPENWAKEWORD_MODEL_PATH` ke model openWakeWord custom, atau pakai Porcupine dengan file `.ppn` (`WAKE_WORD_ENGINE=porcupine` + `PICOVOICE_ACCESS_KEY`).

### Fitur Suggestion Flow (Typo / Salah Dengar)

Jika STT salah dengar (misal "buka breif" -> "brave browser"), Elysia akan:
1. Coba fuzzy match otomatis
2. Jika tidak yakin, tanya LLM (choose_app)
3. Tanyakan ke user: "Maksudmu 'brave browser'?"
4. User jawab "iya" -> langsung buka (tanpa re-LLM)
5. User jawab "bukan" -> dengar ulang (max 2x retry per sesi)

### Perintah yang Dideskripsikan

- "Buka brave browser"
- "Buka file manager"
- "Buka terminal"
- "Buka spotify"
- "Matikan komputer" (membutuhkan konfirmasi verbal "ya")
- "Kunci layar" (membutuhkan konfirmasi verbal "ya")

---

## Pengujian (Test Suite)

Perintah cepat (dari root repo, virtualenv aktif):

```bash
make install-dev   # sekali saja: dependency + ruff
make lint          # ruff check
make test          # pytest
make ci            # lint + test (sama seperti GitHub Actions)
```

Setara manual:

```bash
pytest tests/ -v
```

Test berjalan **tanpa audio server / mikrofon** (headless-safe): `sounddevice` diimpor lazy, jadi CI maupun laptop tanpa PulseAudio tidak error saat koleksi test.

Total: 206 test cases (test_apps, test_tools, test_suggest, test_main, test_state, test_vad_stt, test_wake_word, test_resilience, test_debug_audio, test_devices, test_utils, test_llm, test_diagnostics, test_recorder).

---

## Fitur Keamanan

- **Zero Free-form Shell Execution**: Output LLM dilarang keras menjalankan string shell bebas (shell=True).
- **Strict Allowlist**: LLM hanya memilih dari nama aplikasi yang terdaftar di execution/apps.py.
- **Allowlist Struktural**: `execution/linux.py` menolak `argv` apa pun yang tidak ada di registry, tepat sebelum `Popen` — bukan hanya di layer tool.
- **Konfirmasi Dua Langkah**: Perintah destruktif (shutdown, reboot, lock screen) memerlukan jawaban "ya" secara verbal sebelum dieksekusi.
- **Mic Anti-Echo**: Mikrofon di-mute dan wake word di-reset secara otomatis selama playback TTS agar suara Elysia tidak memicu loop perintah sendiri.
- **Token-Guard**: Exact match hanya diakui jika token benar-benar ada di STT asli (mencegah LLM "mengoreksi" sendiri tanpa konfirmasi).
- **Prompt Injection Guard**: System prompt memblok instruksi tersembunyi dari user input.
- **Supply-chain**: Model Silero VAD dibundel di wheel `silero-vad`; tidak ada `torch.hub` atau eksekusi kode remote saat startup.
- **Secret Isolation**: Aplikasi yang diluncurkan tidak mewarisi `GOOGLE_API_KEY` / `PICOVOICE_ACCESS_KEY`.
- **Log Privasi**: Transkrip suara hanya dicatat sebagai panjang + fingerprint di level INFO; teks penuh hanya saat `LOG_LEVEL=DEBUG`. File `.debug_audio/` dibuat `0700` dan WAV `0600`.