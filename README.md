# Elysia Personal Voice Assistant 🎙️

Elysia adalah asisten pribadi *speech-to-speech* modular berbasis Python yang berjalan lokal di laptop Linux (Hyprland / Wayland). Asisten ini mendengarkan kata pemicu (*wake word*), memproses perintah suara secara aman lewat *allowlist*, dan memberikan respons suara secara *near real-time* (latency E2E ≤ 2–3 detik).

---

## 🏗️ Tech Stack

- **Wake Word Engine**: Picovoice Porcupine (custom keyword `.ppn` / built-in `jarvis`)
- **VAD / Endpointing**: Silero VAD (`snakers4/silero-vad` via Torch)
- **STT (Speech-to-Text)**: `faster-whisper` (CTranslate2, model `small` int8)
- **LLM & Orkestrasi**: Gemini 3.8 Flash via SDK `google-genai` dengan Function Calling
- **Execution Layer**: Python `subprocess` + strict allowlist (`shell=False`)
- **TTS (Text-to-Speech)**: Edge TTS (`id-ID-GadisNeural`) + Piper TTS (offline fallback)
- **Audio Server**: PipeWire / PulseAudio via `sounddevice`

---

## 📋 System Dependencies (Prerequisites)

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

## 🚀 Setup & Instalasi

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

## 🏃 Cara Menjalankan

Dalam keadaan virtual environment aktif (`source .venv/bin/activate`):

```bash
python main.py
```

Elysia akan menginisialisasi model dan mulai mendengarkan wake word. Katakan **"Hei Elysia"** (atau "Jarvis" jika belum memakai file `.ppn` custom) lalu sebutkan perintah Anda, misalnya:

- *"Buka brave browser"*
- *"Buka file manager"*
- *"Buka terminal"*
- *"Buka spotify"*
- *"Matikan komputer"* (membutuhkan konfirmasi verbal "ya")

---

## 🧪 Pengujian (Test Suite)

Jalankan pengujian unit, FSM state transition, allowlist security, dan mock resilience:

```bash
pytest tests/ -v
```

---

## 🔒 Fitur Keamanan

- **Zero Free-form Shell Execution**: Output LLM dilarang keras menjalankan string shell bebas (`shell=True`).
- **Strict Allowlist**: LLM hanya memilih dari nama aplikasi yang terdaftar di `execution/apps.py`.
- **Konfirmasi Dua Langkah**: Perintah destruktif (`shutdown`, `reboot`, `lock screen`) memerlukan jawaban "ya" secara verbal sebelum dieksekusi.
- **Mic Anti-Echo**: Mikrofon di-mute dan Porcupine di-pause secara otomatis selama playback TTS agar suara Elysia tidak memicu loop perintah sendiri.
