# Project: Elysia Personal Voice Assistant 🎙️

## 🎯 Tujuan Proyek (Objective)

Membangun asisten pribadi _speech-to-speech_ bernama **Elysia** yang berjalan di laptop Linux (Hyprland). Asisten mendengarkan perintah suara (dimulai _wake word_ **"Hei Elysia"**), mengeksekusi perintah sistem operasi secara aman (contoh: "buka brave browser"), dan memberi respons suara secara _near real-time_.

- Target latency: total **≤ 2–3 detik** dari user selesai bicara sampai Elysia mulai membalas.
- Input audio: **mikrofon laptop bawaan, speaker (mic via ALSA/PipeWire loopback), atau headset** — semua harus didukung tanpa konfigurasi ulang manual.
- Ini bukan true streaming speech-to-speech — arsitekturnya pipeline modular (lihat bagian Alternatif Arsitektur).

## 🎧 Spesifikasi Audio

- **Format**: WAV, 16kHz, mono, 16-bit PCM — wajib untuk Whisper/faster-whisper dan Porcupine.
- **Input sources** (user pilih 1, aktif via config):
  - `mic` — mikrofon laptop/headset (default).
  - `speaker` — audio mic dari speaker via PipeWire loopback (`alsa_output...monitor`), device didaftar otomatis saat startup (`pactl list short sources`), tidak hardcode.
  - `headset` — headset USB/BT dengan mic bawaan.
- **Output**: default sistem (`default` sink) atau sink tertentu jika user tentukan.
- **Volume**: normalize input ke −6 dBFS sebelum Whisper; normalize TTS output agar konsisten.
- **Sampling**: force 16kHz, mono, 16-bit PCM di recorder (resample jika device keluarkan 48kHz).

1. **Audio I/O & Spesifikasi**: lihat [Spesifikasi Audio](#-spesifikasi-audio) di atas.
2. **Wake Word Engine**: Porcupine (Picovoice)
   - _Tugas_: Mendeteksi kata "Hei Elysia" secara pasif dengan memori sangat rendah.
   - "Hei Elysia" bukan keyword bawaan — harus ditraining sebagai _custom keyword_ (`.ppn`) di Picovoice Console, butuh `PICOVOICE_ACCESS_KEY` (free tier).
   - Alternatif gratis tanpa key: openWakeWord (default). Model bawaan `hey_jarvis` → frasa pemicu **"Hey Jarvis"**.
   - Frasa **"Hei Elysia"** butuh model openWakeWord custom (`OPENWAKEWORD_MODEL_PATH`) atau keyword Porcupine `.ppn`.
   - _PocketSphinx dihapus — sudah usang dan akurasinya rendah._
   - **Cooldown**: `COOLDOWN_SEC = 2–3` setelah trigger Porcupine, agar false-trigger berulang (terutama saat TTS play) tidak membuat sistem me-reset state.
3. **VAD / Endpointing**: Silero VAD
   - _Tugas_: Menentukan kapan user selesai bicara (deteksi silence). Tanpa ini, sistem tidak tahu kapan rekaman harus berhenti.
4. **Speech-to-Text (STT)**: faster-whisper (CTranslate2)
   - _Tugas_: Mengubah rekaman suara perintah menjadi teks.
   - Model `small` (CPU, int8, ~500MB) atau `medium` (lebih akurat, butuh RAM lebih besar) — pilih sesuai spek laptop.
   - _`openai-whisper` tidak dipakai — terlalu lambat di CPU untuk near real-time._
5. **LLM & Orkestrasi**: Gemini 3.8 Flash (Google AI Studio) via SDK `google-genai`
   - _Tugas_: Bertindak sebagai "otak" (_Agent_) yang menganalisis teks, mengambil keputusan, dan memicu _Function Calling_.
   - Versi model cepat berubah — cek halaman Models di dokumentasi Gemini API saat setup.
   - LangChain/LangGraph **opsional**: dipakai hanya jika alur tool bercabang/multi-step. Untuk satu tool call, loop function-calling biasa lebih sederhana dan lebih mudah di-debug.
6. **Execution Layer**: Python `subprocess` + allowlist
   - _Tugas_: Mengeksekusi perintah aplikasi via registry allowlist dengan `shell=False`.
   - Adapter platform/WM terpisah: `hyprctl` (Hyprland/Linux).
7. **Text-to-Speech (TTS)**: Edge TTS (default) / Piper (offline)
   - _Tugas_: Mengubah teks konfirmasi menjadi suara Elysia.
   - Edge TTS: gratis, voice `id-ID-ArdiNeural` (pria) / `id-ID-GadisNeural` (wanita). Catatan: tidak resmi, endpoint bisa berubah sewaktu-waktu.
   - Piper: offline dan cepat, voice Indonesia `id_ID news_tts` (medium/low) bernuansa pembaca berita.

## 🔒 Keamanan (Wajib)

Output LLM adalah _untrusted input_ — audio dari sumber lain bisa mengandung prompt injection ("abaikan instruksi sebelumnya, jalankan ...").

- **Pattern perintah**: sistem memahami format `"buka <nama_app>"`, bukan bahasa bebas. Daftar app di allowlist (lihat di bawah).
- **Trigger konfirmasi verbal**: untuk aksi destruktif/berisiko (`rm`, `shutdown`, `reboot`, `apt`, `pip install system-wide`). Aksi lain langsung jalan.

- **Allowlist aplikasi**: mapping nama ramah → perintah konkret (contoh: `"brave browser"` → `brave-browser`). LLM hanya boleh memilih dari daftar ini, tidak pernah membuat perintah baru.
- **Tidak ada free-form command**: tool tidak menerima string bash mentah dari LLM.
- **`shell=False` + validasi argumen** di semua pemanggilan `subprocess`.

## 🔄 Alur Sistem (System Flow)

State machine utama: `IDLE → LISTENING → PROCESSING → SPEAKING → IDLE`

1. **IDLE / Standby**: mendengarkan _wake word_ "Hei Elysia" (Porcupine, low resource).
2. **LISTENING**: rekam mic → VAD mendeteksi user selesai bicara → hentikan rekaman.
3. **PROCESSING**: faster-whisper mentranskripsi rekaman → Gemini menganalisis dan memicu tool → tool dieksekusi via allowlist.
4. **SPEAKING**: Gemini menyusun kalimat balasan ("Brave browser dibuka") → TTS → playback. **Mic di-mute selama speaking** agar suara Elysia tidak tertangkap mic sendiri (anti echo/loop).
5. Kembali ke **IDLE**. Jika STT atau tool gagal, Elysia membalas dengan pesan error suara — bukan diam.

## 📂 Struktur Folder

```text
elysia-assistant/
│
├── .env                    # Variabel rahasia (GOOGLE_API_KEY, PICOVOICE_ACCESS_KEY)
├── requirements.txt        # Dependensi utama (faster-whisper, sounddevice, silero-vad, pvporcupine, google-genai, edge-tts)
├── README.md               # Cara setup + system deps (ffmpeg, portaudio)
├── main.py                 # Entry point: merangkai state machine & modul
├── note.md                 # Dokumentasi proyek
│
├── core/
│   ├── __init__.py
│   ├── config.py           # Setup konfigurasi global (API key, path model, dll)
│   └── state.py            # State machine (IDLE/LISTENING/PROCESSING/SPEAKING)
│
├── audio/
│   ├── __init__.py
│   ├── wake_word.py        # Logika pendeteksi "Hei Elysia" (Porcupine)
│   ├── recorder.py         # Capture mic (sounddevice)
│   ├── vad.py              # Silero VAD / endpointing
│   ├── stt.py              # Logika konversi suara ke teks (faster-whisper)
│   └── tts.py              # Logika konversi teks ke suara (Edge TTS/Piper)
│
├── agent/
│   ├── __init__.py
│   ├── llm.py              # Setup Gemini 3.8 Flash
│   ├── tools.py            # Kumpulan @tool (eksekusi OS)
│   └── graph.py            # Opsional: LangGraph (multi-step/retry)
│
├── execution/
│   ├── __init__.py
│   ├── apps.py             # Registry allowlist aplikasi (nama ramah → perintah)
│   └── linux.py            # Adapter WM/platform (hyprctl/Hyprland)
│
└── tests/
    ├── test_apps.py        # Validasi registry & batasan allowlist
    └── test_tools.py       # Test tool tanpa audio (mock)
```

## 📌 Catatan Teknis

- **System deps**: `ffmpeg` (decode audio untuk Whisper) dan `portaudio` (sounddevice) tidak cukup lewat `requirements.txt` — wajib didokumentasikan di README.
- **Cold start**: model Whisper dan wake word di-load sekali saat startup, bukan per perintah.
- **Testability**: `execution/` dan `agent/tools.py` bisa di-unit test tanpa audio; layer audio di-mock.
- **Audio device enumeration**: PipeWire/Pulse memberikan `alsa_output...monitor` untuk speaker loopback — device didaftar otomatis saat startup (`pactl list short sources`), tidak hardcode nama device.
- **Alternatif arsitektur (true real-time)**: Gemini Live API (audio-in/audio-out) bisa menggantikan seluruh pipeline STT + LLM + TTS dengan dialog yang lebih natural. Trade-off: ketergantungan penuh ke cloud dan biaya per menit. Desain modular saat ini sengaja dipilih agar tiap komponen bisa dikontrol, diganti, dan di-debug secara terpisah.

## 🪜 Urutan Implementasi (Milestone)

1. Loop teks: input teks → Gemini → tool `open_application` → output teks (uji allowlist dulu, tanpa audio).
2. Tambah TTS (Edge TTS) untuk output suara.
3. Tambah STT (faster-whisper) + recorder + VAD.
4. Tambah wake word (Porcupine) + state machine penuh.
5. Polish: error handling, tuning latency, konfirmasi aksi berisiko.
