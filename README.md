# Voicecover

Voicecover is a command-line video dubbing tool. It transcribes a video's
speech with Whisper, translates each timestamped segment offline, generates a
new voice with Microsoft Edge TTS, and muxes the synchronized dub back into
the original video.

## Requirements

- Python 3.10–3.14
- `ffmpeg` and `ffprobe` available on `PATH`
- An internet connection for initial model downloads, Edge TTS, and YouTube

Whisper and translation run locally. Offline translation uses translation
models in the Argos package format directly through CTranslate2, without
pulling Argos's multi-gigabyte PyTorch and CUDA dependency stack. Speech
synthesis uses the public Edge TTS service.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[visual,clone]'
```

With the Fish shell, activate it with:

```fish
source .venv/bin/activate.fish
```

For an NVIDIA GPU, install a CUDA-compatible CTranslate2 build according to
the `faster-whisper` documentation. CPU mode works without CUDA.

## Use

Launch the terminal interface:

```bash
voicecover
# or: voicecover-tui
```

The TUI accepts either a local path or YouTube URL and shows each pipeline
stage without blocking the interface.

### Web interface

Launch the local web application:

```bash
voicecover-web
```

For the Pondok Huda-style interactive stack manager, run:

```bash
./tui.sh
# equivalent: ./dev.sh
```

The ncurses dashboard shows the web app, local LLM server, and HTTPS tunnel,
including health, PID, uptime, CPU, memory, endpoints, and live component logs.
Use arrow keys to select a service, `Enter` to toggle it, `a` to start the web
service, `t` to toggle the HTTPS tunnel, `l` for a full log, and `?` for help.
Quitting the dashboard does not stop managed services.

The same manager is scriptable and keeps PID records and unbuffered logs under
`.dev-runtime/`:

```bash
./dev.sh start                 # start the web service in the background
./dev.sh start all             # start web, local LLM, and HTTPS tunnel
./dev.sh start llm             # start only the local translation server
./dev.sh start tunnel          # add a temporary HTTPS tunnel
./dev.sh adopt all             # manage matching services already running
./dev.sh status
./dev.sh logs web
./dev.sh restart web           # restores an active tunnel afterward
./dev.sh stop                  # stop the complete managed stack
```

Tunnel service always uses an accountless Cloudflare quick tunnel. Its public
`trycloudflare.com` URL is temporary and changes whenever the tunnel restarts;
no fixed hostname or tunnel credential is used.

Then open <http://127.0.0.1:8000>. The interface accepts a browser upload or a
YouTube URL, runs intensive jobs outside the request thread, reports each
pipeline stage, and provides the finished video and optional SRT as downloads.
Jobs are persisted under `VOICECOVER_JOB_DIR` and run one at a time in upload
order. Interrupted jobs remain visible and require an explicit retry. Multiple
uploaded videos form one durable sequential batch. Set `VOICECOVER_WEB_PORT`
to change the port, and
`VOICECOVER_MAX_UPLOAD_MB` to change the 4096 MB upload limit.

The server listens on `0.0.0.0` by default, so every interface on the machine
answers — LAN and any VPN tunnel. To open it from another machine over a
ZeroTier network, join the same ZeroTier network on both ends, find the
server's ZeroTier address (interface `zt...`, shown by `voicecover-web` at
startup, or `ip -4 -br addr show`), and open `http://<zerotier-ip>:8000`.
Set `VOICECOVER_WEB_HOST=127.0.0.1` to restrict the app to this machine
only, or use the OS firewall to allow just the VPN interface. The app has no
authentication: anyone who can reach the port can start jobs and download
results, so keep the ZeroTier network private.

The Job vault lists every submitted video. Select a row to inspect it in the
processing room; each row can expose its timestamped processing log and offers
the actions valid for its state: pause, resume, discard, retry, download, or
permanently delete the job and its files. A running job must be paused or
cancelled before deletion.

Set `VOICECOVER_OUTPUT_DIR` to publish a durable copy of each finished video
and its optional SRT outside the disposable per-job cache. Publishing is
atomic, and deleting a web job does not remove its published copy:

```bash
export VOICECOVER_OUTPUT_DIR="$HOME/baqi-vid/vid-output"
```

Each job in the processing room has a Pause button. A running job stops at
the next pipeline checkpoint — its transcription, translation, and speech
caches are kept — and a paused job holds the queue until it is resumed or
cancelled. Resume re-enters the pipeline at the saved checkpoint instead of
starting over. A stage already in progress (for example a long Whisper
transcription) finishes before the pause takes effect.

### Job notifications (ntfy)

Set `VOICECOVER_NTFY_URL` to an ntfy topic URL and every job state change —
queued, started, paused, completed, failed — is published to that topic:

```bash
export VOICECOVER_NTFY_URL="https://ntfy.sh/long-random-topic-name"
# optional, for protected topics:
export VOICECOVER_NTFY_TOKEN="tk_..."
# optional, makes notifications tappable links back to this instance:
export VOICECOVER_PUBLIC_URL="http://10.147.17.42:8000"
```

Subscribe to the same topic on your phone with the ntfy app, or in a browser
at <https://ntfy.sh>. Topic names are the only secret — pick a long random
one, since anyone who knows the topic can read it. Notifications are sent by
a background watcher, so a server restart never re-announces old jobs.
Verify the pipe with:

```bash
curl -X POST http://127.0.0.1:8000/api/notifications/test
```

`VOICECOVER_NOTIFY_INTERVAL` changes the watcher's sweep period in seconds
(default `2`).

The default balanced profile uses the multilingual Whisper `small` model with
CPU `int8` inference and a beam size of 5. This is the recommended setting for
older multicore workstations. Choose `tiny` for faster previews; `medium` and
`large-v3` need substantially more processing time and model storage.

Natural speech grouping is enabled by default. Nearby Whisper fragments are
combined into sentence-like utterances before translation and synthesis. This
gives the translator more context and prevents Edge TTS from resetting its
intonation at every short subtitle fragment. Synthesized clips have leading
and trailing silence removed, short speech is padded instead of unnaturally
slowed, and long speech is fitted before the next utterance without shifting
the rest of the timeline. Use `--no-natural-speech` only for diagnostics.

For accuracy-sensitive material, select `medium` or `large-v3`, specify the
dominant source language, and provide names or specialist vocabulary in the
Whisper context field. The implementation follows the decoding controls from
the faster-whisper API, sentence-size limits described by the NLLB model card,
and FFmpeg's `silenceremove`/`atempo` timing guidance.

The default NLLB provider translates directly between languages with a 645 MB
int8 model, improving accuracy over pivot translation. The model is licensed
CC-BY-NC-4.0 and is therefore for non-commercial use. The smaller Argos option
uses direct language pairs where possible and otherwise pivots through English.
For Arabic to Indonesian its two models occupy about 169 MB after extraction.
Install them in advance for offline deployment:

```bash
voicecover-models ar id
```

Set `VOICECOVER_MODELS_DIR` to place models on another disk. Pass
`--no-install-models` to prohibit downloads at runtime. The former Google
provider remains available as `--translator google`, but is unofficial and
not recommended for reliable jobs.

On the tested CPU deployment, the virtual environment occupies about 517 MB,
the Whisper `small` cache about 539 MB, and the Arabic/English/Indonesian
translation route 169 MB. The working installation is therefore roughly
1.2 GB before output videos. Argos model quality varies by language pair;
pivot translation is compact and reliable but may be less fluent than a
larger multilingual model.

The command-line mode remains available for scripts:

```bash
voicecover input.mp4 --target es --output input.spanish.mp4

# YouTube links work directly.
voicecover 'https://www.youtube.com/watch?v=VIDEO_ID' --target es

# Quran-course defaults: local Indonesian translation, cloned presenter voice,
# original Arabic passages, Indonesian SRT, and localized static slides.
voicecover lesson.mp4 --profile quran-id -o lesson.id.mp4

# Translate only static on-screen text and keep the original soundtrack.
voicecover lesson.mp4 -t id --visual-only -o lesson.visual-id.mp4

# Translate and dub speech while preserving the original video pixels.
voicecover lesson.mp4 -t id --sound-only -o lesson.audio-id.mp4
```

### Local LLM translation

The `quran-id` profile uses a local OpenAI-compatible endpoint at
`http://127.0.0.1:8081/v1/chat/completions`. A CPU build of llama.cpp and the
Qwen3.5 4B Q4_K_M model can be installed and started with:

```bash
scripts/setup-llm.sh
scripts/llm-server.sh
```

The model is stored outside the repository under
`~/.local/share/voicecover/models`. Override the endpoint or model identifier
with `VOICECOVER_LLM_URL` and `VOICECOVER_LLM_MODEL`.

On CPU, Voicecover uses up to 16 physical cores for Whisper and runs one F5
clone worker per NUMA socket. Override these conservative defaults with
`VOICECOVER_WHISPER_THREADS` and `VOICECOVER_CLONE_WORKERS` if needed. These
settings change throughput only; model size, beam search, and F5 diffusion
steps remain unchanged.

### Durable sequential batches

Create a UTF-8 JSON manifest:

```json
{
  "version": 1,
  "id": "quran-series",
  "name": "Quran course",
  "jobs": [
    {
      "id": "lesson-01",
      "input": "https://youtu.be/VIDEO_ID",
      "output": "output/lesson-01.id.mp4",
      "profile": "quran-id"
    }
  ]
}
```

Run or resume it sequentially:

```bash
voicecover --batch batch.json --job-store .voicecover-jobs
voicecover --batch batch.json --job-store .voicecover-jobs --retry-failed
```

For mixed-language or specialist material, set the dominant spoken language
and give Whisper a short vocabulary hint. This prevents an Arabic opening
prayer from making an otherwise English lesson get transcribed entirely as
Arabic:

```bash
voicecover lesson.mp4 -t id --source en \
  --initial-prompt "English Quran lesson, Surah Al-Baqarah, Arabic grammar and Quranic vocabulary"
```

Useful options:

```bash
# Use a larger Whisper model and a specific Edge TTS voice.
voicecover movie.mp4 -t fr --model medium --voice fr-FR-DeniseNeural

# Remove the original audio instead of keeping it quietly underneath.
voicecover movie.mp4 -t ja --original-volume 0

# Keep temporary audio files for debugging.
voicecover movie.mp4 -t de --keep-temp
```

Language values may be short codes (`es`, `fr`, `ja`) or locales
(`pt-BR`, `en-GB`). Run `edge-tts --list-voices` after installation to see
available voices. If `--voice` is omitted, Voicecover selects a matching
voice automatically.

If `--output` is omitted, Voicecover writes an MP4 named like
`input.es.mp4`, or `youtube.es.mp4` for a YouTube URL. Playlists are ignored;
only the linked video is downloaded. Explicit output paths must use MP4, MKV,
or MOV.

The command also writes an `.srt` file beside the output video containing the
translated text. Use `--no-srt` to disable it.

## How timing works

Each generated speech clip is fitted into the original transcript segment.
Long clips are accelerated just enough to fit; short clips are padded with
silence. This gives deterministic synchronization without changing the video.
Very dense or highly expressive speech may still sound fast after translation.

By default, the original soundtrack is mixed at 15% volume so music and sound
effects remain audible. This also leaves faint original dialogue. Set
`--original-volume 0` for a clean replacement, or use source separation before
running Voicecover if you need music/effects without the original voices.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
```
