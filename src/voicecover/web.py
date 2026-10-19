from __future__ import annotations

import atexit
import logging
import os
import re
import shutil
import socket
import struct
import threading
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any, Callable

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

from .download import is_youtube_url
from .jobs import (
    InvalidTransition,
    JobPaused,
    JobRecord,
    JobSpec,
    JobStore as DurableJobStore,
)
from .pipeline import Options, run
from .profiles import (
    QURAN_INDONESIAN_PROFILE,
    ProfileError,
    list_profiles,
    load_profile_or_builtin,
)
from .synthesis import available_voices


MAX_UPLOAD_BYTES = int(os.environ.get("VOICECOVER_MAX_UPLOAD_MB", "4096")) * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
# Optional ntfy topic (e.g. https://ntfy.sh/long-secret-name). When set, every
# job state change is published so the instance can be monitored off-machine.
NTFY_URL = os.environ.get("VOICECOVER_NTFY_URL", "").strip()
NTFY_TOKEN = os.environ.get("VOICECOVER_NTFY_TOKEN", "").strip()
PUBLIC_URL = os.environ.get("VOICECOVER_PUBLIC_URL", "").strip().rstrip("/")
NOTIFY_INTERVAL = max(0.5, float(os.environ.get("VOICECOVER_NOTIFY_INTERVAL", "2")))
OUTPUT_DIR = os.environ.get("VOICECOVER_OUTPUT_DIR", "").strip()
_SAFE_PART = re.compile(r"[^a-zA-Z0-9._-]+")
_job_log = logging.getLogger("voicecover.jobs")
_job_log.setLevel(logging.INFO)
_log_dir = Path(os.environ.get("VOICECOVER_LOG_DIR", str(Path(__file__).resolve().parent.parent.parent / ".dev-runtime" / "logs")))
_log_dir.mkdir(parents=True, exist_ok=True)
_job_handler = logging.FileHandler(_log_dir / "jobs.log")
_job_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
_job_log.addHandler(_job_handler)


def _safe_part(value: str, fallback: str) -> str:
    cleaned = _SAFE_PART.sub("-", value.strip()).strip(".-")
    return cleaned[:80] or fallback


def _publish_output(output: Path | str, record: JobRecord) -> Path | None:
    """Atomically copy a finished web result into the configured output vault."""
    if not OUTPUT_DIR or not record.payload.get("web_managed"):
        return None
    source = Path(output)
    destination_dir = Path(OUTPUT_DIR).expanduser()
    destination_dir.mkdir(parents=True, exist_ok=True)
    name = source.name
    if name.startswith("youtube."):
        name = f"youtube-{record.id[:8]}{source.suffix}"
    destination = destination_dir / name

    def publish_file(original: Path, target: Path) -> None:
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            shutil.copy2(original, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    publish_file(source, destination)
    subtitle = source.with_suffix(".srt")
    if subtitle.is_file():
        publish_file(subtitle, destination.with_suffix(".srt"))
    return destination


def _asset(name: str) -> str:
    return files("voicecover.web_assets").joinpath(name).read_text(encoding="utf-8")


def ntfy_send(
    message: str,
    *,
    priority: int = 3,
    tags: str = "",
    click: str | None = None,
) -> None:
    """Publish one message to the configured ntfy topic. Raises on failure."""
    url = NTFY_URL
    if not url:
        raise RuntimeError("VOICECOVER_NTFY_URL is not configured")
    headers = {
        "Content-Type": "text/plain; charset=utf-8",
        "Title": "voicecover",
        "Priority": str(priority),
    }
    if tags:
        headers["Tags"] = tags
    target = PUBLIC_URL if click is None else click
    if target:
        headers["Click"] = target
    if NTFY_TOKEN:
        headers["Authorization"] = f"Bearer {NTFY_TOKEN}"
    request = urllib.request.Request(
        url, data=message.encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()


_NOTIFY_EVENTS: dict[str, tuple[str, int, str]] = {
    "queued": ("queued", 3, "hourglass_flowing_sand"),
    "running": ("started", 3, "gear"),
    "paused": ("paused", 3, "pause_button"),
    "completed": ("completed", 4, "white_check_mark"),
    "failed": ("failed", 5, "x"),
    "cancelled": ("cancelled", 3, "no_entry_sign"),
    "interrupted": ("interrupted", 4, "warning"),
}


class _JobNotifier:
    """Watch job state and publish one ntfy message per transition."""

    def __init__(
        self,
        job_store: "JobStore",
        *,
        send: Callable[..., None] | None = None,
        interval: float = NOTIFY_INTERVAL,
    ) -> None:
        self.store = job_store
        self.interval = interval
        self._send = send
        self._seen: dict[str, str] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def seed(self) -> None:
        """Baseline current jobs silently, e.g. after a server restart."""
        self._seen = {record.id: record.status for record in self.store.durable.all_jobs()}

    def _format(self, record: JobRecord) -> tuple[str, int, str] | None:
        event = _NOTIFY_EVENTS.get(record.status)
        if event is None:
            return None
        label, priority, tags = event
        message = f"{label} · {record.name}"
        if record.status == "failed" and record.error:
            message = f"{message} · {record.error}"
        return message[:500], priority, tags

    def tick(self) -> list[str]:
        sent: list[str] = []
        sender = self._send if self._send is not None else ntfy_send
        for record in self.store.durable.all_jobs():
            if self._seen.get(record.id) == record.status:
                continue
            self._seen[record.id] = record.status
            formatted = self._format(record)
            if formatted is None:
                continue
            message, priority, tags = formatted
            try:
                sender(message, priority=priority, tags=tags, click=PUBLIC_URL)
            except Exception:
                _job_log.exception("job notification failed for %s", record.id[:8])
                continue
            sent.append(message)
        return sent

    def _loop(self) -> None:
        self.seed()
        while not self._stop.wait(self.interval):
            try:
                self.tick()
            except Exception:
                _job_log.exception("job notification sweep failed")

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="voicecover-notify", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


@dataclass(slots=True)
class Job:
    store: "JobStore"
    id: str

    @property
    def _record(self) -> JobRecord:
        return self.store.durable.get_job(self.id)

    @property
    def options(self) -> Options:
        return _options_from_record(self._record)

    @property
    def directory(self) -> Path:
        output = self._record.output
        return Path(output).parent if output else self.store.root / self.id

    @property
    def status(self) -> str:
        return self._record.status

    @property
    def progress(self) -> list[str]:
        return self._record.progress

    @property
    def error(self) -> str | None:
        return self._record.error

    @property
    def output(self) -> Path | None:
        record = self._record
        return Path(record.output) if record.status == "completed" and record.output else None

    @property
    def pause_pending(self) -> bool:
        return self.store.pause_pending(self.id)

    def public(self) -> dict[str, object]:
        record = self._record
        result: dict[str, object] = {
            "id": self.id,
            "name": record.name,
            "status": record.status,
            "progress": record.progress.copy(),
            "error": record.error,
            "attempts": record.attempts,
            "created_at": record.created_at,
            "first_started_at": record.first_started_at,
            "finished_at": record.finished_at,
            "active_seconds": record.active_seconds,
            "active_started_at": record.active_started_at,
            "pause_pending": self.pause_pending,
        }
        if self.output is not None:
            result["video_url"] = f"/api/jobs/{self.id}/video"
            subtitle = self.output.with_suffix(".srt")
            if subtitle.is_file():
                result["subtitle_url"] = f"/api/jobs/{self.id}/subtitles"
        source = record.payload.get("input")
        if source:
            source = str(source)
            # web jobs are named after their uuid directory; show the real source
            result["source"] = source if is_youtube_url(source) else Path(source).name
        return result


class JobStore:
    def __init__(self, root: Path | None = None, workers: int = 4, max_concurrent: int = 4) -> None:
        self.root = root or Path(
            os.environ.get("VOICECOVER_JOB_DIR", str(Path.home() / "baqi-vid" / "jobs"))
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.durable = DurableJobStore(self.root, recover_running="interrupted")
        self.lock = threading.RLock()
        # Pause bookkeeping. A flag is only meaningful while its worker is
        # still inside _execute; the active set closes the race where a pause
        # request lands after the worker already stopped.
        self._pause_flags: set[str] = set()
        self._active_jobs: set[str] = set()
        # process_video is CPU-intensive. A single dispatcher also preserves
        # durable queue order regardless of the legacy constructor arguments.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voicecover")
        self.max_concurrent = 1
        self._closed = False
        self._schedule()

    @property
    def jobs(self) -> dict[str, Job]:
        return {record.id: Job(self, record.id) for record in self.durable.all_jobs()}

    def add(self, options: Options, directory: Path, *, web_managed: bool = False) -> Job:
        record = self.durable.create_job(
            directory.name,
            payload=_options_payload(options, web_managed=web_managed),
            output=options.output,
            job_id=directory.name,
        )
        self._schedule()
        return Job(self, record.id)

    def add_batch(self, name: str, items: list[tuple[str, Options]]) -> list[Job]:
        specs = [
            JobSpec(
                name=Path(options.input).name,
                payload=_options_payload(options, web_managed=True),
                output=options.output,
                id=job_id,
            )
            for job_id, options in items
        ]
        batch = self.durable.create_batch(name, specs)
        self._schedule()
        return [Job(self, job_id) for job_id in batch.job_ids]

    def get(self, job_id: str) -> Job:
        self.durable.get_job(job_id)
        return Job(self, job_id)

    def retry(self, job_id: str) -> Job:
        self.durable.get_job(job_id)
        self._clear_pause(job_id)
        # The web vault exposes an explicit Reprocess action for completed
        # outputs; CLI retry remains conservative by default.
        record = self.durable.retry(job_id, allow_completed=True)
        self._schedule()
        return Job(self, record.id)

    def pause(self, job_id: str) -> Job:
        """Pause a queued job immediately; a running worker stops at its next checkpoint."""
        before = self.durable.get_job(job_id).status
        if before == "running":
            with self.lock:
                self._pause_flags.add(job_id)
        try:
            record = self.durable.pause(job_id)
        except BaseException:
            with self.lock:
                self._pause_flags.discard(job_id)
            raise
        if before in {"queued", "running"}:
            _job_log.info("job %s pause requested (was %s)", job_id[:8], before)
        return Job(self, record.id)

    def resume(self, job_id: str) -> Job:
        self.durable.get_job(job_id)
        self._clear_pause(job_id)
        record = self.durable.retry(job_id)
        _job_log.info("job %s resumed", job_id[:8])
        self._schedule()
        return Job(self, record.id)

    def cancel(self, job_id: str) -> Job:
        """Discard a job; like pause, a running worker stops at its next checkpoint."""
        before = self.durable.get_job(job_id).status
        if before == "running":
            with self.lock:
                self._pause_flags.add(job_id)
        try:
            record = self.durable.cancel(job_id)
        except BaseException:
            with self.lock:
                self._pause_flags.discard(job_id)
            raise
        _job_log.info("job %s cancelled (was %s)", job_id[:8], before)
        # A cancelled paused job may release the queue gate.
        self._schedule()
        return Job(self, record.id)

    def delete(self, job_id: str) -> JobRecord:
        """Remove a job and its web-managed files; a running worker must stop first."""
        self.durable.get_job(job_id)
        if self.pause_pending(job_id):
            raise InvalidTransition(
                "the job is still stopping at a checkpoint; try again in a moment"
            )
        with self.lock:
            self._pause_flags.discard(job_id)
        record = self.durable.delete_job(job_id)
        root = self.root.resolve()
        directory = (self.root / record.id).resolve()
        if directory != root and directory.is_relative_to(root):
            shutil.rmtree(directory, ignore_errors=True)
        _job_log.info("job %s deleted", record.id[:8])
        # Deleting a paused job may release the queue gate.
        self._schedule()
        return record

    def pause_pending(self, job_id: str) -> bool:
        """True while a pause was requested and the worker has not stopped yet."""
        with self.lock:
            return job_id in self._pause_flags and job_id in self._active_jobs

    def _clear_pause(self, job_id: str) -> None:
        if self.pause_pending(job_id):
            raise InvalidTransition(
                "the job is still stopping at a checkpoint; try again in a moment"
            )
        with self.lock:
            self._pause_flags.discard(job_id)

    def _schedule(self) -> None:
        with self.lock:
            if not self._closed:
                self.executor.submit(self._drain)

    def _queue_held(self) -> bool:
        """A paused job holds the queue until it is resumed or cancelled."""
        return any(job.status == "paused" for job in self.durable.all_jobs())

    def _drain(self) -> None:
        while not self._queue_held():
            if self.durable.dispatch_one(self._execute) is None:
                return

    def _execute(self, record: JobRecord, progress) -> Path:
        options = _options_from_record(record)
        with self.lock:
            self._active_jobs.add(record.id)
        try:
            if record.payload.get("web_managed"):
                directory = (self.root / record.id).resolve()
                unsafe_input = not is_youtube_url(str(options.input)) and Path(options.input).resolve().parent != directory
                if unsafe_input or Path(options.output).resolve().parent != directory:
                    raise RuntimeError("persisted web job contains an unsafe path")

            def report(message: str) -> None:
                # The pause flag turns the next progress boundary into a
                # checkpoint: caches persist and resume re-enters the pipeline.
                if record.id in self._pause_flags:
                    raise JobPaused(f"job {record.id} paused on request")
                progress(str(message))
                _job_log.info("job %s · %s", record.id[:8], message)

            try:
                output = run(options, progress=report)
            except JobPaused:
                _job_log.info("job %s paused at a checkpoint", record.id[:8])
                raise
            except Exception:
                _job_log.exception("job %s failed", record.id[:8])
                raise
            published = _publish_output(output, record)
            if published is not None:
                _job_log.info("job %s published · %s", record.id[:8], published)
            _job_log.info("job %s completed · %s", record.id[:8], output)
            return output
        finally:
            with self.lock:
                self._pause_flags.discard(record.id)
                self._active_jobs.discard(record.id)

    @property
    def running_count(self) -> int:
        return sum(1 for job in self.durable.all_jobs() if job.status in ("running", "queued"))

    def close(self) -> None:
        with self.lock:
            self._closed = True
        self.executor.shutdown(wait=False, cancel_futures=True)


_OPTION_FIELDS = {field.name for field in fields(Options)} - {"output"}


def _options_payload(options: Options, *, web_managed: bool = False) -> dict[str, Any]:
    payload = {name: getattr(options, name) for name in _OPTION_FIELDS}
    payload["input"] = os.fspath(options.input)
    payload["protected_terms"] = dict(options.protected_terms)
    payload["web_managed"] = web_managed
    return payload


def _options_from_record(record: JobRecord) -> Options:
    payload = dict(record.payload)
    payload.pop("web_managed", None)
    payload.setdefault("visual_only", False)
    payload.setdefault("sound_only", False)
    if set(payload) != _OPTION_FIELDS or record.output is None:
        raise ValueError("persisted web job has invalid options")
    return Options(**payload, output=Path(record.output))


store = JobStore(
    workers=max(1, int(os.environ.get("VOICECOVER_WEB_WORKERS", "4"))),
    max_concurrent=max(1, int(os.environ.get("VOICECOVER_MAX_CONCURRENT_JOBS", "4"))),
)
atexit.register(store.close)
notifier = _JobNotifier(store)
if NTFY_URL:
    notifier.start()
    atexit.register(notifier.stop)
app = FastAPI(title="Voicecover", docs_url=None, redoc_url=None)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _asset("index.html")


@app.get("/app.css")
def stylesheet() -> Response:
    return Response(_asset("app.css"), media_type="text/css")


@app.get("/app.js")
def javascript() -> Response:
    return Response(_asset("app.js"), media_type="text/javascript")


@app.get("/api/status")
def server_status() -> dict[str, object]:
    jobs = list(store.jobs.values())
    return {
        "running": sum(1 for j in jobs if j.status == "running"),
        "queued": sum(1 for j in jobs if j.status == "queued"),
        "paused": sum(1 for j in jobs if j.status == "paused"),
        "completed": sum(1 for j in jobs if j.status == "completed"),
        "failed": sum(1 for j in jobs if j.status == "failed"),
        "interrupted": sum(1 for j in jobs if j.status == "interrupted"),
        "max_concurrent": store.max_concurrent,
        "notifications": bool(NTFY_URL),
    }


@app.get("/api/voices")
async def voices(language: str = "id") -> dict[str, object]:
    target = language.strip()
    if not target:
        raise HTTPException(400, "language is required")
    try:
        matches = await available_voices(target)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {
        "language": target,
        "voices": [
            {
                "name": str(voice["ShortName"]),
                "locale": str(voice["Locale"]),
                "gender": str(voice.get("Gender", "Unknown")),
            }
            for voice in matches
        ],
    }


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    size = 0
    try:
        with destination.open("wb") as output:
            while chunk := await upload.read(CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "video exceeds the configured upload limit")
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


@app.get("/api/profiles")
def profiles() -> dict[str, object]:
    names = sorted(set([QURAN_INDONESIAN_PROFILE, *list_profiles()]))
    result = []
    for name in names:
        try:
            profile = load_profile_or_builtin(name)
        except (OSError, ProfileError):
            continue
        values = profile.option_overrides()
        values["protected_terms"] = profile.protected_terms
        values["write_visual_track"] = name == QURAN_INDONESIAN_PROFILE
        values["localize_visual_text"] = name == QURAN_INDONESIAN_PROFILE
        result.append({"name": name, "built_in": name == QURAN_INDONESIAN_PROFILE, "options": values})
    return {"profiles": result}


@app.post("/api/jobs", status_code=202)
async def create_job(
    video: Annotated[UploadFile | None, File()] = None,
    videos: Annotated[list[UploadFile] | None, File()] = None,
    youtube_url: Annotated[str, Form()] = "",
    profile: Annotated[str, Form()] = "",
    target_language: Annotated[str, Form()] = "id",
    source_language: Annotated[str, Form()] = "",
    model: Annotated[str, Form()] = "small",
    device: Annotated[str, Form()] = "cpu",
    translator: Annotated[str, Form()] = "auto",
    voice: Annotated[str, Form()] = "",
    initial_prompt: Annotated[str, Form()] = "",
    original_volume: Annotated[float, Form()] = 0.15,
    write_subtitles: Annotated[bool, Form()] = True,
    install_models: Annotated[bool, Form()] = True,
    preserve_arabic: Annotated[bool, Form()] = True,
    natural_speech: Annotated[bool, Form()] = True,
    write_segments: Annotated[bool, Form()] = False,
    write_bilingual: Annotated[bool, Form()] = False,
    visual_context: Annotated[bool, Form()] = True,
    voice_clone: Annotated[bool, Form()] = False,
    clone_model: Annotated[str, Form()] = "",
    clone_nfe: Annotated[int, Form()] = 16,
    write_visual_track: Annotated[bool, Form()] = False,
    localize_visual_text: Annotated[bool, Form()] = False,
    visual_only: Annotated[bool, Form()] = False,
    sound_only: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    url = youtube_url.strip()
    uploads = [item for item in (videos or []) if item.filename]
    if video and video.filename:
        uploads.insert(0, video)
    if bool(uploads) == bool(url):
        raise HTTPException(400, "choose either video upload(s) or a YouTube URL")
    if url and not is_youtube_url(url):
        raise HTTPException(400, "only YouTube URLs are supported")
    profile_name = profile.strip()
    selected_profile = None
    if profile_name:
        try:
            selected_profile = load_profile_or_builtin(profile_name)
        except (FileNotFoundError, ProfileError) as exc:
            raise HTTPException(400, str(exc)) from exc
    target = selected_profile.target_language if selected_profile else target_language.strip()
    if not target:
        raise HTTPException(400, "target language is required")
    if model not in {"tiny", "base", "small", "medium", "large-v3"}:
        raise HTTPException(400, "invalid Whisper model")
    if device not in {"auto", "cpu", "cuda"}:
        raise HTTPException(400, "invalid compute device")
    if translator not in {"auto", "nllb", "argos", "google", "llm"}:
        raise HTTPException(400, "invalid translation provider")
    if not 0 <= original_volume <= 1:
        raise HTTPException(400, "original volume must be between 0 and 1")
    if visual_only and sound_only:
        raise HTTPException(400, "visual-only and sound-only modes are mutually exclusive")

    values: dict[str, Any] = {
        "source_language": source_language.strip() or None,
        "model": model,
        "device": device,
        "voice": voice.strip() or None,
        "original_volume": original_volume,
        "write_subtitles": write_subtitles,
        "translator": translator,
        "install_models": install_models,
        "initial_prompt": initial_prompt.strip() or None,
        "preserve_arabic": preserve_arabic,
        "natural_speech": natural_speech,
        "write_segments": write_segments,
        "write_bilingual": write_bilingual,
        "visual_context": visual_context,
        "voice_clone": voice_clone,
        "clone_model": clone_model.strip() or None,
        "clone_nfe": max(4, min(64, clone_nfe)),
        "protected_terms": {},
        "write_visual_track": write_visual_track,
        "localize_visual_text": localize_visual_text or visual_only,
        "visual_only": visual_only,
        "sound_only": sound_only,
    }
    if selected_profile:
        profile_values = selected_profile.option_overrides()
        profile_values.pop("target_language")
        values.update(profile_values)
        values["protected_terms"] = dict(selected_profile.protected_terms)
        if profile_name == QURAN_INDONESIAN_PROFILE:
            values["write_visual_track"] = True
            values["localize_visual_text"] = True
    if visual_only:
        values["visual_context"] = True
        values["localize_visual_text"] = True
        values["voice_clone"] = False
    elif sound_only:
        values["visual_context"] = False
        values["write_visual_track"] = False
        values["localize_visual_text"] = False

    prepared: list[tuple[str, Options]] = []
    directories: list[Path] = []
    try:
        if uploads:
            for upload in uploads:
                job_id = uuid.uuid4().hex
                directory = store.root / job_id
                directory.mkdir()
                directories.append(directory)
                filename = _safe_part(Path(upload.filename or "").name, "upload.mp4")
                input_value = directory / filename
                await _save_upload(upload, input_value)
                stem = _safe_part(Path(filename).stem, "dubbed")
                output = directory / f"{stem}.{_safe_part(target, 'dubbed')}.mp4"
                prepared.append((job_id, Options(input=input_value, output=output, target_language=target, **values)))
        else:
            job_id = uuid.uuid4().hex
            directory = store.root / job_id
            directory.mkdir()
            directories.append(directory)
            output = directory / f"youtube.{_safe_part(target, 'dubbed')}.mp4"
            prepared.append((job_id, Options(input=url, output=output, target_language=target, **values)))

        if len(prepared) == 1:
            job_id, options = prepared[0]
            job = store.add(options, store.root / job_id, web_managed=True)
            return job.public()
        jobs = store.add_batch(f"{len(prepared)} uploaded videos", prepared)
        return {
            "jobs": [job.public() for job in jobs],
            "batch": True,
            "batch_id": jobs[0]._record.batch_id,
        }
    except Exception:
        for directory in directories:
            shutil.rmtree(directory, ignore_errors=True)
        raise


@app.get("/api/jobs")
def list_jobs() -> dict[str, object]:
    jobs = [job.public() for job in store.jobs.values()]
    return {"jobs": jobs}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, object]:
    try:
        return store.get(job_id).public()
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, "job not found") from exc


@app.post("/api/jobs/{job_id}/retry", status_code=202)
def retry_job(job_id: str) -> dict[str, object]:
    try:
        return store.retry(job_id).public()
    except KeyError as exc:
        raise HTTPException(404, "job not found") from exc
    except (InvalidTransition, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/jobs/{job_id}/pause", status_code=202)
def pause_job(job_id: str) -> dict[str, object]:
    try:
        return store.pause(job_id).public()
    except InvalidTransition as exc:
        raise HTTPException(409, str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, "job not found") from exc


@app.post("/api/jobs/{job_id}/resume", status_code=202)
def resume_job(job_id: str) -> dict[str, object]:
    try:
        return store.resume(job_id).public()
    except InvalidTransition as exc:
        raise HTTPException(409, str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, "job not found") from exc


@app.post("/api/jobs/{job_id}/cancel", status_code=202)
def cancel_job(job_id: str) -> dict[str, object]:
    try:
        return store.cancel(job_id).public()
    except InvalidTransition as exc:
        raise HTTPException(409, str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, "job not found") from exc


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> Response:
    try:
        record = store.delete(job_id)
    except InvalidTransition as exc:
        raise HTTPException(409, str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, "job not found") from exc
    if NTFY_URL:
        try:
            ntfy_send(f"deleted · {record.name}", priority=3, tags="wastebasket", click=PUBLIC_URL)
        except Exception:
            _job_log.info("job %s delete notice failed", job_id[:8])
    return Response(status_code=204)


@app.get("/api/jobs/{job_id}/log")
def job_log(job_id: str) -> dict[str, object]:
    try:
        job = store.get(job_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, "job not found") from exc
    marker = f"job {job_id[:8]}"
    lines: list[str] = []
    try:
        with (_log_dir / "jobs.log").open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if marker in line:
                    lines.append(line.rstrip())
    except OSError:
        pass
    return {
        "id": job_id,
        "status": job.public()["status"],
        "progress": list(job.public()["progress"]),
        # durable progress caps at 100 lines; the file keeps the full history
        "lines": lines[-500:],
    }


@app.post("/api/notifications/test")
def test_notification() -> dict[str, object]:
    if not NTFY_URL:
        raise HTTPException(400, "VOICECOVER_NTFY_URL is not configured")
    try:
        ntfy_send("test notification from voicecover", priority=3, tags="bell", click=PUBLIC_URL)
    except Exception as exc:
        raise HTTPException(502, f"ntfy publish failed: {exc}") from exc
    return {"sent": True}


def _result(job_id: str, subtitles: bool = False) -> FileResponse:
    try:
        job = store.get(job_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, "job not found") from exc
    if job.status != "completed" or job.output is None:
        raise HTTPException(409, "job has not completed")
    path = job.output.with_suffix(".srt") if subtitles else job.output
    directory = (store.root / job.id).resolve()
    if path.resolve().parent != directory:
        raise HTTPException(404, "result file not found")
    if not path.is_file():
        raise HTTPException(404, "result file not found")
    media_type = "application/x-subrip" if subtitles else "video/mp4"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/api/jobs/{job_id}/video")
def download_video(job_id: str) -> FileResponse:
    return _result(job_id)


@app.get("/api/jobs/{job_id}/subtitles")
def download_subtitles(job_id: str) -> FileResponse:
    return _result(job_id, subtitles=True)


def _interface_addresses() -> list[tuple[str, str]]:
    """Return non-loopback IPv4 addresses as (interface, address) pairs."""
    results: list[tuple[str, str]] = []
    try:
        import fcntl

        socket_ioctl = 0x8915  # SIOCGIFADDR
        for _, name in socket.if_nameindex():
            if name == "lo":
                continue
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                packed = fcntl.ioctl(
                    probe.fileno(), socket_ioctl, struct.pack("256s", name[:15].encode())
                )
            except OSError:
                continue
            finally:
                probe.close()
            results.append((name, socket.inet_ntoa(packed[20:24])))
    except (AttributeError, OSError, ValueError):
        pass
    return results


def main() -> None:
    host = os.environ.get("VOICECOVER_WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("VOICECOVER_WEB_PORT", "8000"))
    if host in {"0.0.0.0", "::"}:
        print(
            "voicecover web is reachable from every interface on this machine:",
            flush=True,
        )
        for name, address in _interface_addresses():
            label = "zerotier" if name.startswith("zt") else name
            print(f"  http://{address}:{port}   ({label})", flush=True)
        print(
            "  restricted to this machine only with VOICECOVER_WEB_HOST=127.0.0.1",
            flush=True,
        )
    uvicorn.run(
        "voicecover.web:app",
        host=host,
        port=port,
    )


if __name__ == "__main__":
    main()
