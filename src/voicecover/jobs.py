from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping


MANIFEST_VERSION = 1
JOB_STATUSES = frozenset(
    {"queued", "running", "paused", "completed", "failed", "cancelled", "interrupted"}
)
BATCH_STATUSES = JOB_STATUSES
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted"})

RecoveryMode = Literal["interrupted", "queued"]
Worker = Callable[["JobRecord", Callable[[str], None]], object]

_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_TRANSITIONS = {
    "queued": frozenset({"running", "paused", "cancelled"}),
    "running": frozenset({"completed", "failed", "cancelled", "interrupted", "paused"}),
    "paused": frozenset({"queued", "cancelled", "completed"}),
    "completed": frozenset({"queued"}),
    "failed": frozenset({"queued", "cancelled"}),
    "cancelled": frozenset({"queued"}),
    "interrupted": frozenset({"queued", "cancelled"}),
}


class ManifestError(ValueError):
    """Raised when a persisted manifest is invalid or unsupported."""


class InvalidTransition(ValueError):
    """Raised when a job cannot move directly to the requested status."""


class JobPaused(RuntimeError):
    """Raised by a worker at a checkpoint after a pause was requested."""


def validate_id(value: str) -> str:
    """Validate an identifier before it is used as part of a path."""
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ValueError("id must contain only letters, numbers, '.', '_' and '-'")
    if value in {".", ".."}:
        raise ValueError("id cannot be a path component")
    return value


def validate_name(value: str) -> str:
    """Validate a human-readable name without allowing path-like values."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("name must be a nonempty, trimmed string")
    if len(value) > 255 or value in {".", ".."}:
        raise ValueError("name is invalid")
    if "/" in value or "\\" in value or any(ord(character) < 32 for character in value):
        raise ValueError("name cannot contain path separators or control characters")
    return value


def new_id(prefix: str = "job") -> str:
    validate_id(prefix)
    return f"{prefix}-{uuid.uuid4().hex}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_status(status: str, allowed: frozenset[str]) -> str:
    if not isinstance(status, str) or status not in allowed:
        raise ManifestError(f"unknown status: {status!r}")
    return status


def _require_keys(data: Mapping[str, Any], keys: set[str], kind: str) -> None:
    missing = keys - data.keys()
    extra = data.keys() - keys
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unknown {sorted(extra)}")
        raise ManifestError(f"invalid {kind} manifest: {', '.join(details)}")


@dataclass(slots=True)
class JobRecord:
    id: str
    name: str
    status: str = "queued"
    payload: dict[str, Any] = field(default_factory=dict)
    output: str | None = None
    batch_id: str | None = None
    sequence: int = 0
    attempts: int = 0
    progress: list[str] = field(default_factory=list)
    error: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    first_started_at: str | None = None
    finished_at: str | None = None
    active_seconds: float = 0.0
    active_started_at: str | None = None
    version: int = MANIFEST_VERSION

    def __post_init__(self) -> None:
        validate_id(self.id)
        validate_name(self.name)
        _require_status(self.status, JOB_STATUSES)
        if self.batch_id is not None:
            validate_id(self.batch_id)
        if not isinstance(self.payload, dict):
            raise ManifestError("job payload must be an object")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ManifestError("job sequence must be a nonnegative integer")
        if type(self.attempts) is not int or self.attempts < 0:
            raise ManifestError("job attempts must be a nonnegative integer")
        if not isinstance(self.progress, list) or not all(
            isinstance(entry, str) for entry in self.progress
        ):
            raise ManifestError("job progress must be a list of strings")
        if self.output is not None and not isinstance(self.output, str):
            raise ManifestError("job output must be a string or null")
        if self.error is not None and not isinstance(self.error, str):
            raise ManifestError("job error must be a string or null")
        if not isinstance(self.created_at, str) or not isinstance(self.updated_at, str):
            raise ManifestError("job timestamps must be strings")
        if self.first_started_at is not None and not isinstance(self.first_started_at, str):
            raise ManifestError("job first_started_at must be a string or null")
        if self.finished_at is not None and not isinstance(self.finished_at, str):
            raise ManifestError("job finished_at must be a string or null")
        if isinstance(self.active_seconds, bool) or not isinstance(self.active_seconds, (int, float)):
            raise ManifestError("job active_seconds must be a number")
        if self.active_seconds < 0:
            raise ManifestError("job active_seconds must be nonnegative")
        self.active_seconds = float(self.active_seconds)
        if self.active_started_at is not None and not isinstance(self.active_started_at, str):
            raise ManifestError("job active_started_at must be a string or null")
        if type(self.version) is not int or self.version != MANIFEST_VERSION:
            raise ManifestError(f"unsupported job manifest version: {self.version!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JobRecord":
        fields = set(cls.__dataclass_fields__)
        values = dict(data)
        unknown = set(values) - fields
        required = fields - {
            "first_started_at", "finished_at", "active_seconds", "active_started_at"
        }
        missing_required = required - set(values)
        if unknown or missing_required:
            _require_keys(values, fields, "job")
        if "first_started_at" not in values:
            attempts = values.get("attempts", 0)
            values["first_started_at"] = values.get("created_at") if attempts else None
        if "finished_at" not in values:
            values["finished_at"] = (
                values.get("updated_at") if values.get("status") in TERMINAL_STATUSES else None
            )
        if "active_seconds" not in values:
            # Exact historical active time cannot be reconstructed from old
            # manifests. Start at zero rather than counting offline wall time.
            values["active_seconds"] = 0.0
        if "active_started_at" not in values:
            values["active_started_at"] = (
                values.get("updated_at") if values.get("status") == "running" else None
            )
        try:
            return cls(**values)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ManifestError):
                raise
            raise ManifestError(f"invalid job manifest: {exc}") from exc


@dataclass(slots=True)
class BatchRecord:
    id: str
    name: str
    job_ids: list[str]
    status: str = "queued"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    version: int = MANIFEST_VERSION

    def __post_init__(self) -> None:
        validate_id(self.id)
        validate_name(self.name)
        _require_status(self.status, BATCH_STATUSES)
        if not isinstance(self.job_ids, list) or not all(
            isinstance(job_id, str) for job_id in self.job_ids
        ):
            raise ManifestError("batch job_ids must be a list of strings")
        for job_id in self.job_ids:
            validate_id(job_id)
        if not self.job_ids:
            raise ManifestError("batch must contain at least one job")
        if len(set(self.job_ids)) != len(self.job_ids):
            raise ManifestError("batch job_ids must be unique")
        if not isinstance(self.created_at, str) or not isinstance(self.updated_at, str):
            raise ManifestError("batch timestamps must be strings")
        if type(self.version) is not int or self.version != MANIFEST_VERSION:
            raise ManifestError(f"unsupported batch manifest version: {self.version!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BatchRecord":
        fields = set(cls.__dataclass_fields__)
        _require_keys(data, fields, "batch")
        try:
            return cls(**dict(data))
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ManifestError):
                raise
            raise ManifestError(f"invalid batch manifest: {exc}") from exc


@dataclass(frozen=True, slots=True)
class JobSpec:
    name: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    output: str | os.PathLike[str] | None = None
    id: str | None = None


def _atomic_write(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError(f"manifest {path} must contain a JSON object")
    return data


class JobStore:
    """Durable, deterministic queue of jobs grouped into optional batches."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        recover_running: RecoveryMode = "interrupted",
        progress_limit: int = 100,
    ) -> None:
        if recover_running not in {"interrupted", "queued"}:
            raise ValueError("recover_running must be 'interrupted' or 'queued'")
        if not isinstance(progress_limit, int) or progress_limit < 1:
            raise ValueError("progress_limit must be a positive integer")
        self.root = Path(root)
        self.jobs_dir = self.root / "jobs"
        self.batches_dir = self.root / "batches"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.batches_dir.mkdir(parents=True, exist_ok=True)
        self.progress_limit = progress_limit
        self._lock = threading.RLock()
        self._dispatch_lock = threading.Lock()
        self._jobs = self._load_jobs()
        self._batches = self._load_batches()
        self._validate_batch_links()
        self._next_sequence = max((job.sequence for job in self._jobs.values()), default=-1) + 1
        self._recover(recover_running)

    def _load_jobs(self) -> dict[str, JobRecord]:
        records: dict[str, JobRecord] = {}
        for path in sorted(self.jobs_dir.glob("*.json")):
            record = JobRecord.from_dict(_read_json(path))
            if path.stem != record.id:
                raise ManifestError(f"job filename does not match id: {path}")
            records[record.id] = record
        return records

    def _load_batches(self) -> dict[str, BatchRecord]:
        records: dict[str, BatchRecord] = {}
        for path in sorted(self.batches_dir.glob("*.json")):
            record = BatchRecord.from_dict(_read_json(path))
            if path.stem != record.id:
                raise ManifestError(f"batch filename does not match id: {path}")
            records[record.id] = record
        return records

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{validate_id(job_id)}.json"

    def _batch_path(self, batch_id: str) -> Path:
        return self.batches_dir / f"{validate_id(batch_id)}.json"

    def _validate_batch_links(self) -> None:
        for batch in self._batches.values():
            for job_id in batch.job_ids:
                if job_id not in self._jobs:
                    raise ManifestError(f"batch {batch.id!r} refers to unknown job {job_id!r}")
                if self._jobs[job_id].batch_id != batch.id:
                    raise ManifestError(f"job {job_id!r} does not refer back to batch {batch.id!r}")
        for job in self._jobs.values():
            if job.batch_id is not None and job.batch_id not in self._batches:
                raise ManifestError(f"job {job.id!r} refers to unknown batch {job.batch_id!r}")

    def _save_job(self, job: JobRecord) -> None:
        job.updated_at = _now()
        _atomic_write(self._job_path(job.id), job.to_dict())

    def _save_batch(self, batch: BatchRecord) -> None:
        batch.updated_at = _now()
        _atomic_write(self._batch_path(batch.id), batch.to_dict())

    def _recover(self, mode: RecoveryMode) -> None:
        for job in self._jobs.values():
            changed = False
            if job.status == "running":
                self._stop_active_timer(job, job.updated_at)
                job.status = mode
                job.error = "interrupted during previous process" if mode == "interrupted" else None
                job.finished_at = _now() if mode == "interrupted" else None
                changed = True
            elif job.status == "completed" and not self.output_exists(job):
                job.status = "queued"
                job.error = None
                job.finished_at = None
                changed = True
            if changed:
                self._save_job(job)
        # Re-derive every batch in case the process stopped between its job and
        # batch manifest writes.
        for batch_id in self._batches:
            self._refresh_batch(batch_id)

    def create_job(
        self,
        name: str,
        *,
        payload: Mapping[str, Any] | None = None,
        output: str | os.PathLike[str] | None = None,
        job_id: str | None = None,
        batch_id: str | None = None,
    ) -> JobRecord:
        with self._lock:
            identifier = validate_id(job_id) if job_id is not None else new_id()
            if identifier in self._jobs:
                raise ValueError(f"job already exists: {identifier}")
            if batch_id is not None:
                validate_id(batch_id)
            record = JobRecord(
                id=identifier,
                name=validate_name(name),
                payload=dict(payload or {}),
                output=os.fspath(output) if output is not None else None,
                batch_id=batch_id,
                sequence=self._next_sequence,
            )
            # Fail before publishing an unserializable payload into memory.
            json.dumps(record.to_dict())
            self._next_sequence += 1
            self._save_job(record)
            self._jobs[identifier] = record
            return record

    enqueue = create_job

    def create_batch(
        self,
        name: str,
        items: Iterable[JobSpec | Mapping[str, Any]],
        *,
        batch_id: str | None = None,
    ) -> BatchRecord:
        with self._lock:
            identifier = validate_id(batch_id) if batch_id is not None else new_id("batch")
            if identifier in self._batches:
                raise ValueError(f"batch already exists: {identifier}")
            specs = [item if isinstance(item, JobSpec) else JobSpec(**item) for item in items]
            if not specs:
                raise ValueError("batch must contain at least one job")
            prepared: list[tuple[JobSpec, str]] = []
            prepared_ids: set[str] = set()
            for spec in specs:
                validate_name(spec.name)
                job_id = validate_id(spec.id) if spec.id is not None else new_id()
                if job_id in self._jobs or job_id in prepared_ids:
                    raise ValueError(f"job already exists: {job_id}")
                if not isinstance(spec.payload, Mapping):
                    raise TypeError("job payload must be a mapping")
                json.dumps(dict(spec.payload))
                prepared.append((spec, job_id))
                prepared_ids.add(job_id)
            job_ids: list[str] = []
            for spec, job_id in prepared:
                job = self.create_job(
                    spec.name,
                    payload=spec.payload,
                    output=spec.output,
                    job_id=job_id,
                    batch_id=identifier,
                )
                job_ids.append(job.id)
            batch = BatchRecord(id=identifier, name=validate_name(name), job_ids=job_ids)
            self._save_batch(batch)
            self._batches[identifier] = batch
            return batch

    def get_job(self, job_id: str) -> JobRecord:
        with self._lock:
            try:
                return self._jobs[validate_id(job_id)]
            except KeyError as exc:
                raise KeyError(f"unknown job: {job_id}") from exc

    def get_batch(self, batch_id: str) -> BatchRecord:
        with self._lock:
            try:
                return self._batches[validate_id(batch_id)]
            except KeyError as exc:
                raise KeyError(f"unknown batch: {batch_id}") from exc

    def queued_jobs(self) -> list[JobRecord]:
        with self._lock:
            return sorted(
                (job for job in self._jobs.values() if job.status == "queued"),
                key=lambda job: (job.sequence, job.id),
            )

    def all_jobs(self) -> list[JobRecord]:
        """Return every job in stable submission order."""
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: (job.sequence, job.id))

    @staticmethod
    def output_exists(job: JobRecord) -> bool:
        if not job.output:
            return False
        try:
            path = Path(job.output)
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    def is_complete(self, job: JobRecord) -> bool:
        return job.status == "completed" and self.output_exists(job)

    def add_progress(self, job_id: str, message: str) -> JobRecord:
        if not isinstance(message, str):
            raise TypeError("progress message must be a string")
        with self._lock:
            job = self.get_job(job_id)
            job.progress.append(message)
            del job.progress[:-self.progress_limit]
            self._save_job(job)
            return job

    def _transition(self, job: JobRecord, status: str, *, error: str | None = None) -> None:
        _require_status(status, JOB_STATUSES)
        if status not in _TRANSITIONS[job.status]:
            raise InvalidTransition(f"cannot transition job from {job.status!r} to {status!r}")
        now = _now()
        if job.status == "running" and status != "running":
            self._stop_active_timer(job, now)
        job.status = status
        job.error = error
        if status == "running":
            if job.first_started_at is None:
                job.first_started_at = now
            job.active_started_at = now
            job.finished_at = None
        elif status == "queued":
            job.finished_at = None
        elif status in TERMINAL_STATUSES:
            job.finished_at = now
        self._save_job(job)
        if job.batch_id and job.batch_id in self._batches:
            self._refresh_batch(job.batch_id)

    @staticmethod
    def _stop_active_timer(job: JobRecord, stopped_at: str) -> None:
        if job.active_started_at is None:
            return
        try:
            started = datetime.fromisoformat(job.active_started_at)
            stopped = datetime.fromisoformat(stopped_at)
            job.active_seconds += max(0.0, (stopped - started).total_seconds())
        except ValueError:
            pass
        job.active_started_at = None

    def retry(self, job_id: str, *, allow_completed: bool = False) -> JobRecord:
        with self._lock:
            job = self.get_job(job_id)
            if job.status == "completed" and self.output_exists(job) and not allow_completed:
                raise InvalidTransition("cannot retry a completed job with a valid output")
            self._transition(job, "queued")
            return job

    def pause(self, job_id: str) -> JobRecord:
        """Hold a queued or running job; a running worker stops at its next checkpoint."""
        with self._lock:
            job = self.get_job(job_id)
            if job.status == "paused":
                return job
            self._transition(job, "paused")
            return job

    def delete_job(self, job_id: str) -> JobRecord:
        """Remove a stopped job from the queue; running jobs must stop first."""
        with self._lock:
            job = self.get_job(job_id)
            if job.status == "running":
                raise InvalidTransition("cannot delete a running job")
            if job.batch_id is not None and job.batch_id in self._batches:
                batch = self._batches[job.batch_id]
                batch.job_ids = [kept for kept in batch.job_ids if kept != job.id]
                if batch.job_ids:
                    self._save_batch(batch)
                    self._refresh_batch(batch.id)
                else:
                    self._batch_path(batch.id).unlink(missing_ok=True)
                    del self._batches[batch.id]
            self._jobs.pop(job.id, None)
            self._job_path(job.id).unlink(missing_ok=True)
            return job

    def cancel(self, job_id: str) -> JobRecord:
        with self._lock:
            job = self.get_job(job_id)
            self._transition(job, "cancelled")
            return job

    def cancel_batch(self, batch_id: str) -> BatchRecord:
        with self._lock:
            batch = self.get_batch(batch_id)
            for job_id in batch.job_ids:
                job = self._jobs[job_id]
                if job.status in {"queued", "running", "paused", "failed", "interrupted"}:
                    self._transition(job, "cancelled")
            self._refresh_batch(batch.id)
            return batch

    def retry_batch(self, batch_id: str) -> BatchRecord:
        with self._lock:
            batch = self.get_batch(batch_id)
            for job_id in batch.job_ids:
                job = self._jobs[job_id]
                if job.status in {"failed", "cancelled", "interrupted", "paused"} or (
                    job.status == "completed" and not self.output_exists(job)
                ):
                    self._transition(job, "queued")
            self._refresh_batch(batch.id)
            return batch

    def _refresh_batch(self, batch_id: str) -> None:
        batch = self._batches[batch_id]
        jobs = [self._jobs[job_id] for job_id in batch.job_ids]
        if all(self.is_complete(job) for job in jobs):
            status = "completed"
        elif any(job.status == "running" for job in jobs):
            status = "running"
        elif any(job.status == "paused" for job in jobs):
            status = "paused"
        elif any(job.status == "failed" for job in jobs):
            status = "failed"
        elif any(job.status == "interrupted" for job in jobs):
            status = "interrupted"
        elif all(job.status in {"completed", "cancelled"} for job in jobs):
            status = "cancelled"
        else:
            status = "queued"
        if batch.status != status:
            batch.status = status
            self._save_batch(batch)

    def dispatch_one(self, worker: Worker) -> JobRecord | None:
        """Run at most one queued job synchronously and persist every transition."""
        if not self._dispatch_lock.acquire(blocking=False):
            return None
        try:
            with self._lock:
                for candidate in sorted(
                    self._jobs.values(),
                    key=lambda queued_job: (queued_job.sequence, queued_job.id),
                ):
                    if candidate.status == "completed" and not self.output_exists(candidate):
                        self._transition(candidate, "queued")
                queued = self.queued_jobs()
                if not queued:
                    return None
                job = queued[0]
                self._transition(job, "running")
                job.attempts += 1
                self._save_job(job)

            def progress(message: str) -> None:
                self.add_progress(job.id, message)

            heartbeat_stop = threading.Event()

            def heartbeat() -> None:
                while not heartbeat_stop.wait(5):
                    with self._lock:
                        if job.status != "running":
                            return
                        self._save_job(job)

            heartbeat_thread = threading.Thread(
                target=heartbeat, name=f"job-heartbeat-{job.id[:8]}", daemon=True
            )
            heartbeat_thread.start()
            try:
                result = worker(job, progress)
            except JobPaused:
                with self._lock:
                    if job.status == "running":
                        self._transition(job, "paused")
                return job
            except Exception as exc:
                with self._lock:
                    if job.status == "running":
                        self._transition(job, "failed", error=str(exc))
                    elif job.status == "paused":
                        # The pause already won the race; keep it paused but
                        # record why the winding-down worker exited.
                        job.error = str(exc)
                        self._save_job(job)
                return job
            finally:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=1)

            with self._lock:
                if job.status == "running":
                    if isinstance(result, (str, os.PathLike)):
                        job.output = os.fspath(result)
                    if not self.output_exists(job):
                        self._transition(job, "failed", error="worker did not produce a nonempty output")
                    else:
                        self._transition(job, "completed")
                elif job.status == "paused" and isinstance(result, (str, os.PathLike)):
                    # The worker finished before the pause took effect. Publish
                    # the output instead of forcing a redundant re-run.
                    job.output = os.fspath(result)
                    if self.output_exists(job):
                        self._transition(job, "completed")
                return job
        finally:
            self._dispatch_lock.release()


__all__ = [
    "BATCH_STATUSES",
    "JOB_STATUSES",
    "MANIFEST_VERSION",
    "BatchRecord",
    "InvalidTransition",
    "JobPaused",
    "JobRecord",
    "JobSpec",
    "JobStore",
    "ManifestError",
    "new_id",
    "validate_id",
    "validate_name",
]
