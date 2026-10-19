from __future__ import annotations

import json
from pathlib import Path

import pytest

import voicecover.jobs as jobs_module
from voicecover.jobs import (
    MANIFEST_VERSION,
    InvalidTransition,
    JobPaused,
    JobRecord,
    JobSpec,
    JobStore,
    ManifestError,
    validate_id,
    validate_name,
)


def test_records_are_strict_and_versioned() -> None:
    record = JobRecord(id="job-1", name="First job")
    assert record.to_dict()["version"] == MANIFEST_VERSION

    invalid = record.to_dict()
    invalid["status"] = "waiting"
    with pytest.raises(ManifestError, match="unknown status"):
        JobRecord.from_dict(invalid)

    invalid = record.to_dict()
    invalid["future_field"] = True
    with pytest.raises(ManifestError, match="unknown"):
        JobRecord.from_dict(invalid)

    invalid = record.to_dict()
    invalid["version"] = MANIFEST_VERSION + 1
    with pytest.raises(ManifestError, match="unsupported"):
        JobRecord.from_dict(invalid)


def test_legacy_job_manifest_gains_durable_timing_fields() -> None:
    record = JobRecord(id="legacy-job", name="Legacy", status="failed", attempts=1)
    legacy = record.to_dict()
    legacy.pop("first_started_at")
    legacy.pop("finished_at")
    legacy.pop("active_seconds")
    legacy.pop("active_started_at")

    loaded = JobRecord.from_dict(legacy)

    assert loaded.first_started_at == loaded.created_at
    assert loaded.finished_at == loaded.updated_at
    assert loaded.active_seconds == 0
    assert loaded.active_started_at is None


def test_active_time_accumulates_only_while_running(monkeypatch, tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    job = store.create_job("Timed", job_id="timed")
    clock = ["2026-01-01T00:00:00+00:00"]
    monkeypatch.setattr(jobs_module, "_now", lambda: clock[0])
    store._transition(job, "running")
    clock[0] = "2026-01-01T00:00:10+00:00"
    store._transition(job, "failed")
    store._transition(job, "queued")
    clock[0] = "2026-01-01T01:00:00+00:00"
    store._transition(job, "running")
    clock[0] = "2026-01-01T01:00:05+00:00"
    store._transition(job, "cancelled")

    assert job.active_seconds == 15
    assert job.active_started_at is None


def test_recovery_stops_active_timer_at_last_heartbeat(monkeypatch, tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    job = store.create_job("Recover timer", job_id="recover-timer")
    job.status = "running"
    job.active_seconds = 3
    job.active_started_at = "2026-01-01T00:00:00+00:00"
    monkeypatch.setattr(jobs_module, "_now", lambda: "2026-01-01T00:00:07+00:00")
    store._save_job(job)

    recovered = JobStore(tmp_path).get_job(job.id)

    assert recovered.status == "interrupted"
    assert recovered.active_seconds == 10
    assert recovered.active_started_at is None

@pytest.mark.parametrize("value", ["../job", "a/b", "a\\b", "", "."])
def test_ids_reject_unsafe_path_components(value: str) -> None:
    with pytest.raises(ValueError):
        validate_id(value)


@pytest.mark.parametrize("value", ["../video", "folder/video", " bad", "bad\x00name"])
def test_names_reject_path_like_or_untrimmed_values(value: str) -> None:
    with pytest.raises(ValueError):
        validate_name(value)


def test_manifests_are_atomic_json_and_reload(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    job = store.create_job("Encode", payload={"language": "id"}, job_id="encode-1")

    manifest = tmp_path / "jobs" / "encode-1.json"
    assert json.loads(manifest.read_text(encoding="utf-8"))["id"] == job.id
    assert not list((tmp_path / "jobs").glob("*.tmp"))
    assert JobStore(tmp_path).get_job(job.id).payload == {"language": "id"}


def test_queue_order_is_persisted_and_deterministic(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    second = store.create_job("Second", job_id="z-job")
    first = store.create_job("First", job_id="a-job")

    assert [job.id for job in store.queued_jobs()] == [second.id, first.id]
    reloaded = JobStore(tmp_path)
    assert [job.id for job in reloaded.queued_jobs()] == [second.id, first.id]


def test_startup_recovers_running_jobs_using_selected_mode(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    job = store.create_job("Recover me", job_id="recover-me")
    job.status = "running"
    store._save_job(job)

    recovered = JobStore(tmp_path).get_job(job.id)
    assert recovered.status == "interrupted"
    assert recovered.error == "interrupted during previous process"

    recovered.status = "running"
    JobStore(tmp_path)._save_job(recovered)
    assert JobStore(tmp_path, recover_running="queued").get_job(job.id).status == "queued"


def test_completed_job_is_skipped_only_with_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "result.mp4"
    store = JobStore(tmp_path / "store")
    job = store.create_job("Dub", output=output, job_id="dub")
    job.status = "completed"
    store._save_job(job)

    assert JobStore(tmp_path / "store").get_job(job.id).status == "queued"

    output.write_bytes(b"video")
    queued = JobStore(tmp_path / "store")
    queued_job = queued.get_job(job.id)
    queued_job.status = "completed"
    queued._save_job(queued_job)
    complete = JobStore(tmp_path / "store")
    assert complete.get_job(job.id).status == "completed"
    assert complete.queued_jobs() == []


def test_dispatches_one_job_and_bounds_persisted_progress(tmp_path: Path) -> None:
    output = tmp_path / "one.out"
    store = JobStore(tmp_path / "store", progress_limit=2)
    first = store.create_job("One", output=output, job_id="one")
    store.create_job("Two", output=tmp_path / "two.out", job_id="two")

    def worker(job: JobRecord, progress) -> Path:
        progress("one")
        progress("two")
        progress("three")
        output.write_bytes(b"finished")
        return output

    result = store.dispatch_one(worker)
    assert result is first
    assert result.status == "completed"
    assert result.attempts == 1
    assert result.first_started_at is not None
    assert result.finished_at is not None
    assert result.progress == ["two", "three"]
    assert [job.id for job in store.queued_jobs()] == ["two"]

    persisted = JobStore(tmp_path / "store").get_job("one")
    assert persisted.status == "completed"
    assert persisted.progress == ["two", "three"]


def test_failed_dispatch_can_retry_and_queued_job_can_cancel(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    failed = store.create_job("Failure", output=tmp_path / "missing", job_id="failed")
    cancelled = store.create_job("Cancel", job_id="cancelled")

    store.dispatch_one(lambda job, progress: (_ for _ in ()).throw(RuntimeError("boom")))
    assert failed.status == "failed"
    assert failed.error == "boom"
    assert failed.first_started_at is not None
    assert failed.finished_at is not None
    first_started_at = failed.first_started_at
    retried = store.retry(failed.id)
    assert retried.status == "queued"
    assert retried.first_started_at == first_started_at
    assert retried.finished_at is None
    assert store.cancel(cancelled.id).status == "cancelled"
    with pytest.raises(InvalidTransition):
        store.cancel(cancelled.id)


def test_batch_tracks_order_completion_retry_and_cancel(tmp_path: Path) -> None:
    outputs = [tmp_path / "a.out", tmp_path / "b.out"]
    store = JobStore(tmp_path / "store")
    batch = store.create_batch(
        "Two videos",
        [
            JobSpec("A", output=outputs[0], id="a"),
            {"name": "B", "output": outputs[1], "id": "b"},
        ],
        batch_id="batch-1",
    )
    assert batch.job_ids == ["a", "b"]

    def worker(job: JobRecord, progress) -> Path:
        path = Path(job.output or "")
        path.write_bytes(job.id.encode())
        return path

    assert store.dispatch_one(worker).id == "a"
    assert batch.status == "queued"
    assert store.dispatch_one(worker).id == "b"
    assert batch.status == "completed"

    outputs[0].unlink()
    reloaded = JobStore(tmp_path / "store")
    assert reloaded.get_job("a").status == "queued"
    assert reloaded.get_batch(batch.id).status == "queued"
    reloaded.cancel_batch(batch.id)
    assert reloaded.get_job("a").status == "cancelled"
    reloaded.retry_batch(batch.id)
    assert reloaded.get_job("a").status == "queued"


def test_pause_holds_a_queued_job_until_resumed(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "store")
    job = store.create_job("Later", output=tmp_path / "later.out", job_id="later")

    assert store.pause(job.id).status == "paused"
    assert store.pause(job.id).status == "paused"
    assert store.queued_jobs() == []
    assert store.dispatch_one(lambda candidate, progress: None) is None

    assert store.retry(job.id).status == "queued"
    assert [queued.id for queued in store.queued_jobs()] == ["later"]


def test_pause_rejects_terminal_jobs(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    job = store.create_job("Failure", job_id="failure")

    store.dispatch_one(lambda candidate, progress: (_ for _ in ()).throw(RuntimeError("boom")))
    assert job.status == "failed"

    with pytest.raises(InvalidTransition):
        store.pause(job.id)


def test_running_worker_pauses_at_checkpoint_then_resumes(tmp_path: Path) -> None:
    output = tmp_path / "resume.out"
    store = JobStore(tmp_path / "store")
    job = store.create_job("Resume", output=output, job_id="resume")
    calls = 0

    def worker(candidate: JobRecord, progress) -> Path:
        nonlocal calls
        calls += 1
        if calls == 1:
            progress("first stage")
            raise JobPaused("checkpoint reached")
        output.write_bytes(b"finished")
        return output

    paused = store.dispatch_one(worker)
    assert paused.id == job.id
    assert paused.status == "paused"
    assert paused.error is None
    assert paused.progress == ["first stage"]
    assert store.dispatch_one(worker) is None

    assert store.retry(job.id).status == "queued"
    finished = store.dispatch_one(worker)
    assert finished.status == "completed"
    assert finished.output == str(output)
    assert calls == 2


def test_worker_error_after_pause_keeps_the_job_paused(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "store")
    job = store.create_job("Racing", output=tmp_path / "racing.out", job_id="racing")

    def worker(candidate: JobRecord, progress) -> Path:
        # The pause lands first; the winding-down worker then fails for real.
        store.pause(candidate.id)
        raise RuntimeError("stage blew up")

    result = store.dispatch_one(worker)
    assert result.status == "paused"
    assert result.error == "stage blew up"

    store.retry(job.id)
    assert store.get_job(job.id).status == "queued"
    assert store.get_job(job.id).error is None


def test_batch_reports_a_paused_member(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "store")
    batch = store.create_batch(
        "Two videos",
        [JobSpec("A", id="a"), JobSpec("B", id="b")],
        batch_id="batch-paused",
    )

    store.pause("a")
    assert store.get_batch(batch.id).status == "paused"
    store.retry("a")
    assert store.get_batch(batch.id).status == "queued"


def test_delete_job_removes_the_manifest(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "store")
    store.create_job("Gone", output=tmp_path / "gone.out", job_id="gone")

    store.delete_job("gone")
    with pytest.raises(KeyError, match="unknown job"):
        store.get_job("gone")
    assert not (tmp_path / "store" / "gone.json").exists()
    assert [job.id for job in JobStore(tmp_path / "store").all_jobs()] == []


def test_completed_job_requires_explicit_reprocess_permission(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "store")
    output = tmp_path / "done.mp4"
    output.write_bytes(b"done")
    job = store.create_job("Done", output=output, job_id="done")
    store.dispatch_one(lambda candidate, progress: output)

    with pytest.raises(InvalidTransition, match="completed"):
        store.retry(job.id)
    assert store.retry(job.id, allow_completed=True).status == "queued"


def test_delete_job_rejects_a_running_worker(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "store")
    job = store.create_job("Busy", output=tmp_path / "busy.out", job_id="busy")
    job.status = "running"

    with pytest.raises(InvalidTransition, match="running"):
        store.delete_job("busy")
    assert store.get_job("busy").status == "running"


def test_delete_job_updates_or_removes_batches(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "store")
    store.create_batch(
        "Two videos",
        [JobSpec("A", id="a"), JobSpec("B", id="b")],
        batch_id="batch-del",
    )

    store.delete_job("a")
    assert store.get_batch("batch-del").job_ids == ["b"]

    # a half-empty batch must survive a restart like any other manifest
    reloaded = JobStore(tmp_path / "store")
    assert reloaded.get_batch("batch-del").job_ids == ["b"]

    reloaded.delete_job("b")
    with pytest.raises(KeyError, match="unknown batch"):
        JobStore(tmp_path / "store").get_batch("batch-del")
