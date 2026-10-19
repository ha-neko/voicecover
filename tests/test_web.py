from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

import voicecover.web as web
from voicecover.jobs import JobStore as DurableJobStore
from voicecover.pipeline import Options


def test_safe_part_removes_path_and_shell_characters() -> None:
    assert web._safe_part(" ../../my video;$(no).mp4 ", "upload") == "my-video-no-.mp4"
    assert web._safe_part("...", "upload") == "upload"


def test_job_store_runs_pipeline_outside_request(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    directory = tmp_path / "job-id"
    directory.mkdir()
    output = directory / "result.mp4"
    options = Options(input=source, output=output, target_language="id")

    def fake_run(given: Options, progress) -> Path:
        progress("extracting audio...")
        given.output.write_bytes(b"dub")
        return given.output

    monkeypatch.setattr(web, "run", fake_run)
    jobs = web.JobStore(root=tmp_path / "jobs", workers=1)
    try:
        job = jobs.add(options, directory)
        deadline = time.monotonic() + 2
        while job.status not in {"completed", "failed"} and time.monotonic() < deadline:
            time.sleep(0.01)
        public = job.public()
        assert public["status"] == "completed"
        assert public["progress"] == ["extracting audio..."]
        assert public["attempts"] == 1
        assert public["first_started_at"] is not None
        assert public["finished_at"] is not None
        assert public["active_seconds"] >= 0
        assert public["active_started_at"] is None
        assert public["video_url"] == "/api/jobs/job-id/video"
    finally:
        jobs.close()


def test_job_store_records_pipeline_failure(monkeypatch, tmp_path: Path) -> None:
    directory = tmp_path / "failed-job"
    directory.mkdir()
    options = Options(input="video.mp4", output=directory / "out.mp4", target_language="ja")

    def fail(*args, **kwargs) -> Path:
        raise RuntimeError("ffmpeg went missing")

    monkeypatch.setattr(web, "run", fail)
    jobs = web.JobStore(root=tmp_path / "jobs", workers=1)
    try:
        job = jobs.add(options, directory)
        deadline = time.monotonic() + 2
        while job.status not in {"completed", "failed"} and time.monotonic() < deadline:
            time.sleep(0.01)
        assert job.status == "failed"
        assert job.error == "ffmpeg went missing"
    finally:
        jobs.close()


def test_job_store_persists_options_progress_and_output(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    directory = root / "persisted"
    directory.mkdir(parents=True)
    source = directory / "source.mp4"
    source.write_bytes(b"source")
    output = directory / "result.mp4"
    options = Options(
        input=source,
        output=output,
        target_language="id",
        translator="llm",
        preserve_arabic=True,
        voice_clone=True,
        write_visual_track=True,
        localize_visual_text=True,
    )

    def fake_run(given: Options, progress) -> Path:
        progress("durable progress")
        given.output.write_bytes(b"result")
        return given.output

    monkeypatch.setattr(web, "run", fake_run)
    jobs = web.JobStore(root=root)
    try:
        job = jobs.add(options, directory, web_managed=True)
        deadline = time.monotonic() + 2
        while job.status not in {"completed", "failed"} and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        jobs.close()

    reloaded = DurableJobStore(root).get_job("persisted")
    assert reloaded.status == "completed"
    assert reloaded.progress == ["durable progress"]
    assert reloaded.output == str(output)
    assert reloaded.payload["translator"] == "llm"
    assert reloaded.payload["write_visual_track"] is True
    assert not list((root / "jobs").glob("*.tmp"))


def test_job_store_runs_exactly_one_pipeline_at_a_time(monkeypatch, tmp_path: Path) -> None:
    active = 0
    maximum = 0
    order: list[str] = []
    guard = threading.Lock()

    def fake_run(options: Options, progress) -> Path:
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
            order.append(Path(options.input).name)
        time.sleep(0.04)
        options.output.write_bytes(b"done")
        with guard:
            active -= 1
        return options.output

    monkeypatch.setattr(web, "run", fake_run)
    jobs = web.JobStore(root=tmp_path / "jobs", workers=8, max_concurrent=8)
    try:
        submitted = []
        for index in range(3):
            directory = tmp_path / "jobs" / f"job-{index}"
            directory.mkdir()
            source = directory / f"{index}.mp4"
            source.write_bytes(b"source")
            submitted.append(
                jobs.add(
                    Options(input=source, output=directory / "out.mp4", target_language="id"),
                    directory,
                    web_managed=True,
                )
            )
        deadline = time.monotonic() + 3
        while any(job.status not in {"completed", "failed"} for job in submitted) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert [job.status for job in submitted] == ["completed"] * 3
        assert maximum == 1
        assert order == ["0.mp4", "1.mp4", "2.mp4"]
        assert jobs.max_concurrent == 1
    finally:
        jobs.close()


def test_running_job_reloads_as_interrupted_until_explicit_retry(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    directory = root / "recover"
    directory.mkdir(parents=True)
    source = directory / "source.mp4"
    source.write_bytes(b"source")
    output = directory / "out.mp4"
    options = Options(input=source, output=output, target_language="id")
    durable = DurableJobStore(root)
    record = durable.create_job(
        "recover", payload=web._options_payload(options, web_managed=True), output=output, job_id="recover"
    )
    record.status = "running"
    durable._save_job(record)
    calls = 0

    def fake_run(given: Options, progress) -> Path:
        nonlocal calls
        calls += 1
        given.output.write_bytes(b"done")
        return given.output

    monkeypatch.setattr(web, "run", fake_run)
    jobs = web.JobStore(root=root)
    try:
        time.sleep(0.05)
        assert jobs.get("recover").status == "interrupted"
        assert calls == 0
        jobs.retry("recover")
        deadline = time.monotonic() + 2
        while jobs.get("recover").status not in {"completed", "failed"} and time.monotonic() < deadline:
            time.sleep(0.01)
        assert jobs.get("recover").status == "completed"
        assert calls == 1
    finally:
        jobs.close()


def test_web_managed_job_rejects_persisted_path_escape(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    directory = root / "unsafe"
    directory.mkdir(parents=True)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"source")
    options = Options(input=outside, output=directory / "out.mp4", target_language="id")
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(web, "run", fake_run)
    jobs = web.JobStore(root=root)
    try:
        job = jobs.add(options, directory, web_managed=True)
        deadline = time.monotonic() + 2
        while job.status not in {"completed", "failed"} and time.monotonic() < deadline:
            time.sleep(0.01)
        assert job.status == "failed"
        assert job.error == "persisted web job contains an unsafe path"
        assert called is False
    finally:
        jobs.close()


def test_profiles_expose_quran_series_workflow(monkeypatch) -> None:
    monkeypatch.setattr(web, "list_profiles", lambda: [])
    payload = web.profiles()
    quran = next(item for item in payload["profiles"] if item["name"] == "quran-id")
    assert quran["built_in"] is True
    assert quran["options"]["translator"] == "llm"
    assert quran["options"]["preserve_arabic"] is True
    assert quran["options"]["voice_clone"] is True
    assert quran["options"]["write_visual_track"] is True
    assert quran["options"]["localize_visual_text"] is True


def test_web_assets_include_batch_and_series_controls() -> None:
    html = web._asset("index.html")
    assert 'name="videos"' in html and "multiple" in html
    assert 'id="profile-picker"' in html
    assert '<option value="llm">' in html
    assert 'name="preserve_arabic"' in html
    assert 'name="voice_clone" checked' in html
    assert "F5-TTS (not an LLM voice)" in html
    assert 'name="write_visual_track"' in html
    assert 'name="localize_visual_text"' in html
    assert 'name="visual_only"' in html
    assert 'name="sound_only"' in html
    assert 'name="clone_nfe"' in html
    assert 'Fast · 16 steps' in html
    assert 'name="localization_mode" value="full" checked' in html
    assert 'name="localization_mode" value="sound"' in html
    assert 'name="localization_mode" value="visual"' in html
    assert 'id="source-language"' in html
    assert '<option value="" selected>Auto detect</option>' in html
    assert 'id="progress-track"' in html
    assert 'role="progressbar"' in html
    assert 'id="elapsed-time"' in html
    assert 'id="started-time"' in html
    assert 'id="attempt-count"' in html
    javascript = web._asset("app.js")
    assert "active_seconds" in javascript
    assert "active_started_at" in javascript
    assert "item.name === 'quran-id'" in javascript
    assert "function estimatedProgress" in javascript
    assert "clone voice\\s+" in javascript
    assert "function syncLocalizationMode" in javascript
    assert "Create visual localization" in javascript
    assert "Create sound dub" in javascript
    assert "function formatDuration" in javascript
    assert "function renderTiming" in javascript


def test_batch_upload_uses_profile_and_generated_paths_in_upload_order(
    monkeypatch, tmp_path: Path
) -> None:
    jobs = web.JobStore(root=tmp_path / "jobs")

    def fake_run(options: Options, progress) -> Path:
        options.output.write_bytes(b"done")
        return options.output

    monkeypatch.setattr(web, "store", jobs)
    monkeypatch.setattr(web, "run", fake_run)
    try:
        with TestClient(web.app) as client:
            response = client.post(
                "/api/jobs",
                data={"profile": "quran-id"},
                files=[
                    ("videos", ("../../lesson 1.mp4", b"first", "video/mp4")),
                    ("videos", ("lesson 2.mp4", b"second", "video/mp4")),
                ],
            )
        assert response.status_code == 202
        body = response.json()
        assert body["batch"] is True
        assert body["batch_id"].startswith("batch-")
        ids = [item["id"] for item in body["jobs"]]
        records = [jobs.durable.get_job(job_id) for job_id in ids]
        assert [Path(record.payload["input"]).name for record in records] == [
            "lesson-1.mp4",
            "lesson-2.mp4",
        ]
        for record in records:
            options = web._options_from_record(record)
            assert Path(options.input).parent == jobs.root / record.id
            assert options.target_language == "id"
            assert options.translator == "llm"
            assert options.preserve_arabic is True
            assert options.voice_clone is True
            assert options.write_visual_track is True
            assert options.localize_visual_text is True
    finally:
        jobs.close()


def test_upload_limit_is_enforced_before_job_is_queued(monkeypatch, tmp_path: Path) -> None:
    jobs = web.JobStore(root=tmp_path / "jobs")
    monkeypatch.setattr(web, "store", jobs)
    monkeypatch.setattr(web, "MAX_UPLOAD_BYTES", 3)
    try:
        with TestClient(web.app) as client:
            response = client.post(
                "/api/jobs",
                files={"videos": ("too-large.mp4", b"four", "video/mp4")},
            )
        assert response.status_code == 413
        assert jobs.durable.all_jobs() == []
        assert [path for path in jobs.root.iterdir() if path.name not in {"jobs", "batches"}] == []
    finally:
        jobs.close()


def _quiet_run(options: Options, progress) -> Path:
    options.output.write_bytes(b"done")
    return options.output


def _web_job(jobs: web.JobStore, name: str, *, web_managed: bool = True) -> "web.Job":
    directory = jobs.root / name
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "source.mp4"
    if web_managed and not source.exists():
        source.write_bytes(b"source")
    options = Options(input=source, output=directory / "out.mp4", target_language="id")
    return jobs.add(options, directory, web_managed=web_managed)


def test_pause_and_resume_a_running_job_through_api(monkeypatch, tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()

    def fake_run(options: Options, progress) -> Path:
        progress("stage one")
        entered.set()
        release.wait(5)
        progress("stage two")
        options.output.write_bytes(b"done")
        return options.output

    monkeypatch.setattr(web, "run", fake_run)
    jobs = web.JobStore(root=tmp_path / "jobs")
    monkeypatch.setattr(web, "store", jobs)
    try:
        job = _web_job(jobs, "pausable")
        with TestClient(web.app) as client:
            assert entered.wait(2)
            paused = client.post(f"/api/jobs/{job.id}/pause")
            assert paused.status_code == 202
            body = paused.json()
            assert body["status"] == "paused"
            assert body["pause_pending"] is True

            release.set()
            deadline = time.monotonic() + 2
            while jobs.pause_pending(job.id) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert jobs.pause_pending(job.id) is False
            assert jobs.get(job.id).status == "paused"

            resumed = client.post(f"/api/jobs/{job.id}/resume")
            assert resumed.status_code == 202
            deadline = time.monotonic() + 2
            while jobs.get(job.id).status not in {"completed", "failed"} and time.monotonic() < deadline:
                time.sleep(0.01)
            assert jobs.get(job.id).status == "completed"
    finally:
        release.set()
        jobs.close()


def test_resume_is_rejected_while_the_worker_is_still_stopping(monkeypatch, tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()

    def fake_run(options: Options, progress) -> Path:
        entered.set()
        release.wait(5)
        raise RuntimeError("unreachable")

    monkeypatch.setattr(web, "run", fake_run)
    jobs = web.JobStore(root=tmp_path / "jobs")
    monkeypatch.setattr(web, "store", jobs)
    try:
        job = _web_job(jobs, "stalling")
        with TestClient(web.app) as client:
            assert entered.wait(2)
            assert client.post(f"/api/jobs/{job.id}/pause").status_code == 202
            blocked = client.post(f"/api/jobs/{job.id}/resume")
            assert blocked.status_code == 409
            assert "stopping" in blocked.json()["detail"]

            release.set()
            deadline = time.monotonic() + 2
            while jobs.pause_pending(job.id) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert client.post(f"/api/jobs/{job.id}/resume").status_code == 202
    finally:
        release.set()
        jobs.close()


def test_paused_job_holds_the_queue_until_resumed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web, "run", _quiet_run)
    jobs = web.JobStore(root=tmp_path / "jobs")
    monkeypatch.setattr(web, "store", jobs)
    try:
        # Drain the constructor's scheduling pass so nothing can pick the
        # seeded job up before it is paused.
        jobs.executor.submit(lambda: None).result(timeout=2)
        held_dir = jobs.root / "held"
        held_dir.mkdir(parents=True)
        held_options = Options(
            input=held_dir / "source.mp4", output=held_dir / "out.mp4", target_language="id"
        )
        jobs.durable.create_job(
            "held",
            payload=web._options_payload(held_options),
            output=held_options.output,
            job_id="held",
        )
        jobs.pause("held")

        blocked = _web_job(jobs, "blocked")
        time.sleep(0.15)
        assert jobs.get(blocked.id).status == "queued"

        jobs.resume("held")
        deadline = time.monotonic() + 3
        while (
            jobs.get("held").status != "completed" or jobs.get(blocked.id).status != "completed"
        ) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert jobs.get("held").status == "completed"
        assert jobs.get(blocked.id).status == "completed"
    finally:
        jobs.close()


def test_pause_endpoint_reports_unknown_and_terminal_jobs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web, "run", _quiet_run)
    jobs = web.JobStore(root=tmp_path / "jobs")
    monkeypatch.setattr(web, "store", jobs)
    try:
        with TestClient(web.app) as client:
            assert client.post("/api/jobs/no-such-job/pause").status_code == 404
            done = _web_job(jobs, "finished")
            deadline = time.monotonic() + 2
            while jobs.get(done.id).status not in {"completed", "failed"} and time.monotonic() < deadline:
                time.sleep(0.01)
            assert client.post(f"/api/jobs/{done.id}/pause").status_code == 409
    finally:
        jobs.close()


def test_cancelling_a_paused_job_releases_the_queue(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web, "run", _quiet_run)
    jobs = web.JobStore(root=tmp_path / "jobs")
    monkeypatch.setattr(web, "store", jobs)
    try:
        jobs.executor.submit(lambda: None).result(timeout=2)
        parked_dir = jobs.root / "parked"
        parked_dir.mkdir(parents=True)
        parked_options = Options(
            input=parked_dir / "source.mp4", output=parked_dir / "out.mp4", target_language="id"
        )
        jobs.durable.create_job(
            "parked",
            payload=web._options_payload(parked_options),
            output=parked_options.output,
            job_id="parked",
        )
        jobs.pause("parked")

        blocked = _web_job(jobs, "after-parked")
        time.sleep(0.15)
        assert jobs.get(blocked.id).status == "queued"

        with TestClient(web.app) as client:
            cancelled = client.post("/api/jobs/parked/cancel")
            assert cancelled.status_code == 202
            assert cancelled.json()["status"] == "cancelled"

        deadline = time.monotonic() + 3
        while jobs.get(blocked.id).status not in {"completed", "failed"} and time.monotonic() < deadline:
            time.sleep(0.01)
        assert jobs.get(blocked.id).status == "completed"
        assert jobs.get("parked").status == "cancelled"
    finally:
        jobs.close()


def test_notifier_publishes_each_job_transition(monkeypatch, tmp_path: Path) -> None:
    jobs = web.JobStore(root=tmp_path / "jobs")
    try:
        record = jobs.durable.create_job("lesson 1", job_id="lesson-1")
        sent: list[tuple[str, dict]] = []
        notifier = web._JobNotifier(
            jobs, send=lambda message, **kwargs: sent.append((message, kwargs))
        )

        notifier.seed()
        assert notifier.tick() == []
        assert sent == []

        jobs.durable._transition(record, "running")
        jobs.durable._transition(record, "completed")
        notifier.tick()
        # Fast jobs only announce the state that survived until the sweep.
        assert [message for message, _ in sent] == ["completed · lesson 1"]
        assert sent[0][1]["priority"] == 4
        assert sent[0][1]["tags"] == "white_check_mark"

        jobs.durable.create_job("lesson 2", job_id="lesson-2")
        notifier.tick()
        assert any("queued · lesson 2" in message for message, _ in sent)

        restarted = web._JobNotifier(
            jobs, send=lambda message, **kwargs: (_ for _ in ()).throw(AssertionError(message))
        )
        restarted.seed()
        assert restarted.tick() == []
    finally:
        jobs.close()


def test_notification_endpoint_requires_configured_topic(monkeypatch) -> None:
    monkeypatch.setattr(web, "NTFY_URL", "")
    with TestClient(web.app) as client:
        response = client.post("/api/notifications/test")
    assert response.status_code == 400
    assert "VOICECOVER_NTFY_URL" in response.json()["detail"]


def test_notification_endpoint_publishes_test_message(monkeypatch) -> None:
    monkeypatch.setattr(web, "NTFY_URL", "https://ntfy.example/voicecover-test")
    sent: list[tuple[str, dict]] = []
    monkeypatch.setattr(web, "ntfy_send", lambda message, **kwargs: sent.append((message, kwargs)))
    with TestClient(web.app) as client:
        response = client.post("/api/notifications/test")
    assert response.status_code == 200
    assert response.json() == {"sent": True}
    assert sent and sent[0][0].startswith("test notification")


def test_ntfy_send_posts_topic_headers_and_body(monkeypatch) -> None:
    monkeypatch.setattr(web, "NTFY_URL", "https://ntfy.example/voicecover-topic")
    monkeypatch.setattr(web, "NTFY_TOKEN", "tk_secret")
    monkeypatch.setattr(web, "PUBLIC_URL", "http://10.147.17.42:8000")
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args) -> bool:
            return False

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request, timeout=None):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(web.urllib.request, "urlopen", fake_urlopen)
    web.ntfy_send("started · lesson 1", priority=4, tags="gear")

    request = captured["request"]
    headers = {key.lower(): value for key, value in request.header_items()}
    assert request.full_url == "https://ntfy.example/voicecover-topic"
    assert request.data == "started · lesson 1".encode("utf-8")
    assert request.get_method() == "POST"
    assert headers["title"] == "voicecover"
    assert headers["priority"] == "4"
    assert headers["tags"] == "gear"
    assert headers["click"] == "http://10.147.17.42:8000"
    assert headers["authorization"] == "Bearer tk_secret"
    assert captured["timeout"] == 10


def test_web_assets_expose_pause_controls() -> None:
    html = web._asset("index.html")
    assert 'id="pause"' in html
    assert 'id="discard"' in html
    javascript = web._asset("app.js")
    assert "pause_pending" in javascript
    assert "'resume'" in javascript
    assert "/api/jobs/${currentJob.id}/${action}" in javascript
    assert "/api/jobs/${currentJob.id}/cancel" in javascript
    assert "Stopping current stage" in javascript
    stylesheet = web._asset("app.css")
    assert ".job-progress-track.paused" in stylesheet
    assert ".retry:disabled" in stylesheet


def test_delete_endpoint_removes_job_record_and_files(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web, "run", _quiet_run)
    jobs = web.JobStore(root=tmp_path / "jobs")
    monkeypatch.setattr(web, "store", jobs)
    try:
        job = _web_job(jobs, "deleteme")
        deadline = time.monotonic() + 2
        while jobs.get(job.id).status not in {"completed", "failed"} and time.monotonic() < deadline:
            time.sleep(0.01)
        directory = jobs.root / job.id
        assert directory.is_dir()

        with TestClient(web.app) as client:
            assert client.delete(f"/api/jobs/{job.id}").status_code == 204
            assert client.delete(f"/api/jobs/{job.id}").status_code == 404
        assert not directory.exists()
        assert all(kept.id != job.id for kept in jobs.durable.all_jobs())
    finally:
        jobs.close()


def test_delete_endpoint_refuses_running_and_unknown_jobs(monkeypatch, tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()

    def fake_run(options: Options, progress) -> Path:
        entered.set()
        release.wait(5)
        raise RuntimeError("unreachable")

    monkeypatch.setattr(web, "run", fake_run)
    jobs = web.JobStore(root=tmp_path / "jobs")
    monkeypatch.setattr(web, "store", jobs)
    try:
        job = _web_job(jobs, "in-flight")
        with TestClient(web.app) as client:
            assert entered.wait(2)
            blocked = client.delete(f"/api/jobs/{job.id}")
            assert blocked.status_code == 409
            assert "running" in blocked.json()["detail"]
            assert client.delete("/api/jobs/no-such-job").status_code == 404

            release.set()
            deadline = time.monotonic() + 2
            while jobs.get(job.id).status not in {"failed", "cancelled"} and time.monotonic() < deadline:
                time.sleep(0.01)
            assert client.delete(f"/api/jobs/{job.id}").status_code == 204
    finally:
        release.set()
        jobs.close()


def test_deleting_a_paused_job_releases_the_queue(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web, "run", _quiet_run)
    jobs = web.JobStore(root=tmp_path / "jobs")
    monkeypatch.setattr(web, "store", jobs)
    try:
        jobs.executor.submit(lambda: None).result(timeout=2)
        parked_dir = jobs.root / "parked-vault"
        parked_dir.mkdir(parents=True)
        parked_options = Options(
            input=parked_dir / "source.mp4", output=parked_dir / "out.mp4", target_language="id"
        )
        jobs.durable.create_job(
            "parked-vault",
            payload=web._options_payload(parked_options),
            output=parked_options.output,
            job_id="parked-vault",
        )
        jobs.pause("parked-vault")

        blocked = _web_job(jobs, "after-vault-delete")
        time.sleep(0.15)
        assert jobs.get(blocked.id).status == "queued"

        with TestClient(web.app) as client:
            assert client.delete("/api/jobs/parked-vault").status_code == 204

        deadline = time.monotonic() + 3
        while jobs.get(blocked.id).status not in {"completed", "failed"} and time.monotonic() < deadline:
            time.sleep(0.01)
        assert jobs.get(blocked.id).status == "completed"
    finally:
        jobs.close()


def test_job_log_endpoint_returns_timed_lines_and_progress(monkeypatch, tmp_path: Path) -> None:
    def fake_run(options: Options, progress) -> Path:
        progress("vault log marker")
        return _quiet_run(options, progress)

    monkeypatch.setattr(web, "run", fake_run)
    jobs = web.JobStore(root=tmp_path / "jobs")
    monkeypatch.setattr(web, "store", jobs)
    try:
        job = _web_job(jobs, "logged")
        deadline = time.monotonic() + 2
        while jobs.get(job.id).status not in {"completed", "failed"} and time.monotonic() < deadline:
            time.sleep(0.01)
        with TestClient(web.app) as client:
            response = client.get(f"/api/jobs/{job.id}/log")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "completed"
            assert "vault log marker" in body["progress"]
            assert any("vault log marker" in line for line in body["lines"])
            assert client.get("/api/jobs/no-such-job/log").status_code == 404
    finally:
        jobs.close()


def test_web_assets_expose_the_job_vault() -> None:
    html = web._asset("index.html")
    assert 'id="job-list"' in html
    assert "Job vault" in html
    javascript = web._asset("app.js")
    assert "/api/jobs/${id}/log" in javascript
    assert "data-action" in javascript
    assert "? 'DELETE' : 'POST'" in javascript
    stylesheet = web._asset("app.css")
    assert ".job-vault" in stylesheet
    assert ".job-log" in stylesheet


def test_publish_output_copies_video_and_subtitle_atomically(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "job" / "lesson.id.mp4"
    source.parent.mkdir()
    source.write_bytes(b"video")
    source.with_suffix(".srt").write_text("subtitle", encoding="utf-8")
    destination = tmp_path / "published"
    monkeypatch.setattr(web, "OUTPUT_DIR", str(destination))
    record = web.JobRecord(
        id="publish-job",
        name="lesson",
        payload={"web_managed": True},
        output=str(source),
    )

    published = web._publish_output(source, record)

    assert published == destination / "lesson.id.mp4"
    assert published.read_bytes() == b"video"
    assert published.with_suffix(".srt").read_text(encoding="utf-8") == "subtitle"
    assert not list(destination.glob("*.tmp"))


def test_publish_output_keeps_youtube_jobs_distinct(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "youtube.id.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(web, "OUTPUT_DIR", str(tmp_path / "published"))
    record = web.JobRecord(
        id="abcdef123456",
        name="youtube",
        payload={"web_managed": True},
        output=str(source),
    )

    published = web._publish_output(source, record)

    assert published == tmp_path / "published" / "youtube-abcdef12.mp4"
    assert published.read_bytes() == b"video"
