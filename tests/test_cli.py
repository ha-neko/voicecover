from __future__ import annotations

import json
from pathlib import Path

from voicecover import cli
from voicecover.jobs import JobStore
from voicecover.profiles import quran_indonesian_profile


def write_manifest(path: Path, jobs: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"version": 1, "id": "batch-test", "name": "Test batch", "jobs": jobs}),
        encoding="utf-8",
    )


def test_batch_processes_in_manifest_order_and_persists(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "batch.json"
    outputs = [tmp_path / "first.mp4", tmp_path / "second.mp4"]
    write_manifest(
        manifest,
        [
            {"id": "first", "input": "first.mov", "output": str(outputs[0]), "target": "id"},
            {"id": "second", "input": "second.mov", "output": str(outputs[1]), "target": "fr"},
        ],
    )
    seen: list[str] = []

    def process_video(options, progress):
        seen.append(str(options.input))
        options.output.write_bytes(b"video")
        return options.output

    monkeypatch.setattr(cli, "process_video", process_video)
    store_path = tmp_path / "store"
    assert cli.main(["--batch", str(manifest), "--job-store", str(store_path)]) == 0
    assert seen == ["first.mov", "second.mov"]

    reloaded = JobStore(store_path)
    assert reloaded.get_batch("batch-test").status == "completed"
    assert [reloaded.get_job(job_id).attempts for job_id in ["first", "second"]] == [1, 1]


def test_batch_resumes_running_job_and_retries_failure(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "batch.json"
    output = tmp_path / "result.mp4"
    write_manifest(
        manifest,
        [{"id": "recover", "input": "source.mov", "output": str(output), "target": "id"}],
    )
    store_path = tmp_path / "store"
    store = JobStore(store_path)
    data = cli._load_batch(manifest)
    batch = store.create_batch(
        "Test batch",
        cli._batch_specs(data, cli.parser().parse_args([])),
        batch_id="batch-test",
    )
    job = store.get_job(batch.job_ids[0])
    job.status = "running"
    store._save_job(job)

    calls = 0

    def process_video(options, progress):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary failure")
        options.output.write_bytes(b"recovered")
        return options.output

    monkeypatch.setattr(cli, "process_video", process_video)
    args = ["--batch", str(manifest), "--job-store", str(store_path)]
    assert cli.main(args) == 1
    assert JobStore(store_path).get_job("recover").status == "failed"
    assert cli.main([*args, "--retry-failed"]) == 0
    recovered = JobStore(store_path).get_job("recover")
    assert recovered.status == "completed"
    assert recovered.attempts == 2


def test_batch_applies_profile_then_safe_item_options(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "batch.json"
    output = tmp_path / "profiled.mp4"
    write_manifest(
        manifest,
        [
            {
                "input": "lesson.mov",
                "output": str(output),
                "profile": "quran-id",
                "options": {"model": "tiny", "write_subtitles": False},
            }
        ],
    )
    captured = []

    def process_video(options, progress):
        captured.append(options)
        options.output.write_bytes(b"video")
        return options.output

    monkeypatch.setattr(cli, "process_video", process_video)
    monkeypatch.setattr(cli, "load_profile_or_builtin", lambda name: quran_indonesian_profile())
    assert cli.main(["--batch", str(manifest), "--job-store", str(tmp_path / "store")]) == 0

    options = captured[0]
    assert options.target_language == "id"
    assert options.model == "tiny"
    assert options.write_subtitles is False
    assert options.translator == "llm"
    assert options.protected_terms["Allah"] == "Allah"
    assert options.localize_visual_text is True


def test_visual_only_flag_enables_visual_localization(tmp_path: Path) -> None:
    args = cli.parser().parse_args(["lesson.mp4", "--target", "id", "--visual-only"])
    options = cli._options_from_args(
        args, "lesson.mp4", tmp_path / "lesson.id.mp4", "id", None
    )

    assert options.visual_only is True
    assert options.localize_visual_text is True


def test_sound_only_flag_disables_visual_processing(tmp_path: Path) -> None:
    args = cli.parser().parse_args(["lesson.mp4", "--target", "id", "--sound-only"])
    options = cli._options_from_args(
        args, "lesson.mp4", tmp_path / "lesson.id.mp4", "id", "quran-id"
    )

    assert options.sound_only is True
    assert options.visual_only is False
    assert options.visual_context is False
    assert options.localize_visual_text is False
    assert options.write_visual_track is False
