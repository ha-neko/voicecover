from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import voicecover.clone as clone_module
from voicecover.clone import (
    CloneConfig,
    ensure_clone_files,
    pick_reference_window,
    select_reference,
)
from voicecover.models import Segment


def _tone_wav(path: Path, seconds: float = 10.0) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(path),
        ],
        check=True,
    )


def test_pick_reference_prefers_clean_language() -> None:
    arabic = Segment(0.0, 5.0, "بسم الله الرحمن الرحيم")
    english = Segment(6.0, 14.0, "Welcome back to the course, let us continue where we stopped")
    shorter = Segment(15.0, 18.0, "another spoken sentence for review")
    start, end, best = pick_reference_window([arabic, english, shorter], 20.0)
    assert (start, end) == (6.0, 14.0)
    assert best is english


def test_pick_reference_prefers_complete_sentence_over_long_fragment() -> None:
    complete = Segment(0.0, 7.3, "Learn weak verbs that occur once in every line of the Quran.")
    fragment = Segment(8.0, 18.9, "And this is related to grammar and spoken")

    start, end, best = pick_reference_window([fragment, complete], 20.0)

    assert best is complete
    assert (start, end) == (0.0, 7.3)


def test_pick_reference_falls_back_when_only_arabic(tmp_path: Path) -> None:
    wav = tmp_path / "tone.wav"
    _tone_wav(wav)
    segments = [Segment(0.0, 10.0, "الحمد لله رب العالمين")]
    start, end, _ = pick_reference_window(segments, 10.0)
    assert start == 0.0
    assert end == 10.0
    assert (start, end) == (0.0, 10.0)


def test_pick_reference_rejects_tiny_segments() -> None:
    segments = [Segment(0.0, 0.5, "hi"), Segment(1.0, 9.0, "a long and clean example sentence here")]
    start, end, best = pick_reference_window(segments, 10.0)
    assert best is segments[1]
    assert (start, end) == (1.0, 9.0)


def test_pick_reference_raises_without_speech() -> None:
    with pytest.raises(RuntimeError):
        pick_reference_window([], 10.0)


def test_select_reference_cuts_and_transcribes(tmp_path: Path) -> None:
    wav = tmp_path / "tone.wav"
    _tone_wav(wav, seconds=12.0)
    segments = [Segment(1.0, 11.0, "This is the reference sentence the clone will be conditioned on")]
    ref_wav, ref_text = select_reference(wav, segments, tmp_path)
    assert ref_text == segments[0].text
    assert ref_wav.is_file()
    assert ref_wav.stat().st_size > 1000


def test_select_reference_reuses_persisted_reference(tmp_path: Path) -> None:
    wav = tmp_path / "tone.wav"
    _tone_wav(wav, seconds=12.0)
    cache = tmp_path / "fixed-reference"
    segments = [Segment(1.0, 11.0, "This reference must remain fixed across every retry.")]

    first_wav, first_text = select_reference(
        wav, segments, tmp_path, reference_cache_dir=cache
    )
    messages: list[str] = []
    second_wav, second_text = select_reference(
        wav, [], tmp_path, messages.append, reference_cache_dir=cache
    )

    assert first_wav == second_wav == cache / "reference.wav"
    assert first_text == second_text == segments[0].text
    assert messages == [f"restored fixed clone reference: {cache / 'reference.wav'}"]


def test_reference_preparation_preserves_internal_timing(monkeypatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(clone_module.media, "run", lambda command: commands.append(list(command)))
    monkeypatch.setattr(clone_module.media, "duration", lambda path: 6.0)

    clone_module._trim_and_normalize(tmp_path / "source.wav", 2.0, 8.0, tmp_path)

    audio_filter = commands[0][commands[0].index("-af") + 1]
    assert "loudnorm" in audio_filter
    assert "silenceremove" not in audio_filter


def test_old_reference_cache_is_regenerated(monkeypatch, tmp_path: Path) -> None:
    cache = tmp_path / "reference"
    cache.mkdir()
    (cache / "reference.wav").write_bytes(b"old audio")
    (cache / "reference.json").write_text(
        '{"start":0,"end":6,"text":"stale reference"}', encoding="utf-8"
    )
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    generated = tmp_path / "generated.wav"
    generated.write_bytes(b"new aligned audio")
    monkeypatch.setattr(clone_module.media, "duration", lambda path: 10.0)
    monkeypatch.setattr(
        clone_module,
        "_trim_and_normalize",
        lambda source_wav, start, end, work: generated,
    )

    _, text = select_reference(
        source,
        [Segment(1, 9, "new aligned reference sentence")],
        tmp_path,
        reference_cache_dir=cache,
    )

    assert text == "new aligned reference sentence"
    assert '"version": 2' in (cache / "reference.json").read_text(encoding="utf-8")


def test_ensure_clone_files_uses_explicit_paths(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    vocab = tmp_path / "vocab.txt"
    ckpt.write_bytes(b"checkpoint")
    vocab.write_text("a b c", encoding="utf-8")
    config = CloneConfig(ref_wav=tmp_path / "no.wav", ref_text="x", ckpt=ckpt, vocab=vocab)
    assert ensure_clone_files(config) == (ckpt, vocab)
