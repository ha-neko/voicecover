import sys
from pathlib import Path
from types import SimpleNamespace

from voicecover.transcription import cpu_thread_count, resolve_device, transcribe


def test_explicit_device_is_preserved() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_accurate_transcription_profile_is_passed_to_whisper(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    initialized: dict[str, object] = {}

    class FakeModel:
        def __init__(self, *args, **kwargs):
            initialized.update(kwargs)

        def transcribe(self, audio, **kwargs):
            captured.update(kwargs)
            raw = [SimpleNamespace(start=0.0, end=1.0, text=" hello ")]
            return raw, SimpleNamespace(language="en")

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeModel))
    segments, language = transcribe(tmp_path / "audio.wav", language=None, device="cpu")
    assert segments[0].text == "hello"
    assert language == "en"
    assert captured["beam_size"] == 5
    assert captured["multilingual"] is False
    assert initialized["cpu_threads"] == cpu_thread_count()
    assert initialized["num_workers"] == 1
    assert captured["language_detection_segments"] == 3
    assert captured["vad_parameters"] == {
        "threshold": 0.35,
        "min_silence_duration_ms": 1000,
        "speech_pad_ms": 400,
    }


def test_mixed_language_detection_can_be_enabled(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio, **kwargs):
            captured.update(kwargs)
            return [SimpleNamespace(start=0.0, end=1.0, text=" الحمد لله ")], SimpleNamespace(language="en")

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeModel))
    segments, _ = transcribe(
        tmp_path / "audio.wav", language="en", device="cpu", multilingual=True
    )
    assert captured["multilingual"] is True
    assert segments[0].text == "الحمد لله"


def test_whisper_thread_override(monkeypatch) -> None:
    monkeypatch.setenv("VOICECOVER_WHISPER_THREADS", "12")
    assert cpu_thread_count() == 12
