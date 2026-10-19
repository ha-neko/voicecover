from pathlib import Path

from voicecover import media
from voicecover.media import _atempo_chain


def test_atempo_chain_handles_large_speed() -> None:
    assert _atempo_chain(250) == "atempo=100.00000000,atempo=2.50000000"


def test_atempo_chain_handles_normal_speed() -> None:
    assert _atempo_chain(1.25) == "atempo=1.25000000"


def test_fit_audio_no_trim(monkeypatch, tmp_path: Path) -> None:
    """fit_audio should not trim or pad — just normalize format."""
    commands: list[list[str]] = []
    monkeypatch.setattr(media, "duration", lambda path: 2.0)
    monkeypatch.setattr(media, "run", lambda command: commands.append(list(command)))
    media.fit_audio(tmp_path / "source.wav", tmp_path / "output.wav", 4.0)
    filters = commands[0][commands[0].index("-af") + 1]
    assert "atempo" not in filters
    assert "apad" not in filters
    assert "atrim" not in filters


def test_fit_audio_caps_tempo_correction_at_twenty_five_percent(
    monkeypatch, tmp_path: Path
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(media, "duration", lambda path: 8.0)
    monkeypatch.setattr(media, "run", lambda command: commands.append(list(command)))

    media.fit_audio(tmp_path / "source.wav", tmp_path / "output.wav", 4.0)

    filters = commands[0][commands[0].index("-af") + 1]
    assert "atempo=1.25000000" in filters


def test_extract_audio_range_uses_exact_interval(monkeypatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(media, "run", lambda command: commands.append(list(command)))
    media.extract_audio_range(tmp_path / "source.wav", tmp_path / "arabic.wav", 2.5, 5.75)
    command = commands[0]
    assert command[command.index("-ss") + 1] == "2.500000"
    assert command[command.index("-t") + 1] == "3.250000"
    assert command[command.index("-ar") + 1] == "48000"
