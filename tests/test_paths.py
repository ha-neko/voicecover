from pathlib import Path

from voicecover.paths import default_output


def test_default_output_for_local_video() -> None:
    assert default_output("clips/demo.webm", "fr") == Path("clips/demo.fr.mp4")


def test_default_output_for_youtube() -> None:
    assert default_output("https://youtu.be/abc", "ja") == Path("youtube.ja.mp4")
