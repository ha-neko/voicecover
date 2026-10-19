from __future__ import annotations

from pathlib import Path

from .download import is_youtube_url


def default_output(input_value: str, target_language: str) -> Path:
    if is_youtube_url(input_value):
        return Path(f"youtube.{target_language}.mp4")
    input_path = Path(input_value)
    return input_path.with_name(f"{input_path.stem}.{target_language}.mp4")
