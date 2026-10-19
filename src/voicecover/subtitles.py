from __future__ import annotations

from pathlib import Path

from .models import Segment


def _timestamp(seconds: float) -> str:
    milliseconds = round(max(0, seconds) * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments: list[Segment], output: Path, bilingual: bool = False) -> None:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        # preserved Arabic passages carry their original text in translated_text;
        # everything else carries the real translation.
        text = segment.translated_text or segment.text
        if bilingual and segment.translated_text and segment.translated_text != segment.text:
            text = f"{segment.text}\n{segment.translated_text}"
        blocks.append(
            f"{index}\n{_timestamp(segment.start)} --> {_timestamp(segment.end)}\n{text}\n"
        )
    output.write_text("\n".join(blocks), encoding="utf-8")
