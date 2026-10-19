from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import Mock

from voicecover.visual import (
    _ahash,
    _dedupe,
    _parse_ocr_box,
    build_prompt,
    cluster_slide_states,
    extract_frames,
)
from voicecover.visual_models import FrameSample, SlideState, VisualTrack


def _lavfi_frame(path: Path, source: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"{source}=s=320x180:d=0.1",
            "-frames:v",
            "1",
            str(path),
        ],
        check=True,
    )


def test_build_prompt_keeps_arabic_and_english_terms() -> None:
    terms = build_prompt(
        ["Surah Al-Baqarah weak sound page", "فعل ماضي and إسن"], None
    )
    assert "Surah" in terms
    assert "Baqarah" in terms
    assert "فعل" in terms
    assert "ماضي" in terms
    assert "إسن" in terms


def test_build_prompt_filters_stopwords() -> None:
    terms = build_prompt(["the and this that with page size"], None)
    assert "the" not in terms
    assert "page" not in terms


def test_build_prompt_keeps_existing_user_prompt() -> None:
    combined = build_prompt(["Surah Al-Baqarah"], "Quranic Arabic lesson")
    assert combined.startswith("Quranic Arabic lesson")


def test_dedupe_keeps_only_distinct_frames(tmp_path: Path) -> None:
    first = tmp_path / "a.jpg"
    second = tmp_path / "b.jpg"
    third = tmp_path / "c.jpg"
    _lavfi_frame(first, "testsrc2")
    _lavfi_frame(second, "testsrc2")
    _lavfi_frame(third, "smptebars")
    kept = _dedupe([first, second, third])
    assert len(kept) == 2


def test_adjacent_clustering_keeps_a_slide_that_reappears() -> None:
    frames = [
        FrameSample("a1.jpg", 0.0, 480, 270, average_hash=0),
        FrameSample("a2.jpg", 3.0, 480, 270, average_hash=0),
        FrameSample("b.jpg", 6.0, 480, 270, average_hash=(1 << 256) - 1),
        FrameSample("a3.jpg", 9.0, 480, 270, average_hash=0),
    ]

    slides = cluster_slide_states(frames, sample_interval=3.0)

    assert [slide.frame_indices for slide in slides] == [(0, 1), (2,), (3,)]


def test_ahash_uses_all_256_pixels(monkeypatch, tmp_path: Path) -> None:
    pixels = bytes([0] * 128 + [255] * 128)
    monkeypatch.setattr(subprocess, "run", Mock(return_value=Mock(stdout=pixels)))

    value = _ahash(tmp_path / "frame.jpg")

    assert value.bit_count() == 128
    assert value >> 128 == (1 << 128) - 1


def test_visual_track_round_trip_retains_ocr_geometry_and_metadata() -> None:
    box = _parse_ocr_box(
        [[[1, 2], [101, 2], [101, 22], [1, 22]], "زوروا https://BrandX.com", 0.97],
        480,
        270,
    )
    assert box is not None
    track = VisualTrack(
        frames=(FrameSample("frame.jpg", 3.0, 480, 270, boxes=(box,), average_hash=42),),
        slides=(SlideState(0, 3.0, 6.0, (0,), 0),),
        sample_interval=3.0,
    )

    restored = VisualTrack.from_json(track.to_json())

    assert restored == track
    assert restored.version == 1
    assert restored.frames[0].boxes[0].points[2].x == 101
    assert restored.frames[0].boxes[0].source_width == 480
    assert restored.frames[0].boxes[0].scripts == ("arabic", "latin")
    assert set(restored.frames[0].boxes[0].protection_reasons) == {
        "arabic-script",
        "url",
        "likely-branding",
    }
    assert "https://BrandX.com" in build_prompt(restored)
    assert "زوروا" in build_prompt(restored)


def test_extract_frames_produces_expected_count(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x180:r=2:d=4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    frames = extract_frames(source, tmp_path / "out", interval=1.0)
    assert len(frames) == 4
