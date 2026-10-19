from __future__ import annotations

from pathlib import Path

from voicecover import visual_replace
from voicecover.visual_models import FrameSample, OCRBox, Point, SlideState, VisualTrack
from voicecover.visual_replace import RenderedOverlay


def _box(text: str, confidence: float = 0.95, *, protected: bool = False) -> OCRBox:
    return OCRBox(
        points=(Point(10, 10), Point(110, 10), Point(110, 30), Point(10, 30)),
        text=text,
        confidence=confidence,
        source_width=120,
        source_height=60,
        scripts=("latin",),
        protected=protected,
    )


def _track(*boxes: OCRBox) -> VisualTrack:
    frames = (
        FrameSample("a.jpg", 0, 120, 60, boxes=boxes),
        FrameSample("ignored.jpg", 3, 120, 60, boxes=(_box("Not representative"),)),
        FrameSample("b.jpg", 6, 120, 60, boxes=(_box("Repeated text"),)),
    )
    return VisualTrack(
        frames=frames,
        slides=(SlideState(0, 0, 6, (0, 1), 0), SlideState(1, 6, 9, (2,), 2)),
        sample_interval=3,
    )


def test_selection_is_conservative_and_deduplicates_translation() -> None:
    invalid = _box("Outside")
    invalid = OCRBox(
        points=(Point(-1, 10), *invalid.points[1:]),
        text=invalid.text,
        confidence=invalid.confidence,
        source_width=invalid.source_width,
        source_height=invalid.source_height,
    )
    track = _track(
        _box("Repeated text"),
        _box("نص عربي"),
        _box("https://example.com"),
        _box("BrandName"),
        _box("Low confidence", 0.4),
        _box("Explicitly protected", protected=True),
        invalid,
    )
    calls: list[list[str]] = []

    replacements = visual_replace.select_replacements(
        track, lambda values: calls.append(values) or ["Teks berulang"]
    )

    assert calls == [["Repeated text"]]
    assert [(item.slide_index, item.box_index) for item in replacements] == [(0, 0), (1, 0)]
    assert all(item.translated == "Teks berulang" for item in replacements)


def test_scale_polygon_scales_and_clamps() -> None:
    points = (Point(-2, 5), Point(50, 5), Point(50, 25), Point(102, 25))

    scaled = visual_replace.scale_polygon(points, (100, 50), (1920, 1080))

    assert scaled == ((0, 108), (960, 108), (960, 540), (1919, 540))


def test_wrap_text_wraps_words_and_splits_an_overlong_word() -> None:
    measure = lambda value: len(value) * 10

    assert visual_replace.wrap_text("one two three", 70, measure) == ["one two", "three"]
    assert visual_replace.wrap_text("abcdefgh", 30, measure) == ["abc", "def", "gh"]


def test_extract_representatives_uses_slide_timestamp_and_no_scaling(
    monkeypatch, tmp_path: Path
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(visual_replace.media, "run", lambda command: commands.append(list(command)))

    frames = visual_replace.extract_representative_frames(
        tmp_path / "input.mp4", _track(_box("Some text")), tmp_path / "frames", {1}
    )

    assert set(frames) == {1}
    assert commands[0][commands[0].index("-ss") + 1] == "7.500000"
    assert commands[0].index("-ss") > commands[0].index("-i")
    assert "-vf" not in commands[0]


def test_apply_overlays_constructs_timed_h264_command(monkeypatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(visual_replace.media, "run", lambda command: commands.append(list(command)))
    overlays = [RenderedOverlay(2, 6.0, 9.5, tmp_path / "overlay.png")]

    result = visual_replace.apply_overlays(
        tmp_path / "input.mp4", overlays, tmp_path / "output.mp4"
    )

    command = commands[0]
    filters = command[command.index("-filter_complex") + 1]
    assert "between(t,6.000000,9.500000)" in filters
    assert command[command.index("-crf") + 1] == "18"
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-c:a") + 1] == "copy"
    assert "0:a?" in command
    assert result == tmp_path / "output.mp4"


def test_apply_overlays_copies_input_when_nothing_rendered(monkeypatch, tmp_path: Path) -> None:
    source, output = tmp_path / "input.mp4", tmp_path / "output.mp4"
    source.write_bytes(b"unchanged")
    monkeypatch.setattr(
        visual_replace.media,
        "run",
        lambda command: (_ for _ in ()).throw(AssertionError("ffmpeg should not run")),
    )

    visual_replace.apply_overlays(source, [], output)

    assert output.read_bytes() == b"unchanged"


def test_unchanged_translation_is_not_redrawn() -> None:
    replacements = visual_replace.select_replacements(
        _track(_box("Already translated")), lambda values: values
    )

    assert replacements == []


def test_render_overlay_masks_glyphs_instead_of_the_whole_ocr_box(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    frame = tmp_path / "slide.png"
    image = Image.new("RGB", (640, 360))
    image.putdata(
        [
            (shade, shade, shade)
            for _y in range(360)
            for x in range(640)
            for shade in [180 + round(75 * x / 639)]
        ]
    )
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(visual_replace._font_path(False), 34)
    draw.text((100, 145), "Original text", font=font, fill="black")
    image.save(frame)
    box = OCRBox(
        points=(Point(95, 138), Point(345, 138), Point(345, 188), Point(95, 188)),
        text="Original text",
        confidence=0.99,
        source_width=640,
        source_height=360,
        scripts=("latin",),
    )
    output = tmp_path / "overlay.png"

    rendered = visual_replace.render_slide_overlay(
        frame,
        [visual_replace.BoxReplacement(0, 0, "Original text", "Teks asli", box)],
        output,
    )

    assert rendered
    alpha = Image.open(output).getchannel("A")
    opaque_pixels = sum(
        count
        for value, count in enumerate(alpha.crop((95, 138, 345, 188)).histogram())
        if value > 16
    )
    box_pixels = (345 - 95) * (188 - 138)
    # A full-box inpaint approaches 100%; glyph-shaped cleaning stays sparse.
    assert 0 < opaque_pixels < box_pixels * 0.55
