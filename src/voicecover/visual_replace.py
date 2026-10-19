from __future__ import annotations

import math
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import media
from .visual_models import OCRBox, Point, VisualTrack

MIN_CONFIDENCE = 0.75
_ARABIC = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
_LATIN = re.compile(r"[A-Za-z]")
_URL = re.compile(r"(?:https?://|www\.)\S+|\b\S+\.(?:com|org|net|io|id)\b", re.I)


@dataclass(frozen=True)
class BoxReplacement:
    """A translated OCR box, identified within its slide representative."""

    slide_index: int
    box_index: int
    source: str
    translated: str
    box: OCRBox


@dataclass(frozen=True)
class RenderedOverlay:
    slide_index: int
    start: float
    end: float
    path: Path


def _looks_like_brand(text: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z0-9]*", text)
    return any(
        (word.isupper() and 2 <= len(word) <= 12)
        or (any(char.isupper() for char in word[1:]) and any(char.islower() for char in word))
        or (any(char.isalpha() for char in word) and any(char.isdigit() for char in word))
        for word in words
    ) or "™" in text or "®" in text


def _valid_box(box: OCRBox, min_confidence: float) -> bool:
    protected_reasons = {"arabic-script", "url", "likely-branding"}
    if (
        box.confidence < min_confidence
        or not box.text.strip()
        or box.protected
        or "arabic" in box.scripts
        or bool(protected_reasons.intersection(box.protection_reasons))
        or len(box.points) != 4
        or box.source_width <= 0
        or box.source_height <= 0
    ):
        return False
    if _ARABIC.search(box.text) or _URL.search(box.text) or _looks_like_brand(box.text):
        return False
    if not _LATIN.search(box.text):
        return False
    coordinates = [(point.x, point.y) for point in box.points]
    if not all(math.isfinite(value) for point in coordinates for value in point):
        return False
    if not all(
        0 <= x <= box.source_width and 0 <= y <= box.source_height
        for x, y in coordinates
    ):
        return False
    xs, ys = zip(*coordinates)
    area = abs(
        sum(
            xs[index] * ys[(index + 1) % 4] - xs[(index + 1) % 4] * ys[index]
            for index in range(4)
        )
    )
    box_width, box_height = max(xs) - min(xs), max(ys) - min(ys)
    # Tiny OCR inside screenshots, diagrams, and thumbnails cannot be replaced
    # legibly after translation. It is substantially better left untouched.
    return (
        box_width > 0
        and box_height > 0
        and area > 0
        and box_height / box.source_height >= 0.018
        and box_width / box.source_width >= 0.02
        and (box_width * box_height) / (box.source_width * box.source_height) >= 0.0015
    )


def select_replacements(
    track: VisualTrack,
    translate: Callable[[list[str]], list[str]],
    min_confidence: float = MIN_CONFIDENCE,
) -> list[BoxReplacement]:
    """Select representative-frame boxes and translate each exact string once."""
    selected: list[tuple[int, int, OCRBox]] = []
    for slide in track.slides:
        if not (0 <= slide.representative_frame_index < len(track.frames)):
            continue
        frame = track.frames[slide.representative_frame_index]
        selected.extend(
            (slide.index, box_index, box)
            for box_index, box in enumerate(frame.boxes)
            if _valid_box(box, min_confidence)
        )

    sources = list(dict.fromkeys(box.text.strip() for _, _, box in selected))
    if not sources:
        return []
    translated = list(translate(sources))
    if len(translated) != len(sources):
        raise ValueError("translate() must return one result for every source string")
    translations = dict(zip(sources, translated))
    return [
        BoxReplacement(slide_index, box_index, source, target.strip(), box)
        for slide_index, box_index, box in selected
        if (source := box.text.strip())
        and (target := str(translations[source])).strip()
        # A failed visual-translation batch falls back to its source text.
        # Redrawing that source in a different font only damages the slide.
        and _normalized_text(source) != _normalized_text(target)
    ]


def _normalized_text(text: str) -> str:
    return re.sub(r"\W+", "", text, flags=re.UNICODE).casefold()


def scale_polygon(
    points: Sequence[Point], source_size: tuple[int, int], target_size: tuple[int, int]
) -> tuple[tuple[int, int], ...]:
    """Scale OCR coordinates to source-video pixels, clamped to its bounds."""
    source_width, source_height = source_size
    target_width, target_height = target_size
    if source_width <= 0 or source_height <= 0 or target_width <= 0 or target_height <= 0:
        raise ValueError("source and target dimensions must be positive")
    return tuple(
        (
            min(target_width - 1, max(0, round(point.x * target_width / source_width))),
            min(target_height - 1, max(0, round(point.y * target_height / source_height))),
        )
        for point in points
    )


def wrap_text(text: str, max_width: float, measure: Callable[[str], float]) -> list[str]:
    """Greedily wrap words, splitting a single overlong word when necessary."""
    if max_width <= 0:
        return []
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if not current or measure(candidate) <= max_width:
            current = candidate
            if measure(current) <= max_width:
                continue
        if current and current != word:
            lines.append(current)
            current = word
        if measure(current) > max_width:
            chunk = ""
            for character in current:
                if chunk and measure(chunk + character) > max_width:
                    lines.append(chunk)
                    chunk = character
                else:
                    chunk += character
            current = chunk
    if current:
        lines.append(current)
    return lines


def extract_representative_frames(
    video: Path | str,
    track: VisualTrack,
    work: Path,
    slide_indices: set[int] | None = None,
) -> dict[int, Path]:
    """Extract full-resolution frames at selected slide timestamps."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    frames: dict[int, Path] = {}
    for slide in track.slides:
        if slide_indices is not None and slide.index not in slide_indices:
            continue
        if slide.start < 0 or slide.end <= slide.start:
            continue
        output = work / f"slide-{slide.index:04d}.png"
        representative = track.frames[slide.representative_frame_index]
        # The fps filter chooses the frame nearest each interval centre while
        # FrameSample timestamps identify interval starts. Keep the centre
        # correction, but use accurate output-side seeking below.
        timestamp = representative.timestamp + track.sample_interval / 2
        media.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                # Output-side seeking decodes from the preceding keyframe and
                # lands accurately; input-side -ss can return an earlier frame
                # while slide elements are still animating into position.
                "-ss", f"{timestamp:.6f}", "-frames:v", "1", str(output),
            ]
        )
        frames[slide.index] = output
    return frames


def _imaging_modules():
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError(
            "Visual text replacement requires OpenCV, numpy, and Pillow; "
            "install voicecover[visual]"
        ) from exc
    return cv2, np, Image, ImageDraw, ImageFont


@lru_cache(maxsize=2)
def _font_path(bold: bool) -> str:
    family = "Noto Sans" if not bold else "Noto Sans:style=Bold"
    try:
        result = subprocess.run(
            ["fc-match", "-f", "%{file}", family],
            check=True,
            capture_output=True,
            text=True,
        )
        path = result.stdout.strip()
        if path and Path(path).is_file():
            return path
    except (OSError, subprocess.CalledProcessError):
        pass
    candidates = (
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    raise RuntimeError("a fontconfig-managed sans-serif font is required for visual replacement")


def _font(ImageFont, size: int, bold: bool):
    try:
        return ImageFont.truetype(_font_path(bold), size)
    except OSError as exc:
        raise RuntimeError("the selected visual replacement font could not be loaded") from exc


def _estimate_text_color(image, mask, polygon, cv2, np) -> tuple[int, int, int]:
    outer = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
    ring = (outer > 0) & (mask == 0)
    background_pixels = image[ring]
    if not len(background_pixels):
        background_pixels = image.reshape(-1, 3)
    background = np.median(background_pixels, axis=0)
    raw = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(raw, [polygon], 255)
    candidates = image[raw > 0]
    if not len(candidates):
        color = 255 - background
    else:
        distance = np.linalg.norm(candidates.astype(float) - background, axis=1)
        contrasting = candidates[distance >= np.percentile(distance, 75)]
        color = np.median(contrasting if len(contrasting) else candidates, axis=0)
    return tuple(int(value) for value in color[::-1])


def _draw_text_with_outline(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int],
    outline_width: int = 2,
    outline_color: tuple[int, int, int, int] | None = None,
) -> None:
    """Draw text with a contrasting outline for readability on any background."""
    x, y = xy
    if outline_color is None:
        # pick black or white outline based on text luminance
        lum = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
        outline_color = (0, 0, 0, 200) if lum > 128 else (255, 255, 255, 180)
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx or dy:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    draw.text((x, y), text, font=font, fill=fill)


def _estimate_background_color(image, mask, cv2, np) -> tuple[int, int, int]:
    """Estimate the dominant background color near a masked region."""
    outer = cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=2)
    ring = (outer > 0) & (mask == 0)
    pixels = image[ring]
    if not len(pixels):
        pixels = image.reshape(-1, 3)
    return tuple(int(c) for c in np.median(pixels, axis=0)[::-1])


def _source_glyph_mask(image, polygon_mask, polygon, cv2, np):
    """Segment source letter strokes without erasing the whole OCR rectangle.

    OCR polygons include the spaces between letters and often a generous strip
    of surrounding slide art.  Treating that polygon as an inpaint mask creates
    the large smeared rectangles the old renderer was known for.  This uses the
    colors immediately outside the polygon as a background model, then retains
    only interior pixels that are materially closer to the likely foreground.
    An uncertain mask is rejected rather than damaging the frame.
    """
    box_height = max(1, int(polygon[:, 1].max() - polygon[:, 1].min()))
    ring_size = max(5, min(17, round(box_height * 0.35)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_size, ring_size))
    outer = cv2.dilate(polygon_mask, kernel, iterations=1)
    ring = (outer > 0) & (polygon_mask == 0)
    inside = polygon_mask > 0
    background_pixels = image[ring]
    interior_pixels = image[inside]
    if len(background_pixels) < 12 or len(interior_pixels) < 12:
        return None

    interior = interior_pixels.astype(np.float32)
    # OCR boxes around white letters on a colored banner often have a white
    # *outside* ring. The dominant interior color is the reliable background;
    # using the outside ring would identify the banner itself as foreground.
    background = np.median(interior, axis=0)
    interior_distance = np.linalg.norm(interior - background, axis=1)
    flat_background = float(np.mean(interior_distance < 22)) >= 0.80
    distance_from_background = np.linalg.norm(interior - background, axis=1)
    # Letter strokes may occupy less than 20% of a generous OCR rectangle, so
    # a percentile over *all* interior pixels often sees only background.
    changed = distance_from_background[distance_from_background >= 8]
    if len(changed) < max(4, round(len(interior) * 0.003)):
        return None
    contrast = float(np.percentile(changed, 65))

    # The most background-distant interior pixels are usually the glyph color.
    foreground_pool = interior[distance_from_background >= np.percentile(changed, 55)]
    foreground = np.median(foreground_pool, axis=0)
    image_f = image.astype(np.float32)
    from_background = np.linalg.norm(image_f - background, axis=2)
    from_foreground = np.linalg.norm(image_f - foreground, axis=2)
    threshold = max(10.0, min(42.0, contrast * 0.32))
    likely_glyph = (
        inside
        & (from_background >= threshold)
        & (from_foreground <= from_background * 0.9)
    )
    mask = likely_glyph.astype(np.uint8) * 255

    # Remove isolated compression noise, but preserve thin character strokes.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    minimum_component = max(2, round(cv2.contourArea(polygon) * 0.0008))
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_component:
            cleaned[labels == label] = 255

    polygon_area = max(1, int(np.count_nonzero(inside)))
    coverage = np.count_nonzero(cleaned) / polygon_area
    if coverage < 0.004 or coverage > 0.42:
        return None
    dilation = max(1, min(3, round(box_height * 0.035)))
    glyph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1)
    )
    glyph_mask = cv2.dilate(cleaned, glyph_kernel, iterations=1)
    return glyph_mask, background, flat_background


def render_slide_overlay(
    frame: Path | str,
    replacements: Sequence[BoxReplacement],
    output: Path,
) -> bool:
    """Render inpainted translated patches into one transparent PNG."""
    cv2, np, Image, ImageDraw, ImageFont = _imaging_modules()
    image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not read representative frame: {frame}")
    height, width = image.shape[:2]
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    inpaint_mask = np.zeros((height, width), dtype=np.uint8)
    flat_fills: list[tuple[object, object]] = []
    plans: list[dict[str, object]] = []

    for replacement in replacements:
        box = replacement.box
        scaled = scale_polygon(
            box.points, (box.source_width, box.source_height), (width, height)
        )
        polygon = np.asarray(scaled, dtype=np.int32)
        if polygon.shape != (4, 2) or abs(cv2.contourArea(polygon)) < 4:
            continue
        polygon_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [polygon], 255)
        segmented = _source_glyph_mask(image, polygon_mask, polygon, cv2, np)
        if segmented is None:
            continue
        glyph_mask, background_bgr, flat_background = segmented
        xs, ys = polygon[:, 0], polygon[:, 1]
        box_height = max(1, int(ys.max() - ys.min()))
        padding = max(2, round(box_height * 0.08))
        left, top = max(0, int(xs.min()) - padding), max(0, int(ys.min()) - padding)
        right, bottom = min(width, int(xs.max()) + padding + 1), min(height, int(ys.max()) + padding + 1)
        if right - left < 10 or bottom - top < 10:
            continue

        color = _estimate_text_color(image, glyph_mask, polygon, cv2, np)
        background = tuple(int(value) for value in background_bgr[::-1])
        inner_width = max(1, right - left - 2 * padding)
        inner_height = max(1, bottom - top - 2 * padding)
        minimum_size = max(10, round(height * 0.011))
        glyph_coverage = np.count_nonzero(glyph_mask) / max(1, np.count_nonzero(polygon_mask))
        bold = glyph_coverage >= 0.22
        layout = _layout_text(
            ImageFont,
            replacement.translated,
            inner_width,
            inner_height,
            min_size=minimum_size,
            max_size=max(minimum_size, min(round(box_height * 0.82), round(height * 0.09))),
            bold=bold,
        )
        if layout is None:
            continue
        font, lines, line_height = layout
        box_center = (int(xs.min()) + int(xs.max())) / 2
        centered = (right - left) >= width * 0.28 and abs(box_center - width / 2) <= width * 0.18
        plans.append(
            {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "padding": padding,
                "font": font,
                "lines": lines,
                "line_height": line_height,
                "color": color,
                "background": background,
                "centered": centered,
            }
        )
        if flat_background:
            # On a solid title bar, replacing the complete OCR strip with its
            # dominant color removes anti-aliased source edges more faithfully
            # than inpainting. Textured regions still use stroke-only masks.
            removal_mask = polygon_mask
            flat_fills.append((polygon_mask, background_bgr))
        else:
            removal_mask = glyph_mask
            inpaint_mask = cv2.max(inpaint_mask, glyph_mask)
        combined_mask = cv2.max(combined_mask, removal_mask)

    if not plans:
        return False

    # Clean every source glyph in one operation. Per-box inpainting could paste
    # original pixels back over a neighboring translated box when OCR polygons
    # overlapped.
    inpaint_radius = max(2, round(min(width, height) * 0.002))
    inpainted = (
        cv2.inpaint(image, inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)
        if np.any(inpaint_mask)
        else image.copy()
    )
    for fill_mask, background_bgr in flat_fills:
        inpainted[fill_mask > 0] = background_bgr
    alpha_mask = cv2.GaussianBlur(
        combined_mask.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=max(0.65, inpaint_radius * 0.4),
    )
    overlay = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay.putalpha(
        Image.fromarray(np.clip(alpha_mask * 255, 0, 255).astype(np.uint8))
    )
    draw = ImageDraw.Draw(overlay)
    for plan in plans:
        left, top = int(plan["left"]), int(plan["top"])
        right, bottom = int(plan["right"]), int(plan["bottom"])
        padding = int(plan["padding"])
        font = plan["font"]
        lines = plan["lines"]
        line_height = int(plan["line_height"])
        color = plan["color"]
        background = plan["background"]
        total_text_height = len(lines) * line_height
        y = top + max(padding, (bottom - top - total_text_height) // 2)
        text_luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        background_luminance = 0.299 * background[0] + 0.587 * background[1] + 0.114 * background[2]
        outline_w = 1 if abs(text_luminance - background_luminance) < 90 and font.size >= 15 else 0
        for line in lines:
            line_width = font.getlength(line)
            x = (
                max(left + padding, left + (right - left - round(line_width)) // 2)
                if plan["centered"]
                else left + padding
            )
            _draw_text_with_outline(draw, (x, y), line, font, (*color, 255), outline_w)
            y += line_height

    output.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output, format="PNG")
    return True


def _layout_text(
    ImageFont,
    text: str,
    width: int,
    height: int,
    *,
    min_size: int = 8,
    max_size: int | None = None,
    bold: bool = False,
):
    """Find the largest font that fits the text within the given dimensions."""
    ceiling = max_size if max_size is not None else min(height, 80)
    for size in range(max(min_size, ceiling), min_size - 1, -1):
        font = _font(ImageFont, size, bold=bold)
        lines = wrap_text(text, width, font.getlength)
        bounds = font.getbbox("Ag")
        line_height = max(1, bounds[3] - bounds[1] + max(2, round(size * 0.15)))
        if lines and len(lines) <= 4 and len(lines) * line_height <= height:
            return font, lines, line_height
    return None


def render_overlays(
    video: Path | str,
    track: VisualTrack,
    replacements: Sequence[BoxReplacement],
    work: Path,
) -> list[RenderedOverlay]:
    """Extract representatives and render overlays for nonempty slide plans."""
    grouped: dict[int, list[BoxReplacement]] = {}
    for replacement in replacements:
        grouped.setdefault(replacement.slide_index, []).append(replacement)
    if not grouped:
        return []
    work = Path(work)
    frames = extract_representative_frames(video, track, work / "frames", set(grouped))
    slides = {slide.index: slide for slide in track.slides}
    overlays: list[RenderedOverlay] = []
    for slide_index, items in grouped.items():
        slide, frame = slides.get(slide_index), frames.get(slide_index)
        if slide is None or frame is None:
            continue
        output = work / "overlays" / f"slide-{slide_index:04d}.png"
        if render_slide_overlay(frame, items, output):
            # Slide boundaries already lie halfway between sampled frames.
            # Only representative-frame extraction needs the half-interval
            # correction; shifting playback intervals causes visible lag.
            overlays.append(RenderedOverlay(slide_index, slide.start, slide.end, output))
    return overlays


def apply_overlays(
    video: Path | str, overlays: Sequence[RenderedOverlay], output: Path | str
) -> Path:
    """Apply static overlays during slide intervals and preserve source audio."""
    source, destination = Path(video), Path(output)
    if not overlays:
        if source.resolve() != destination.resolve():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-i", str(source)]
    for overlay in overlays:
        command.extend(["-loop", "1", "-i", str(overlay.path)])
    filters: list[str] = []
    previous = "0:v"
    for index, overlay in enumerate(overlays, start=1):
        output_label = f"v{index}"
        filters.append(
            f"[{previous}][{index}:v]overlay=0:0:shortest=1:"
            f"enable='between(t,{overlay.start:.6f},{overlay.end:.6f})'[{output_label}]"
        )
        previous = output_label
    command.extend(
        [
            "-filter_complex", ";".join(filters), "-map", f"[{previous}]", "-map", "0:a?",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy",
            "-map_metadata", "0", "-movflags", "+faststart", str(destination),
        ]
    )
    media.run(command)
    return destination


def replace_visual_text(
    video: Path | str,
    track: VisualTrack,
    translate: Callable[[list[str]], list[str]],
    output: Path | str,
    work: Path,
    min_confidence: float = MIN_CONFIDENCE,
) -> Path:
    """Replace static Latin slide text; this does not support complex animation."""
    replacements = select_replacements(track, translate, min_confidence)
    overlays = render_overlays(video, track, replacements, work)
    return apply_overlays(video, overlays, output)
