from __future__ import annotations

import re
import subprocess
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path

from . import media
from .visual_models import FrameSample, OCRBox, Point, SlideState, VisualTrack

FRAME_INTERVAL = 3.0
FRAME_WIDTH = 480
_MAX_PROMPT_TERMS = 80
_AHASH_BITS = 256
_AHASH_THRESHOLD = 16
_SLIDE_HASH_WIDTH_RATIO = 0.72

_STOPWORDS = {
    "and", "the", "this", "that", "with", "from", "you", "your", "have", "has",
    "are", "was", "will", "not", "but", "for", "what", "when", "where", "which",
    "into", "than", "then", "there", "their", "each", "about", "after", "before",
    "page", "size", "shape", "form", "here", "actually", "really", "just",
}

_WORD = re.compile(r"[A-Za-z]{4,}|[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+")
_URL = re.compile(r"(?:https?://|www\.)[^\s,]+", re.IGNORECASE)
_ARABIC = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_LATIN = re.compile(r"[A-Za-z]")


def extract_frames(
    video: Path | str, work: Path, interval: float = FRAME_INTERVAL, width: int = FRAME_WIDTH
) -> list[Path]:
    """Sample one downscaled frame every `interval` seconds."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    pattern = work / "frame-%06d.jpg"
    media.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"fps=1/{interval},scale={width}:-1",
            "-q:v",
            "3",
            str(pattern),
        ]
    )
    return sorted(work.glob("frame-*.jpg"))


def _ahash(frame: Path) -> int:
    """16x16 grayscale average hash; two frames are near-duplicates when close."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(frame),
            "-vf",
            f"crop=iw*{_SLIDE_HASH_WIDTH_RATIO}:ih:0:0,scale=16:16,format=gray",
            "-f",
            "rawvideo",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    pixels = result.stdout
    if len(pixels) < _AHASH_BITS:
        return 0
    pixels = pixels[:_AHASH_BITS]
    mean = sum(pixels) / _AHASH_BITS
    value = 0
    for index, pixel in enumerate(pixels):
        if pixel >= mean:
            value |= 1 << index
    return value


def _hamming(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def _dedupe(frames: list[Path], threshold: int = _AHASH_THRESHOLD) -> list[Path]:
    """Collapse adjacent near-identical frames, preserving later reappearances."""
    kept: list[Path] = []
    previous_hash: int | None = None
    for frame in frames:
        try:
            frame_hash = _ahash(frame)
        except (subprocess.CalledProcessError, OSError):
            continue
        if previous_hash is None or _hamming(frame_hash, previous_hash) > threshold:
            kept.append(frame)
        previous_hash = frame_hash
    return kept


def _image_dimensions(frame: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(frame),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    width, height = result.stdout.strip().split("x", 1)
    return int(width), int(height)


def extract_frame_samples(
    video: Path | str, work: Path, interval: float = FRAME_INTERVAL, width: int = FRAME_WIDTH
) -> list[FrameSample]:
    """Extract timestamped frame samples while retaining the legacy extractor."""
    samples: list[FrameSample] = []
    for index, frame in enumerate(extract_frames(video, work, interval, width)):
        try:
            frame_width, frame_height = _image_dimensions(frame)
            frame_hash = _ahash(frame)
        except (subprocess.CalledProcessError, OSError, ValueError):
            frame_width, frame_height, frame_hash = 0, 0, None
        samples.append(
            FrameSample(
                path=str(frame),
                timestamp=index * interval,
                width=frame_width,
                height=frame_height,
                average_hash=frame_hash,
            )
        )
    return samples


def cluster_slide_states(
    frames: list[FrameSample] | tuple[FrameSample, ...],
    threshold: int = _AHASH_THRESHOLD,
    sample_interval: float = FRAME_INTERVAL,
) -> list[SlideState]:
    """Cluster only neighboring samples so the same slide can reappear later."""
    if not frames:
        return []

    def same_slide(previous: FrameSample, current: FrameSample) -> bool:
        previous_text = " ".join(
            re.sub(r"\W+", " ", box.text.casefold()).strip() for box in previous.boxes
        ).strip()
        current_text = " ".join(
            re.sub(r"\W+", " ", box.text.casefold()).strip() for box in current.boxes
        ).strip()
        if previous_text or current_text:
            if not previous_text or not current_text:
                return False
            return SequenceMatcher(None, previous_text, current_text).ratio() >= 0.72
        if previous.average_hash is None or current.average_hash is None:
            return False
        return _hamming(previous.average_hash, current.average_hash) <= threshold

    groups: list[list[int]] = [[0]]
    for index in range(1, len(frames)):
        if not same_slide(frames[index - 1], frames[index]):
            groups.append([index])
        else:
            groups[-1].append(index)
    slides: list[SlideState] = []
    for index, group in enumerate(groups):
        representative = max(
            group,
            key=lambda frame_index: (
                len(frames[frame_index].boxes),
                sum(box.confidence for box in frames[frame_index].boxes),
            ),
        )
        slides.append(
            SlideState(
                index=index,
                start=frames[group[0]].timestamp,
                end=frames[group[-1]].timestamp + sample_interval,
                frame_indices=tuple(group),
                representative_frame_index=representative,
            )
        )
    return slides


def _text_metadata(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    scripts: list[str] = []
    reasons: list[str] = []
    if _ARABIC.search(text):
        scripts.append("arabic")
        reasons.append("arabic-script")
    if _LATIN.search(text):
        scripts.append("latin")
    if any(character.isdigit() for character in text):
        scripts.append("numeric")
    if _URL.search(text):
        reasons.append("url")
    words = re.findall(r"[A-Za-z][A-Za-z0-9]*", text)
    likely_brand = any(
        (word.isupper() and 2 <= len(word) <= 12)
        or (any(char.isupper() for char in word[1:]) and any(char.islower() for char in word))
        or (any(char.isalpha() for char in word) and any(char.isdigit() for char in word))
        for word in words
    ) or "™" in text or "®" in text
    if likely_brand:
        reasons.append("likely-branding")
    return tuple(scripts), tuple(reasons)


def _parse_ocr_box(item: object, width: int, height: int) -> OCRBox | None:
    try:
        raw_points, raw_text, raw_confidence = item  # type: ignore[misc]
        points = tuple(Point(float(point[0]), float(point[1])) for point in raw_points)
        text = str(raw_text).strip()
        confidence = float(raw_confidence)
    except (TypeError, ValueError, IndexError):
        return None
    if not text or len(points) != 4:
        return None
    scripts, reasons = _text_metadata(text)
    return OCRBox(
        points=points,
        text=text,
        confidence=confidence,
        source_width=width,
        source_height=height,
        scripts=scripts,
        protected=bool(reasons),
        protection_reasons=reasons,
    )


def _box_overlap(first: OCRBox, second: OCRBox) -> float:
    def bounds(box: OCRBox) -> tuple[float, float, float, float]:
        xs = [point.x for point in box.points]
        ys = [point.y for point in box.points]
        return min(xs), min(ys), max(xs), max(ys)

    left_a, top_a, right_a, bottom_a = bounds(first)
    left_b, top_b, right_b, bottom_b = bounds(second)
    width = max(0.0, min(right_a, right_b) - max(left_a, left_b))
    height = max(0.0, min(bottom_a, bottom_b) - max(top_a, top_b))
    intersection = width * height
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, right_a - left_a) * max(0.0, bottom_a - top_a)
    area_b = max(0.0, right_b - left_b) * max(0.0, bottom_b - top_b)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _rapid_items(raw_result: object) -> list[object]:
    if all(hasattr(raw_result, name) for name in ("boxes", "txts", "scores")):
        boxes = raw_result.boxes
        if boxes is None:
            return []
        return list(zip(boxes, raw_result.txts, raw_result.scores))
    result, _ = raw_result  # type: ignore[misc]
    return list(result or [])


def ocr_frame_samples(
    frames: list[FrameSample] | tuple[FrameSample, ...],
    progress: Callable[[str], None] = print,
) -> list[FrameSample]:
    """Attach OCR boxes to samples; unavailable or failed OCR is nonfatal."""
    try:
        try:
            from rapidocr import RapidOCR
        except ImportError:
            from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        progress("  visual context disabled: RapidOCR is not installed")
        return list(frames)
    try:
        engine = RapidOCR()
    except Exception:
        return list(frames)
    arabic_engine = None
    try:
        from rapidocr import LangRec, ModelType, OCRVersion

        arabic_engine = RapidOCR(
            params={
                "Rec.lang_type": LangRec.ARABIC,
                "Rec.model_type": ModelType.MOBILE,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
            }
        )
    except (ImportError, Exception):
        # Legacy RapidOCR and offline installs can still provide Latin OCR.
        arabic_engine = None
    output: list[FrameSample] = []
    for index, frame in enumerate(frames, start=1):
        progress(f"  reading on-screen text {index}/{len(frames)}...")
        try:
            result = _rapid_items(engine(frame.path))
        except Exception:
            output.append(frame)
            continue
        parsed = [
            box
            for item in result or []
            if (box := _parse_ocr_box(item, frame.width, frame.height)) is not None
        ]
        if arabic_engine is not None:
            try:
                arabic = [
                    box
                    for item in _rapid_items(arabic_engine(frame.path))
                    if (box := _parse_ocr_box(item, frame.width, frame.height)) is not None
                    and "arabic" in box.scripts
                    and box.confidence >= 0.50
                ]
            except Exception:
                arabic = []
            used: set[int] = set()
            merged: list[OCRBox] = []
            for box in parsed:
                matches = [
                    (position, candidate)
                    for position, candidate in enumerate(arabic)
                    if _box_overlap(box, candidate) >= 0.45
                ]
                if matches:
                    position, candidate = max(
                        matches, key=lambda item: _box_overlap(box, item[1])
                    )
                    used.add(position)
                    merged.append(candidate)
                else:
                    merged.append(box)
            merged.extend(box for position, box in enumerate(arabic) if position not in used)
            parsed = merged
        output.append(replace(frame, boxes=tuple(parsed)))
    return output


def ocr_frames(frames: list[Path], progress: Callable[[str], None] = print) -> list[str]:
    """Read text from paths, preserving the original text-list API."""
    samples: list[FrameSample] = []
    for index, frame in enumerate(frames):
        try:
            width, height = _image_dimensions(frame)
        except (subprocess.CalledProcessError, OSError, ValueError):
            width, height = 0, 0
        samples.append(FrameSample(str(frame), index * FRAME_INTERVAL, width, height))
    enriched = ocr_frame_samples(samples, progress)
    return [box.text for frame in enriched for box in frame.boxes]


def build_visual_track(
    video: Path | str,
    work: Path,
    interval: float = FRAME_INTERVAL,
    width: int = FRAME_WIDTH,
    progress: Callable[[str], None] = print,
) -> VisualTrack:
    """Build a timestamped, adjacent-clustered slide and OCR track."""
    frames = ocr_frame_samples(
        extract_frame_samples(video, work, interval, width), progress
    )
    slides = cluster_slide_states(frames, sample_interval=interval)
    return VisualTrack(tuple(frames), tuple(slides), interval)


def build_prompt(
    texts: list[str] | tuple[str, ...] | VisualTrack,
    existing_prompt: str | None = None,
) -> str:
    """Turn OCR terms into a Whisper vocabulary hint, keeping the user prompt."""
    protected: list[str] = []
    if isinstance(texts, VisualTrack):
        raw_texts = [box.text for frame in texts.frames for box in frame.boxes]
        for frame in texts.frames:
            for box in frame.boxes:
                if "url" in box.protection_reasons:
                    protected.extend(_URL.findall(box.text))
                elif "likely-branding" in box.protection_reasons:
                    protected.append(box.text)
    else:
        raw_texts = list(texts)
    tokens = [token for token in _WORD.findall(" ".join(raw_texts))]
    frequencies = Counter(
        token for token in tokens if token.lower() not in _STOPWORDS
    )
    top_terms = list(dict.fromkeys(protected))
    top_terms.extend(
        token
        for token, _ in frequencies.most_common(_MAX_PROMPT_TERMS)
        if token not in top_terms
    )
    top_terms = top_terms[:_MAX_PROMPT_TERMS]
    vocabulary = ", ".join(top_terms)
    base = (existing_prompt or "").strip()
    if base and vocabulary:
        return f"{base}, {vocabulary}"
    return base or vocabulary


def visual_context(
    video: Path | str,
    work: Path,
    existing_prompt: str | None = None,
    progress: Callable[[str], None] = print,
) -> str:
    """Sample the video, OCR the on-screen text, return a Whisper prompt."""
    progress("visual context: extracting frames...")
    track = build_visual_track(video, work, progress=progress)
    if not track.frames:
        return (existing_prompt or "").strip()
    progress(
        f"visual context: {len(track.frames)} frames -> {len(track.slides)} slide states"
    )
    if not any(frame.boxes for frame in track.frames):
        return (existing_prompt or "").strip()
    prompt = build_prompt(track, existing_prompt)
    progress(f"visual context: {len(top_terms(prompt))} on-screen terms")
    return prompt


def top_terms(prompt: str) -> list[str]:
    return [term.strip() for term in prompt.split(",") if term.strip()]
