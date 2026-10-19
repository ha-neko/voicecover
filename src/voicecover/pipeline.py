from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import media
from .argos import translate_segments as translate_segments_offline
from .clone import CloneConfig, DEFAULT_CLONE_MODEL, select_reference
from .download import download_youtube, is_youtube_url
from .detect import has_english_latin_text
from .languages import whisper_language
from .models import Segment, is_arabic_passage, is_garbled_text
from .nllb import source_tags as segment_source_tags
from .nllb import translate_segments as translate_segments_nllb
from .subtitles import write_srt
from .synthesis import choose_voice, synthesize
from .transcription import transcribe
from .translation import translate_segments
from .visual import build_prompt, build_visual_track, extract_frame_samples
from .visual_models import VisualTrack
from .visual_replace import replace_visual_text


MAX_TIMELINE_SPEED = 1.25
MAX_UTTERANCE_DURATION = 12.0
MAX_UTTERANCE_CHARACTERS = 320
MAX_UTTERANCE_GAP = 0.45
MAX_CLONE_PHRASE_DURATION = 11.0
MAX_CLONE_PHRASE_CHARACTERS = 280
TERMINAL_PUNCTUATION = (".", "!", "?", "؟", "。", "！", "？")
CHECKPOINT_VERSION = 1
SOURCE_CACHE_VERSION = 1


@dataclass(slots=True)
class Options:
    input: Path | str
    output: Path
    target_language: str
    source_language: str | None = None
    model: str = "small"
    device: str = "cpu"
    voice: str | None = None
    original_volume: float = 0.15
    write_subtitles: bool = True
    keep_temp: bool = False
    translator: str = "auto"
    install_models: bool = True
    initial_prompt: str | None = None
    preserve_arabic: bool = True
    natural_speech: bool = True
    write_segments: bool = False
    write_bilingual: bool = False
    visual_context: bool = True
    voice_clone: bool = False
    clone_model: str | None = None
    clone_nfe: int = 16
    protected_terms: dict[str, str] = field(default_factory=dict)
    write_visual_track: bool = False
    localize_visual_text: bool = False
    visual_only: bool = False
    sound_only: bool = False
    visual_interval: float = 3.0
    visual_width: int = 960


def _checkpoint_path(output: Path) -> Path:
    return output.with_suffix(".checkpoint.json")


def _source_cache_dir(output: Path) -> Path:
    return output.parent / f"{output.stem}.source-cache"


def _load_youtube_source_cache(output: Path, url: str) -> Path | None:
    """Return a validated durable YouTube download for this exact URL."""
    directory = _source_cache_dir(output)
    manifest = directory / "source.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("version") != SOURCE_CACHE_VERSION
            or payload.get("url") != url
            or not isinstance(payload.get("filename"), str)
            or Path(payload["filename"]).name != payload["filename"]
            or not isinstance(payload.get("size"), int)
        ):
            return None
        source = directory / payload["filename"]
        if not source.is_file() or source.stat().st_size != payload["size"] or payload["size"] <= 0:
            return None
        return source
    except (OSError, json.JSONDecodeError):
        return None


def _write_youtube_source_cache(downloaded: Path, output: Path, url: str) -> Path:
    """Atomically persist a downloaded source video beside its job output."""
    directory = _source_cache_dir(output)
    directory.mkdir(parents=True, exist_ok=True)
    suffix = downloaded.suffix.lower() if downloaded.suffix else ".mp4"
    destination = directory / f"source{suffix}"
    temporary = directory / f".{destination.name}.tmp"
    shutil.copy2(downloaded, temporary)
    os.replace(temporary, destination)
    payload = {
        "version": SOURCE_CACHE_VERSION,
        "url": url,
        "filename": destination.name,
        "size": destination.stat().st_size,
    }
    manifest = directory / "source.json"
    manifest_tmp = directory / ".source.json.tmp"
    manifest_tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, manifest)
    return destination


def _segment_payload(segment: Segment) -> dict[str, object]:
    return {
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "translated_text": segment.translated_text,
    }


def _decode_checkpoint_segments(value: object, *, translated: bool) -> list[Segment]:
    if not isinstance(value, list) or not value:
        raise ValueError("checkpoint segments must be a non-empty array")
    result: list[Segment] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("checkpoint segment must be an object")
        start, end = item.get("start"), item.get("end")
        text, translated_text = item.get("text"), item.get("translated_text")
        if (
            not isinstance(start, (int, float))
            or isinstance(start, bool)
            or not isinstance(end, (int, float))
            or isinstance(end, bool)
            or float(start) < 0
            or float(end) < float(start)
            or not isinstance(text, str)
            or not text.strip()
            or (translated and (not isinstance(translated_text, str) or not translated_text.strip()))
            or (translated_text is not None and not isinstance(translated_text, str))
        ):
            raise ValueError("checkpoint contains an invalid segment")
        result.append(Segment(float(start), float(end), text, translated_text))
    return result


def _load_translation_checkpoint(
    options: Options,
) -> tuple[list[Segment], list[Segment], str, str] | None:
    path = _checkpoint_path(options.output)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("checkpoint root must be an object")
        if payload.get("version") != CHECKPOINT_VERSION or payload.get("stage") != "translated":
            raise ValueError("unsupported checkpoint version or stage")
        target = payload.get("target_language")
        if target is not None and target != options.target_language:
            return None
        configured_source = payload.get("configured_source_language")
        if configured_source is not None and configured_source != options.source_language:
            return None
        segments = _decode_checkpoint_segments(payload.get("segments"), translated=True)
        asr_segments = _decode_checkpoint_segments(payload.get("asr_segments"), translated=False)
        detected = payload.get("detected_language")
        source = payload.get("source_language")
        if not isinstance(detected, str) or not detected or not isinstance(source, str) or not source:
            raise ValueError("checkpoint languages are invalid")
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return None
    return segments, asr_segments, detected, source


def _write_translation_checkpoint(
    options: Options,
    segments: list[Segment],
    asr_segments: list[Segment],
    detected_language: str,
    source_language: str,
) -> Path:
    path = _checkpoint_path(options.output)
    payload = {
        "version": CHECKPOINT_VERSION,
        "stage": "translated",
        "target_language": options.target_language,
        "configured_source_language": options.source_language,
        "detected_language": detected_language,
        "source_language": source_language,
        "segments": [_segment_payload(segment) for segment in segments],
        "asr_segments": [_segment_payload(segment) for segment in asr_segments],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def _restore_visual_checkpoint(
    video: Path,
    checkpoint: Path,
    work: Path,
    interval: float,
    width: int,
) -> VisualTrack | None:
    """Restore saved OCR data while recreating only its disposable frame images."""
    if not checkpoint.is_file():
        return None
    try:
        cached = VisualTrack.from_json(checkpoint.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if not cached.frames or abs(cached.sample_interval - interval) > 1e-6:
        return None
    frames = extract_frame_samples(video, work, interval, width)
    if len(frames) != len(cached.frames):
        return None
    restored_frames = tuple(
        replace(cached_frame, path=fresh.path)
        for cached_frame, fresh in zip(cached.frames, frames, strict=True)
    )
    return VisualTrack(
        frames=restored_frames,
        slides=cached.slides,
        sample_interval=cached.sample_interval,
    )


def _translate_with_preserved_arabic(
    segments: list[Segment],
    preserve_arabic: bool,
    translator,
    progress=print,
) -> list[Segment]:
    # every Arabic-dominant utterance (recitations, prayers, Quranic terms)
    # stays in its original script so it can be voiced by the Arabic voice;
    # Arabic spans inside otherwise-English sentences are protected later.
    arabic = [
        is_arabic_passage(segment.text) and not has_english_latin_text(segment.text)
        for segment in segments
    ]
    # an all-Arabic video is meant to be dubbed, not preserved
    if not preserve_arabic or all(arabic):
        return translator(segments)

    translatable = [segment for segment, keep in zip(segments, arabic, strict=True) if not keep]
    count = sum(arabic)
    if count:
        progress(f"preserving {count} Arabic passage(s) without translation...")

    translated = iter(translator(translatable)) if translatable else iter(())
    return [
        segment.translated(segment.text) if keep else next(translated)
        for segment, keep in zip(segments, arabic, strict=True)
    ]


def group_utterances(segments: list[Segment]) -> list[Segment]:
    """Merge adjacent ASR fragments into sentence-like units for context/prosody."""
    if not segments:
        return []
    groups: list[Segment] = []
    start = segments[0].start
    end = segments[0].end
    texts = [segments[0].text.strip()]
    arabic = is_arabic_passage(segments[0].text)

    def flush() -> None:
        groups.append(Segment(start, end, " ".join(texts).strip()))

    for segment in segments[1:]:
        text = segment.text.strip()
        combined = " ".join([*texts, text])
        can_join = (
            segment.start - end <= MAX_UTTERANCE_GAP
            and segment.end - start <= MAX_UTTERANCE_DURATION
            and len(combined) <= MAX_UTTERANCE_CHARACTERS
            and not texts[-1].endswith(TERMINAL_PUNCTUATION)
            and is_arabic_passage(text) == arabic
        )
        if can_join:
            texts.append(text)
            end = segment.end
            continue
        flush()
        start, end, texts = segment.start, segment.end, [text]
        arabic = is_arabic_passage(text)
    flush()
    return groups


def _natural_text_parts(text: str, count: int) -> list[str]:
    """Split text near punctuation into a requested number of spoken phrases."""
    words = text.split()
    if count <= 1 or len(words) < 4:
        return [text.strip()]
    count = min(count, max(1, len(words) // 3))
    cuts: list[int] = []
    previous = 0
    for part in range(1, count):
        target = round(len(words) * part / count)
        lower = max(previous + 2, target - 4)
        upper = min(len(words) - 2, target + 4)
        candidates = [
            index for index in range(lower, upper + 1)
            if re.search(r"[,.!?;:]$", words[index - 1])
        ]
        # An arbitrary word-count cut produces a standalone fragment with a
        # visibly severed cadence. If no clause boundary exists, leave the
        # utterance intact and give F5 enough generation headroom instead.
        if not candidates:
            return [text.strip()]
        cut = min(candidates, key=lambda index: abs(index - target))
        cut = max(previous + 1, min(cut, len(words) - 1))
        cuts.append(cut)
        previous = cut
    parts: list[str] = []
    start = 0
    for cut in [*cuts, len(words)]:
        part = " ".join(words[start:cut]).strip()
        if part:
            parts.append(part)
        start = cut
    return parts


def split_clone_phrases(segments: list[Segment]) -> list[Segment]:
    """Use short complete phrases for stable, efficient cloned-voice prosody."""
    result: list[Segment] = []
    for segment in segments:
        target = segment.translated_text if segment.translated_text is not None else segment.text
        if is_arabic_passage(target):
            result.append(segment)
            continue
        count = max(
            1,
            math.ceil(segment.duration / MAX_CLONE_PHRASE_DURATION),
            math.ceil(len(target) / MAX_CLONE_PHRASE_CHARACTERS),
        )
        target_parts = _natural_text_parts(target, count)
        if len(target_parts) == 1:
            result.append(segment)
            continue
        source_parts = _natural_text_parts(segment.text, len(target_parts))
        if len(source_parts) != len(target_parts):
            source_parts = [segment.text] * len(target_parts)
        weights = [max(1, len(part.split())) for part in target_parts]
        total_weight = sum(weights)
        cursor = segment.start
        for index, (source_part, target_part, weight) in enumerate(
            zip(source_parts, target_parts, weights, strict=True)
        ):
            end = (
                segment.end
                if index + 1 == len(target_parts)
                else cursor + segment.duration * weight / total_weight
            )
            result.append(Segment(cursor, end, source_part, target_part))
            cursor = end
    return result


def merge_legacy_clone_fragments(segments: list[Segment]) -> list[Segment]:
    """Repair old checkpoints split at arbitrary words with a synthetic comma."""
    merged: list[Segment] = []
    for segment in segments:
        if merged:
            previous = merged[-1]
            previous_target = previous.translated_text or previous.text
            contiguous = abs(previous.end - segment.start) <= 0.002
            synthetic_comma = (
                previous.text.rstrip().endswith(",")
                and previous_target.rstrip().endswith(",")
            )
            if contiguous and synthetic_comma:
                current_target = segment.translated_text or segment.text
                merged[-1] = Segment(
                    previous.start,
                    segment.end,
                    f"{previous.text.rstrip(', ')} {segment.text}".strip(),
                    f"{previous_target.rstrip(', ')} {current_target}".strip(),
                )
                continue
        merged.append(segment)
    return merged


def _build_timeline(
    segments: list[Segment], speech_files: list[Path], total_duration: float, work: Path,
    progress=print,
) -> Path:
    parts: list[Path] = []
    cursor = 0.0
    part_number = 0

    def silence(seconds: float) -> None:
        nonlocal part_number
        if seconds <= 0.001:
            return
        path = work / f"part-{part_number:05d}.wav"
        media.make_silence(path, seconds)
        parts.append(path)
        part_number += 1

    for index, (segment, speech) in enumerate(zip(segments, speech_files, strict=True)):
        nominal_start = min(total_duration, max(0.0, segment.start))
        start = max(cursor, nominal_start)

        # fill any gap before this segment with silence
        gap = start - cursor
        if gap > 0.001:
            silence(gap)

        source_duration = media.duration(speech)
        next_start = segments[index + 1].start if index + 1 < len(segments) else total_duration
        available = max(0.01, next_start - start)
        fitted = work / f"part-{part_number:05d}.wav"
        media.fit_audio(speech, fitted, available)
        parts.append(fitted)
        part_number += 1
        fitted_duration = media.duration(fitted)
        cursor = start + fitted_duration
        drift = start - nominal_start
        if drift > 0.5:
            progress(f"  timing: utterance {index + 1} delayed {drift:.2f}s (no words cut)")

    silence(total_duration - cursor)
    if not parts:
        silence(total_duration)
    output = work / "dub.wav"
    media.concatenate(parts, work / "concat.txt", output)
    return output


def _fd_count() -> int:
    """Return the number of open file descriptors for diagnostics."""
    import os
    try:
        return len(os.listdir(f"/proc/{os.getpid()}/fd"))
    except OSError:
        return -1


def _run_translator(
    provider: str,
    options: Options,
    source: str,
    selected: list[Segment],
    progress=print,
) -> list[Segment]:
    if provider == "nllb":
        return translate_segments_nllb(
            selected,
            options.source_language or "auto",
            options.target_language,
            auto_install=options.install_models,
            progress=progress,
        )
    if provider == "argos":
        return translate_segments_offline(
            selected,
            source,
            options.target_language,
            auto_install=options.install_models,
            progress=progress,
        )
    if provider == "llm":
        from .llm import translate_segments as translate_segments_llm

        return translate_segments_llm(
            selected,
            options.source_language or source or "auto",
            options.target_language,
            protected_terms=options.protected_terms,
        )
    if provider in {"google", "auto"}:
        try:
            return translate_segments(selected, options.target_language, source)
        except RuntimeError:
            if provider == "google":
                raise
            progress("  Google unavailable - falling back to offline NLLB...")
            return translate_segments_nllb(
                selected,
                options.source_language or "auto",
                options.target_language,
                auto_install=options.install_models,
                progress=progress,
            )
    raise ValueError(f"unknown translation provider: {provider}")


def _is_translatable_visual_text(text: str) -> bool:
    """Reject symbols-only and non-Latin OCR noise before asking the LLM."""
    value = text.strip()
    if len(value) < 2 or not any(character.isalpha() for character in value):
        return False
    supported = sum(
        1 for character in value if character.isascii() or "\u0600" <= character <= "\u06ff"
    )
    return supported / len(value) >= 0.3


def _visual_translator(options: Options, source: str, progress=print):
    """Build a cached translator that checkpoints after each small group."""
    cache_path = options.output.with_suffix(".visual-translations.json")
    cache: dict[str, str] = {}
    cache_version = 2  # v2 adds fit-aware concise visual translations
    source_language = options.source_language or source
    if cache_path.is_file():
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            translations = loaded.get("translations") if isinstance(loaded, dict) else None
            if (
                isinstance(loaded, dict)
                and loaded.get("version") == cache_version
                and loaded.get("source_language") == source_language
                and loaded.get("target_language") == options.target_language
                and isinstance(translations, dict)
                and all(isinstance(key, str) and isinstance(value, str) for key, value in translations.items())
            ):
                cache = translations
                progress(f"  loaded {len(cache)} cached visual translations")
        except (json.JSONDecodeError, OSError):
            pass

    def save() -> None:
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "version": cache_version,
                    "source_language": source_language,
                    "target_language": options.target_language,
                    "translations": cache,
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, cache_path)

    def translate_visual(texts: list[str]) -> list[str]:
        translatable = [text for text in texts if _is_translatable_visual_text(text)]
        passthrough = {text: text for text in texts if text not in translatable}
        uncached = [text for text in translatable if text not in cache]
        if uncached:
            from .llm import translate_texts

            total = (len(uncached) + 19) // 20
            for offset in range(0, len(uncached), 20):
                chunk = uncached[offset : offset + 20]
                progress(
                    f"  visual translation batch {offset // 20 + 1}/{total} "
                    f"({len(chunk)} strings)..."
                )
                translated = translate_texts(
                    chunk, source_language, options.target_language
                )
                fallback_indices = [
                    index
                    for index, (original, target) in enumerate(
                        zip(chunk, translated, strict=True)
                    )
                    if target.strip().casefold() == original.strip().casefold()
                ]
                if fallback_indices:
                    progress(
                        f"  retrying {len(fallback_indices)} visual string(s) "
                        "individually after a batch fallback..."
                    )
                    for index in fallback_indices:
                        retried = translate_texts(
                            [chunk[index]], source_language, options.target_language
                        )[0]
                        if retried.strip().casefold() != chunk[index].strip().casefold():
                            translated[index] = retried
                # Same-as-source values are the LLM client's failure fallback.
                # Do not make a transient failure permanent or redraw unchanged
                # English in a mismatched font.
                cache.update(
                    (original, target)
                    for original, target in zip(chunk, translated, strict=True)
                    if target.strip().casefold() != original.strip().casefold()
                )
                save()
                progress(f"  visual translation checkpoint: {len(cache)} entries")
        combined = {**cache, **passthrough}
        return [combined.get(text, text) for text in texts]

    return translate_visual


def run(options: Options, progress=print) -> Path:
    media.require_ffmpeg()
    if options.visual_only and options.sound_only:
        raise ValueError("visual-only and sound-only modes are mutually exclusive")
    if not 0 <= options.original_volume <= 1:
        raise ValueError("original volume must be between 0 and 1")
    if options.output.suffix.lower() not in {".mp4", ".mkv", ".mov"}:
        raise ValueError("output must use an .mp4, .mkv, or .mov container")
    input_value = str(options.input)
    youtube = is_youtube_url(input_value)
    local_input = None if youtube else Path(input_value)
    if local_input is not None and not local_input.is_file():
        raise FileNotFoundError(f"Input video does not exist: {local_input}")
    if local_input is not None and local_input.resolve() == options.output.resolve():
        raise ValueError("output path must be different from the input video")
    options.output.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix="voicecover-"))
    try:
        if youtube:
            input_video = _load_youtube_source_cache(options.output, input_value)
            if input_video is not None:
                progress(f"restored cached YouTube source: {input_video}")
            else:
                progress("downloading YouTube video...")
                downloaded = download_youtube(input_value, temp_dir)
                input_video = _write_youtube_source_cache(downloaded, options.output, input_value)
                progress(f"YouTube source checkpoint written to: {input_video}")
        else:
            assert local_input is not None
            input_video = local_input

        # ── visual-only fast path ──────────────────────────────────────
        if options.visual_only:
            progress("visual-only mode: translating on-screen text only (original audio preserved)")

            visual_track: VisualTrack | None = None
            visual_dump = options.output.with_suffix(".visual.json")
            try:
                visual_track = _restore_visual_checkpoint(
                    input_video,
                    visual_dump,
                    temp_dir / "visual-samples",
                    options.visual_interval,
                    options.visual_width,
                )
                if visual_track is None:
                    progress("visual context: extracting slide states...")
                    visual_track = build_visual_track(
                        input_video,
                        temp_dir / "visual-samples",
                        interval=options.visual_interval,
                        width=options.visual_width,
                        progress=progress,
                    )
                else:
                    progress(
                        f"visual context: restored {len(visual_track.frames)} saved OCR samples "
                        "(OCR skipped)"
                    )
                progress(
                    f"visual context: {len(visual_track.frames)} samples -> "
                    f"{len(visual_track.slides)} slide states"
                )
            except Exception as exc:
                raise RuntimeError(f"visual-only mode requires visual context: {exc}") from exc

            if visual_track is None or not visual_track.slides:
                raise RuntimeError("no visual slide states found — nothing to translate")

            portable_track = VisualTrack(
                frames=tuple(replace(frame, path=Path(frame.path).name) for frame in visual_track.frames),
                slides=visual_track.slides,
                sample_interval=visual_track.sample_interval,
            )
            visual_dump.write_text(portable_track.to_json(indent=2) + "\n", encoding="utf-8")
            progress(f"visual track written to: {visual_dump}")

            progress("localizing static on-screen text...")
            translate_visual = _visual_translator(options, options.source_language or "auto", progress)

            localized = temp_dir / "visual-localized.mp4"
            progress("rendering localized visual overlays...")
            render_input = replace_visual_text(
                input_video,
                visual_track,
                translate_visual,
                localized,
                temp_dir / "visual-replacement",
            )

            # the localized video already has original audio — just copy to output
            shutil.copy2(render_input, options.output)
            progress(f"done: {options.output}")
            return options.output
        # ── end visual-only ────────────────────────────────────────────

        if options.sound_only:
            progress("sound-only mode: translating speech only (original video preserved)")

        audio = temp_dir / "source.wav"
        progress("extracting audio...")
        media.extract_audio(input_video, audio)

        initial_prompt = options.initial_prompt
        visual_track: VisualTrack | None = None
        visual_dump = options.output.with_suffix(".visual.json")
        if not options.sound_only and (
            options.visual_context or options.write_visual_track or options.localize_visual_text
        ):
            try:
                visual_track = _restore_visual_checkpoint(
                    input_video,
                    visual_dump,
                    temp_dir / "visual-samples",
                    options.visual_interval,
                    options.visual_width,
                )
                if visual_track is None:
                    progress("visual context: extracting slide states...")
                    visual_track = build_visual_track(
                        input_video,
                        temp_dir / "visual-samples",
                        interval=options.visual_interval,
                        width=options.visual_width,
                        progress=progress,
                    )
                else:
                    progress(
                        f"visual context: restored {len(visual_track.frames)} saved OCR samples "
                        "(OCR skipped)"
                    )
                initial_prompt = build_prompt(visual_track, options.initial_prompt)
                progress(
                    f"visual context: {len(visual_track.frames)} samples -> "
                    f"{len(visual_track.slides)} slide states"
                )
            except Exception as exc:
                progress(f"  visual context unavailable: {exc}")

        if options.write_visual_track and visual_track is not None:
            portable_frames = tuple(
                replace(frame, path=Path(frame.path).name) for frame in visual_track.frames
            )
            portable_track = VisualTrack(
                frames=portable_frames,
                slides=visual_track.slides,
                sample_interval=visual_track.sample_interval,
            )
            visual_dump.write_text(portable_track.to_json(indent=2) + "\n", encoding="utf-8")
            progress(f"visual track written to: {visual_dump}")

        checkpoint = _load_translation_checkpoint(options)
        if checkpoint is not None:
            segments, asr_segments, detected_language, source = checkpoint
            checkpoint_count = len(segments)
            segments = merge_legacy_clone_fragments(segments)
            if len(segments) != checkpoint_count:
                progress(
                    f"repairing legacy clone phrases: {checkpoint_count} -> "
                    f"{len(segments)} natural utterances"
                )
            progress(
                f"resuming from translated checkpoint: {len(segments)} utterances "
                "(skipping transcription and translation)"
            )
        else:
            progress("transcribing speech...")
            fd_before = _fd_count()
            try:
                segments, detected_language = transcribe(
                    audio,
                    options.model,
                    whisper_language(options.source_language),
                    options.device,
                    initial_prompt,
                    multilingual=options.preserve_arabic,
                )
            except Exception:
                fd_after = _fd_count()
                progress(f"  [fd diagnostics: {fd_before} -> {fd_after} open fds]")
                raise
            if not segments:
                raise RuntimeError("No speech was detected in the video")
            progress(f"detected source language: {detected_language}")
            asr_segments = list(segments)

            if options.natural_speech:
                source_count = len(segments)
                segments = group_utterances(segments)
                progress(
                    f"grouping {source_count} transcript fragments into {len(segments)} natural utterances..."
                )

            source = options.source_language or detected_language
            progress(f"translating {source} -> {options.target_language}...")
            fd_before = _fd_count()

            def translate_selected(selected: list[Segment]) -> list[Segment]:
                return _run_translator(
                    options.translator, options, source, selected, progress
                )

            segments = _translate_with_preserved_arabic(
                segments,
                options.preserve_arabic,
                translate_selected,
                progress,
            )
            if options.voice_clone:
                phrase_count = len(segments)
                segments = split_clone_phrases(segments)
                if len(segments) != phrase_count:
                    progress(
                        f"splitting {phrase_count} translated utterances into "
                        f"{len(segments)} short clone phrases..."
                    )
            checkpoint_path = _write_translation_checkpoint(
                options, segments, asr_segments, detected_language, source
            )
            progress(f"translated checkpoint written to: {checkpoint_path}")

        clone_config = None
        if options.voice_clone:
            ref_wav, ref_text = select_reference(
                audio,
                asr_segments,
                temp_dir,
                progress,
                options.output.with_suffix(".clone-reference"),
            )
            clone_config = CloneConfig(
                ref_wav=ref_wav,
                ref_text=ref_text,
                model=options.clone_model or DEFAULT_CLONE_MODEL,
                nfe_step=options.clone_nfe,
                # F5 estimates output length from reference/text byte ratios.
                # Indonesian loanwords and final consonants need substantial
                # headroom or the generated tensor can end mid-phoneme.
                speed=0.88,
                device=options.device,
            )
            progress(
                f"generating {len(segments)} utterance(s) with the cloned original voice "
                f"({clone_config.model})..."
            )
            selected_voice = ""
        else:
            selected_voice = choose_voice(options.target_language, options.voice)
            progress(f"generating voice with {selected_voice}...")
        preserve_audio = None
        if any(
            is_arabic_passage(segment.translated_text or segment.text)
            or is_garbled_text(segment.translated_text or segment.text)
            for segment in segments
        ):
            preserve_audio = temp_dir / "source-timeline.wav"
            progress("extracting full-quality source passages...")
            media.extract_timeline_audio(input_video, preserve_audio)

        speech_cache = options.output.parent / f"{options.output.stem}.speech-cache"
        speech_cache.mkdir(parents=True, exist_ok=True)
        speech_files = synthesize(
            segments,
            temp_dir,
            selected_voice,
            progress,
            clone_config,
            preserve_audio,
            speech_cache_dir=speech_cache,
        )

        if options.write_segments:
            tags = segment_source_tags(segments, options.source_language or "auto")
            metadata = [
                {
                    "index": index,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "translated_text": segment.translated_text,
                    "detected_source": tag,
                }
                for index, (segment, tag) in enumerate(zip(segments, tags, strict=True))
            ]
            dump = options.output.with_suffix(".segments.json")
            dump.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            progress(f"segment metadata written to: {dump}")

        progress("synchronizing speech...")
        video_duration = media.duration(input_video)
        dub = _build_timeline(
            segments, speech_files, video_duration, temp_dir, progress
        )
        dub_duration = media.duration(dub)
        if dub_duration > video_duration + 0.05:
            # Delaying crowded utterances preserves every clip, but accumulated
            # drift can leave the final words beyond the last video frame. A
            # tiny whole-track tempo correction keeps them inside the playable
            # timeline without trimming. Typical corrections are below 2%.
            ratio = dub_duration / video_duration
            progress(
                f"  fitting final speech timeline ({dub_duration:.2f}s -> "
                f"{video_duration:.2f}s, {ratio:.3f}x; no words cut)..."
            )
            fitted_dub = temp_dir / "dub-fitted.wav"
            media.fit_audio(dub, fitted_dub, video_duration)
            dub = fitted_dub

        render_input = input_video
        if not options.sound_only and options.localize_visual_text and visual_track is not None:
            progress("localizing static on-screen text...")
            translate_visual = _visual_translator(options, options.source_language or source, progress)

            localized = temp_dir / "visual-localized.mp4"
            progress("rendering localized visual overlays...")
            render_input = replace_visual_text(
                input_video,
                visual_track,
                translate_visual,
                localized,
                temp_dir / "visual-replacement",
            )

        progress("creating dubbed video...")
        media.mux(render_input, dub, options.output, options.original_volume)
        if options.write_subtitles:
            write_srt(segments, options.output.with_suffix(".srt"), options.write_bilingual)
        progress(f"done: {options.output}")
        return options.output
    finally:
        if options.keep_temp:
            progress(f"temporary files kept at: {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
