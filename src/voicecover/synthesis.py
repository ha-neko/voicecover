from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import shutil
import threading
import time
import unicodedata
from pathlib import Path

from .clone import CloneConfig, CloneEngine
from .detect import has_english_latin_text
from .media import concatenate, decode_speech, duration, extract_audio_range, make_silence
from .languages import tts_language
from .models import Segment, is_arabic_passage, is_garbled_text


_LOG = logging.getLogger("voicecover.synthesis")

# Scripts a given voice locale can actually pronounce. Latin is always allowed;
# Arabic is allowed everywhere because Indonesian lessons deliberately mix
# Arabic vocabulary into their narration.
_VOICE_SCRIPTS = {
    "ar": ("ARABIC",),
    "he": ("HEBREW",),
    "ru": ("CYRILLIC",),
    "uk": ("CYRILLIC",),
    "bg": ("CYRILLIC",),
    "sr": ("CYRILLIC",),
    "el": ("GREEK",),
    "ja": ("CJK", "HIRAGANA", "KATAKANA"),
    "ko": ("HANGUL",),
    "zh": ("CJK",),
    "th": ("THAI",),
    "hi": ("DEVANAGARI",),
    "mr": ("DEVANAGARI",),
}


def unpronounceable_characters(text: str, voice: str) -> list[str]:
    """Return alphabetic characters the target voice cannot pronounce.

    Whisper hallucinations mix unrelated scripts (Cyrillic, CJK, Greek...) into
    otherwise Latin or Arabic lesson text. TTS rejects such input, and speaking
    it would be meaningless anyway, so callers treat these segments as noise.
    """
    base = voice.split("-")[0].lower()
    allowed = {"LATIN", "ARABIC"} | set(_VOICE_SCRIPTS.get(base, ()))
    foreign: list[str] = []
    for character in text:
        if not character.isalpha():
            continue
        try:
            name = unicodedata.name(character)
        except ValueError:
            continue
        if not any(name.startswith(script) for script in allowed):
            foreign.append(character)
    return foreign


def _cache_key_path(cache_dir: Path, index: int) -> Path:
    return cache_dir / f"speech-{index:05d}.sha256"


def _copy_to_cache(
    source: Path, cache_dir: Path | None, index: int, cache_key: str
) -> None:
    """Copy a completed speech WAV to the persistent cache directory."""
    if cache_dir is None:
        return
    dest = cache_dir / f"speech-{index:05d}.wav"
    try:
        shutil.copy2(source, dest)
        key_path = _cache_key_path(cache_dir, index)
        temporary = key_path.with_suffix(".sha256.tmp")
        temporary.write_text(f"{cache_key}\n", encoding="ascii")
        temporary.replace(key_path)
    except OSError:
        pass


def _load_cached_speech(
    cache_dir: Path | None, cache_keys: list[str], progress=print
) -> dict[int, Path]:
    """Return a mapping of utterance index to cached WAV path."""
    cached: dict[int, Path] = {}
    if cache_dir is None or not cache_dir.is_dir():
        return cached
    for index, expected_key in enumerate(cache_keys):
        wav = cache_dir / f"speech-{index:05d}.wav"
        try:
            valid_key = _cache_key_path(cache_dir, index).read_text(encoding="ascii").strip()
            if valid_key == expected_key and wav.is_file() and wav.stat().st_size > 0:
                cached[index] = wav
        except (OSError, UnicodeError):
            continue
    if cached:
        progress(f"  loaded {len(cached)}/{len(cache_keys)} verified cached speech files")
    return cached


def _speech_cache_keys(segments: list[Segment], clone: CloneConfig) -> list[str]:
    """Bind cached audio to the exact text, reference, and F5 settings."""
    reference_hash = hashlib.sha256(clone.ref_wav.read_bytes()).hexdigest()
    keys: list[str] = []
    for segment in segments:
        text = segment.translated_text if segment.translated_text is not None else segment.text
        arabic = is_arabic_passage(text)
        payload = {
            "cache_version": 3,
            "kind": "original-arabic" if arabic else "f5-clone",
            "text": text if arabic else normalize_clone_text(text),
            "start": segment.start if arabic else None,
            "end": segment.end if arabic else None,
            "model": clone.model,
            "nfe_step": clone.nfe_step,
            "speed": clone.speed,
            "ref_text": clone.ref_text,
            "ref_audio_sha256": reference_hash,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        keys.append(hashlib.sha256(encoded).hexdigest())
    return keys


MAX_TTS_RATE_PERCENT = 20

_ARABIC_CLONE_PRONUNCIATIONS = {
    # Frequent grammar terms plus common Whisper/OCR substitutions found in
    # mixed English-Arabic lessons. Full Arabic passages use source audio.
    "فعل مفعول فعل": "fi'il maf'ul fi'il",
    "ماضي مضاري أمر": "madhi, mudhari, amr",
    "قلوب نفس": "qulub, nafs",
    "ما لا": "ma, la",
    "إسماء": "asma",
    "أسماء": "asma",
    "أفعل": "af'al",
    "أفعال": "af'al",
    "مفعول": "maf'ul",
    "مضارع": "mudhari",
    "مضاري": "mudhari",
    "ماضي": "madhi",
    "أمر": "amr",
    "حروف": "huruf",
    "حرف": "harf",
    "إسن": "isim",
    "اسم": "isim",
    "فعل": "fi'il",
    "حر": "harf",
    "علم": "allim",
    "قلوب": "qulub",
    "قلب": "qalb",
    "نفس": "nafs",
    "فوس": "nufus",
    "فتحة": "fataha",
    "نصرة": "nasara",
    "ضربة": "daraba",
    "سمعة": "sami'a",
}

_INDONESIAN_UNITS = (
    "nol", "satu", "dua", "tiga", "empat", "lima",
    "enam", "tujuh", "delapan", "sembilan", "sepuluh", "sebelas",
)


def _number_in_indonesian(value: int) -> str:
    if value < 0:
        return "minus " + _number_in_indonesian(-value)
    if value < 12:
        return _INDONESIAN_UNITS[value]
    if value < 20:
        return _number_in_indonesian(value - 10) + " belas"
    if value < 100:
        tens, rest = divmod(value, 10)
        return _number_in_indonesian(tens) + " puluh" + (
            " " + _number_in_indonesian(rest) if rest else ""
        )
    if value < 200:
        return "seratus" + (" " + _number_in_indonesian(value - 100) if value > 100 else "")
    if value < 1000:
        hundreds, rest = divmod(value, 100)
        return _number_in_indonesian(hundreds) + " ratus" + (
            " " + _number_in_indonesian(rest) if rest else ""
        )
    if value < 2000:
        return "seribu" + (" " + _number_in_indonesian(value - 1000) if value > 1000 else "")
    if value < 1_000_000:
        thousands, rest = divmod(value, 1000)
        return _number_in_indonesian(thousands) + " ribu" + (
            " " + _number_in_indonesian(rest) if rest else ""
        )
    return str(value)


def normalize_clone_text(text: str) -> str:
    """Make Indonesian F5-TTS pronounce digits and inline Arabic predictably."""
    # The Indonesian checkpoint often treats the apostrophe in ``Qur'an`` as
    # a stop and drops the final syllable (audibly producing "Al Qur...").
    # This spelling is only sent to TTS; subtitles retain the proper spelling.
    normalized = re.sub(r"(?i)\bal[ -]?qur(?:'|’)?an\b", "Al Kuran", text)
    for arabic, pronunciation in sorted(
        _ARABIC_CLONE_PRONUNCIATIONS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        normalized = normalized.replace(arabic, pronunciation)

    def replace_number(match: re.Match[str]) -> str:
        words = _number_in_indonesian(int(match.group(1)))
        return words + (" persen" if match.group(2) else "")

    normalized = re.sub(r"(?<!\w)(\d{1,6})(\s*%)?(?!\w)", replace_number, normalized)
    return " ".join(normalized.split())


def adaptive_rate(source_duration: float, target_duration: float) -> str | None:
    """Return an Edge TTS rate that moves speech toward its timestamp slot."""
    if source_duration <= 0 or target_duration <= 0:
        return None
    factor = source_duration / target_duration
    if factor <= 1.08:
        return None
    percent = round((factor - 1) * 100)
    # Edge voices become difficult to understand when pushed much beyond this.
    # Longer speech is allowed to overflow its source timestamp in the timeline
    # instead of sacrificing clarity here.
    percent = min(MAX_TTS_RATE_PERCENT, percent)
    return f"{percent:+d}%"


async def _save_with_retry(communicate, output: Path, attempts: int = 8) -> None:
    for attempt in range(attempts):
        try:
            await communicate().save(str(output))
            return
        except Exception:
            output.unlink(missing_ok=True)
            if attempt + 1 == attempts:
                raise
            # Edge TTS outages can last longer than a few seconds; back off up
            # to ~30s per attempt so transient drops do not kill a long job.
            await asyncio.sleep(min(30.0, 2**attempt) + random.uniform(0.0, 0.5))


def matching_voices(voices: list[dict[str, object]], target_language: str) -> list[dict[str, object]]:
    """Return locale-matching voices with male voices listed first."""
    wanted = tts_language(target_language)
    base = wanted.split("-")[0]
    exact = [voice for voice in voices if str(voice["Locale"]).lower() == wanted]
    candidates = exact or [
        voice
        for voice in voices
        if str(voice["Locale"]).lower().split("-")[0] == base
    ]
    return sorted(candidates, key=lambda voice: str(voice.get("Gender")) != "Male")


async def available_voices(target_language: str) -> list[dict[str, object]]:
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("edge-tts is not installed") from exc

    try:
        voices = await edge_tts.list_voices()
    except Exception as exc:
        raise RuntimeError(f"Could not retrieve Edge TTS voices: {exc}") from exc
    candidates = matching_voices(voices, target_language)
    if not candidates:
        raise RuntimeError(f"No Edge TTS voice found for language {target_language!r}")
    return candidates


async def select_voice(target_language: str) -> str:
    candidates = await available_voices(target_language)
    preferred = next((v for v in candidates if v.get("Gender") == "Male"), candidates[0])
    return str(preferred["ShortName"])


def chunk_text(text: str, max_bytes: int = 2800) -> list[str]:
    """Split text losslessly at natural boundaries, measured as UTF-8 bytes."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    chunks: list[str] = []
    remaining = normalized
    while len(remaining.encode("utf-8")) > max_bytes:
        byte_count = 0
        character_limit = 0
        for character_limit, character in enumerate(remaining, start=1):
            byte_count += len(character.encode("utf-8"))
            if byte_count > max_bytes:
                character_limit -= 1
                break
        prefix = remaining[:character_limit]
        boundaries = [match.end() for match in re.finditer(r"[.!?؟;؛,:،]\s+", prefix)]
        split_at = boundaries[-1] if boundaries else prefix.rfind(" ")
        if split_at <= 0:
            split_at = character_limit
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    if " ".join(chunks) != normalized:
        raise RuntimeError("TTS chunking changed the source text")
    return chunks


async def _synthesize_one(
    index: int,
    segment: Segment,
    output_dir: Path,
    voice: str,
    arabic_voice: str | None,
    semaphore: asyncio.Semaphore,
    progress,
    completed: list[int],
    progress_lock: asyncio.Lock,
    source_audio: Path | None = None,
) -> Path:
    """Synthesize a single segment with TTS, trimming and rate adjustment."""
    output = output_dir / f"speech-{index:05d}.wav"
    text = segment.translated_text if segment.translated_text is not None else segment.text
    if not text.strip():
        raise RuntimeError(f"Speech segment {index + 1} is empty")

    # Whisper occasionally emits non-speech placeholders such as "???" for
    # pauses or unclear audio. There is nothing to pronounce, so hold silence
    # for the original duration instead of crashing the job.
    if not any(character.isalpha() for character in text):
        silence = output_dir / f"speech-{index:05d}.wav"
        await asyncio.to_thread(make_silence, silence, max(0.2, segment.duration))
        async with progress_lock:
            completed[0] += 1
            progress(f"  silence {completed[0]}/{completed[1]} utterances")
        return silence

    # Hallucinated tokens (scripts mixed inside one word, foreign scripts)
    # cannot be pronounced coherently. Keep the real recording, which is what
    # actually happened at this timestamp, instead of a garbled rendering.
    if is_garbled_text(text):
        if source_audio is not None:
            _LOG.warning(
                "utterance %d keeps original audio; garbled text %r",
                index + 1,
                text[:160],
            )
            await asyncio.to_thread(
                extract_audio_range, source_audio, output, segment.start, segment.end
            )
            async with progress_lock:
                completed[0] += 1
                progress(f"  source audio {completed[0]}/{completed[1]} utterances")
            return output
        _LOG.warning(
            "utterance %d held as silence; garbled text with no source audio %r",
            index + 1,
            text[:160],
        )
        silence = output_dir / f"speech-{index:05d}.wav"
        await asyncio.to_thread(make_silence, silence, max(0.2, segment.duration))
        async with progress_lock:
            completed[0] += 1
            progress(f"  silence {completed[0]}/{completed[1]} utterances")
        return silence

    # Whisper hallucinations mix unrelated scripts into lesson text (for
    # example Arabic + Cyrillic + garbled Latin in a 2s noise gap). TTS rejects
    # those, and speaking them would be meaningless, so hold silence instead.
    foreign = unpronounceable_characters(text, voice)
    if foreign:
        _LOG.warning(
            "utterance %d held as silence; unpronounceable characters %r in %r",
            index + 1,
            "".join(foreign),
            text[:160],
        )
        silence = output_dir / f"speech-{index:05d}.wav"
        await asyncio.to_thread(make_silence, silence, max(0.2, segment.duration))
        async with progress_lock:
            completed[0] += 1
            progress(f"  silence {completed[0]}/{completed[1]} utterances")
        return silence

    # Quran recitation and other Arabic passages should remain the original
    # recording even when Indonesian narration uses a generic neural voice.
    if (
        source_audio is not None
        and is_arabic_passage(segment.text)
        and not has_english_latin_text(segment.text)
    ):
        await asyncio.to_thread(
            extract_audio_range, source_audio, output, segment.start, segment.end
        )
        async with progress_lock:
            completed[0] += 1
            progress(f"  original Arabic {completed[0]}/{completed[1]} utterances")
        return output

    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("edge-tts is not installed") from exc

    def contains_arabic(value: str) -> bool:
        return any("ARABIC" in unicodedata.name(c, "") for c in value if c.isalpha())

    # Mixed Indonesian/Arabic utterances need both voices: Indonesian prose is
    # translated and spoken normally, while Arabic tokens remain Arabic and are
    # pronounced by the Arabic neural voice.
    voiced_parts: list[tuple[str, str]] = []
    for token in re.findall(r"\S+\s*", text):
        selected = arabic_voice if contains_arabic(token) and arabic_voice else voice
        if voiced_parts and voiced_parts[-1][1] == selected:
            voiced_parts[-1] = (voiced_parts[-1][0] + token, selected)
        else:
            voiced_parts.append((token, selected))
    chunks = [
        (chunk, selected)
        for part, selected in voiced_parts
        for chunk in chunk_text(part)
    ]
    decoded: list[Path] = []

    async def render_chunks(rate: str | None) -> None:
        """Generate every chunk of this utterance, optionally at an Edge TTS rate."""
        decoded.clear()
        for chunk_index, (chunk, selected_voice) in enumerate(chunks):
            raw = output_dir / f"speech-{index:05d}-{chunk_index:03d}.mp3"
            wav = output_dir / f"speech-{index:05d}-{chunk_index:03d}.wav"
            async with semaphore:
                await _save_with_retry(
                    lambda value=chunk, voice=selected_voice, tempo=rate: (
                        edge_tts.Communicate(value, voice, rate=tempo)
                        if tempo
                        else edge_tts.Communicate(value, voice)
                    ),
                    raw,
                )
            if raw.stat().st_size < 256:
                raise RuntimeError("TTS returned an incomplete audio file")
            decode_speech(raw, wav)
            if duration(wav) <= 0.05:
                raise RuntimeError("TTS returned audio with no usable duration")
            decoded.append(wav)
        if len(decoded) == 1:
            decoded[0].replace(output)
        else:
            parts: list[Path] = []
            for chunk_index, wav in enumerate(decoded):
                parts.append(wav)
                if chunk_index + 1 < len(decoded):
                    pause = output_dir / f"speech-{index:05d}-pause-{chunk_index:03d}.wav"
                    make_silence(pause, 0.08)
                    parts.append(pause)
            concatenate(parts, output_dir / f"speech-{index:05d}.txt", output)

    try:
        await render_chunks(None)
        # Speech that overruns its slot is what pushes the dub behind the
        # video: the timeline only recovers part of the overrun, and the rest
        # accumulates for every following utterance. Regenerate at a faster
        # Edge TTS rate while the speech is still clear.
        initial_duration = duration(output)
        tempo = adaptive_rate(initial_duration, max(segment.duration, 0.1))
        if tempo:
            await render_chunks(tempo)
            progress(
                f"  tempo {tempo} utterance {index + 1} "
                f"({initial_duration:.2f}s speech > {segment.duration:.2f}s slot)"
            )
            _LOG.info(
                "utterance %d regenerated at rate %s (%.2fs speech -> %.2fs slot)",
                index + 1,
                tempo,
                initial_duration,
                segment.duration,
            )
    except Exception as exc:
        raise RuntimeError(
            f"Speech synthesis failed for utterance {index + 1}; no text was dropped: {exc}"
        ) from exc

    async with progress_lock:
        completed[0] += 1
        progress(f"  voice {completed[0]}/{completed[1]} utterances")

    return output


TTS_CONCURRENCY = 2


async def synthesize_segments(
    segments: list[Segment],
    output_dir: Path,
    voice: str,
    progress=print,
    clone: CloneConfig | None = None,
    source_audio: Path | None = None,
    speech_cache_dir: Path | None = None,
) -> list[Path]:
    semaphore = asyncio.Semaphore(TTS_CONCURRENCY)
    arabic_voice = None
    if clone is None and any(
        is_arabic_passage(segment.translated_text or segment.text) for segment in segments
    ):
        try:
            arabic_voice = await select_voice("ar")
        except RuntimeError:
            arabic_voice = "ar-SA-HamedNeural"
    completed = [0, len(segments)]
    if clone is None:
        progress_lock = asyncio.Lock()
        tasks = [
            _synthesize_one(
                index, segment, output_dir, voice, arabic_voice, semaphore,
                progress, completed, progress_lock, source_audio,
            )
            for index, segment in enumerate(segments)
        ]
        return list(await asyncio.gather(*tasks))
    progress_lock = threading.Lock()
    results: list[Path | None] = [None] * len(segments)
    cache_keys = _speech_cache_keys(segments, clone)
    cached = _load_cached_speech(speech_cache_dir, cache_keys, progress)
    for index in cached:
        results[index] = cached[index]
    completed[0] = len(cached)
    clone_jobs: list[tuple[int, str]] = []
    for index, segment in enumerate(segments):
        if results[index] is not None:
            continue
        text = segment.translated_text if segment.translated_text is not None else segment.text
        if is_arabic_passage(text) or is_garbled_text(text):
            if source_audio is None:
                raise RuntimeError("Original source audio is required to preserve Arabic speech")
            results[index] = await asyncio.to_thread(
                _preserve_original_one, index, segment, source_audio, output_dir,
                progress, completed, progress_lock,
            )
            _copy_to_cache(results[index], speech_cache_dir, index, cache_keys[index])
        else:
            clone_jobs.append((index, normalize_clone_text(text)))
    if not clone_jobs:
        return [result for result in results if result is not None]
    cpu_groups = _synthesis_worker_cpus()
    # Split each NUMA socket in half to allow up to 4 workers.
    split_groups: list[list[int]] = []
    for group in cpu_groups:
        mid = max(1, len(group) // 2)
        split_groups.append(group[:mid])
        split_groups.append(group[mid:])
    requested = int(os.environ.get("VOICECOVER_CLONE_WORKERS", "4"))
    workers = max(1, min(requested, len(clone_jobs), len(split_groups)))
    if workers == 1:
        engine = CloneEngine(clone)
        for index, text in clone_jobs:
            results[index] = _generate_clone_wav(engine, text, output_dir, index)
            _copy_to_cache(results[index], speech_cache_dir, index, cache_keys[index])
            with progress_lock:
                completed[0] += 1
                progress(f"  clone voice {completed[0]}/{completed[1]} utterances")
        return [result for result in results if result is not None]
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    splits = _balanced_clone_splits(clone_jobs, workers)
    processes = [
        context.Process(
            target=_clone_worker,
            args=(
                clone, jobs, output_dir, split_groups[pidx % len(split_groups)],
                speech_cache_dir, cache_keys,
            ),
        )
        for pidx, jobs in enumerate(splits)
    ]
    try:
        for process in processes:
            process.start()
        last_reported = -1
        while any(process.is_alive() for process in processes):
            done = sum(
                1
                for index, _ in clone_jobs
                if (output_dir / f"speech-{index:05d}.wav").is_file()
            )
            if done and done != last_reported:
                with progress_lock:
                    progress(f"  clone voice {done + completed[0]}/{completed[1]} utterances")
                last_reported = done
            time.sleep(4)
        for process in processes:
            process.join()
        for index, _ in clone_jobs:
            output = output_dir / f"speech-{index:05d}.wav"
            if not output.is_file():
                raise RuntimeError(f"clone worker produced no audio for utterance {index + 1}")
            results[index] = output
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
    return [result for result in results if result is not None]


def _synthesis_worker_cpus() -> list[list[int]]:
    """Group online CPUs per NUMA node so each clone worker pins to one socket."""
    import glob

    online = set(os.sched_getaffinity(0))
    groups: list[list[int]] = []
    for path in sorted(
        glob.glob("/sys/devices/system/node/node*/cpulist"),
        key=lambda p: int(re.search(r"node(\d+)", p).group(1)),
    ):
        cpus: list[int] = []
        for part in Path(path).read_text().strip().split(","):
            if "-" in part:
                start, stop = part.split("-", 1)
                cpus.extend(range(int(start), int(stop) + 1))
            elif part:
                cpus.append(int(part))
        cpus = [cpu for cpu in cpus if cpu in online]
        if cpus:
            groups.append(cpus)
    if not groups:
        available = sorted(online)
        middle = len(available) // 2
        groups = [available[:middle], available[middle:]]
    return [group for group in groups if group]


def _balanced_clone_splits(
    jobs: list[tuple[int, str]], workers: int
) -> list[list[tuple[int, str]]]:
    """Greedily balance estimated F5 work so workers finish at similar times."""
    if workers < 1:
        raise ValueError("workers must be positive")
    buckets: list[list[tuple[int, str]]] = [[] for _ in range(workers)]
    weights = [0] * workers
    for job in sorted(jobs, key=lambda item: len(item[1]), reverse=True):
        target = min(range(workers), key=lambda index: weights[index])
        buckets[target].append(job)
        weights[target] += max(1, len(job[1]))
    # Keep the LPT-balanced totals, but finish short clips first. This creates
    # useful checkpoints quickly and avoids losing several long in-flight
    # clips if a CPU-only run is interrupted.
    for bucket in buckets:
        bucket.sort(key=lambda item: len(item[1]))
    return buckets


def _clone_worker(
    config: CloneConfig,
    jobs: list[tuple[int, str]],
    output_dir: Path,
    cpus: list[int],
    speech_cache_dir: Path | None = None,
    cache_keys: list[str] | None = None,
) -> int:
    """F5-TTS worker in its own process, pinned to one CPU group (spawn)."""
    if cpus:
        os.sched_setaffinity(0, set(cpus))
        thread_count = len(cpus)
        try:
            import torch

            torch.set_num_threads(thread_count)
        except Exception:
            pass
    engine = CloneEngine(config)
    for index, text in jobs:
        output = _generate_clone_wav(engine, text, output_dir, index)
        if cache_keys is not None:
            _copy_to_cache(output, speech_cache_dir, index, cache_keys[index])
    return len(jobs)


def _generate_clone_wav(
    engine: CloneEngine, text: str, output_dir: Path, index: int
) -> Path:
    """Run F5-TTS for one utterance, chunking long text and joining the audio."""
    output = output_dir / f"speech-{index:05d}.wav"
    if not text.strip():
        raise RuntimeError(f"Speech segment {index + 1} is empty")
    chunks = chunk_text(text)
    decoded: list[Path] = []
    try:
        for chunk_index, chunk in enumerate(chunks):
            chunk_wav = output_dir / f"speech-{index:05d}-{chunk_index:03d}.wav"
            engine.generate(chunk, chunk_wav, progress=lambda _message: None)
            if duration(chunk_wav) <= 0.05:
                raise RuntimeError("clone TTS returned audio with no usable duration")
            decoded.append(chunk_wav)
        if len(decoded) == 1:
            decoded[0].replace(output)
        else:
            parts: list[Path] = []
            for chunk_index, chunk_wav in enumerate(decoded):
                parts.append(chunk_wav)
                if chunk_index + 1 < len(decoded):
                    pause = output_dir / f"speech-{index:05d}-pause-{chunk_index:03d}.wav"
                    make_silence(pause, 0.08)
                    parts.append(pause)
            concatenate(parts, output_dir / f"speech-{index:05d}.txt", output)
    except Exception as exc:
        raise RuntimeError(
            f"Clone speech synthesis failed for utterance {index + 1}; no text was dropped: {exc}"
        ) from exc
    return output


def _preserve_original_one(
    index: int,
    segment: Segment,
    source_audio: Path,
    output_dir: Path,
    progress,
    completed: list[int],
    progress_lock: threading.Lock,
) -> Path:
    """Use the speaker's original recording for an Arabic passage."""
    output = output_dir / f"speech-{index:05d}.wav"
    extract_audio_range(source_audio, output, segment.start, segment.end)
    with progress_lock:
        completed[0] += 1
        progress(f"  original Arabic {completed[0]}/{completed[1]} utterances")
    return output


def _synthesize_clone_one(
    index: int,
    segment: Segment,
    output_dir: Path,
    engine: CloneEngine,
    progress,
    completed: list[int],
    progress_lock: threading.Lock,
) -> Path:
    """Synthesize a translated utterance with the original speaker's cloned voice."""
    text = segment.translated_text if segment.translated_text is not None else segment.text
    output = _generate_clone_wav(engine, normalize_clone_text(text), output_dir, index)
    with progress_lock:
        completed[0] += 1
        progress(f"  clone voice {completed[0]}/{completed[1]} utterances")
    return output


def choose_voice(target_language: str, voice: str | None) -> str:
    return voice or asyncio.run(select_voice(target_language))


def synthesize(
    segments: list[Segment],
    output_dir: Path,
    voice: str,
    progress=print,
    clone: CloneConfig | None = None,
    source_audio: Path | None = None,
    speech_cache_dir: Path | None = None,
) -> list[Path]:
    return asyncio.run(
        synthesize_segments(segments, output_dir, voice, progress, clone, source_audio, speech_cache_dir)
    )
