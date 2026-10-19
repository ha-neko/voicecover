from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


class MediaError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
    if missing:
        raise MediaError(f"Missing required program(s): {', '.join(missing)}")


def run(command: Sequence[str]) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise MediaError(detail) from exc


def duration(path: Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(json.loads(result.stdout)["format"]["duration"])
    except (subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise MediaError(f"Could not determine duration of {path}") from exc


def extract_audio(video: Path, output: Path) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )


def extract_timeline_audio(video: Path, output: Path) -> None:
    """Extract full-quality PCM used when source passages are kept in the dub."""
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )


def extract_audio_range(source: Path, output: Path, start: float, end: float) -> None:
    """Copy an exact source-audio interval into the dub's canonical format."""
    if start < 0 or end <= start:
        raise MediaError("An audio interval must have a positive duration")
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{end - start:.6f}",
            "-i",
            str(source),
            "-af",
            "asetpts=PTS-STARTPTS",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )


def _atempo_chain(speed: float) -> str:
    """Build an atempo filter chain; each factor must be within 0.5–100."""
    factors: list[float] = []
    while speed > 100:
        factors.append(100.0)
        speed /= 100
    while speed < 0.5:
        factors.append(0.5)
        speed /= 0.5
    factors.append(speed)
    return ",".join(f"atempo={factor:.8f}" for factor in factors)


def fit_audio(source: Path, output: Path, target_duration: float) -> None:
    if target_duration <= 0:
        raise MediaError("A speech segment must have a positive duration")
    source_duration = duration(source)
    filters: list[str] = []
    speed = source_duration / target_duration
    # Bounded tempo correction. Never trim spoken audio; any remaining overrun
    # is handled by the timeline scheduler rather than deleting words.
    if speed > 1.05:
        # Indonesian localization is often slightly longer than the English
        # source. A 15% ceiling allowed small overruns to accumulate into more
        # than a minute of A/V drift on lecture videos. Twenty-five percent is
        # still intelligible while giving the scheduler enough room to recover.
        filters.append(_atempo_chain(min(speed, 1.25)))
    filters.extend(["asetpts=PTS-STARTPTS"])
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-af",
            ",".join(filters),
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )


def decode_speech(source: Path, output: Path) -> None:
    """Decode a complete TTS response without destructive silence removal."""
    run(
        [
            "ffmpeg", "-y", "-i", str(source),
            "-af", "asetpts=PTS-STARTPTS",
            "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(output),
        ]
    )


def trim_speech(source: Path, output: Path) -> None:
    """Decode TTS audio and remove only encoder artifacts and excessive silence."""
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-af",
            (
                "silenceremove=start_periods=1:start_duration=0.15:"
                "start_threshold=-38dB:stop_periods=1:stop_duration=0.2:"
                "stop_threshold=-38dB:detection=rms,"
                "afade=t=in:d=0.02,areverse,afade=t=in:d=0.02,areverse"
            ),
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )


def make_silence(output: Path, seconds: float) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            f"{max(seconds, 0.001):.6f}",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )


def concatenate(parts: list[Path], manifest: Path, output: Path) -> None:
    # Paths are generated inside one temporary directory and cannot contain quotes.
    manifest.write_text("".join(f"file '{part.name}'\n" for part in parts), encoding="utf-8")
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest),
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )


def mux(video: Path, dub: Path, output: Path, original_volume: float) -> None:
    if original_volume > 0:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-i",
            str(dub),
            "-filter_complex",
            (
                f"[0:a]volume={original_volume:.4f}[original];"
                "[1:a]asplit=2[dub][side];"
                "[original][side]sidechaincompress=threshold=0.025:ratio=8:"
                "attack=20:release=300[ducked];"
                "[ducked][dub]amix=inputs=2:duration=longest:normalize=0,"
                "alimiter=limit=0.95[a]"
            ),
            "-map",
            "0:v:0",
            "-map",
            "[a]",
        ]
    else:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-i",
            str(dub),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
        ]
    command.extend(
        [
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output),
        ]
    )
    run(command)
