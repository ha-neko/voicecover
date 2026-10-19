from __future__ import annotations

import os
from pathlib import Path

from .models import Segment


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except (ImportError, RuntimeError):
        return "cpu"


def cpu_thread_count() -> int:
    """Use physical cores by default; SMT adds little to large-model inference."""
    configured = os.environ.get("VOICECOVER_WHISPER_THREADS")
    if configured is not None:
        try:
            value = int(configured)
        except ValueError as exc:
            raise ValueError("VOICECOVER_WHISPER_THREADS must be an integer") from exc
        if value < 1:
            raise ValueError("VOICECOVER_WHISPER_THREADS must be positive")
        return value
    try:
        logical = len(os.sched_getaffinity(0))
    except AttributeError:
        logical = os.cpu_count() or 4
    return max(1, min(16, logical // 2 if logical >= 8 else logical))


def transcribe(
    audio: Path,
    model_name: str = "small",
    language: str | None = None,
    device: str = "auto",
    initial_prompt: str | None = None,
    multilingual: bool = False,
) -> tuple[list[Segment], str]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is not installed") from exc

    selected_device = resolve_device(device)
    compute_type = "int8" if selected_device == "cpu" else "float16"
    model = WhisperModel(
        model_name,
        device=selected_device,
        compute_type=compute_type,
        cpu_threads=cpu_thread_count() if selected_device == "cpu" else 0,
        num_workers=1,
    )
    raw_segments, info = model.transcribe(
        str(audio),
        language=language,
        vad_filter=True,
        vad_parameters={
            "threshold": 0.35,
            "min_silence_duration_ms": 1000,
            "speech_pad_ms": 400,
        },
        beam_size=5,
        initial_prompt=initial_prompt,
        multilingual=multilingual,
        language_detection_segments=3 if language is None else 1,
    )
    segments = [
        Segment(float(item.start), float(item.end), item.text.strip())
        for item in raw_segments
        if item.text.strip() and item.end > item.start
    ]
    return segments, info.language
