from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import media
from .detect import detect_language
from .models import Segment

DEFAULT_CLONE_MODEL = "Eempostor/F5-TTS-INDO-FINETUNE-V2"
MAX_REFERENCE_SECONDS = 12.0
MIN_REFERENCE_SECONDS = 4.0
REFERENCE_CACHE_VERSION = 2


@dataclass(slots=True)
class CloneConfig:
    """How to clone the original speaker with an Indonesian F5-TTS model."""

    ref_wav: Path
    ref_text: str
    model: str = DEFAULT_CLONE_MODEL
    cache_dir: Path | None = None
    nfe_step: int = 16
    speed: float = 1.0
    device: str = "cpu"
    ckpt: Path | None = None
    vocab: Path | None = None


def _tag_of(segment: Segment) -> str:
    return detect_language(segment.text)


def _cleanest_language(segments: list[Segment]) -> str | None:
    """Most common non-Arabic source among the reference candidates."""
    counts: dict[str, int] = {}
    for segment in segments:
        tag = _tag_of(segment)
        if tag in {"ar", "arb_Arab"}:
            continue
        counts[tag] = counts.get(tag, 0) + 1
    return max(counts, key=counts.get) if counts else None


def pick_reference_window(
    segments: list[Segment], total_duration: float
) -> tuple[float, float, Segment]:
    """Pick the cleanest, longest speech window to use as the clone reference."""
    candidates = [
        segment
        for segment in segments
        if MIN_REFERENCE_SECONDS <= segment.end - segment.start <= MAX_REFERENCE_SECONDS
        and segment.end <= total_duration + 0.1
    ]
    if not candidates:
        candidates = [
            segment
            for segment in segments
            if 1.5 <= segment.end - segment.start <= MAX_REFERENCE_SECONDS
            and segment.end <= total_duration + 0.1
        ]
    if not candidates:
        raise RuntimeError("no usable speech window found for voice cloning")
    preferred = _cleanest_language(candidates)
    if preferred is not None:
        in_language = [s for s in candidates if _tag_of(s) == preferred]
        if in_language:
            candidates = in_language
    def reference_score(segment: Segment) -> tuple[int, int, int, int, float, float]:
        text = segment.text.strip()
        duration = segment.end - segment.start
        # A complete declarative sentence gives F5 a stable cadence. Long
        # fragments ending mid-thought tend to leak uncertain prosody into
        # every generated utterance.
        declarative = int(text.endswith((".", "!")))
        complete = int(text.endswith((".", "!", "?")))
        sentence_marks = len(re.findall(r"[.!?](?:\s|$)", text))
        natural_sentence_count = 2 if sentence_marks == 2 else int(sentence_marks == 1)
        trailing_connector = int(
            bool(re.search(r"\b(and|or|but|as|to|the|a|an|of|for|with|spoken)\W*$", text, re.I))
        )
        word_rate = len(text.split()) / max(duration, 0.1)
        natural_rate = int(1.4 <= word_rate <= 3.6)
        target_duration = -abs(duration - 8.0)
        return (
            declarative, natural_sentence_count, complete, natural_rate,
            -trailing_connector, target_duration,
        )

    best = max(candidates, key=reference_score)
    return best.start, best.end, best


def _trim_and_normalize(source_wav: Path, start: float, end: float, work: Path) -> Path:
    """Cut an aligned reference window and normalize it without changing its timing."""
    output = work / "clone-reference.wav"
    media.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(source_wav),
            "-af",
            # Internal silence belongs to the transcript and is part of the
            # speaker's natural rate. Removing it while retaining the complete
            # reference text makes F5 allocate generated clips far too short.
            "loudnorm=I=-20:TP=-1.5:LRA=11",
            "-ar",
            "24000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )
    if media.duration(output) > MAX_REFERENCE_SECONDS + 0.5:
        capped = work / "clone-reference-trim.wav"
        media.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(output),
                "-t",
                f"{MAX_REFERENCE_SECONDS:.3f}",
                "-c:a",
                "pcm_s16le",
                str(capped),
            ]
        )
        output = capped
    return output


def select_reference(
    source_wav: Path,
    segments: list[Segment],
    work: Path,
    progress=print,
    reference_cache_dir: Path | None = None,
) -> tuple[Path, str]:
    """Choose the original speaker's voice window and its transcript."""
    if reference_cache_dir is not None:
        cached_wav = reference_cache_dir / "reference.wav"
        cached_metadata = reference_cache_dir / "reference.json"
        try:
            metadata = json.loads(cached_metadata.read_text(encoding="utf-8"))
            cached_text = metadata["text"]
            if (
                isinstance(cached_text, str)
                and cached_text.strip()
                and metadata.get("version") == REFERENCE_CACHE_VERSION
                and cached_wav.is_file()
                and cached_wav.stat().st_size > 44
                and media.duration(cached_wav) >= MIN_REFERENCE_SECONDS
            ):
                progress(f"restored fixed clone reference: {cached_wav}")
                return cached_wav, cached_text
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    if not segments:
        raise RuntimeError("no speech segments available to build a voice reference")
    total = media.duration(source_wav)
    start, end, best = pick_reference_window(segments, total)
    progress(
        f"cloning original voice from {start:.1f}s-{end:.1f}s "
        f"(lang={_tag_of(best)}, {best.end - best.start:.1f}s)"
    )
    ref_wav = _trim_and_normalize(source_wav, start, end, work)
    ref_text = best.text.strip()
    if not ref_text:
        raise RuntimeError("the reference window has no transcript to condition the clone")
    if reference_cache_dir is not None:
        reference_cache_dir.mkdir(parents=True, exist_ok=True)
        cached_wav = reference_cache_dir / "reference.wav"
        shutil.copy2(ref_wav, cached_wav)
        (reference_cache_dir / "reference.json").write_text(
            json.dumps(
                {
                    "version": REFERENCE_CACHE_VERSION,
                    "start": start,
                    "end": end,
                    "text": ref_text,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        ref_wav = cached_wav
    return ref_wav, ref_text


def ensure_clone_files(config: CloneConfig, progress=print) -> tuple[Path, Path]:
    """Download or locate the checkpoint and vocabulary for the clone model."""
    if config.ckpt and config.vocab:
        return config.ckpt, config.vocab
    cache_dir = config.cache_dir or Path.home() / ".cache" / "voicecover" / "clone"
    repo_dir = cache_dir / config.model.replace("/", "--")
    if not (repo_dir / "vocab.txt").is_file():
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is not installed; run: pip install 'voicecover[clone]'"
            ) from exc
        progress(f"downloading clone model {config.model}...")
        snapshot_download(
            config.model,
            local_dir=str(repo_dir),
            allow_patterns=["*.pt", "*.safetensors", "vocab.txt", "*.model"],
        )
    ckpt = config.ckpt
    if ckpt is None:
        candidates = sorted(repo_dir.glob("*.pt")) + sorted(repo_dir.glob("*.safetensors"))
        if not candidates:
            raise RuntimeError(f"no checkpoint found in {repo_dir}")
        ckpt = candidates[0]
    vocab = config.vocab or repo_dir / "vocab.txt"
    if not vocab.is_file():
        raise RuntimeError(f"vocabulary file not found in {repo_dir}")
    return ckpt, vocab


@dataclass(slots=True)
class CloneEngine:
    """Lazy F5-TTS wrapper that generates speech with the cloned voice."""

    config: CloneConfig
    _engine: object = field(default=None, init=False, repr=False)

    def load(self, progress=print) -> None:
        if self._engine is not None:
            return
        try:
            from f5_tts.api import F5TTS
        except ImportError as exc:
            raise RuntimeError("f5-tts is not installed; run: pip install 'voicecover[clone]'") from exc
        ckpt, vocab = ensure_clone_files(self.config, progress)
        progress("loading F5-TTS clone model...")
        self._engine = F5TTS(
            model="F5TTS_v1_Base",
            ckpt_file=str(ckpt),
            vocab_file=str(vocab),
            device=self.config.device,
        )
        progress("clone model ready")

    def generate(self, text: str, output: Path, progress=print) -> Path:
        if self._engine is None:
            self.load(progress)
        wav, sample_rate, _ = self._engine.infer(
            ref_file=str(self.config.ref_wav),
            ref_text=self.config.ref_text,
            gen_text=text,
            show_info=lambda _message: None,
            progress=None,
            nfe_step=self.config.nfe_step,
            speed=self.config.speed,
            remove_silence=False,
            seed=42,
        )
        import soundfile as sf
        import numpy as np

        samples = np.asarray(wav).copy()
        # F5 returns a fixed-size tensor and can finish on an active sample.
        # A tiny release removes the click; post-roll keeps the final consonant
        # away from the timeline boundary without deleting speech.
        release = min(len(samples), max(1, round(sample_rate * 0.008)))
        if release:
            samples[-release:] *= np.linspace(1.0, 0.0, release, dtype=samples.dtype)
        post_roll = np.zeros(round(sample_rate * 0.18), dtype=samples.dtype)
        samples = np.concatenate((samples, post_roll))
        sf.write(str(output), samples, sample_rate)
        return output
