from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .jobs import JobRecord, JobSpec, JobStore, ManifestError
from .paths import default_output
from .pipeline import Options, run as process_video
from .profiles import load_profile_or_builtin


BATCH_MANIFEST_VERSION = 1
_BATCH_OPTION_FIELDS = frozenset(
    {
        "source_language",
        "model",
        "device",
        "voice",
        "original_volume",
        "write_subtitles",
        "keep_temp",
        "translator",
        "install_models",
        "initial_prompt",
        "preserve_arabic",
        "natural_speech",
        "write_segments",
        "write_bilingual",
        "visual_context",
        "voice_clone",
        "clone_model",
        "clone_nfe",
        "write_visual_track",
        "localize_visual_text",
        "visual_only",
        "sound_only",
        "visual_interval",
        "visual_width",
    }
)
_PAYLOAD_FIELDS = _BATCH_OPTION_FIELDS | {"input", "target_language", "protected_terms"}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="voicecover", description="Translate and dub a video into another language."
    )
    result.add_argument("input", nargs="?", help="local video path or YouTube URL")
    result.add_argument("--batch", type=Path, help="run a versioned UTF-8 JSON batch manifest")
    result.add_argument(
        "--job-store",
        type=Path,
        default=Path(".voicecover-jobs"),
        help="durable batch state directory (default: .voicecover-jobs)",
    )
    result.add_argument(
        "--retry-failed",
        action="store_true",
        help="requeue failed, cancelled, or interrupted jobs in an existing batch",
    )
    result.add_argument("-t", "--target", help="target language code or locale")
    result.add_argument("-o", "--output", type=Path, help="output video path")
    result.add_argument("--profile", help="saved series profile name (quran-id is built in)")
    result.add_argument("--source", help="source language; auto-detected by default")
    result.add_argument(
        "--initial-prompt",
        help="context and vocabulary hint supplied to Whisper",
    )
    result.add_argument("--model", default="small", help="Whisper model (default: small)")
    result.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="cpu", help="Whisper device"
    )
    result.add_argument("--voice", help="Edge TTS voice name; selected automatically by default")
    result.add_argument(
        "--translator",
        choices=("auto", "nllb", "argos", "google", "llm"),
        default="auto",
        help="translation provider (default: Google with offline NLLB fallback)",
    )
    result.add_argument(
        "--no-install-models",
        action="store_true",
        help="fail instead of downloading missing offline translation models",
    )
    result.add_argument(
        "--original-volume",
        type=float,
        default=0.15,
        metavar="0..1",
        help="original soundtrack volume in the mix (default: 0.15)",
    )
    result.add_argument("--no-srt", action="store_true", help="do not write translated subtitles")
    result.add_argument(
        "--bilingual-srt",
        action="store_true",
        help="write subtitles showing the original and the translation",
    )
    result.add_argument(
        "--no-visual-context",
        action="store_true",
        help="do not OCR on-screen text to build a Whisper vocabulary hint",
    )
    result.add_argument(
        "--clone-voice",
        action="store_true",
        help="dub the translation with the original speaker's cloned voice (F5-TTS Indonesian)",
    )
    result.add_argument(
        "--clone-model",
        default=None,
        help="Hugging Face repo with an F5-TTS checkpoint for the clone voice",
    )
    result.add_argument("--clone-nfe", type=int, default=16, help="F5-TTS diffusion steps (default 16)")
    result.add_argument(
        "--localize-visual-text",
        action="store_true",
        help="replace static Latin slide text with translated text",
    )
    mode = result.add_mutually_exclusive_group()
    mode.add_argument(
        "--visual-only",
        action="store_true",
        help="translate on-screen text only and preserve the original audio",
    )
    mode.add_argument(
        "--sound-only",
        action="store_true",
        help="translate and dub speech only, preserving the original video pixels",
    )
    result.add_argument(
        "--write-visual-track",
        action="store_true",
        help="write detected slide/OCR metadata as .visual.json",
    )
    result.add_argument("--keep-temp", action="store_true", help="keep intermediate audio files")
    result.add_argument(
        "--translate-arabic",
        action="store_true",
        help="translate Arabic passages in mixed-language material instead of preserving them",
    )
    result.add_argument(
        "--no-natural-speech",
        action="store_true",
        help="keep raw Whisper fragments instead of grouping complete utterances",
    )
    result.add_argument("--tui", action="store_true", help="open the interactive terminal interface")
    return result


def _options_from_args(
    args: argparse.Namespace,
    input_value: str,
    output: Path,
    target: str,
    profile_name: str | None,
) -> Options:
    profile = load_profile_or_builtin(profile_name) if profile_name else None
    profile_values = profile.option_overrides() if profile else {}
    return Options(
        input=input_value,
        output=output,
        target_language=target,
        source_language=profile_values.get("source_language") if profile else args.source,
        model=str(profile_values.get("model", args.model)),
        device=str(profile_values.get("device", args.device)),
        voice=profile_values.get("voice") if profile else args.voice,
        original_volume=float(profile_values.get("original_volume", args.original_volume)),
        write_subtitles=bool(profile_values.get("write_subtitles", not args.no_srt)),
        keep_temp=args.keep_temp,
        translator=str(profile_values.get("translator", args.translator)),
        install_models=bool(profile_values.get("install_models", not args.no_install_models)),
        initial_prompt=profile_values.get("initial_prompt") if profile else args.initial_prompt,
        preserve_arabic=bool(profile_values.get("preserve_arabic", not args.translate_arabic)),
        natural_speech=bool(profile_values.get("natural_speech", not args.no_natural_speech)),
        write_segments=bool(profile_values.get("write_segments", False)),
        write_bilingual=bool(profile_values.get("write_bilingual", args.bilingual_srt)),
        visual_context=(
            bool(profile_values.get("visual_context", not args.no_visual_context))
            and not args.sound_only
        ),
        voice_clone=bool(profile_values.get("voice_clone", args.clone_voice)),
        clone_model=profile_values.get("clone_model") if profile else args.clone_model,
        clone_nfe=int(profile_values.get("clone_nfe", args.clone_nfe)),
        protected_terms=profile.protected_terms if profile else {},
        localize_visual_text=(
            args.localize_visual_text or args.visual_only or profile_name == "quran-id"
        ) and not args.sound_only,
        visual_only=args.visual_only,
        sound_only=args.sound_only,
        write_visual_track=(args.write_visual_track or profile_name == "quran-id") and not args.sound_only,
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ManifestError(f"invalid JSON constant: {value}")


def _load_batch(path: Path) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(
                handle,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_constant,
            )
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read batch manifest {path}: {exc}") from exc
    if type(data) is not dict:
        raise ManifestError("batch manifest must be an object")
    expected = {"version", "id", "name", "jobs"}
    if set(data) != expected:
        raise ManifestError(
            f"batch manifest fields must be exactly: {', '.join(sorted(expected))}"
        )
    if data["version"] != BATCH_MANIFEST_VERSION:
        raise ManifestError(f"unsupported batch input version: {data['version']!r}")
    if type(data["jobs"]) is not list or not data["jobs"]:
        raise ManifestError("batch jobs must be a nonempty array")
    return data


def _validate_option_types(values: Mapping[str, Any]) -> None:
    optional_strings = {"source_language", "voice", "initial_prompt", "clone_model"}
    strings = {"model", "device", "translator"}
    booleans = _BATCH_OPTION_FIELDS - optional_strings - strings - {
        "original_volume", "clone_nfe", "visual_interval", "visual_width"
    }
    for name in optional_strings:
        if name in values and values[name] is not None and type(values[name]) is not str:
            raise ManifestError(f"batch option {name} must be a string or null")
    for name in strings:
        if name in values and type(values[name]) is not str:
            raise ManifestError(f"batch option {name} must be a string")
    for name in booleans:
        if name in values and type(values[name]) is not bool:
            raise ManifestError(f"batch option {name} must be a boolean")
    if "original_volume" in values and type(values["original_volume"]) not in {int, float}:
        raise ManifestError("batch option original_volume must be a number")
    if "original_volume" in values and not 0 <= values["original_volume"] <= 1:
        raise ManifestError("batch option original_volume must be between 0 and 1")
    for name in {"clone_nfe", "visual_width"}:
        if name in values and type(values[name]) is not int:
            raise ManifestError(f"batch option {name} must be an integer")
    if "visual_interval" in values and type(values["visual_interval"]) not in {int, float}:
        raise ManifestError("batch option visual_interval must be a number")
    if values.get("clone_nfe", 1) < 1 or values.get("visual_width", 1) < 1:
        raise ManifestError("batch option clone_nfe and visual_width must be positive")
    if values.get("visual_interval", 1) <= 0:
        raise ManifestError("batch option visual_interval must be positive")
    if values.get("device", "cpu") not in {"auto", "cpu", "cuda"}:
        raise ManifestError("batch option device must be auto, cpu, or cuda")
    if values.get("translator", "auto") not in {"auto", "nllb", "argos", "google", "llm"}:
        raise ManifestError("batch option translator is invalid")
    if values.get("visual_only", False) and values.get("sound_only", False):
        raise ManifestError("batch options visual_only and sound_only are mutually exclusive")


def _options_payload(options: Options) -> dict[str, Any]:
    return {
        "input": str(options.input),
        "target_language": options.target_language,
        **{name: getattr(options, name) for name in _BATCH_OPTION_FIELDS},
        "protected_terms": dict(options.protected_terms),
    }


def _options_from_payload(payload: Mapping[str, Any], output: str | None) -> Options:
    payload = dict(payload)
    payload.setdefault("visual_only", False)
    payload.setdefault("sound_only", False)
    if set(payload) != _PAYLOAD_FIELDS or output is None:
        raise ManifestError("persisted batch job has invalid option fields")
    _validate_option_types(payload)
    if type(payload["input"]) is not str or type(payload["target_language"]) is not str:
        raise ManifestError("persisted batch input and target_language must be strings")
    protected = payload["protected_terms"]
    if type(protected) is not dict or any(
        type(key) is not str or type(value) is not str for key, value in protected.items()
    ):
        raise ManifestError("persisted protected_terms must map strings to strings")
    return Options(
        input=payload["input"],
        output=Path(output),
        target_language=payload["target_language"],
        source_language=payload["source_language"],
        model=payload["model"],
        device=payload["device"],
        voice=payload["voice"],
        original_volume=float(payload["original_volume"]),
        write_subtitles=payload["write_subtitles"],
        keep_temp=payload["keep_temp"],
        translator=payload["translator"],
        install_models=payload["install_models"],
        initial_prompt=payload["initial_prompt"],
        preserve_arabic=payload["preserve_arabic"],
        natural_speech=payload["natural_speech"],
        write_segments=payload["write_segments"],
        write_bilingual=payload["write_bilingual"],
        visual_context=payload["visual_context"],
        voice_clone=payload["voice_clone"],
        clone_model=payload["clone_model"],
        clone_nfe=payload["clone_nfe"],
        protected_terms=dict(protected),
        write_visual_track=payload["write_visual_track"],
        localize_visual_text=payload["localize_visual_text"],
        visual_only=payload["visual_only"],
        sound_only=payload["sound_only"],
        visual_interval=float(payload["visual_interval"]),
        visual_width=payload["visual_width"],
    )


def _batch_specs(data: Mapping[str, Any], args: argparse.Namespace) -> list[JobSpec]:
    specs: list[JobSpec] = []
    allowed = {"id", "name", "input", "output", "target", "profile", "options"}
    for index, item in enumerate(data["jobs"], 1):
        if type(item) is not dict or not {"input", "output"} <= set(item) or set(item) - allowed:
            raise ManifestError(
                f"batch job {index} requires input/output and only supports {', '.join(sorted(allowed))}"
            )
        if type(item["input"]) is not str or type(item["output"]) is not str:
            raise ManifestError(f"batch job {index} input/output must be strings")
        profile_name = item.get("profile", args.profile)
        target = item.get("target", args.target)
        if profile_name is not None and type(profile_name) is not str:
            raise ManifestError(f"batch job {index} profile must be a string")
        if target is None and profile_name:
            target = load_profile_or_builtin(profile_name).target_language
        if type(target) is not str or not target:
            raise ManifestError(f"batch job {index} requires target or profile")
        options = _options_from_args(args, item["input"], Path(item["output"]), target, profile_name)
        overrides = item.get("options", {})
        if type(overrides) is not dict or set(overrides) - _BATCH_OPTION_FIELDS:
            raise ManifestError(f"batch job {index} contains unknown or unsafe options")
        _validate_option_types(overrides)
        options = replace(options, **overrides)
        name = item.get("name", f"Job {index}")
        specs.append(
            JobSpec(
                name=name,
                payload=_options_payload(options),
                output=item["output"],
                id=item.get("id"),
            )
        )
    return specs


def _run_batch(args: argparse.Namespace) -> int:
    data = _load_batch(args.batch)
    store = JobStore(args.job_store, recover_running="queued")
    try:
        batch = store.get_batch(data["id"])
    except KeyError:
        batch = store.create_batch(
            data["name"], _batch_specs(data, args), batch_id=data["id"]
        )
    if args.retry_failed:
        batch = store.retry_batch(batch.id)

    def worker(job: JobRecord, progress) -> Path:
        options = _options_from_payload(job.payload, job.output)
        return process_video(options, progress=progress)

    while store.dispatch_one(worker) is not None:
        pass
    batch = store.get_batch(batch.id)
    print(f"batch {batch.id}: {batch.status}")
    return 0 if batch.status == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        if args.batch:
            if args.input is not None or args.tui:
                argument_parser.error("--batch cannot be combined with input or --tui")
            return _run_batch(args)
        if args.retry_failed:
            argument_parser.error("--retry-failed requires --batch")
        if args.tui or (args.input is None and args.target is None):
            from .tui import VoicecoverApp

            app = VoicecoverApp()
            app.run()
            return app.return_code or 0
        profile = load_profile_or_builtin(args.profile) if args.profile else None
        target = args.target or (profile.target_language if profile else None)
        if args.input is None or target is None:
            argument_parser.error("input and --target (or --profile) are required in command-line mode")
        output = args.output or default_output(args.input, target)
        process_video(_options_from_args(args, args.input, output, target, args.profile))
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, ManifestError) as exc:
        print(f"voicecover: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
