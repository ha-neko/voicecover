from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


PROFILE_SCHEMA_VERSION = 1
QURAN_INDONESIAN_PROFILE = "quran-id"
_PROFILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


class ProfileError(ValueError):
    """A series profile is invalid or unsafe to access."""


@dataclass(frozen=True, slots=True)
class SeriesProfile:
    name: str
    schema_version: int
    source_language: str | None
    target_language: str
    model: str
    device: str
    voice: str | None
    original_volume: float
    write_subtitles: bool
    translator: str
    install_models: bool
    initial_prompt: str | None
    preserve_arabic: bool
    natural_speech: bool
    write_segments: bool
    write_bilingual: bool
    visual_context: bool
    voice_clone: bool
    clone_model: str | None
    clone_nfe: int
    protected_terms: dict[str, str]

    def __post_init__(self) -> None:
        _validate_values(asdict(self))

    def option_overrides(self) -> dict[str, object]:
        """Return values safe to pass as keyword overrides to pipeline.Options."""
        values = asdict(self)
        values.pop("name")
        values.pop("schema_version")
        values.pop("protected_terms")
        return values


_PROFILE_FIELDS = frozenset(field.name for field in fields(SeriesProfile))
_OPTION_FIELDS = _PROFILE_FIELDS - {"name", "schema_version", "protected_terms"}
_STRING_FIELDS = {
    "target_language",
    "model",
    "device",
    "translator",
}
_OPTIONAL_STRING_FIELDS = {
    "source_language",
    "voice",
    "initial_prompt",
    "clone_model",
}
_BOOL_FIELDS = {
    "write_subtitles",
    "install_models",
    "preserve_arabic",
    "natural_speech",
    "write_segments",
    "write_bilingual",
    "visual_context",
    "voice_clone",
}


def default_profile_directory() -> Path:
    """Return the XDG-compliant directory used for series profiles."""
    config_home = os.environ.get("XDG_CONFIG_HOME")
    root = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return root / "voicecover" / "profiles"


def validate_profile_name(name: str) -> str:
    if type(name) is not str or not _PROFILE_NAME.fullmatch(name):
        raise ProfileError(
            "profile name must be 1-64 ASCII letters, digits, dots, underscores, or "
            "hyphens, and must start with a letter or digit"
        )
    return name


def profile_path(name: str, directory: Path | str | None = None) -> Path:
    """Build a profile path without allowing a name to escape its directory."""
    validate_profile_name(name)
    root = Path(directory) if directory is not None else default_profile_directory()
    path = root / f"{name}.json"
    if path.parent.resolve() != root.resolve():
        raise ProfileError("profile path escapes the profile directory")
    return path


def save_profile(profile: SeriesProfile, directory: Path | str | None = None) -> Path:
    """Atomically write a complete profile as versioned JSON."""
    if type(profile) is not SeriesProfile:
        raise TypeError("profile must be a SeriesProfile")
    _validate_values(asdict(profile))
    path = profile_path(profile.name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(profile), ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_profile(name: str, directory: Path | str | None = None) -> SeriesProfile:
    """Load and strictly validate a profile from its scoped profile directory."""
    path = profile_path(name, directory)
    root = path.parent.resolve()
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"profile not found: {name}") from None
    if resolved.parent != root:
        raise ProfileError("profile path escapes the profile directory")

    try:
        with resolved.open(encoding="utf-8") as handle:
            data = json.load(
                handle,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
    except json.JSONDecodeError as error:
        raise ProfileError(f"invalid profile JSON: {error.msg}") from error

    if type(data) is not dict:
        raise ProfileError("profile JSON must be an object")
    _validate_values(data)
    if data["name"] != name:
        raise ProfileError("profile name does not match its filename")
    return SeriesProfile(**data)


def list_profiles(directory: Path | str | None = None) -> list[str]:
    root = Path(directory) if directory is not None else default_profile_directory()
    if not root.exists():
        return []
    if not root.is_dir():
        raise ProfileError("profile directory is not a directory")

    names: list[str] = []
    for path in root.glob("*.json"):
        name = path.stem
        try:
            validate_profile_name(name)
        except ProfileError:
            continue
        names.append(name)
    return sorted(names)


def delete_profile(name: str, directory: Path | str | None = None) -> bool:
    path = profile_path(name, directory)
    if not path.exists() and not path.is_symlink():
        return False
    if path.is_symlink() and path.resolve().parent != path.parent.resolve():
        raise ProfileError("profile path escapes the profile directory")
    if not path.is_file():
        raise ProfileError("profile path is not a file")
    path.unlink()
    return True


def option_overrides(profile: SeriesProfile) -> dict[str, object]:
    return profile.option_overrides()


def quran_indonesian_profile(name: str = QURAN_INDONESIAN_PROFILE) -> SeriesProfile:
    """Return the quality-oriented profile for the Understand Al-Qur'an series."""
    return SeriesProfile(
        name=name,
        schema_version=PROFILE_SCHEMA_VERSION,
        source_language="en",
        target_language="id",
        model="large-v3",
        device="cpu",
        voice=None,
        original_volume=0.0,
        write_subtitles=True,
        translator="llm",
        install_models=True,
        initial_prompt=(
            "Understand Al-Qur'an course, Surah Al-Fatihah, Surah Al-Baqarah, "
            "Quranic Arabic grammar and Islamic educational vocabulary"
        ),
        preserve_arabic=True,
        natural_speech=True,
        write_segments=True,
        write_bilingual=False,
        visual_context=True,
        voice_clone=True,
        clone_model="Eempostor/F5-TTS-INDO-FINETUNE-V2",
        clone_nfe=32,
        protected_terms={
            "Allah": "Allah",
            "Al-Qur'an": "Al-Qur'an",
            "Qur'an": "Al-Qur'an",
            "Al-Fatihah": "Al-Fatihah",
            "Al-Baqarah": "Al-Baqarah",
            "Juz": "Juz",
            "Muttaqeen": "Muttaqin",
            "inshaAllah": "insyaallah",
        },
    )


def load_profile_or_builtin(
    name: str, directory: Path | str | None = None
) -> SeriesProfile:
    """Load a saved profile, with the maintained Quran course profile built in."""
    try:
        return load_profile(name, directory)
    except FileNotFoundError:
        if name == QURAN_INDONESIAN_PROFILE:
            return quran_indonesian_profile()
        raise


def _validate_values(data: dict[str, Any]) -> None:
    keys = set(data)
    missing = _PROFILE_FIELDS - keys
    unknown = keys - _PROFILE_FIELDS
    if missing:
        raise ProfileError(f"missing profile field(s): {', '.join(sorted(missing))}")
    if unknown:
        raise ProfileError(f"unknown profile field(s): {', '.join(sorted(unknown))}")

    validate_profile_name(data["name"])
    if type(data["schema_version"]) is not int:
        raise ProfileError("schema_version must be an integer")
    if data["schema_version"] != PROFILE_SCHEMA_VERSION:
        raise ProfileError(f"unsupported profile schema version: {data['schema_version']}")

    for field in _STRING_FIELDS:
        if type(data[field]) is not str:
            raise ProfileError(f"{field} must be a string")
    for field in _OPTIONAL_STRING_FIELDS:
        if data[field] is not None and type(data[field]) is not str:
            raise ProfileError(f"{field} must be a string or null")
    for field in _BOOL_FIELDS:
        if type(data[field]) is not bool:
            raise ProfileError(f"{field} must be a boolean")
    if type(data["original_volume"]) is not float:
        raise ProfileError("original_volume must be a float")
    if type(data["clone_nfe"]) is not int:
        raise ProfileError("clone_nfe must be an integer")

    protected_terms = data["protected_terms"]
    if type(protected_terms) is not dict:
        raise ProfileError("protected_terms must be an object")
    if any(type(key) is not str or type(value) is not str for key, value in protected_terms.items()):
        raise ProfileError("protected_terms keys and values must be strings")

    if set(data) - {"name", "schema_version", "protected_terms"} != _OPTION_FIELDS:
        raise ProfileError("profile option fields do not match the supported options")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProfileError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ProfileError(f"invalid JSON constant: {value}")
