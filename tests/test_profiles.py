from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from voicecover.profiles import (
    PROFILE_SCHEMA_VERSION,
    ProfileError,
    SeriesProfile,
    default_profile_directory,
    delete_profile,
    list_profiles,
    load_profile,
    option_overrides,
    profile_path,
    save_profile,
)


def profile(name: str = "quran-course") -> SeriesProfile:
    return SeriesProfile(
        name=name,
        schema_version=PROFILE_SCHEMA_VERSION,
        source_language="ar",
        target_language="en",
        model="large-v3",
        device="cuda",
        voice="en-US-GuyNeural",
        original_volume=0.1,
        write_subtitles=True,
        translator="nllb",
        install_models=False,
        initial_prompt="Quran lesson with Arabic recitation.",
        preserve_arabic=True,
        natural_speech=True,
        write_segments=True,
        write_bilingual=True,
        visual_context=False,
        voice_clone=False,
        clone_model=None,
        clone_nfe=24,
        protected_terms={"القرآن": "Quran", "تجويد": "tajwid"},
    )


def test_profile_is_frozen_and_slotted() -> None:
    item = profile()
    with pytest.raises(FrozenInstanceError):
        item.model = "small"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        getattr(item, "__dict__")


def test_default_directory_honors_xdg_config_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert default_profile_directory() == tmp_path / "voicecover" / "profiles"


def test_default_directory_falls_back_to_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert default_profile_directory() == tmp_path / ".config" / "voicecover" / "profiles"


@pytest.mark.parametrize("name", ["../escape", "nested/name", ".hidden", "", "/absolute"])
def test_profile_names_cannot_traverse_directories(tmp_path: Path, name: str) -> None:
    with pytest.raises(ProfileError):
        profile_path(name, tmp_path)


def test_save_load_list_and_delete_round_trip(tmp_path: Path) -> None:
    first = profile("lesson-02")
    second = profile("lesson-01")

    path = save_profile(first, tmp_path)
    save_profile(second, tmp_path)

    assert path == tmp_path / "lesson-02.json"
    assert load_profile("lesson-02", tmp_path) == first
    assert list_profiles(tmp_path) == ["lesson-01", "lesson-02"]
    assert delete_profile("lesson-02", tmp_path) is True
    assert delete_profile("lesson-02", tmp_path) is False
    assert list_profiles(tmp_path) == ["lesson-01"]


def test_save_is_atomic_and_leaves_no_temporary_files(tmp_path: Path) -> None:
    save_profile(profile(), tmp_path)
    save_profile(profile(), tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["quran-course.json"]


def test_option_overrides_contains_only_pipeline_fields() -> None:
    item = profile()
    overrides = option_overrides(item)

    assert overrides["target_language"] == "en"
    assert overrides["clone_nfe"] == 24
    assert set(overrides).isdisjoint({"name", "schema_version", "protected_terms"})
    assert len(overrides) == 18


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"extra": "nope"}, "unknown profile field"),
        ({"schema_version": 2}, "unsupported profile schema version"),
        ({"write_subtitles": 1}, "write_subtitles must be a boolean"),
        ({"clone_nfe": True}, "clone_nfe must be an integer"),
        ({"original_volume": 1}, "original_volume must be a float"),
        ({"protected_terms": {"Quran": 1}}, "keys and values must be strings"),
    ],
)
def test_load_rejects_unknown_version_and_mistyped_fields(
    tmp_path: Path, change: dict[str, object], message: str
) -> None:
    data = profile().__dict__ if hasattr(profile(), "__dict__") else {
        field: getattr(profile(), field)
        for field in profile().__slots__
    }
    data.update(change)
    (tmp_path / "quran-course.json").write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ProfileError, match=message):
        load_profile("quran-course", tmp_path)


def test_load_requires_every_field_instead_of_applying_defaults(tmp_path: Path) -> None:
    data = {
        field: getattr(profile(), field)
        for field in profile().__slots__
        if field != "visual_context"
    }
    (tmp_path / "quran-course.json").write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ProfileError, match="missing profile field.*visual_context"):
        load_profile("quran-course", tmp_path)


def test_load_rejects_filename_metadata_mismatch(tmp_path: Path) -> None:
    save_profile(profile("original"), tmp_path)
    (tmp_path / "renamed.json").write_bytes((tmp_path / "original.json").read_bytes())

    with pytest.raises(ProfileError, match="does not match"):
        load_profile("renamed", tmp_path)


def test_load_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-profile.json"
    outside.write_text("{}", encoding="utf-8")
    (tmp_path / "escaped.json").symlink_to(outside)

    with pytest.raises(ProfileError, match="escapes"):
        load_profile("escaped", tmp_path)
