from __future__ import annotations


def _normalized(value: str) -> str:
    return value.strip().replace("_", "-").lower()


def whisper_language(value: str | None) -> str | None:
    """Convert a locale into the ISO-style language code Whisper expects."""
    if value is None:
        return None
    base = _normalized(value).split("-")[0]
    return {"iw": "he", "fil": "tl"}.get(base, base)


def translation_language(value: str) -> str:
    """Convert common locales and aliases into Google Translate codes."""
    normalized = _normalized(value)
    if normalized in {"zh-cn", "zh-sg", "zh-hans"}:
        return "zh-CN"
    if normalized in {"zh-tw", "zh-hk", "zh-mo", "zh-hant"}:
        return "zh-TW"
    base = normalized.split("-")[0]
    return {"he": "iw", "fil": "tl"}.get(base, base)


def tts_language(value: str) -> str:
    """Convert provider aliases into the language/locale family Edge uses."""
    normalized = _normalized(value)
    base = normalized.split("-")[0]
    if base == "iw":
        return normalized.replace("iw", "he", 1)
    if base == "tl":
        return normalized.replace("tl", "fil", 1)
    if normalized == "zh":
        return "zh-cn"
    return normalized
