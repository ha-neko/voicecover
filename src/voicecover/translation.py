from __future__ import annotations

import time
from collections.abc import Callable

from .languages import translation_language
from .models import Segment
from .protect import protect, restore


def _valid_translation(text: object) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    lowered = text.lower().replace("’", "'")
    error_markers = (
        "error 500",
        "that's an error",
        "there was an error. please try again later",
        "service unavailable",
        "too many requests",
    )
    return not any(marker in lowered for marker in error_markers)


def _translate_with_retry(translator: object, text: str, attempts: int = 4) -> str:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            result = translator.translate(text)  # type: ignore[attr-defined]
            if _valid_translation(result):
                return result
            last_error = RuntimeError("translation provider returned an error response")
        except Exception as exc:
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(0.75 * (attempt + 1))
    assert last_error is not None
    raise last_error


def translate_segments(
    segments: list[Segment],
    target_language: str,
    source_language: str = "auto",
    translator_factory: Callable[..., object] | None = None,
) -> list[Segment]:
    if translator_factory is None:
        try:
            from deep_translator import GoogleTranslator
        except ImportError as exc:
            raise RuntimeError("deep-translator is not installed") from exc
        translator_factory = GoogleTranslator

    target = translation_language(target_language)
    source = "auto" if source_language == "auto" else translation_language(source_language)
    try:
        translator = translator_factory(source=source, target=target)
    except Exception as exc:
        raise RuntimeError(
            f"Could not initialize translation for {source!r} -> {target!r}: {exc}"
        ) from exc
    translated: list[Segment] = []
    automatic_translator = None
    for segment in segments:
        protected, tokens = protect(segment.text)
        try:
            text = _translate_with_retry(translator, protected)
        except Exception as exc:
            if source == "auto":
                raise RuntimeError(f"Translation failed: {exc}") from exc
            try:
                if automatic_translator is None:
                    automatic_translator = translator_factory(source="auto", target=target)
                text = _translate_with_retry(automatic_translator, protected)
            except Exception as retry_exc:
                raise RuntimeError(
                    f"Translation failed with source {source!r} and auto-detection: {retry_exc}"
                ) from retry_exc
        translated.append(segment.translated(restore(str(text), tokens)))
        # The public endpoint becomes unreliable under a tight request loop.
        time.sleep(0.05)
    return translated
