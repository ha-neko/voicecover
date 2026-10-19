from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .detect import has_english_latin_text
from .models import Segment


DEFAULT_ENDPOINT = "http://127.0.0.1:8081/v1/chat/completions"
DEFAULT_MODEL = "local-model"
_LOG = logging.getLogger("voicecover.llm")
_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_CLEAN_TEXT = re.compile(r"[\x00-\x08\x0b-\x1c\x1e-\x1f\x7f\ufeff\ufffd]+")
_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "en": "English",
    "id": "Bahasa Indonesia (Indonesian)",
    "ms": "Bahasa Melayu (Malay)",
}


class LLMTranslationError(RuntimeError):
    """Raised when the local chat-completions translator cannot return valid output."""


def _visual_messages(
    texts: list[str],
    source_language: str,
    target_language: str,
) -> list[dict[str, str]]:
    source_code = source_language.lower().replace("_", "-").split("-")[0]
    target_code = target_language.lower().replace("_", "-").split("-")[0]
    source_name = _LANGUAGE_NAMES.get(source_code, source_language)
    target_name = _LANGUAGE_NAMES.get(target_code, target_language)
    system = (
        f"Translate each short text from {source_name} to {target_name}. "
        "These are on-screen slide labels that must fit back into the original text boxes. "
        "Preserve Arabic text exactly. Use concise, natural wording: no explanation, no added "
        "context, and normally no more than the max_chars supplied for that item. Prefer a "
        "short familiar phrase over a literal long translation. Return only a strict JSON object: "
        '{"translations":[{"id":0,"text":"..."}]}. One entry per input, same IDs, no extras.'
    )
    user = json.dumps(
        [
            {
                "id": i,
                "text": text,
                # Indonesian often needs modest expansion, but unbounded prose
                # cannot be made legible inside a small source label.
                "max_chars": max(len(text), round(len(text) * 1.35)),
            }
            for i, text in enumerate(texts)
        ],
        ensure_ascii=False,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def translate_texts(
    texts: list[str],
    source_language: str = "auto",
    target_language: str = "id",
    endpoint: str | None = None,
    model: str | None = None,
    timeout: float = 3600.0,
) -> list[str]:
    """Translate plain text strings through the local LLM, batched into small groups.
    On failure, falls back to the original text for that batch."""
    if not texts:
        return []
    endpoint = endpoint or os.environ.get("VOICECOVER_LLM_URL", DEFAULT_ENDPOINT)
    model = model or os.environ.get("VOICECOVER_LLM_MODEL", DEFAULT_MODEL)
    results: dict[int, str] = {}
    for i in range(0, len(texts), 5):
        chunk = texts[i:i + 5]
        expected_ids = list(range(len(chunk)))
        messages = _visual_messages(chunk, source_language, target_language)
        content = ""
        try:
            content = _request(endpoint, model, messages, timeout)
            _LOG.info("visual batch %d raw (first 200): %r", i, content[:200])
            result = _validated_translations(content, expected_ids)
        except (LLMTranslationError, ValueError) as first_error:
            _LOG.warning("visual batch %d failed: %s", i, first_error)
            try:
                correction = messages + [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            f"Your response was invalid: {first_error}. Correct it now. "
                            f"Return only JSON with exactly these IDs: {expected_ids}."
                        ),
                    },
                ]
                corrected = _request(endpoint, model, correction, timeout)
                _LOG.info("visual batch %d correction (first 200): %r", i, corrected[:200])
                result = _validated_translations(corrected, expected_ids)
            except Exception as second_error:
                _LOG.warning("visual batch %d fallback to originals: %s", i, second_error)
                for idx, text in enumerate(chunk):
                    results[i + idx] = text
                continue
        for idx, text in result.items():
            results[i + idx] = text
    return [results.get(j, texts[j]) for j in range(len(texts))]


def _batches(
    segments: list[Segment], character_budget: int, max_items: int = 10
) -> list[list[tuple[int, Segment]]]:
    if character_budget <= 0:
        raise ValueError("character_budget must be greater than zero")
    if max_items <= 0:
        raise ValueError("max_items must be greater than zero")
    batches: list[list[tuple[int, Segment]]] = []
    current: list[tuple[int, Segment]] = []
    size = 0
    for item in enumerate(segments):
        text_size = len(item[1].text)
        if current and (size + text_size > character_budget or len(current) >= max_items):
            batches.append(current)
            current, size = [], 0
        current.append(item)
        size += text_size
    if current:
        batches.append(current)
    return batches


def _messages(
    batch: list[tuple[int, Segment]],
    source_language: str,
    target_language: str,
    protected_terms: Mapping[str, str],
) -> list[dict[str, str]]:
    source_code = source_language.lower().replace("_", "-").split("-")[0]
    target_code = target_language.lower().replace("_", "-").split("-")[0]
    source_name = _LANGUAGE_NAMES.get(source_code, source_language)
    target_name = _LANGUAGE_NAMES.get(target_code, target_language)
    inputs = [
        {
            "id": index,
            "start": segment.start,
            "end": segment.end,
            "duration": segment.duration,
            # Whisper occasionally emits control characters or replacement
            # glyphs that derail JSON generation; normalize them away.
            "text": _CLEAN_TEXT.sub(" ", segment.text).strip(),
        }
        for index, segment in batch
    ]
    target_guidance = (
        "Use natural Indonesian educational localization, not rigid word-for-word translation."
        if target_code in {"id", "indonesian", "bahasa indonesia"}
        else f"Use natural educational localization appropriate for {target_name}."
    )
    system = (
        "You are a careful audiovisual translator. "
        f"Translate from {source_name} to {target_name}. Every non-Arabic source sentence must be "
        f"written in {target_name}, never copied in {source_name}. {target_guidance} "
        "Preserve every Arabic scripture passage exactly, including its spelling and diacritics. "
        "Preserve or render the protected terms exactly as specified. Keep each translation concise "
        "enough to be spoken during that segment's duration. Use the complete chronological batch as "
        "local context, but return one translation for each input ID. Return only a strict JSON object "
        'with this shape: {"translations":[{"id":0,"text":"..."}]}. Do not add or omit IDs.'
    )
    user = (
        "Protected terms (source term to required rendering):\n"
        f"{json.dumps(dict(protected_terms), ensure_ascii=False, sort_keys=True)}\n"
        "Chronological segments with IDs, timings, and full local context:\n"
        f"{json.dumps(inputs, ensure_ascii=False)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _request(endpoint: str, model: str, messages: list[dict[str, str]], timeout: float) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 8192,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        response = urlopen(request, timeout=timeout)
        raw = response.read()
        close = getattr(response, "close", None)
        if close is not None:
            close()
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace").strip()
        except OSError:
            detail = ""
        suffix = f": {detail[:500]}" if detail else ""
        raise LLMTranslationError(
            f"Local LLM server at {endpoint} returned HTTP {exc.code}: {exc.reason}{suffix}"
        ) from exc
    except (URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise LLMTranslationError(f"Could not reach local LLM server at {endpoint}: {reason}") from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise LLMTranslationError(
            f"Local LLM server at {endpoint} returned a malformed chat-completions response"
        ) from exc
    if not isinstance(content, str):
        raise LLMTranslationError(
            f"Local LLM server at {endpoint} returned non-text message content"
        )
    return content


def _json_object(content: str) -> dict[str, Any]:
    # strip<think>...</think>blocks (Qwen reasoning output)
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    fenced = _FENCED_JSON.search(cleaned)
    candidate = fenced.group(1).strip() if fenced else cleaned
    decoder = json.JSONDecoder()
    starts = [0] if candidate.startswith("{") else []
    starts.extend(index for index, character in enumerate(candidate) if character == "{")
    for start in dict.fromkeys(starts):
        try:
            value, _ = decoder.raw_decode(candidate, start)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("response content does not contain a JSON object")


def _validated_translations(content: str, expected_ids: list[int]) -> dict[int, str]:
    value = _json_object(content)
    if "translations" not in value or not isinstance(value["translations"], list):
        raise ValueError("JSON must contain a translations array")
    extra = sorted(set(value) - {"translations"})
    if extra:
        # Qwen sometimes wraps the array in a verbose object. Extra sibling
        # keys are harmless as long as the translations themselves are valid.
        _LOG.warning("ignoring unexpected keys from local LLM: %s", extra)
    translations: dict[int, str] = {}
    for item in value["translations"]:
        if not isinstance(item, dict) or not {"id", "text"}.issubset(item):
            raise ValueError("each translation must contain id and text")
        identifier, text = item["id"], item["text"]
        if not isinstance(identifier, int) or isinstance(identifier, bool):
            raise ValueError("translation IDs must be integers")
        if identifier in translations:
            raise ValueError(f"duplicate translation ID {identifier}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"translation {identifier} is empty")
        translations[identifier] = text.strip()
    if set(translations) != set(expected_ids) or len(translations) != len(expected_ids):
        missing = sorted(set(expected_ids) - set(translations))
        unexpected = sorted(set(translations) - set(expected_ids))
        raise ValueError(f"translation IDs do not match inputs (missing={missing}, unexpected={unexpected})")
    return translations


def _duplicate_target_indices(
    segments: list[Segment], translated: Mapping[int, str]
) -> list[int]:
    """Find different sources that received identical translations.

    The local LLM occasionally labels a batch's translations with the wrong
    IDs, shifting every entry in it. The shift is invisible to ID validation,
    but it always leaves one target duplicated across sources that clearly
    differ, which exposes the whole batch.
    """
    by_target: dict[str, list[int]] = {}
    for index, text in translated.items():
        by_target.setdefault(re.sub(r"\W+", "", text).casefold(), []).append(index)
    suspects: list[int] = []
    for indexes in by_target.values():
        if len(indexes) < 2:
            continue
        sources = {
            re.sub(r"\W+", "", segments[index].text).casefold() for index in indexes
        }
        if len(sources) > 1:
            suspects.extend(indexes)
    return sorted(suspects)


def translate_segments(
    segments: list[Segment],
    source_language: str = "auto",
    target_language: str = "id",
    protected_terms: Mapping[str, str] | None = None,
    character_budget: int = 2400,
    max_batch_items: int = 10,
    endpoint: str | None = None,
    model: str | None = None,
    timeout: float = 3600.0,
) -> list[Segment]:
    """Translate segments through a local OpenAI-compatible chat-completions server."""
    if not segments:
        return []
    endpoint = endpoint or os.environ.get("VOICECOVER_LLM_URL", DEFAULT_ENDPOINT)
    model = model or os.environ.get("VOICECOVER_LLM_MODEL", DEFAULT_MODEL)
    terms = protected_terms or {}
    translated: dict[int, str] = {}

    def untranslated_ids(
        batch: list[tuple[int, Segment]], result: Mapping[int, str]
    ) -> list[int]:
        if target_language.lower().replace("_", "-").split("-")[0] != "id":
            return []
        return [
            index
            for index, segment in batch
            if has_english_latin_text(segment.text)
            and has_english_latin_text(result[index])
        ]

    def translate_batch(
        batch: list[tuple[int, Segment]], *, semantic_retry: bool = False
    ) -> dict[int, str]:
        expected_ids = [index for index, _ in batch]
        messages = _messages(batch, source_language, target_language, terms)
        if semantic_retry:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Translate every English instruction and clause into natural Indonesian. "
                        "Keep only Arabic scripture, Arabic grammatical forms, names, and explicitly "
                        "protected terms unchanged. The answer must contain no leftover English prose."
                    ),
                }
            )
        content = _request(endpoint, model, messages, timeout)
        # A single stubborn item gets extra attempts: temperature sampling means
        # each retry is a fresh chance to emit the expected shape.
        attempts = 3 if len(batch) == 1 else 2
        conversation = messages
        result: dict[int, str] | None = None
        validation_error: ValueError | None = None
        for _attempt in range(attempts):
            try:
                result = _validated_translations(content, expected_ids)
                break
            except ValueError as error:
                validation_error = error
                _LOG.warning(
                    "invalid LLM translation response for IDs %s (%s): %.400s",
                    expected_ids,
                    error,
                    content,
                )
                conversation = conversation + [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            f"Your response was invalid: {error}. Correct it now. Return only the "
                            f'required JSON object: {{"translations":[{{"id":<id>,"text":"<translation>"}}]}} '
                            f"with exactly these IDs once each: {expected_ids}."
                        ),
                    },
                ]
                content = _request(endpoint, model, conversation, timeout)
        if result is None:
            assert validation_error is not None
            if len(batch) > 1:
                midpoint = len(batch) // 2
                _LOG.warning(
                    "speech translation batch with %d items remained invalid; "
                    "retrying as %d + %d items",
                    len(batch),
                    midpoint,
                    len(batch) - midpoint,
                )
                return {
                    **translate_batch(batch[:midpoint]),
                    **translate_batch(batch[midpoint:]),
                }
            raise LLMTranslationError(
                f"Local LLM returned invalid translations after repeated corrections: "
                f"{validation_error}"
            ) from validation_error
        bad = untranslated_ids(batch, result)
        if bad:
            if semantic_retry:
                raise LLMTranslationError(
                    f"Local LLM left English untranslated for IDs {bad} after a focused retry"
                )
            repaired = dict(result)
            by_id = dict(batch)
            for identifier in bad:
                repaired.update(
                    translate_batch([(identifier, by_id[identifier])], semantic_retry=True)
                )
            return repaired
        return result

    for batch in _batches(segments, character_budget, max_batch_items):
        translated.update(translate_batch(batch))

    for index in _duplicate_target_indices(segments, translated):
        # Re-ask one segment at a time so the model cannot shift IDs again
        # across a whole batch; single-item calls get up to three attempts.
        _LOG.warning(
            "segment %d received a translation duplicated from another source; "
            "re-translating it individually",
            index,
        )
        translated.update(translate_batch([(index, segments[index])]))

    return [segment.translated(translated[index]) for index, segment in enumerate(segments)]
