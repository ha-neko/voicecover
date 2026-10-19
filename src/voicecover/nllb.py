from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path

from .detect import detect_language
from .models import Segment
from .protect import protect, restore


MODEL_ID = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
LANGUAGE_TAGS = {
    "ar": "arb_Arab",
    "de": "deu_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "id": "ind_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ms": "zsm_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "th": "tha_Thai",
    "tl": "tgl_Latn",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "ur": "urd_Arab",
    "vi": "vie_Latn",
    "zh": "zho_Hans",
}

_MAX_WORDS_PER_INPUT = 60
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?؟。！？])\s+")


def language_tag(value: str) -> str:
    normalized = value.strip().replace("_", "-").lower()
    if normalized in {"zh-tw", "zh-hk", "zh-hant"}:
        return "zho_Hant"
    base = normalized.split("-")[0]
    base = {"iw": "he", "fil": "tl"}.get(base, base)
    try:
        return LANGUAGE_TAGS[base]
    except KeyError as exc:
        raise RuntimeError(f"NLLB language {value!r} is not configured in Voicecover") from exc


def model_directory() -> Path:
    override = os.environ.get("VOICECOVER_MODELS_DIR")
    if override:
        return Path(override).expanduser() / "nllb-600m-int8"
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return data_home / "voicecover" / "nllb-600m-int8"


def install_model(progress: Callable[[str], None] = print) -> Path:
    destination = model_directory()
    if (destination / "model.bin").is_file():
        return destination
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required to download NLLB") from exc
    progress("downloading the 645 MB offline NLLB translation model...")
    try:
        snapshot_download(
            MODEL_ID,
            local_dir=destination,
            allow_patterns=[
                "config.json",
                "model.bin",
                "sentencepiece.bpe.model",
                "shared_vocabulary.txt",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        )
    except Exception as exc:
        raise RuntimeError(f"Could not install the NLLB model: {exc}") from exc
    return destination


def source_tags(segments: list[Segment], source_language: str) -> list[str]:
    """Resolve a supported NLLB source tag for every segment.

    The ASR-detected language is only a hint: each utterance is detected
    independently so mixed-language videos route every segment correctly.
    """
    hint = None if source_language == "auto" else source_language
    tags: list[str] = []
    for segment in segments:
        iso = detect_language(segment.text, hint)
        try:
            tags.append(language_tag(iso))
        except RuntimeError:
            try:
                tags.append(language_tag(hint))
            except RuntimeError:
                tags.append("eng_Latn")
    return tags


def _flatten(texts: list[str]) -> tuple[list[str], list[list[int]]]:
    """Split oversized inputs into model-safe rows, keeping a row index per text."""
    rows: list[str] = []
    mapping: list[list[int]] = []
    for text in texts:
        if len(text.split()) <= _MAX_WORDS_PER_INPUT:
            row_index = len(rows)
            rows.append(text)
            mapping.append([row_index])
            continue
        pieces: list[str] = []
        current: list[str] = []
        word_count = 0
        for sentence in _SENTENCE_BOUNDARY.split(text):
            words = sentence.split()
            if not words:
                continue
            if word_count + len(words) > _MAX_WORDS_PER_INPUT and current:
                pieces.append(" ".join(current))
                current, word_count = [sentence], len(words)
            else:
                current.append(sentence)
                word_count += len(words)
        if current:
            pieces.append(" ".join(current))
        bounded: list[str] = []
        for piece in pieces:
            words = piece.split()
            while len(words) > _MAX_WORDS_PER_INPUT:
                bounded.append(" ".join(words[:_MAX_WORDS_PER_INPUT]))
                words = words[_MAX_WORDS_PER_INPUT:]
            if words:
                bounded.append(" ".join(words))
        start = len(rows)
        rows.extend(bounded)
        mapping.append(list(range(start, len(rows))))
    return rows, mapping


def _normalized(text: str) -> str:
    return " ".join(text.lower().replace("’", "'").split())


def _online_fallback(text: str, target_language: str) -> str | None:
    """Best-effort online retry for rows NLLB echoes back unchanged."""
    try:
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source="auto", target=target_language)
        result = translator.translate(text)
    except Exception:
        return None
    if not result or not result.strip():
        return None
    return result.strip()


def _truncated(source: str, translated: str) -> bool:
    """True when the output is far too short to carry the source meaning."""
    return len(translated.split()) < max(2, int(len(source.split()) * 0.35))


def translate_segments(
    segments: list[Segment],
    source_language: str = "auto",
    target_language: str = "id",
    auto_install: bool = True,
    progress: Callable[[str], None] = print,
) -> list[Segment]:
    target_tag = language_tag(target_language)
    tags = source_tags(segments, source_language)
    path = install_model(progress) if auto_install else model_directory()
    if not (path / "model.bin").is_file():
        raise RuntimeError("The offline NLLB model is not installed")
    try:
        import ctranslate2
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("NLLB requires ctranslate2 and sentencepiece") from exc

    sp = spm.SentencePieceProcessor(model_file=str(path / "sentencepiece.bpe.model"))
    translator = ctranslate2.Translator(
        str(path), device="cpu", compute_type="int8", inter_threads=8
    )
    translated_texts: list[str] = [""] * len(segments)
    start = 0
    while start < len(segments):
        end = start + 1
        while end < len(segments) and tags[end] == tags[start]:
            end += 1
        source_tag = tags[start]
        if source_tag == target_tag:
            for index in range(start, end):
                translated_texts[index] = segments[index].text
            start = end
            continue
        progress(
            f"  translating {end - start} utterance(s) ({source_tag} -> {target_tag})..."
        )
        rows, mapping = _flatten([segment.text for segment in segments[start:end]])
        # protect Arabic spans (doa, recitation) inside otherwise-translatable rows
        protected_rows: list[str] = []
        row_tokens: list[list[str]] = []
        for row in rows:
            protected, tokens = protect(row)
            protected_rows.append(protected)
            row_tokens.append(tokens)
        tokenized = [
            [source_tag] + sp.encode(row, out_type=str) + ["</s>"] for row in protected_rows
        ]
        results = translator.translate_batch(
            tokenized,
            target_prefix=[[target_tag] for _ in tokenized],
            beam_size=4,
            max_batch_size=32,
            batch_type="tokens",
        )
        decoded: list[str] = []
        for row, result in zip(protected_rows, results, strict=True):
            if not row.strip():
                decoded.append("")
                continue
            tokens = result.hypotheses[0]
            if tokens and tokens[0] == target_tag:
                tokens = tokens[1:]
            text = sp.decode_pieces(tokens).strip()
            if not text:
                raise RuntimeError("NLLB returned an empty translation")
            decoded.append(text)
        restored = [
            restore(text, tokens) for text, tokens in zip(decoded, row_tokens, strict=True)
        ]
        # NLLB echoes input, or truncates it badly, on hard rows.
        for row_index, (row, text) in enumerate(zip(rows, restored, strict=True)):
            echoed = source_tag != target_tag and row.strip() and _normalized(text) == _normalized(row)
            english_row = source_tag.startswith("eng")
            truncated = english_row and _truncated(row, text)
            if row.strip() and (echoed or truncated):
                fallback = _online_fallback(row, target_language)
                if fallback and (echoed or len(fallback.split()) >= len(text.split())):
                    restored[row_index] = fallback
                elif fallback is None:
                    progress(f"  note: utterance {row_index + 1} kept with words missing")
        for local_index, row_indices in enumerate(mapping):
            translated_texts[start + local_index] = " ".join(
                restored[row_index] for row_index in row_indices
            )
        start = end
    return [
        segment.translated(text or segment.text)
        for segment, text in zip(segments, translated_texts, strict=True)
    ]