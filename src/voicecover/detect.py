from __future__ import annotations

import re
import unicodedata


_NON_LATIN_SCRIPTS = {
    "ar": ("ARABIC",),
    "ru": ("CYRILLIC",),
    "he": ("HEBREW",),
    "hi": ("DEVANAGARI",),
    "ko": ("HANGUL",),
    "zh": ("CJK",),
    "ja": ("HIRAGANA", "KATAKANA"),
    "th": ("THAI",),
    "el": ("GREEK",),
}

_LATIN_STOPWORDS = {
    "en": {
        "the", "and", "is", "are", "of", "to", "in", "that", "it", "for", "with",
        "you", "this", "have", "be", "was", "on", "not", "but", "what", "we",
        "they", "so", "do", "don't", "don", "can", "will", "just", "like", "about",
        "he", "she", "his", "her", "does", "did", "say", "means", "example",
        "repeat", "after", "now", "right", "from", "all", "me", "side",
    },
    "id": {
        "yang", "di", "ke", "dan", "ini", "itu", "aku", "saya", "kamu", "anda",
        "kita", "kami", "tidak", "ada", "dengan", "dari", "untuk", "pada", "adalah",
        "sebuah", "sudah", "akan", "bisa", "juga", "karena", "atau", "kalau",
        "jangan", "kenapa", "apa", "siapa", "banyak", "sangat", "hanya", "lagi",
    },
    "es": {
        "el", "la", "los", "las", "de", "que", "y", "en", "un", "una", "es", "por",
        "con", "para", "del", "como", "más", "ser", "su", "no", "lo", "se", "este",
    },
    "fr": {
        "le", "la", "les", "de", "des", "du", "et", "en", "un", "une", "est", "que",
        "pour", "dans", "avec", "pas", "sur", "il", "elle", "nous", "vous", "c'est",
        "plus", "tout", "mais", "aussi",
    },
    "de": {
        "der", "die", "das", "und", "ist", "ein", "eine", "zu", "mit", "den", "für",
        "von", "auf", "nicht", "ich", "du", "sie", "wir", "das", "auch", "sich",
        "dass", "als", "es",
    },
    "pt": {
        "o", "a", "os", "as", "de", "do", "da", "que", "e", "em", "um", "uma",
        "é", "para", "com", "por", "não", "no", "na", "ser", "mais", "este",
    },
    "it": {
        "il", "lo", "la", "i", "gli", "le", "di", "che", "e", "in", "un", "una",
        "è", "per", "con", "non", "sono", "questo", "qui", "anche", "se",
    },
    "nl": {
        "de", "het", "een", "en", "van", "te", "in", "dat", "je", "is", "niet",
        "voor", "op", "met", "aan", "ook", "maar", "geen", "als",
    },
    "tr": {
        "bir", "bu", "ve", "için", "ile", "ben", "sen", "o", "değil", "çok",
        "şey", "biz", "siz", "onlar", "gibi", "daha", "sonra", "ama",
    },
    "tl": {
        "ang", "ng", "sa", "ako", "mo", "po", "sila", "kami", "kaniya", "hindi",
        "para", "mga", "ito", "iyan", "kay", "namin", "ninyo", "wala", "may",
    },
    "vi": {
        "và", "là", "có", "không", "của", "người", "một", "những", "cho",
        "được", "trong", "với", "này", "bạn", "tôi", "chúng", "các", "đây",
        "cũng", "nếu", "rồi", "thì",
    },
}

_WORD_TOKEN = re.compile(r"\w+", re.UNICODE)

_LATIN_LANGUAGES = frozenset(_LATIN_STOPWORDS) | {"ms", "pl"}


def _script_language(text: str) -> str | None:
    """Return the language for a clearly dominant non-Latin script, if any."""
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return None
    counts: dict[str, int] = {}
    for character in letters:
        name = unicodedata.name(character, "")
        for language, markers in _NON_LATIN_SCRIPTS.items():
            if any(marker in name for marker in markers):
                counts[language] = counts.get(language, 0) + 1
                break
        else:
            counts["la"] = counts.get("la", 0) + 1
    dominant, count = max(counts.items(), key=lambda entry: entry[1])
    if dominant != "la" and count / len(letters) >= 0.5:
        return dominant
    return None


def detect_language(text: str, hint: str | None = None) -> str:
    """Detect the language of a short utterance without external models."""
    script = _script_language(text)
    if script is not None:
        return script
    tokens = [match.group(0).lower() for match in _WORD_TOKEN.finditer(text)]
    if not tokens:
        return hint or "en"
    scores = {
        language: sum(token in words for token in tokens) / len(tokens)
        for language, words in _LATIN_STOPWORDS.items()
    }
    best, best_score = max(scores.items(), key=lambda entry: entry[1])
    runner_up = sorted(scores.values())[-2] if len(scores) > 1 else 0.0
    if best_score >= 0.1 and best_score - runner_up >= 0.03:
        return best
    # Latin-script text can never be Arabic: an Arabic hint must not win here.
    if hint in _LATIN_LANGUAGES:
        return hint
    return "en"


def has_english_latin_text(text: str) -> bool:
    """Detect meaningful English inside Latin or mixed-script lesson text."""
    latin = " ".join(re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text))
    if not latin:
        return False
    return detect_language(latin, "id") == "en"
