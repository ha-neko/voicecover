from __future__ import annotations

import unicodedata
from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class Segment:
    start: float
    end: float
    text: str
    translated_text: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def translated(self, text: str) -> "Segment":
        return replace(self, translated_text=text)


def is_arabic_passage(text: str, threshold: float = 0.5) -> bool:
    """Return true when Arabic-script letters dominate a text segment."""
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False
    arabic = sum("ARABIC" in unicodedata.name(character, "") for character in letters)
    return arabic >= 2 and arabic / len(letters) >= threshold


def has_latin_letters(text: str) -> bool:
    """Return true when the text contains any Latin-script letter."""
    return any("LATIN" in unicodedata.name(character, "") for character in text if character.isalpha())


def is_garbled_text(text: str) -> bool:
    """Detect ASR/LLM hallucination noise inside otherwise normal speech.

    Legitimate transliteration always separates the scripts ("fataha فَتَحَ"),
    so a single token mixing Arabic and Latin letters ("أculture", "تُفggervere")
    or any token using an unrelated script (Cyrillic, Hangul, kana...) inside a
    Latin/Arabic lesson is noise. Such segments should keep the original
    recording instead of being sent to TTS, which garbles them audibly.
    """
    for token in text.split():
        arabic = latin = foreign = False
        for character in token:
            if not character.isalpha():
                continue
            try:
                name = unicodedata.name(character)
            except ValueError:
                continue
            if name.startswith("ARABIC"):
                arabic = True
            elif name.startswith("LATIN"):
                latin = True
            else:
                foreign = True
        if foreign or (arabic and latin):
            return True
    return False
