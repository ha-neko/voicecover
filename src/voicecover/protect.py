from __future__ import annotations

import re

# Arabic script block plus tatweel and ZWNJ/ZWJ.
_ARABIC_BLOCK = "\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\u200c\u200d"
# a span must start and end with an Arabic character; whitespace is only
# allowed between Arabic characters, so plain spaces never match on their own.
_ARABIC_SPAN = re.compile(f"[{_ARABIC_BLOCK}](?:[{_ARABIC_BLOCK}\\s]*[{_ARABIC_BLOCK}])?")

_PLACEHOLDER = re.compile(r"zzz\s*hold\s*(\d+)\s*zzz", re.IGNORECASE)


def protect(text: str) -> tuple[str, list[str]]:
    """Replace Arabic spans with numbered placeholders before translation.

    Returns the protected text plus the original spans, in index order.
    Used so recitations and prayers embedded in a foreign sentence survive
    translation byte-for-byte instead of being transliterated or dropped.
    """
    tokens: list[str] = []

    def replace(match: re.Match[str]) -> str:
        tokens.append(match.group(0).strip())
        return f" zzzhold{len(tokens) - 1}zzz "

    return _ARABIC_SPAN.sub(replace, text), tokens


def restore(text: str, tokens: list[str]) -> str:
    """Put the protected Arabic spans back into a translated string."""
    if not tokens:
        return text

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if 0 <= index < len(tokens):
            return tokens[index]
        return match.group(0)

    return _PLACEHOLDER.sub(replace, text)