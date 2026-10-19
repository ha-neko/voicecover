from voicecover.models import Segment
from voicecover.translation import _valid_translation, translate_segments


class FakeTranslator:
    def __init__(self, source: str, target: str):
        assert source == "en"
        assert target == "pt"

    def translate(self, text: str) -> str:
        return f"translated {text}"


def test_translate_segments_normalizes_locale() -> None:
    result = translate_segments(
        [Segment(0, 1, "hello")], "pt-BR", "en-US", FakeTranslator
    )
    assert result[0].translated_text == "translated hello"


class ChineseTranslator:
    def __init__(self, source: str, target: str):
        assert source == "en"
        assert target == "zh-TW"

    def translate(self, text: str) -> str:
        return "你好"


def test_translate_segments_preserves_chinese_variant() -> None:
    result = translate_segments(
        [Segment(0, 1, "hello")], "zh-TW", "en-US", ChineseTranslator
    )
    assert result[0].translated_text == "你好"


class MixedLanguageTranslator:
    def __init__(self, source: str, target: str):
        self.source = source

    def translate(self, text: str) -> str:
        if self.source != "auto":
            raise ValueError("mixed-language segment")
        return "hasil terjemahan"


def test_translation_retries_mixed_language_segment_with_auto_detection(monkeypatch) -> None:
    monkeypatch.setattr("voicecover.translation.time.sleep", lambda _: None)
    result = translate_segments(
        [Segment(0, 1, "نص mixed text")], "id", "ar", MixedLanguageTranslator
    )
    assert result[0].translated_text == "hasil terjemahan"


def test_provider_error_page_is_not_accepted_as_translation() -> None:
    assert not _valid_translation(
        "Error 500 (Server Error)!! That's an error. Please try again later."
    )
