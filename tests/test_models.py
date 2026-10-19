from voicecover.models import Segment


def test_segment_translation_is_immutable() -> None:
    original = Segment(1, 3.5, "hello")
    translated = original.translated("hola")

    assert original.translated_text is None
    assert translated.translated_text == "hola"
    assert translated.duration == 2.5
