from voicecover.languages import translation_language, tts_language, whisper_language


def test_provider_language_normalization() -> None:
    assert whisper_language("en-US") == "en"
    assert whisper_language("zh-CN") == "zh"
    assert translation_language("zh-TW") == "zh-TW"
    assert translation_language("he-IL") == "iw"
    assert translation_language("fil-PH") == "tl"
    assert tts_language("iw") == "he"
    assert tts_language("tl-PH") == "fil-ph"
