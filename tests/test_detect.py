from voicecover.detect import detect_language, has_english_latin_text
from voicecover.models import Segment
from voicecover.nllb import _flatten, language_tag, source_tags


def test_arabic_script_is_detected() -> None:
    assert detect_language("بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ", "en") == "ar"


def test_script_overrides_whisper_hint() -> None:
    assert detect_language("مرحبا بكم في الدرس", "en") == "ar"


def test_english_stopwords_override_arabic_hint() -> None:
    assert detect_language("This is the best lesson you will ever watch", "ar") == "en"


def test_indonesian_stopwords_are_recognized() -> None:
    assert detect_language("ini adalah video yang sangat bagus untuk kalian", "en") == "id"


def test_ambiguous_utterance_falls_back_to_hint() -> None:
    assert detect_language("xyz qzw", "fr") == "fr"


def test_ambiguous_latin_text_ignores_arabic_hint() -> None:
    assert detect_language("page nine", "ar") == "en"


def test_english_is_found_inside_arabic_script() -> None:
    assert has_english_latin_text("For example هذا بيت كبير")
    assert not has_english_latin_text("هذا بيت كبير")
    assert not has_english_latin_text("wahaba yahabu hab")


def test_source_tags_route_mixed_content_per_segment() -> None:
    segments = [
        Segment(0, 1, "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"),
        Segment(1, 2, "Welcome to the lesson"),
        Segment(2, 3, "ini adalah teks Indonesia"),
        Segment(3, 4, "..."),
    ]
    assert source_tags(segments, "ar") == [
        "arb_Arab",
        "eng_Latn",
        "ind_Latn",
        "arb_Arab",
    ]


def test_flatten_splits_long_inputs_without_losing_text() -> None:
    text = " ".join(f"word{index}" for index in range(150))
    rows, mapping = _flatten([text])
    assert len(rows) > 1
    assert len(mapping[0]) == len(rows)
    assert " ".join(rows) == text


def test_flatten_short_input_stays_single_row() -> None:
    rows, mapping = _flatten(["hello world"])
    assert rows == ["hello world"]
    assert mapping == [[0]]


def test_language_tag_normalizes_locales() -> None:
    assert language_tag("ar-SA") == "arb_Arab"
    assert language_tag("zh-TW") == "zho_Hant"


def test_garbled_text_detects_mixed_script_hallucinations() -> None:
    from voicecover.models import is_garbled_text

    # tokens mixing scripts inside one word, or foreign scripts, are noise
    assert is_garbled_text("سمت أculture له وأ tocotه")
    assert is_garbled_text("فنحن مسلمون.Very simply.")
    assert is_garbled_text("Wa 나오legitsom Warahmatullahi")
    assert is_garbled_text("ت много тُفggervere")
    # legitimate text always separates scripts
    assert not is_garbled_text("Bacalah بِسْمِ اللَّهِ")
    assert not is_garbled_text("contohnya فَتَحَ يَفْتَحُ seperti وَعَدَ")
    assert not is_garbled_text("Belajar membaca Al-Qur'an dan Tajweed.")
    assert not is_garbled_text("???")
