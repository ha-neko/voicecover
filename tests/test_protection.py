from voicecover.nllb import _truncated
from voicecover.protect import protect, restore


def test_protect_extracts_arabic_spans() -> None:
    protected, tokens = protect("in which 3 are إسن and 3 are فعل and the rest")
    assert "إسن" not in protected
    assert "فعل" not in protected
    assert tokens == ["إسن", "فعل"]
    assert "zzzhold0zzz" in protected
    assert "zzzhold1zzz" in protected


def test_protect_keeps_harakat_and_spaces_inside_span() -> None:
    protected, tokens = protect("say صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ after that")
    assert tokens == ["صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ"]


def test_restore_puts_spans_back_after_loose_spacing() -> None:
    protected, tokens = protect("in which 3 are إسن and 3 are فعل")
    # NLLB sometimes re-spaces the placeholder
    translated = "di mana 3 adalah zzz hold 0 zzz dan 3 adalah zzzhold1zzz"
    assert restore(translated, tokens) == (
        "di mana 3 adalah إسن dan 3 adalah فعل"
    )


def test_restore_without_tokens_is_identity() -> None:
    assert restore("hello world", []) == "hello world"


def test_plain_english_passthrough_has_no_tokens() -> None:
    protected, tokens = protect("quick brown fox jumps over the lazy dog")
    assert protected.strip() == "quick brown fox jumps over the lazy dog"
    assert tokens == []


def test_miss_detection_truncated_output() -> None:
    assert _truncated("photosynthesis happens inside the leaves", "halo")


def test_miss_detection_keeps_full_output() -> None:
    assert not _truncated(
        "the quick brown fox jumps over the wall",
        "rubah coklat yang cepat melompati tembok dengan lincah sekali",
    )