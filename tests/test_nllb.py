import pytest

from voicecover.nllb import language_tag


def test_nllb_language_tags() -> None:
    assert language_tag("ar-SA") == "arb_Arab"
    assert language_tag("id-ID") == "ind_Latn"
    assert language_tag("zh-TW") == "zho_Hant"


def test_unknown_nllb_language_is_clear() -> None:
    with pytest.raises(RuntimeError, match="not configured"):
        language_tag("xx")
