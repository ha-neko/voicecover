from voicecover.argos import language_code, translation_route


PACKAGES = [
    {"from_code": "ar", "to_code": "en"},
    {"from_code": "en", "to_code": "id"},
    {"from_code": "en", "to_code": "fr"},
]


def test_argos_language_codes() -> None:
    assert language_code("id-ID") == "id"
    assert language_code("pt-BR") == "pb"
    assert language_code("zh-TW") == "zt"


def test_argos_uses_english_pivot() -> None:
    assert translation_route("ar", "id", PACKAGES) == [("ar", "en"), ("en", "id")]


def test_argos_prefers_direct_route() -> None:
    assert translation_route("en", "id", PACKAGES) == [("en", "id")]
