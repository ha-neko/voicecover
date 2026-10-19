from voicecover.models import Segment
from voicecover.subtitles import write_srt


def test_write_srt_uses_translated_text(tmp_path) -> None:
    output = tmp_path / "captions.srt"
    write_srt([Segment(1.25, 62.5, "hello", "hola")], output)

    assert output.read_text() == "1\n00:00:01,250 --> 00:01:02,500\nhola\n"


def test_write_srt_prefers_translation_for_arabic_original(tmp_path) -> None:
    output = tmp_path / "captions.srt"
    write_srt(
        [Segment(0.0, 1.0, "مرحبا بكم", "selamat datang"), Segment(1.0, 2.0, "hi", None)],
        output,
    )
    content = output.read_text()
    assert "selamat datang" in content
    assert "مرحبا بكم" not in content
    assert "hi" in content


def test_write_srt_bilingual_shows_original_and_translation(tmp_path) -> None:
    output = tmp_path / "captions.srt"
    write_srt(
        [Segment(0.0, 1.0, "welcome", "selamat datang"), Segment(1.0, 2.0, "luar biasa", None)],
        output,
        bilingual=True,
    )
    content = output.read_text()
    assert "welcome\nselamat datang" in content
    assert "luar biasa" in content
    assert "1\n00:00:00,000 --> 00:00:01,000\nwelcome\nselamat datang" in content
