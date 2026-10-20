from pathlib import Path

from voicecover.models import Segment
from voicecover import synthesis
from voicecover.clone import CloneConfig
from voicecover.synthesis import adaptive_rate, chunk_text, matching_voices, normalize_clone_text


def test_matching_voices_prefers_male_and_exact_locale() -> None:
    voices = [
        {"ShortName": "en-US-Female", "Locale": "en-US", "Gender": "Female"},
        {"ShortName": "en-GB-Male", "Locale": "en-GB", "Gender": "Male"},
        {"ShortName": "en-US-Male", "Locale": "en-US", "Gender": "Male"},
    ]
    assert [voice["ShortName"] for voice in matching_voices(voices, "en-US")] == [
        "en-US-Male",
        "en-US-Female",
    ]


def test_adaptive_rate_speeds_up_long_speech() -> None:
    assert adaptive_rate(6, 4) == "+20%"


def test_adaptive_rate_does_not_slow_short_speech() -> None:
    assert adaptive_rate(2, 4) is None


def test_adaptive_rate_ignores_small_difference() -> None:
    assert adaptive_rate(4.1, 4) is None


def test_chunk_text_preserves_every_word_and_punctuation() -> None:
    text = "Kalimat pertama, lengkap. بِسْمِ اللهِ الرَّحْمَنِ الرَّحِيمِ ثم lanjut lagi."
    chunks = chunk_text(text, max_bytes=34)
    assert len(chunks) > 1
    assert " ".join(chunks) == " ".join(text.split())
    assert all(len(chunk.encode("utf-8")) <= 34 for chunk in chunks)


def test_clone_preserves_original_arabic_audio(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[Path, Path, float, float]] = []
    monkeypatch.setattr(
        synthesis,
        "extract_audio_range",
        lambda source, output, start, end: calls.append((source, output, start, end)),
    )
    messages: list[str] = []
    source = tmp_path / "source.wav"
    segment = Segment(1.25, 4.5, "بسم الله الرحمن الرحيم")
    result = synthesis._preserve_original_one(
        3, segment, source, tmp_path, messages.append, [0, 1], synthesis.threading.Lock()
    )
    assert result == tmp_path / "speech-00003.wav"
    assert calls == [(source, result, 1.25, 4.5)]
    assert messages == ["  original Arabic 1/1 utterances"]


def test_normalize_clone_text_expands_numbers_and_inline_arabic() -> None:
    text = "3 adalah إسن, 3 adalah فعل مفعول, dan 95% adalah حروف."
    assert normalize_clone_text(text) == (
        "tiga adalah isim, tiga adalah fi'il maf'ul, dan "
        "sembilan puluh lima persen adalah huruf."
    )


def test_normalize_clone_text_pronounces_quran_and_mixed_terms() -> None:
    assert normalize_clone_text("Alquran: علم, ماضي مضاري أمر") == (
        "Al Kuran: allim, madhi, mudhari, amr"
    )


def test_normalize_clone_text_handles_written_quran_variants() -> None:
    assert normalize_clone_text("Al-Qur'an, al qur’an, dan Al Quran") == (
        "Al Kuran, Al Kuran, dan Al Kuran"
    )


def test_clone_jobs_are_balanced_by_text_workload() -> None:
    jobs = [(0, "a" * 100), (1, "b" * 80), (2, "c" * 40), (3, "d" * 30), (4, "e" * 20)]

    splits = synthesis._balanced_clone_splits(jobs, 2)
    weights = [sum(len(text) for _, text in bucket) for bucket in splits]

    assert sorted(index for bucket in splits for index, _ in bucket) == [0, 1, 2, 3, 4]
    assert max(weights) - min(weights) <= max(len(text) for _, text in jobs)


def test_speech_cache_rejects_changed_text_or_reference(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference-audio")
    config = CloneConfig(reference, "A complete reference sentence.")
    segments = [Segment(0, 2, "source", translated_text="Terjemahan pertama.")]
    keys = synthesis._speech_cache_keys(segments, config)
    generated = tmp_path / "generated.wav"
    generated.write_bytes(b"RIFF-valid-audio")
    cache = tmp_path / "cache"
    cache.mkdir()

    synthesis._copy_to_cache(generated, cache, 0, keys[0])

    assert synthesis._load_cached_speech(cache, keys) == {0: cache / "speech-00000.wav"}
    changed = [Segment(0, 2, "source", translated_text="Terjemahan berbeda.")]
    changed_keys = synthesis._speech_cache_keys(changed, config)
    assert synthesis._load_cached_speech(cache, changed_keys) == {}


def test_non_speech_placeholder_becomes_silence(monkeypatch, tmp_path: Path) -> None:
    import asyncio

    silences: list[float] = []
    monkeypatch.setattr(
        synthesis,
        "make_silence",
        lambda path, seconds: (silences.append(seconds), path.touch()),
    )
    messages: list[str] = []
    result = asyncio.run(
        synthesis._synthesize_one(
            7,
            Segment(635.66, 636.66, "???"),
            tmp_path,
            "id-ID-ArdiNeural",
            None,
            asyncio.Semaphore(1),
            messages.append,
            [0, 1],
            asyncio.Lock(),
        )
    )
    assert result == tmp_path / "speech-00007.wav"
    assert silences == [1.0]
    assert messages == ["  silence 1/1 utterances"]


def test_hallucinated_mixed_scripts_are_detected_as_unpronounceable() -> None:
    # Whisper hallucination from the batch: Arabic + Cyrillic + garbled Latin.
    foreign = synthesis.unpronounceable_characters(
        "ت много تُفggervere", "id-ID-ArdiNeural"
    )
    assert foreign == ["м", "н", "о", "г", "о"]


def test_lesson_text_with_arabic_stays_pronounceable() -> None:
    assert synthesis.unpronounceable_characters(
        "contohnya فَتَحَ يَفْتَحُ dan وَعَدَ", "id-ID-ArdiNeural"
    ) == []


def test_voice_locale_scripts_are_honored() -> None:
    assert synthesis.unpronounceable_characters("привет", "ru-RU-DmitryNeural") == []
    assert synthesis.unpronounceable_characters("hello", "ru-RU-DmitryNeural") == []


def test_hallucinated_segment_becomes_silence(monkeypatch, tmp_path: Path) -> None:
    import asyncio

    silences: list[float] = []
    monkeypatch.setattr(
        synthesis,
        "make_silence",
        lambda path, seconds: (silences.append(seconds), path.touch()),
    )
    messages: list[str] = []
    result = asyncio.run(
        synthesis._synthesize_one(
            9,
            Segment(835.72, 837.72, "ت много تُفggervere"),
            tmp_path,
            "id-ID-ArdiNeural",
            None,
            asyncio.Semaphore(1),
            messages.append,
            [0, 1],
            asyncio.Lock(),
        )
    )
    assert result == tmp_path / "speech-00009.wav"
    assert silences == [2.0]
    assert messages == ["  silence 1/1 utterances"]


def test_garbled_segment_keeps_original_source_audio(monkeypatch, tmp_path: Path) -> None:
    import asyncio

    calls: list[tuple[float, float]] = []
    monkeypatch.setattr(
        synthesis,
        "extract_audio_range",
        lambda source, output, start, end: (calls.append((start, end)), output.touch()),
    )
    messages: list[str] = []
    source = tmp_path / "source.wav"
    result = asyncio.run(
        synthesis._synthesize_one(
            74,
            Segment(504.4, 507.0, "سمت أculture له وأ tocوته"),
            tmp_path,
            "id-ID-ArdiNeural",
            "ar-SA-HamedNeural",
            asyncio.Semaphore(1),
            messages.append,
            [0, 1],
            asyncio.Lock(),
            source,
        )
    )
    assert result == tmp_path / "speech-00074.wav"
    assert calls == [(504.4, 507.0)]
    assert messages == ["  source audio 1/1 utterances"]


def test_garbled_segment_falls_back_to_silence_without_source(monkeypatch, tmp_path: Path) -> None:
    import asyncio

    silences: list[float] = []
    monkeypatch.setattr(
        synthesis,
        "make_silence",
        lambda path, seconds: (silences.append(seconds), path.touch()),
    )
    messages: list[str] = []
    result = asyncio.run(
        synthesis._synthesize_one(
            5,
            Segment(1.0, 4.0, "ف如此"),
            tmp_path,
            "id-ID-ArdiNeural",
            None,
            asyncio.Semaphore(1),
            messages.append,
            [0, 1],
            asyncio.Lock(),
        )
    )
    assert result == tmp_path / "speech-00005.wav"
    assert silences == [3.0]
    assert messages == ["  silence 1/1 utterances"]


def test_overlong_speech_regenerates_at_faster_rate(monkeypatch, tmp_path: Path) -> None:
    """Speech longer than its slot must be redone with an Edge TTS rate."""
    import asyncio
    import sys
    import types

    rates: list[str | None] = []

    class FakeCommunicate:
        def __init__(self, text, voice, rate=None, **kwargs):
            rates.append(rate)
            self._path = None

        def _bind(self, path):
            self._path = path
            return self

        async def save(self, path):
            Path(path).write_bytes(b"x" * 300)

    fake_edge = types.SimpleNamespace(
        Communicate=lambda text, voice, rate=None, **kwargs: FakeCommunicate(
            text, voice, rate
        )._bind(None)
    )
    monkeypatch.setitem(sys.modules, "edge_tts", fake_edge)
    # every duration read reports 6s of speech for a 4s slot -> +20% tempo
    monkeypatch.setattr(synthesis, "duration", lambda path: 6.0)
    monkeypatch.setattr(
        synthesis,
        "decode_speech",
        lambda raw, wav: Path(wav).write_bytes(b"y" * 300),
    )
    messages: list[str] = []
    result = asyncio.run(
        synthesis._synthesize_one(
            3,
            Segment(0.0, 4.0, "Belajar membaca Al-Qur'an dan tajwid."),
            tmp_path,
            "id-ID-ArdiNeural",
            None,
            asyncio.Semaphore(1),
            messages.append,
            [0, 1],
            asyncio.Lock(),
        )
    )
    assert result == tmp_path / "speech-00003.wav"
    assert rates == [None, "+20%"]
    assert messages == ["  voice 1/1 utterances"]


def test_on_tempo_speech_is_not_regenerated(monkeypatch, tmp_path: Path) -> None:
    import asyncio
    import sys
    import types

    rates: list[str | None] = []

    class FakeCommunicate:
        def __init__(self, text, voice, rate=None, **kwargs):
            rates.append(rate)

        async def save(self, path):
            Path(path).write_bytes(b"x" * 300)

    fake_edge = types.SimpleNamespace(
        Communicate=lambda text, voice, rate=None, **kwargs: FakeCommunicate(
            text, voice, rate
        )
    )
    monkeypatch.setitem(sys.modules, "edge_tts", fake_edge)
    # 3.5s of speech fits its 4s slot: single pass, no rate
    monkeypatch.setattr(synthesis, "duration", lambda path: 3.5)
    monkeypatch.setattr(
        synthesis,
        "decode_speech",
        lambda raw, wav: Path(wav).write_bytes(b"y" * 300),
    )
    messages: list[str] = []
    asyncio.run(
        synthesis._synthesize_one(
            4,
            Segment(0.0, 4.0, "Belajar membaca Al-Qur'an dan tajwid."),
            tmp_path,
            "id-ID-ArdiNeural",
            None,
            asyncio.Semaphore(1),
            messages.append,
            [0, 1],
            asyncio.Lock(),
        )
    )
    assert rates == [None]
