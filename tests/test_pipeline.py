import json
from pathlib import Path

from voicecover import pipeline
from voicecover.models import Segment
from voicecover.visual_models import FrameSample, SlideState, VisualTrack


def test_translation_checkpoint_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "lesson.id.mp4"
    options = pipeline.Options(
        input=tmp_path / "lesson.mp4",
        output=output,
        source_language="en",
        target_language="id",
    )
    translated = [Segment(0.0, 2.0, "Welcome", "Selamat datang")]
    raw = [Segment(0.0, 1.0, "Wel"), Segment(1.0, 2.0, "come")]

    path = pipeline._write_translation_checkpoint(options, translated, raw, "en", "en")

    assert path == output.with_suffix(".checkpoint.json")
    assert pipeline._load_translation_checkpoint(options) == (translated, raw, "en", "en")


def test_youtube_source_cache_round_trip_and_url_validation(tmp_path: Path) -> None:
    downloaded = tmp_path / "download.mp4"
    downloaded.write_bytes(b"durable-youtube-video")
    output = tmp_path / "jobs" / "lesson.id.mp4"
    output.parent.mkdir()
    url = "https://youtu.be/example"

    cached = pipeline._write_youtube_source_cache(downloaded, output, url)

    assert cached.read_bytes() == downloaded.read_bytes()
    assert pipeline._load_youtube_source_cache(output, url) == cached
    assert pipeline._load_youtube_source_cache(output, "https://youtu.be/other") is None


def test_youtube_source_cache_rejects_truncated_file(tmp_path: Path) -> None:
    downloaded = tmp_path / "download.webm"
    downloaded.write_bytes(b"complete")
    output = tmp_path / "lesson.id.mp4"
    url = "https://youtu.be/example"
    cached = pipeline._write_youtube_source_cache(downloaded, output, url)
    cached.write_bytes(b"bad")

    assert pipeline._load_youtube_source_cache(output, url) is None


def test_visual_only_skips_audio_pipeline_and_preserves_rendered_video(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "lesson.mp4"
    source.write_bytes(b"source-video")
    output = tmp_path / "lesson.id.mp4"
    track = VisualTrack(
        frames=(FrameSample("frame-000001.jpg", 0.0, 960, 540),),
        slides=(SlideState(0, 0.0, 3.0, (0,), 0),),
        sample_interval=3.0,
    )
    monkeypatch.setattr(pipeline.media, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(pipeline, "_restore_visual_checkpoint", lambda *args: track)
    monkeypatch.setattr(
        pipeline.media,
        "extract_audio",
        lambda *args: (_ for _ in ()).throw(AssertionError("audio extraction must be skipped")),
    )

    def fake_replace(video, visual_track, translate, localized, work):
        assert visual_track == track
        Path(localized).write_bytes(b"localized-with-original-audio")
        return Path(localized)

    monkeypatch.setattr(pipeline, "replace_visual_text", fake_replace)

    result = pipeline.run(
        pipeline.Options(
            input=source,
            output=output,
            target_language="id",
            visual_only=True,
        )
    )

    assert result == output
    assert output.read_bytes() == b"localized-with-original-audio"
    assert output.with_suffix(".visual.json").is_file()


def test_sound_only_skips_visual_pipeline(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "lesson.mp4"
    source.write_bytes(b"source-video")
    output = tmp_path / "lesson.id.mp4"
    segment = Segment(0.0, 1.0, "Welcome", "Selamat datang")
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"wav")

    monkeypatch.setattr(pipeline.media, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(pipeline.media, "extract_audio", lambda video, path: Path(path).write_bytes(b"wav"))
    monkeypatch.setattr(
        pipeline,
        "_restore_visual_checkpoint",
        lambda *args: (_ for _ in ()).throw(AssertionError("visual processing must be skipped")),
    )
    monkeypatch.setattr(
        pipeline,
        "_load_translation_checkpoint",
        lambda options: ([segment], [segment], "en", "en"),
    )
    monkeypatch.setattr(pipeline, "choose_voice", lambda target, voice: "id-ID-test")
    monkeypatch.setattr(pipeline, "synthesize", lambda *args, **kwargs: [audio])
    monkeypatch.setattr(pipeline.media, "duration", lambda video: 1.0)
    monkeypatch.setattr(pipeline, "_build_timeline", lambda *args: audio)

    def fake_mux(video, dub, destination, volume):
        assert Path(video) == source
        Path(destination).write_bytes(b"dubbed-original-video")

    monkeypatch.setattr(pipeline.media, "mux", fake_mux)

    result = pipeline.run(
        pipeline.Options(
            input=source,
            output=output,
            target_language="id",
            sound_only=True,
            visual_context=True,
            localize_visual_text=True,
        )
    )

    assert result == output
    assert output.read_bytes() == b"dubbed-original-video"


def test_visual_translation_supports_automatic_source_detection(
    monkeypatch, tmp_path: Path
) -> None:
    captured = {}

    def fake_translate(texts, source_language, target_language):
        captured.update(source=source_language, target=target_language)
        return ["Halo"]

    monkeypatch.setattr("voicecover.llm.translate_texts", fake_translate)
    options = pipeline.Options(
        input=tmp_path / "lesson.mp4",
        output=tmp_path / "lesson.id.mp4",
        target_language="id",
    )

    translate = pipeline._visual_translator(options, "auto")

    assert translate(["Hello"]) == ["Halo"]
    assert captured == {"source": "auto", "target": "id"}
    assert options.output.with_suffix(".visual-translations.json").is_file()


def test_visual_translation_retries_batch_fallbacks_one_at_a_time(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_translate(texts, source_language, target_language):
        calls.append(list(texts))
        if len(texts) > 1:
            return list(texts)
        return [f"ID: {texts[0]}"]

    monkeypatch.setattr("voicecover.llm.translate_texts", fake_translate)
    options = pipeline.Options(
        input=tmp_path / "lesson.mp4",
        output=tmp_path / "lesson.id.mp4",
        target_language="id",
    )

    translated = pipeline._visual_translator(options, "en")(["Each page", "Four parts"])

    assert translated == ["ID: Each page", "ID: Four parts"]
    assert calls == [["Each page", "Four parts"], ["Each page"], ["Four parts"]]


def test_translation_checkpoint_rejects_different_target(tmp_path: Path) -> None:
    output = tmp_path / "lesson.id.mp4"
    path = output.with_suffix(".checkpoint.json")
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "stage": "translated",
                "target_language": "fr",
                "detected_language": "en",
                "source_language": "en",
                "segments": [
                    {
                        "start": 0,
                        "end": 1,
                        "text": "Welcome",
                        "translated_text": "Bienvenue",
                    }
                ],
                "asr_segments": [
                    {"start": 0, "end": 1, "text": "Welcome", "translated_text": None}
                ],
            }
        ),
        encoding="utf-8",
    )

    options = pipeline.Options(input="lesson.mp4", output=output, target_language="id")
    assert pipeline._load_translation_checkpoint(options) is None


def test_visual_checkpoint_restores_ocr_with_fresh_frame_paths(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "lesson.visual.json"
    cached = VisualTrack(
        frames=(FrameSample("frame-000001.jpg", 0.0, 960, 540),),
        slides=(SlideState(0, 0.0, 3.0, (0,), 0),),
        sample_interval=3.0,
    )
    checkpoint.write_text(cached.to_json(), encoding="utf-8")
    fresh_path = tmp_path / "work" / "frame-000001.jpg"
    monkeypatch.setattr(
        pipeline,
        "extract_frame_samples",
        lambda video, work, interval, width: [
            FrameSample(str(fresh_path), 0.0, 960, 540)
        ],
    )

    restored = pipeline._restore_visual_checkpoint(
        tmp_path / "lesson.mp4", checkpoint, tmp_path / "work", 3.0, 960
    )

    assert restored is not None
    assert restored.frames[0].path == str(fresh_path)
    assert restored.slides == cached.slides


def test_visual_checkpoint_rejects_changed_interval(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "lesson.visual.json"
    checkpoint.write_text(
        VisualTrack(
            frames=(FrameSample("frame.jpg", 0.0, 960, 540),), sample_interval=3.0
        ).to_json(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pipeline,
        "extract_frame_samples",
        lambda *args: (_ for _ in ()).throw(AssertionError("frames should not be extracted")),
    )

    assert (
        pipeline._restore_visual_checkpoint(
            tmp_path / "lesson.mp4", checkpoint, tmp_path / "work", 5.0, 960
        )
        is None
    )


def test_timeline_places_speech_at_natural_position(monkeypatch, tmp_path: Path) -> None:
    """Speech is placed at its timestamp start without trimming or speed changes."""
    speech = tmp_path / "speech.mp3"
    speech.touch()
    slots: list[float] = []

    monkeypatch.setattr(pipeline.media, "duration", lambda path: 5.5)
    monkeypatch.setattr(pipeline.media, "make_silence", lambda path, seconds: path.touch())
    def fit_audio(source, output, target_duration):
        slots.append(target_duration)
        output.touch()
    monkeypatch.setattr(pipeline.media, "fit_audio", fit_audio)
    monkeypatch.setattr(
        pipeline.media,
        "concatenate",
        lambda parts, manifest, output: output.touch(),
    )

    pipeline._build_timeline(
        [Segment(0, 2, "A longer translated sentence")],
        [speech],
        8,
        tmp_path,
    )

    assert slots == [8]


def test_timeline_short_speech_no_fit(monkeypatch, tmp_path: Path) -> None:
    """Short speech is placed at its timestamp without fitting to slot."""
    speech = tmp_path / "speech.mp3"
    speech.touch()
    slots: list[float] = []

    monkeypatch.setattr(pipeline.media, "duration", lambda path: 2.0)
    monkeypatch.setattr(pipeline.media, "make_silence", lambda path, seconds: path.touch())
    def fit_audio(source, output, target_duration):
        slots.append(target_duration)
        output.touch()
    monkeypatch.setattr(pipeline.media, "fit_audio", fit_audio)
    monkeypatch.setattr(
        pipeline.media,
        "concatenate",
        lambda parts, manifest, output: output.touch(),
    )

    pipeline._build_timeline(
        [Segment(1, 5, "Short sentence")],
        [speech],
        8,
        tmp_path,
    )

    assert slots == [7]


def test_mixed_content_preserves_arabic_for_arabic_voice() -> None:
    segments = [
        Segment(0, 1, "Welcome to the lesson"),
        Segment(1, 2, "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"),
    ]

    def translate(selected: list[Segment]) -> list[Segment]:
        assert [segment.text for segment in selected] == ["Welcome to the lesson"]
        return [selected[0].translated("Selamat datang di pelajaran")]

    result = pipeline._translate_with_preserved_arabic(
        segments, True, translate, progress=lambda message: None
    )
    assert result[0].translated_text == "Selamat datang di pelajaran"
    assert result[1].translated_text == "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"
    assert result[1].text == "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"


def test_mixed_english_arabic_utterance_translates_only_the_instruction() -> None:
    segments = [
        Segment(0, 1, "Welcome to the lesson"),
        Segment(1, 2, "for example فتحة نصرة ضربة سمعة"),
    ]

    def translate(selected: list[Segment]) -> list[Segment]:
        assert [segment.text for segment in selected] == [
            "Welcome to the lesson",
            "for example فتحة نصرة ضربة سمعة",
        ]
        return [
            selected[0].translated("Selamat datang di pelajaran"),
            selected[1].translated("contohnya فتحة نصرة ضربة سمعة"),
        ]

    result = pipeline._translate_with_preserved_arabic(
        segments, True, translate, progress=lambda message: None
    )
    assert result[0].translated_text == "Selamat datang di pelajaran"
    assert result[1].translated_text == "contohnya فتحة نصرة ضربة سمعة"


def test_arabic_source_is_still_translated() -> None:
    segment = Segment(0, 1, "مرحبا بكم")
    result = pipeline._translate_with_preserved_arabic(
        [segment],
        True,
        lambda selected: [selected[0].translated("Selamat datang")],
    )
    assert result[0].translated_text == "Selamat datang"


def test_auto_translator_falls_back_to_nllb_when_online_fails(monkeypatch) -> None:
    from voicecover.pipeline import Options, _run_translator

    segment = Segment(0, 1, "hello")
    options = Options(input="x.mp4", output="y.mp4", target_language="id")

    def google_fails(selected, target, source):
        raise RuntimeError("offline")

    def nllb_fallback(selected, source, target, auto_install, progress):
        return [selected[0].translated("halo")]

    monkeypatch.setattr("voicecover.pipeline.translate_segments", google_fails)
    monkeypatch.setattr("voicecover.pipeline.translate_segments_nllb", nllb_fallback)
    result = _run_translator("auto", options, "en", [segment])
    assert result[0].translated_text == "halo"


def test_group_utterances_joins_fragments_until_sentence_end() -> None:
    result = pipeline.group_utterances(
        [
            Segment(0, 1, "This is"),
            Segment(1.1, 2, "one sentence."),
            Segment(2.1, 3, "This is another."),
        ]
    )
    assert [(item.start, item.end, item.text) for item in result] == [
        (0, 2, "This is one sentence."),
        (2.1, 3, "This is another."),
    ]


def test_group_utterances_keeps_arabic_passage_separate() -> None:
    result = pipeline.group_utterances(
        [Segment(0, 1, "Read this"), Segment(1.1, 2, "بسم الله الرحمن الرحيم")]
    )
    assert len(result) == 2


def test_split_clone_phrases_uses_short_complete_target_phrases() -> None:
    segment = Segment(
        10, 22,
        "This is the first explanation. This is the second explanation.",
        "Ini penjelasan pertama. Ini penjelasan kedua.",
    )

    result = pipeline.split_clone_phrases([segment])

    assert len(result) == 2
    assert result[0].start == 10
    assert result[-1].end == 22
    assert result[0].end == result[1].start
    assert [item.translated_text for item in result] == [
        "Ini penjelasan pertama.",
        "Ini penjelasan kedua.",
    ]


def test_split_clone_phrases_preserves_arabic_passages() -> None:
    segment = Segment(0, 12, "بسم الله الرحمن الرحيم", "بسم الله الرحمن الرحيم")
    assert pipeline.split_clone_phrases([segment]) == [segment]


def test_merge_legacy_clone_fragments_repairs_synthetic_comma_split() -> None:
    segments = [
        Segment(0, 3, "hold it at,", "pegang dan tahan di,"),
        Segment(3, 6, "the marked point", "titik yang ditandai."),
        Segment(7, 8, "A genuine clause", "Klausa asli,"),
        Segment(8, 9, "continues here", "berlanjut di sini."),
    ]

    result = pipeline.merge_legacy_clone_fragments(segments)

    assert len(result) == 3
    assert result[0] == Segment(
        0, 6, "hold it at the marked point", "pegang dan tahan di titik yang ditandai."
    )
    assert result[1:] == segments[2:]
