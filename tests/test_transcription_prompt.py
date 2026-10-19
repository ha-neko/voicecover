from pathlib import Path

from voicecover.pipeline import Options


def test_pipeline_options_accept_domain_prompt() -> None:
    options = Options(
        input="lesson.mp4",
        output=Path("lesson.id.mp4"),
        target_language="id",
        initial_prompt="English Quran lesson about Surah Al-Baqarah",
    )
    assert options.initial_prompt == "English Quran lesson about Surah Al-Baqarah"
