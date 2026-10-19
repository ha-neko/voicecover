from voicecover.download import is_youtube_url


def test_detects_supported_youtube_urls() -> None:
    assert is_youtube_url("https://www.youtube.com/watch?v=abc")
    assert is_youtube_url("https://youtu.be/abc")
    assert is_youtube_url("https://m.youtube.com/shorts/abc")
    assert is_youtube_url("https://music.youtube.com/watch?v=abc")


def test_rejects_local_paths_and_lookalike_hosts() -> None:
    assert not is_youtube_url("video.mp4")
    assert not is_youtube_url("https://youtube.com.example.org/watch?v=abc")
