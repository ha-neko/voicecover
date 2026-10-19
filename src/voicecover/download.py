from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
    "youtu.be",
}


def is_youtube_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and (parsed.hostname or "").lower() in YOUTUBE_HOSTS


def download_youtube(url: str, output_dir: Path) -> Path:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is not installed") from exc

    template = str(output_dir / "youtube.%(ext)s")
    options = {
        "format": "bv*+ba/b",
        "outtmpl": template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
            prepared = Path(downloader.prepare_filename(info))
    except Exception as exc:
        raise RuntimeError(f"YouTube download failed: {exc}") from exc

    if prepared.is_file():
        return prepared
    candidates = [
        path
        for path in output_dir.glob("youtube.*")
        if path.is_file() and path.suffix not in {".part", ".ytdl"}
    ]
    if not candidates:
        raise RuntimeError("YouTube download completed but no video file was found")
    return max(candidates, key=lambda path: path.stat().st_size)
