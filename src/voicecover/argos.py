from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from .models import Segment


INDEX_URL = "https://raw.githubusercontent.com/argosopentech/argospm-index/main/index.json"


def language_code(value: str) -> str:
    normalized = value.strip().replace("_", "-").lower()
    if normalized in {"zh-tw", "zh-hk", "zh-hant"}:
        return "zt"
    if normalized == "pt-br":
        return "pb"
    base = normalized.split("-")[0]
    return {"iw": "he", "fil": "tl"}.get(base, base)


def models_directory() -> Path:
    override = os.environ.get("VOICECOVER_MODELS_DIR")
    if override:
        return Path(override).expanduser()
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return data_home / "voicecover" / "argos"


def _fetch_index() -> list[dict[str, object]]:
    try:
        with urllib.request.urlopen(INDEX_URL, timeout=30) as response:
            return json.load(response)
    except Exception as exc:
        raise RuntimeError(f"Could not retrieve the Argos model index: {exc}") from exc


def translation_route(
    source: str, target: str, packages: list[dict[str, object]]
) -> list[tuple[str, str]]:
    source = language_code(source)
    target = language_code(target)
    available = {(str(item["from_code"]), str(item["to_code"])) for item in packages}
    if source == target:
        return []
    if (source, target) in available:
        return [(source, target)]
    if source != "en" and target != "en" and (source, "en") in available and ("en", target) in available:
        return [(source, "en"), ("en", target)]
    raise RuntimeError(f"No offline Argos translation route is available for {source} -> {target}")


def _package_for(
    source: str, target: str, packages: list[dict[str, object]]
) -> dict[str, object]:
    return next(
        item
        for item in packages
        if item.get("from_code") == source and item.get("to_code") == target
    )


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            member_path = (destination / member.filename).resolve()
            if destination_root not in member_path.parents and member_path != destination_root:
                raise RuntimeError("Argos model archive contains an unsafe path")
        package.extractall(destination)


def install_models(
    source: str,
    target: str,
    progress: Callable[[str], None] = print,
    packages: list[dict[str, object]] | None = None,
) -> list[Path]:
    packages = packages or _fetch_index()
    route = translation_route(source, target, packages)
    root = models_directory()
    root.mkdir(parents=True, exist_ok=True)
    installed: list[Path] = []

    for from_code, to_code in route:
        destination = root / f"{from_code}_{to_code}"
        if (destination / "metadata.json").is_file():
            installed.append(destination)
            continue
        package = _package_for(from_code, to_code, packages)
        links = package.get("links")
        if not isinstance(links, list):
            raise RuntimeError(f"Argos model {from_code} -> {to_code} has no download URL")
        url = next((str(link) for link in links if str(link).startswith("https://")), None)
        if url is None:
            raise RuntimeError(f"Argos model {from_code} -> {to_code} has no HTTPS download URL")

        progress(f"downloading offline translation model {from_code} -> {to_code}...")
        work = Path(tempfile.mkdtemp(prefix="argos-install-", dir=root))
        archive = work / "model.argosmodel"
        extracted = work / "extracted"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Voicecover/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as file:
                shutil.copyfileobj(response, file)
            extracted.mkdir()
            _safe_extract(archive, extracted)
            metadata_files = list(extracted.rglob("metadata.json"))
            if len(metadata_files) != 1:
                raise RuntimeError("Downloaded Argos archive has an unexpected structure")
            model_root = metadata_files[0].parent
            shutil.move(str(model_root), str(destination))
        except Exception as exc:
            shutil.rmtree(destination, ignore_errors=True)
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError(f"Could not install Argos model {from_code} -> {to_code}: {exc}") from exc
        finally:
            shutil.rmtree(work, ignore_errors=True)
        installed.append(destination)
    return installed


class ArgosModel:
    def __init__(self, path: Path) -> None:
        try:
            import ctranslate2
            import sentencepiece
        except ImportError as exc:
            raise RuntimeError("offline translation requires ctranslate2 and sentencepiece") from exc

        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        self.target_prefix = str(metadata.get("target_prefix", ""))
        sentencepiece_path = path / "sentencepiece.model"
        if sentencepiece_path.is_file():
            processor = sentencepiece.SentencePieceProcessor(model_file=str(sentencepiece_path))
            self.encode = lambda text: processor.encode(text, out_type=str)
            self.decode = processor.decode_pieces
        else:
            try:
                from sacremoses.normalize import MosesPunctNormalizer
                from sacremoses.tokenize import MosesDetokenizer, MosesTokenizer
                from subword_nmt.apply_bpe import BPE
            except ImportError as exc:
                raise RuntimeError("BPE translation models require sacremoses and subword-nmt") from exc
            from_code = str(metadata["from_code"])
            to_code = str(metadata["to_code"])
            normalizer = MosesPunctNormalizer(from_code)
            tokenizer = MosesTokenizer(from_code)
            detokenizer = MosesDetokenizer(to_code)
            with (path / "bpe.model").open(encoding="utf-8") as codes:
                bpe = BPE(codes)

            def encode_bpe(text: str) -> list[str]:
                normalized = normalizer.normalize(text)
                words = tokenizer.tokenize(normalized)
                return bpe.segment_tokens(words)

            def decode_bpe(tokens: list[str]) -> str:
                return detokenizer.detokenize(" ".join(tokens).replace("@@ ", "").split())

            self.encode = encode_bpe
            self.decode = decode_bpe
        self.translator = ctranslate2.Translator(
            str(path / "model"), device="cpu", compute_type="int8", inter_threads=2
        )

    def translate(self, texts: list[str]) -> list[str]:
        tokenized = [self.encode(text) for text in texts]
        prefixes = None
        if self.target_prefix:
            prefixes = [[self.target_prefix] for _ in tokenized]
        results = self.translator.translate_batch(
            tokenized,
            target_prefix=prefixes,
            replace_unknowns=True,
            max_batch_size=32,
            batch_type="tokens",
            beam_size=3,
        )
        translated = []
        for result in results:
            value = self.decode(result.hypotheses[0]).replace("▁", " ").replace("_", " ").strip()
            if self.target_prefix and value.startswith(self.target_prefix):
                value = value[len(self.target_prefix) :].strip()
            translated.append(value)
        return translated


def translate_segments(
    segments: list[Segment],
    source_language: str,
    target_language: str,
    auto_install: bool = True,
    progress: Callable[[str], None] = print,
) -> list[Segment]:
    packages = _fetch_index() if auto_install else []
    if auto_install:
        install_models(source_language, target_language, progress, packages)
    source = language_code(source_language)
    target = language_code(target_language)
    if source == target:
        return [segment.translated(segment.text) for segment in segments]

    root = models_directory()
    if (root / f"{source}_{target}" / "metadata.json").is_file():
        route = [(source, target)]
    elif (root / f"{source}_en" / "metadata.json").is_file() and (
        root / f"en_{target}" / "metadata.json"
    ).is_file():
        route = [(source, "en"), ("en", target)]
    else:
        raise RuntimeError(
            f"Offline models for {source} -> {target} are not installed; enable automatic model installation"
        )

    texts = [segment.text for segment in segments]
    for from_code, to_code in route:
        progress(f"running offline translation {from_code} -> {to_code}...")
        texts = ArgosModel(root / f"{from_code}_{to_code}").translate(texts)
    return [segment.translated(text) for segment, text in zip(segments, texts, strict=True)]


def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Install offline Voicecover translation models.")
    parser.add_argument("source", help="source language code")
    parser.add_argument("target", help="target language code")
    args = parser.parse_args()
    paths = install_models(args.source, args.target)
    for path in paths:
        print(path)
