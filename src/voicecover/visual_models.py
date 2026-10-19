from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


VISUAL_TRACK_VERSION = 1


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Point:
        return cls(x=float(data["x"]), y=float(data["y"]))


@dataclass(frozen=True)
class OCRBox:
    points: tuple[Point, ...]
    text: str
    confidence: float
    source_width: int
    source_height: int
    scripts: tuple[str, ...] = ()
    protected: bool = False
    protection_reasons: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OCRBox:
        return cls(
            points=tuple(Point.from_dict(point) for point in data["points"]),
            text=str(data["text"]),
            confidence=float(data["confidence"]),
            source_width=int(data["source_width"]),
            source_height=int(data["source_height"]),
            scripts=tuple(data.get("scripts", ())),
            protected=bool(data.get("protected", False)),
            protection_reasons=tuple(data.get("protection_reasons", ())),
        )


@dataclass(frozen=True)
class FrameSample:
    path: str
    timestamp: float
    width: int
    height: int
    boxes: tuple[OCRBox, ...] = ()
    average_hash: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameSample:
        return cls(
            path=str(data["path"]),
            timestamp=float(data["timestamp"]),
            width=int(data["width"]),
            height=int(data["height"]),
            boxes=tuple(OCRBox.from_dict(box) for box in data.get("boxes", ())),
            average_hash=(
                None if data.get("average_hash") is None else int(data["average_hash"])
            ),
        )


@dataclass(frozen=True)
class SlideState:
    index: int
    start: float
    end: float
    frame_indices: tuple[int, ...]
    representative_frame_index: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SlideState:
        return cls(
            index=int(data["index"]),
            start=float(data["start"]),
            end=float(data["end"]),
            frame_indices=tuple(int(value) for value in data["frame_indices"]),
            representative_frame_index=int(data["representative_frame_index"]),
        )


@dataclass(frozen=True)
class VisualTrack:
    frames: tuple[FrameSample, ...] = ()
    slides: tuple[SlideState, ...] = ()
    sample_interval: float = 0.0
    version: int = field(default=VISUAL_TRACK_VERSION, init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisualTrack:
        version = int(data.get("version", 0))
        if version != VISUAL_TRACK_VERSION:
            raise ValueError(f"unsupported visual track version: {version}")
        return cls(
            frames=tuple(FrameSample.from_dict(frame) for frame in data.get("frames", ())),
            slides=tuple(SlideState.from_dict(slide) for slide in data.get("slides", ())),
            sample_interval=float(data.get("sample_interval", 0.0)),
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, value: str) -> VisualTrack:
        data = json.loads(value)
        if not isinstance(data, dict):
            raise ValueError("visual track JSON must contain an object")
        return cls.from_dict(data)
