from __future__ import annotations

import json
from io import BytesIO
from urllib.error import URLError

import pytest

from voicecover.llm import LLMTranslationError, translate_segments
from voicecover.models import Segment


def _response(content: str) -> BytesIO:
    return BytesIO(
        json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    )


def test_payload_batches_context_and_returns_original_order(monkeypatch) -> None:
    requests = []
    replies = iter(
        [
            _response('{"translations":[{"id":1,"text":"dua"},{"id":0,"text":"satu"}]}'),
            _response('```json\n{"translations":[{"id":2,"text":"tiga"}]}\n```'),
        ]
    )

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return next(replies)

    monkeypatch.setattr("voicecover.llm.urlopen", fake_urlopen)
    monkeypatch.setenv("VOICECOVER_LLM_MODEL", "test-model")
    segments = [
        Segment(0, 1.5, "one"),
        Segment(1.5, 3, "two"),
        Segment(3, 5, "three-long"),
    ]

    result = translate_segments(segments, "en", "id", character_budget=8)

    assert [segment.translated_text for segment in result] == ["satu", "dua", "tiga"]
    assert len(requests) == 2
    first = json.loads(requests[0][0].data)
    assert first["model"] == "test-model"
    assert first["max_tokens"] == 8192
    assert "response_format" not in first
    assert requests[0][0].full_url == "http://127.0.0.1:8081/v1/chat/completions"
    assert requests[0][0].get_method() == "POST"
    assert requests[0][1] == 3600.0
    prompt = first["messages"][1]["content"]
    assert '"id": 0' in prompt and '"start": 0' in prompt and '"end": 1.5' in prompt
    assert "one" in prompt and "two" in prompt
    assert "three-long" not in prompt


def test_prompt_preserves_arabic_and_protected_terms(monkeypatch) -> None:
    payloads = []

    def fake_urlopen(request, timeout):
        payloads.append(json.loads(request.data))
        return _response('{"translations":[{"id":0,"text":"بسم الله Quran"}]}')

    monkeypatch.setattr("voicecover.llm.urlopen", fake_urlopen)
    translate_segments(
        [Segment(0, 4, "Recite بسم الله from the Qur'an")],
        source_language="English",
        target_language="Indonesian",
        protected_terms={"Qur'an": "Quran"},
    )

    text = json.dumps(payloads[0], ensure_ascii=False)
    assert "natural Indonesian educational localization" in text
    assert "Arabic scripture" in text
    assert "بسم الله" in text
    assert "Qur'an" in text and "Quran" in text
    assert "duration" in text


def test_leftover_english_is_retried_individually(monkeypatch) -> None:
    calls = []
    replies = iter(
        [
            '{"translations":[{"id":0,"text":"For example هذا بيت كبير"}]}',
            '{"translations":[{"id":0,"text":"Contohnya هذا بيت كبير"}]}',
        ]
    )

    def fake_urlopen(request, timeout):
        calls.append(json.loads(request.data))
        return _response(next(replies))

    monkeypatch.setattr("voicecover.llm.urlopen", fake_urlopen)
    result = translate_segments([Segment(0, 2, "For example هذا بيت كبير")], "en", "id")

    assert result[0].translated_text == "Contohnya هذا بيت كبير"
    assert len(calls) == 2
    assert "no leftover English prose" in calls[1]["messages"][-1]["content"]


@pytest.mark.parametrize(
    "invalid",
    [
        '{"translations":[{"id":0,"text":"ok"},{"id":0,"text":"again"}]}',
        '{"translations":[{"id":0,"text":"ok"}]}',
        '{"translations":[{"id":0,"text":""},{"id":1,"text":"ok"}]}',
    ],
)
def test_invalid_result_retries_once_then_accepts_correction(monkeypatch, invalid) -> None:
    calls = []
    replies = iter(
        [invalid, '{"translations":[{"id":1,"text":"B"},{"id":0,"text":"A"}]}'],
    )

    def fake_urlopen(request, timeout):
        calls.append(json.loads(request.data))
        return _response(next(replies))

    monkeypatch.setattr("voicecover.llm.urlopen", fake_urlopen)
    result = translate_segments([Segment(0, 1, "a"), Segment(1, 2, "b")])

    assert [item.translated_text for item in result] == ["A", "B"]
    assert len(calls) == 2
    assert calls[1]["messages"][-2] == {"role": "assistant", "content": invalid}
    assert "exactly these IDs once each: [0, 1]" in calls[1]["messages"][-1]["content"]


def test_invalid_correction_has_clear_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "voicecover.llm.urlopen", lambda request, timeout: _response('{"translations":[]}')
    )
    with pytest.raises(LLMTranslationError, match="after repeated corrections"):
        translate_segments([Segment(0, 1, "hello")])


def test_extra_json_keys_are_tolerated(monkeypatch) -> None:
    monkeypatch.setattr(
        "voicecover.llm.urlopen",
        lambda request, timeout: _response(
            '{"notes":"verbose wrapper","translations":[{"id":0,"text":"Halo","confidence":1}]}'
        ),
    )
    result = translate_segments([Segment(0, 1, "hello")])
    assert result[0].translated_text == "Halo"


def test_control_characters_are_stripped_from_prompt(monkeypatch) -> None:
    bodies = []

    def fake_urlopen(request, timeout):
        bodies.append(json.loads(request.data))
        return _response('{"translations":[{"id":0,"text":"Halo"}]}')

    monkeypatch.setattr("voicecover.llm.urlopen", fake_urlopen)
    translate_segments([Segment(0, 1, "hello\x00\x1b\xff\xfdworld")])
    sent = bodies[0]["messages"][-1]["content"]
    assert "\x00" not in sent and "\x1b" not in sent and "\ufffd" not in sent
    assert "hello" in sent and "world" in sent


def test_invalid_large_batch_recovers_by_recursive_split(monkeypatch) -> None:
    replies = iter(
        [
            '{"translations":[]}',
            '{"result":[]}',
            '{"translations":[{"id":2,"text":"C"},{"id":3,"text":"D"}]}',
            '{"translations":[{"id":0,"text":"A"},{"id":1,"text":"B"}]}',
            '{"translations":[{"id":2,"text":"C"},{"id":3,"text":"D"}]}',
        ]
    )
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(json.loads(request.data))
        return _response(next(replies))

    monkeypatch.setattr("voicecover.llm.urlopen", fake_urlopen)
    segments = [Segment(index, index + 1, chr(97 + index)) for index in range(4)]

    result = translate_segments(segments, "en", "id")

    assert [item.translated_text for item in result] == ["A", "B", "C", "D"]
    assert len(calls) == 5


def test_malformed_server_response_is_clear(monkeypatch) -> None:
    monkeypatch.setattr("voicecover.llm.urlopen", lambda request, timeout: BytesIO(b"not json"))
    with pytest.raises(LLMTranslationError, match="malformed chat-completions response"):
        translate_segments([Segment(0, 1, "hello")])


def test_network_failure_is_clear(monkeypatch) -> None:
    def fail(request, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr("voicecover.llm.urlopen", fail)
    with pytest.raises(LLMTranslationError, match="Could not reach local LLM server.*connection refused"):
        translate_segments([Segment(0, 1, "hello")])


def test_shifted_batch_is_detected_and_retranslated(monkeypatch) -> None:
    # The model answers a 3-item batch with one duplicated target (IDs shifted).
    replies = iter(
        [
            '{"translations":[{"id":0,"text":"Ketenangan hati."},{"id":1,"text":"Ketenangan hati."},{"id":2,"text":"Doa adalah ibadah."}]}',
            '{"translations":[{"id":0,"text":"Ibadah berarti menyembah."}]}',
            '{"translations":[{"id":1,"text":"Haji menuntut ilmu dan bekerja halal."}]}',
        ]
    )
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(json.loads(request.data))
        return _response(next(replies))

    monkeypatch.setattr("voicecover.llm.urlopen", fake_urlopen)
    result = translate_segments(
        [
            Segment(0, 1, "Ibadah means not only worship"),
            Segment(1, 2, "To go for hajj and seek knowledge"),
            Segment(2, 3, "The Prophet said dua is worship"),
        ],
        "en",
        "id",
    )

    assert result[0].translated_text == "Ibadah berarti menyembah."
    assert result[1].translated_text == "Haji menuntut ilmu dan bekerja halal."
    assert result[2].translated_text == "Doa adalah ibadah."
    assert len(calls) == 3
    # the repairs were asked one segment at a time
    assert [m for m in calls[1]["messages"] if m["role"] == "user"][-1]["content"].count('"id"') >= 1


def test_identical_sources_are_not_flagged_as_shifted(monkeypatch) -> None:
    replies = iter(
        ['{"translations":[{"id":0,"text":"Kamu sahaja."},{"id":1,"text":"Kamu sahaja."}]}']
    )
    monkeypatch.setattr(
        "voicecover.llm.urlopen", lambda request, timeout: _response(next(replies))
    )
    result = translate_segments(
        [Segment(0, 1, "You alone"), Segment(1, 2, "You alone.")], "en", "id"
    )
    assert [item.translated_text for item in result] == ["Kamu sahaja.", "Kamu sahaja."]
