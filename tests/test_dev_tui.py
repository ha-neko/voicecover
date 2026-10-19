from __future__ import annotations

import json

import dev_tui


class _Response:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_poll_jobs_puts_active_work_first_and_ignores_old_attempt_stage(monkeypatch) -> None:
    payload = {
        "jobs": [
            {
                "status": "completed",
                "progress": ["extracting audio...", "done: old.mp4"],
            },
            {
                "status": "running",
                "progress": [
                    "extracting audio...",
                    "done: stale.mp4",
                    "extracting audio...",
                    "generating 99 utterance(s) with the cloned original voice...",
                ],
            },
        ]
    }
    monkeypatch.setattr(dev_tui.urllib.request, "urlopen", lambda *args, **kwargs: _Response(payload))

    jobs = dev_tui.poll_jobs()

    assert [job.status for job in jobs] == ["running", "completed"]
    assert jobs[0].stage == "voice"
    assert jobs[1].stage == "done"
