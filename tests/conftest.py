from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


class FakeSettings:
    def __init__(self) -> None:
        self.api_key = ""
        self.gpu_device = "cuda"
        self.preload_on_startup = False
        self.app_log_level = "INFO"


class FakeService:
    def __init__(self, snapshot_payload: dict, process_result=None, process_error: Exception | None = None) -> None:
        self._snapshot_payload = snapshot_payload
        self._process_result = process_result
        self._process_error = process_error
        self.calls: list[dict] = []

    def snapshot(self):
        class _Snapshot:
            def __init__(self, payload: dict) -> None:
                self.payload = payload

            def model_dump(self, mode: str = "json") -> dict:
                return self.payload

        return _Snapshot(self._snapshot_payload)

    async def process(self, **kwargs):
        self.calls.append(kwargs)
        if self._process_error is not None:
            raise self._process_error
        return self._process_result


class FakeContext:
    def __init__(self, *, snapshot_payload: dict, demucs: FakeService, whisperx: FakeService) -> None:
        self._snapshot_payload = snapshot_payload
        self.demucs = demucs
        self.whisperx = whisperx

    def snapshot(self) -> dict:
        return self._snapshot_payload


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    get_settings.cache_clear()
    monkeypatch.setenv("PRELOAD_ON_STARTUP", "false")
    monkeypatch.setenv("API_KEY", "")

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    get_settings.cache_clear()

