from __future__ import annotations

import io
import json
import math
import os
import wave
import zipfile
from pathlib import Path

import httpx
import pytest


LIVE_SERVER_URL = os.getenv("LIVE_SERVER_URL", "").rstrip("/")
LIVE_SERVER_API_KEY = os.getenv("LIVE_SERVER_API_KEY", "")
LIVE_TIMEOUT_SECONDS = float(os.getenv("LIVE_TEST_TIMEOUT_SECONDS", "600"))
LIVE_AUDIO_PATH = Path(os.getenv("LIVE_TEST_AUDIO_PATH", "tests/voice-sample.mp3"))
LIVE_OUTPUT_DIR = Path(os.getenv("LIVE_TEST_OUTPUT_DIR", "tests/output"))
LIVE_SOURCE_URL = os.getenv(
    "LIVE_TEST_SOURCE_URL",
    "https://tmpfiles.org/dl/29079124/voice-sample.mp3",
)

pytestmark = pytest.mark.live_server


def _require_live_server() -> None:
    if not LIVE_SERVER_URL:
        pytest.skip("Set LIVE_SERVER_URL to run real server integration tests.")


def _require_source_url() -> None:
    if not LIVE_SOURCE_URL:
        pytest.skip("Set LIVE_TEST_SOURCE_URL to run processing tests via source_url.")


def _headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if LIVE_SERVER_API_KEY:
        headers["X-API-Key"] = LIVE_SERVER_API_KEY
    return headers


def _generate_wav_bytes(
    *,
    duration_seconds: float = 1.5,
    sample_rate: int = 16000,
    frequency_hz: float = 440.0,
    amplitude: float = 0.35,
) -> bytes:
    frame_count = int(duration_seconds * sample_rate)
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frames = bytearray()
        for index in range(frame_count):
            sample = amplitude * math.sin(2 * math.pi * frequency_hz * index / sample_rate)
            pcm_value = int(max(-1.0, min(1.0, sample)) * 32767)
            frames.extend(pcm_value.to_bytes(2, byteorder="little", signed=True))
        wav_file.writeframes(bytes(frames))

    return buffer.getvalue()


def _load_test_audio_bytes() -> bytes:
    if LIVE_AUDIO_PATH.exists():
        return LIVE_AUDIO_PATH.read_bytes()
    return _generate_wav_bytes()


def _ensure_output_dir() -> Path:
    LIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return LIVE_OUTPUT_DIR


def _write_output_bytes(filename: str, content: bytes) -> Path:
    output_dir = _ensure_output_dir()
    destination = output_dir / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return destination


def _write_output_json(filename: str, payload: dict) -> Path:
    output_dir = _ensure_output_dir()
    destination = output_dir / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def _extract_zip_to_dir(content: bytes, directory_name: str) -> tuple[Path, list[str]]:
    output_dir = _ensure_output_dir() / directory_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(content), "r") as archive:
        archive.extractall(output_dir)
        return output_dir, archive.namelist()


def _assert_zip_has_files(content: bytes) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(content), "r") as archive:
        names = archive.namelist()
        assert names, "Archive is empty."
        return names


def test_live_health_and_status() -> None:
    _require_live_server()

    with httpx.Client(timeout=LIVE_TIMEOUT_SECONDS) as client:
        health = client.get(f"{LIVE_SERVER_URL}/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        status = client.get(f"{LIVE_SERVER_URL}/status")
        assert status.status_code == 200
        payload = status.json()
        assert "demucs" in payload
        assert "whisperx" in payload
        assert payload["gpu_device"] == "cuda"


def test_live_ready_endpoint() -> None:
    _require_live_server()

    with httpx.Client(timeout=LIVE_TIMEOUT_SECONDS) as client:
        ready = client.get(f"{LIVE_SERVER_URL}/ready")
        assert ready.status_code in {200, 503}
        payload = ready.json()
        assert "overall_ready" in payload
        assert "demucs" in payload
        assert "whisperx" in payload


def test_live_demucs_process_archive() -> None:
    _require_live_server()
    _require_source_url()

    with httpx.Client(timeout=LIVE_TIMEOUT_SECONDS, headers=_headers()) as client:
        response = client.post(
            f"{LIVE_SERVER_URL}/demucs/process",
            data={
                "source_url": LIVE_SOURCE_URL,
                "options_json": json.dumps({"response_mode": "archive", "two_stems": "vocals"}),
            },
        )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/zip")
    extracted_dir, names = _extract_zip_to_dir(response.content, "demucs")
    assert extracted_dir.exists()
    assert any(name.endswith(".wav") or name.endswith(".mp3") or name.endswith(".flac") for name in names)


def test_live_whisperx_process_json() -> None:
    _require_live_server()
    _require_source_url()

    with httpx.Client(timeout=LIVE_TIMEOUT_SECONDS, headers=_headers()) as client:
        response = client.post(
            f"{LIVE_SERVER_URL}/whisperx/process",
            data={
                "source_url": LIVE_SOURCE_URL,
                "options_json": json.dumps(
                    {
                        "response_mode": "json",
                        "align": True,
                        "diarize": True,
                    }
                )
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    saved_json = _write_output_json("whisperx/whisperx-output.json", payload)
    assert "job_id" in payload
    assert "segments" in payload
    assert "artifacts" in payload
    assert saved_json.exists()
    if payload["segments"]:
        assert any(segment.get("speaker") for segment in payload["segments"])


def test_live_whisperx_process_archive() -> None:
    _require_live_server()
    _require_source_url()

    with httpx.Client(timeout=LIVE_TIMEOUT_SECONDS, headers=_headers()) as client:
        response = client.post(
            f"{LIVE_SERVER_URL}/whisperx/process",
            data={
                "source_url": LIVE_SOURCE_URL,
                "options_json": json.dumps(
                    {
                        "response_mode": "archive",
                        "align": True,
                        "diarize": True,
                    }
                )
            },
        )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/zip")
    extracted_dir, names = _extract_zip_to_dir(response.content, "whisperx")
    assert extracted_dir.exists()
    assert any(name.endswith("transcript.json") for name in names)
