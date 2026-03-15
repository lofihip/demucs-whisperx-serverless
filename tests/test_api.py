from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from app.errors import InvalidInputError, ServiceBusyError, ServiceNotReadyError
from app.main import get_settings
from app.models import DemucsManifest, OutputArtifact, WhisperXResult, WhisperXSegment
from tests.conftest import FakeContext, FakeService, FakeSettings


def build_status(*, demucs_state: str = "ready", whisperx_state: str = "ready", overall_ready: bool = True) -> dict:
    return {
        "overall_ready": overall_ready,
        "gpu_device": "cuda",
        "demucs": {
            "name": "demucs",
            "state": demucs_state,
            "ready": demucs_state in {"ready", "busy"},
            "busy": demucs_state == "busy",
            "current_job_id": "job-demucs" if demucs_state == "busy" else None,
            "warmed_up_at": None,
            "last_error": None,
            "details": {"model": "htdemucs"},
        },
        "whisperx": {
            "name": "whisperx",
            "state": whisperx_state,
            "ready": whisperx_state in {"ready", "busy"},
            "busy": whisperx_state == "busy",
            "current_job_id": "job-whisperx" if whisperx_state == "busy" else None,
            "warmed_up_at": None,
            "last_error": None,
            "details": {"model": "large-v3"},
        },
    }


def build_zip(path: Path, *, name: str = "result.txt", content: str = "ok") -> Path:
    archive_path = path / "result.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, content)
    return archive_path


def install_fake_context(client, context: FakeContext, *, api_key: str = "") -> None:
    client.app.state.context = context

    def override_settings():
        settings = FakeSettings()
        settings.api_key = api_key
        return settings

    client.app.dependency_overrides[get_settings] = override_settings


def test_health_endpoint(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_returns_503_until_both_services_ready(client) -> None:
    status_payload = build_status(demucs_state="starting", whisperx_state="ready", overall_ready=False)
    context = FakeContext(
        snapshot_payload=status_payload,
        demucs=FakeService(status_payload["demucs"]),
        whisperx=FakeService(status_payload["whisperx"]),
    )
    install_fake_context(client, context)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["overall_ready"] is False


def test_status_endpoints_return_individual_service_states(client) -> None:
    status_payload = build_status(demucs_state="busy", whisperx_state="ready", overall_ready=True)
    context = FakeContext(
        snapshot_payload=status_payload,
        demucs=FakeService(status_payload["demucs"]),
        whisperx=FakeService(status_payload["whisperx"]),
    )
    install_fake_context(client, context)

    response = client.get("/status")
    assert response.status_code == 200
    assert response.json()["demucs"]["busy"] is True

    demucs_response = client.get("/status/demucs")
    assert demucs_response.status_code == 200
    assert demucs_response.json()["state"] == "busy"

    whisperx_response = client.get("/status/whisperx")
    assert whisperx_response.status_code == 200
    assert whisperx_response.json()["state"] == "ready"


def test_demucs_requires_api_key_when_configured(client, tmp_path: Path) -> None:
    status_payload = build_status()
    archive = build_zip(tmp_path)
    context = FakeContext(
        snapshot_payload=status_payload,
        demucs=FakeService(status_payload["demucs"], process_result=(archive, tmp_path)),
        whisperx=FakeService(status_payload["whisperx"]),
    )
    install_fake_context(client, context, api_key="secret")

    response = client.post("/demucs/process", files={"file": ("audio.wav", b"demo", "audio/wav")})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key."


def test_demucs_process_returns_archive(client, tmp_path: Path) -> None:
    status_payload = build_status()
    archive = build_zip(tmp_path, content="demucs")
    demucs = FakeService(status_payload["demucs"], process_result=(archive, tmp_path))
    context = FakeContext(snapshot_payload=status_payload, demucs=demucs, whisperx=FakeService(status_payload["whisperx"]))
    install_fake_context(client, context)

    response = client.post(
        "/demucs/process",
        files={"file": ("song.wav", b"audio-bytes", "audio/wav")},
        data={"options_json": json.dumps({"response_mode": "archive", "two_stems": "vocals"})},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert demucs.calls[0]["options"].response_mode == "archive"
    assert demucs.calls[0]["options"].two_stems == "vocals"


def test_demucs_process_returns_manifest(client, tmp_path: Path) -> None:
    status_payload = build_status()
    manifest = DemucsManifest(
        job_id="job-1",
        output_dir=str(tmp_path / "output"),
        artifacts=[OutputArtifact(name="vocals.wav", relative_path="vocals.wav", size_bytes=123)],
    )
    demucs = FakeService(status_payload["demucs"], process_result=(manifest, tmp_path))
    context = FakeContext(snapshot_payload=status_payload, demucs=demucs, whisperx=FakeService(status_payload["whisperx"]))
    install_fake_context(client, context)

    response = client.post(
        "/demucs/process",
        files={"file": ("song.wav", b"audio-bytes", "audio/wav")},
        data={"options_json": json.dumps({"response_mode": "manifest"})},
    )

    assert response.status_code == 200
    assert response.json()["job_id"] == "job-1"
    assert response.json()["artifacts"][0]["name"] == "vocals.wav"


def test_whisperx_process_returns_json_payload(client, tmp_path: Path) -> None:
    status_payload = build_status()
    result = WhisperXResult(
        job_id="job-wx",
        language="en",
        text="Hello world",
        segments=[WhisperXSegment(start=0.0, end=1.0, text="Hello world", speaker="SPEAKER_00")],
        artifacts=[OutputArtifact(name="transcript.json", relative_path="transcript.json", size_bytes=64)],
    )
    whisperx = FakeService(status_payload["whisperx"], process_result=(result, tmp_path))
    context = FakeContext(snapshot_payload=status_payload, demucs=FakeService(status_payload["demucs"]), whisperx=whisperx)
    install_fake_context(client, context)

    response = client.post(
        "/whisperx/process",
        files={"file": ("speech.wav", b"audio-bytes", "audio/wav")},
        data={"options_json": json.dumps({"response_mode": "json", "diarize": True, "align": True})},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == "job-wx"
    assert payload["segments"][0]["speaker"] == "SPEAKER_00"
    assert whisperx.calls[0]["options"].diarize is True


def test_whisperx_process_returns_archive(client, tmp_path: Path) -> None:
    status_payload = build_status()
    archive = build_zip(tmp_path, name="transcript.txt", content="hello")
    whisperx = FakeService(status_payload["whisperx"], process_result=(archive, tmp_path))
    context = FakeContext(snapshot_payload=status_payload, demucs=FakeService(status_payload["demucs"]), whisperx=whisperx)
    install_fake_context(client, context)

    response = client.post(
        "/whisperx/process",
        files={"file": ("speech.wav", b"audio-bytes", "audio/wav")},
        data={"options_json": json.dumps({"response_mode": "archive"})},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"


def test_processing_endpoints_map_service_errors(client, tmp_path: Path) -> None:
    status_payload = build_status()
    cases = [
        (ServiceNotReadyError("not ready"), 503),
        (ServiceBusyError("busy"), 409),
        (InvalidInputError("bad input"), 400),
        (RuntimeError("boom"), 500),
    ]

    for error, expected_status in cases:
        demucs = FakeService(status_payload["demucs"], process_error=error)
        context = FakeContext(snapshot_payload=status_payload, demucs=demucs, whisperx=FakeService(status_payload["whisperx"]))
        install_fake_context(client, context)

        response = client.post("/demucs/process", files={"file": ("song.wav", b"audio", "audio/wav")})
        assert response.status_code == expected_status


def test_invalid_options_json_returns_400(client) -> None:
    status_payload = build_status()
    context = FakeContext(
        snapshot_payload=status_payload,
        demucs=FakeService(status_payload["demucs"]),
        whisperx=FakeService(status_payload["whisperx"]),
    )
    install_fake_context(client, context)

    response = client.post(
        "/whisperx/process",
        files={"file": ("speech.wav", b"audio", "audio/wav")},
        data={"options_json": "{invalid json"},
    )

    assert response.status_code == 400
    assert "Invalid options JSON" in response.json()["detail"]
