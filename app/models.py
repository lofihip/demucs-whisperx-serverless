from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ServiceStatus(BaseModel):
    name: str
    state: Literal["starting", "ready", "busy", "error"]
    ready: bool
    busy: bool
    current_job_id: str | None = None
    warmed_up_at: datetime | None = None
    last_error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class SystemStatus(BaseModel):
    overall_ready: bool
    gpu_device: str
    demucs: ServiceStatus
    whisperx: ServiceStatus


class DemucsOptions(BaseModel):
    model: str | None = None
    two_stems: str | None = None
    segment: int | None = None
    shifts: int | None = None
    overlap: float | None = None
    jobs: int | None = None
    clip_mode: str | None = None
    filename_template: str | None = None
    mp3: bool = False
    mp3_bitrate: int | None = None
    mp3_preset: int | None = None
    flac: bool = False
    int24: bool = False
    float32: bool = False
    repo: str | None = None
    extra_args: list[str] = Field(default_factory=list)
    response_mode: Literal["archive", "manifest"] = "archive"


class WhisperXOptions(BaseModel):
    model: str | None = None
    language: str | None = None
    batch_size: int | None = None
    compute_type: str | None = None
    align: bool = True
    diarize: bool = True
    min_speakers: int | None = None
    max_speakers: int | None = None
    response_mode: Literal["archive", "json"] = "archive"


class OutputArtifact(BaseModel):
    name: str
    relative_path: str
    size_bytes: int


class DemucsManifest(BaseModel):
    job_id: str
    output_dir: str
    artifacts: list[OutputArtifact]


class WhisperXWord(BaseModel):
    word: str | None = None
    start: float | None = None
    end: float | None = None
    score: float | None = None
    speaker: str | None = None


class WhisperXSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[WhisperXWord] = Field(default_factory=list)


class WhisperXResult(BaseModel):
    job_id: str
    language: str | None = None
    segments: list[WhisperXSegment]
    text: str
    artifacts: list[OutputArtifact]
