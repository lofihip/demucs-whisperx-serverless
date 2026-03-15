from __future__ import annotations

import asyncio
import logging
import subprocess
import time
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.config import Settings
from app.errors import InvalidInputError
from app.models import DemucsManifest, DemucsOptions, OutputArtifact
from app.services.base import BaseRuntimeService
from app.utils.files import collect_artifacts, resolve_input_source, zip_directory


class DemucsService(BaseRuntimeService):
    def __init__(self, settings: Settings, gpu_lock: asyncio.Lock | None = None) -> None:
        super().__init__(name="demucs", settings=settings, gpu_lock=gpu_lock)
        self.logger = logging.getLogger("app.services.demucs")

    async def preload(self) -> None:
        self.status.mark_starting()

        def _preload() -> None:
            started_at = time.perf_counter()
            self.logger.info(
                "Demucs preload started | model=%s | device=%s",
                self.settings.demucs_default_model,
                self.settings.gpu_device,
            )
            warmup_dir = self.make_job_dir("warmup")
            input_path = warmup_dir / "warmup.wav"
            self._generate_silence(input_path)

            cmd = self._build_command(
                input_path=input_path,
                output_dir=warmup_dir / "output",
                options=DemucsOptions(
                    model=self.settings.demucs_default_model,
                    two_stems=self.settings.demucs_default_two_stems or None,
                ),
            )
            self._run_command(cmd)
            self.logger.info(
                "Demucs preload finished | warmup_dir=%s | elapsed=%.2fs",
                warmup_dir,
                time.perf_counter() - started_at,
            )

        try:
            await asyncio.wait_for(asyncio.to_thread(_preload), timeout=self.settings.preload_timeout_seconds)
            self.status.mark_ready(model=self.settings.demucs_default_model, device=self.settings.gpu_device)
        except Exception as exc:
            self.status.mark_error(str(exc))
            raise

    async def process(
        self,
        *,
        upload: UploadFile | None,
        source_url: str | None,
        source_path: str | None,
        options: DemucsOptions,
    ) -> tuple[Path | DemucsManifest, Path]:
        job_id = uuid.uuid4().hex

        async def _work() -> tuple[Path | DemucsManifest, Path]:
            started_at = time.perf_counter()
            job_dir = self.make_job_dir(job_id)
            source_kind = self._source_kind(upload=upload, source_url=source_url, source_path=source_path)
            self.logger.info(
                "Demucs job started | job_id=%s | source_kind=%s | response_mode=%s | model=%s | two_stems=%s",
                job_id,
                source_kind,
                options.response_mode,
                options.model or self.settings.demucs_default_model,
                options.two_stems or self.settings.demucs_default_two_stems or "",
            )
            input_path = await resolve_input_source(
                self.settings,
                upload=upload,
                source_url=source_url,
                source_path=source_path,
                job_dir=job_dir,
            )
            self.logger.info(
                "Demucs input prepared | job_id=%s | input_path=%s | size_bytes=%s",
                job_id,
                input_path,
                input_path.stat().st_size if input_path.exists() else "unknown",
            )
            output_dir = job_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            cmd = self._build_command(input_path=input_path, output_dir=output_dir, options=options)
            separation_started = time.perf_counter()
            await asyncio.to_thread(self._run_command, cmd)
            self.logger.info(
                "Demucs separation finished | job_id=%s | elapsed=%.2fs",
                job_id,
                time.perf_counter() - separation_started,
            )

            artifacts = collect_artifacts(output_dir)
            total_size = sum(size for _, size in artifacts)
            self.logger.info(
                "Demucs outputs prepared | job_id=%s | artifacts=%s | total_size_bytes=%s | output_dir=%s",
                job_id,
                len(artifacts),
                total_size,
                output_dir,
            )

            if options.response_mode == "manifest":
                self.logger.info(
                    "Demucs job completed | job_id=%s | response_mode=manifest | total_elapsed=%.2fs",
                    job_id,
                    time.perf_counter() - started_at,
                )
                return self._build_manifest(job_id=job_id, output_dir=output_dir), job_dir

            archive_path = await asyncio.to_thread(zip_directory, output_dir, job_dir / "demucs-output.zip")
            self.logger.info(
                "Demucs job completed | job_id=%s | response_mode=archive | archive_path=%s | total_elapsed=%.2fs",
                job_id,
                archive_path,
                time.perf_counter() - started_at,
            )
            return archive_path, job_dir

        return await self._run_exclusive(job_id, _work)

    def _build_manifest(self, job_id: str, output_dir: Path) -> DemucsManifest:
        artifacts = [
            OutputArtifact(
                name=path.name,
                relative_path=str(path.relative_to(output_dir)),
                size_bytes=size,
            )
            for path, size in collect_artifacts(output_dir)
        ]
        return DemucsManifest(job_id=job_id, output_dir=str(output_dir), artifacts=artifacts)

    def _build_command(self, *, input_path: Path, output_dir: Path, options: DemucsOptions) -> list[str]:
        model = options.model or self.settings.demucs_default_model
        command = [
            "demucs",
            "-d",
            self.settings.gpu_device,
            "-o",
            str(output_dir),
            "-n",
            model,
        ]

        if options.two_stems or self.settings.demucs_default_two_stems:
            command.extend(["--two-stems", options.two_stems or self.settings.demucs_default_two_stems])
        if options.segment is not None:
            command.extend(["--segment", str(options.segment)])
        if options.shifts is not None:
            command.extend(["--shifts", str(options.shifts)])
        if options.overlap is not None:
            command.extend(["--overlap", str(options.overlap)])
        if options.jobs is not None:
            command.extend(["-j", str(options.jobs)])
        if options.clip_mode:
            command.extend(["--clip-mode", options.clip_mode])
        if options.filename_template:
            command.extend(["--filename", options.filename_template])
        if options.repo:
            command.extend(["--repo", options.repo])
        if options.mp3:
            command.append("--mp3")
        if options.mp3_bitrate is not None:
            command.extend(["--mp3-bitrate", str(options.mp3_bitrate)])
        if options.mp3_preset is not None:
            command.extend(["--mp3-preset", str(options.mp3_preset)])
        if options.flac:
            command.append("--flac")
        if options.int24:
            command.append("--int24")
        if options.float32:
            command.append("--float32")
        command.extend(options.extra_args)
        command.append(str(input_path))
        return command

    def _run_command(self, command: list[str]) -> None:
        self.logger.info("Running Demucs command: %s", " ".join(command))
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            self.logger.error(
                "Demucs command failed | exit_code=%s | stdout=%s | stderr=%s",
                completed.returncode,
                completed.stdout.strip(),
                completed.stderr.strip(),
            )
            raise RuntimeError(
                "Demucs failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
            )
        if completed.stdout.strip():
            self.logger.info("Demucs command stdout | %s", completed.stdout.strip())
        if completed.stderr.strip():
            self.logger.info("Demucs command stderr | %s", completed.stderr.strip())

    def _generate_silence(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=stereo",
            "-t",
            "1",
            str(destination),
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise InvalidInputError(
                f"Unable to generate Demucs warmup audio: {completed.stderr.strip() or completed.stdout.strip()}"
            )

    @staticmethod
    def _source_kind(*, upload: UploadFile | None, source_url: str | None, source_path: str | None) -> str:
        if upload is not None:
            return "upload"
        if source_url:
            return "source_url"
        if source_path:
            return "source_path"
        return "unknown"
