from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from app.config import Settings
from app.models import DemucsManifest, DemucsOptions, OutputArtifact
from app.services.base import BaseRuntimeService
from app.utils.files import collect_artifacts, resolve_input_source, zip_directory


class DemucsService(BaseRuntimeService):
    def __init__(self, settings: Settings, gpu_lock: asyncio.Lock | None = None) -> None:
        super().__init__(name="demucs", settings=settings, gpu_lock=gpu_lock)
        self.logger = logging.getLogger("app.services.demucs")
        self._separators: dict[tuple[str, str | None], object] = {}

    async def preload(self) -> None:
        self.status.mark_starting()

        def _preload() -> None:
            started_at = time.perf_counter()
            self.logger.info(
                "Demucs preload started | model=%s | device=%s",
                self.settings.demucs_default_model,
                self.settings.gpu_device,
            )
            separator = self._get_separator(
                model_name=self.settings.demucs_default_model,
                repo=None,
            )
            import torch

            silence = torch.zeros(2, 44100, dtype=torch.float32)
            separator.separate_tensor(silence, sr=44100)
            self.logger.info(
                "Demucs preload finished | model=%s | elapsed=%.2fs",
                self.settings.demucs_default_model,
                time.perf_counter() - started_at,
            )

        try:
            await asyncio.wait_for(
                asyncio.to_thread(_preload),
                timeout=self.settings.preload_timeout_seconds,
            )
            self.status.mark_ready(
                model=self.settings.demucs_default_model,
                device=self.settings.gpu_device,
            )
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
            source_kind = self._source_kind(
                upload=upload,
                source_url=source_url,
                source_path=source_path,
            )
            model_name = options.model or self.settings.demucs_default_model
            repo_path = Path(options.repo) if options.repo else None

            self.logger.info(
                "Demucs job started | job_id=%s | source_kind=%s | response_mode=%s | model=%s | two_stems=%s | mp3=%s | mp3_bitrate=%s",
                job_id,
                source_kind,
                options.response_mode,
                model_name,
                options.two_stems or self.settings.demucs_default_two_stems or "",
                self._effective_mp3(options),
                options.mp3_bitrate
                if options.mp3_bitrate is not None
                else self.settings.demucs_default_mp3_bitrate,
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

            separation_started = time.perf_counter()
            separator = await asyncio.to_thread(
                self._prepare_separator,
                model_name,
                repo_path,
                options,
            )
            origin, separated = await asyncio.to_thread(
                separator.separate_audio_file,
                input_path,
            )
            self.logger.info(
                "Demucs separation finished | job_id=%s | elapsed=%.2fs | stems=%s",
                job_id,
                time.perf_counter() - separation_started,
                ",".join(sorted(separated.keys())),
            )

            finalized = self._finalize_sources(
                origin=origin,
                separated=separated,
                two_stems=options.two_stems or self.settings.demucs_default_two_stems or None,
            )
            output_root = job_dir / "output" / model_name
            output_root.mkdir(parents=True, exist_ok=True)

            save_started = time.perf_counter()
            await asyncio.to_thread(
                self._save_outputs,
                separator,
                input_path,
                output_root,
                finalized,
                options,
            )
            artifacts = collect_artifacts(output_root)
            total_size = sum(size for _, size in artifacts)
            self.logger.info(
                "Demucs outputs prepared | job_id=%s | artifacts=%s | total_size_bytes=%s | output_dir=%s | save_elapsed=%.2fs",
                job_id,
                len(artifacts),
                total_size,
                output_root,
                time.perf_counter() - save_started,
            )

            if options.response_mode == "manifest":
                self.logger.info(
                    "Demucs job completed | job_id=%s | response_mode=manifest | total_elapsed=%.2fs",
                    job_id,
                    time.perf_counter() - started_at,
                )
                return self._build_manifest(job_id=job_id, output_dir=output_root), job_dir

            archive_path = await asyncio.to_thread(
                zip_directory,
                output_root,
                job_dir / "demucs-output.zip",
            )
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
        return DemucsManifest(
            job_id=job_id,
            output_dir=str(output_dir),
            artifacts=artifacts,
        )

    def _prepare_separator(self, model_name: str, repo_path: Path | None, options: DemucsOptions):
        separator = self._get_separator(model_name=model_name, repo=repo_path)
        separator.update_parameter(
            shifts=options.shifts if options.shifts is not None else 1,
            overlap=options.overlap if options.overlap is not None else 0.25,
            segment=options.segment,
            jobs=options.jobs if options.jobs is not None else 0,
            progress=True,
        )
        return separator

    def _get_separator(self, *, model_name: str, repo: Path | None):
        from demucs.api import Separator

        cache_key = (model_name, str(repo) if repo is not None else None)
        if cache_key in self._separators:
            self.logger.info("Demucs separator reused | model=%s | repo=%s", model_name, repo)
            return self._separators[cache_key]

        load_started = time.perf_counter()
        self.logger.info(
            "Demucs separator loading | model=%s | repo=%s | device=%s",
            model_name,
            repo,
            self.settings.gpu_device,
        )
        separator = Separator(
            model=model_name,
            repo=repo,
            device=self.settings.gpu_device,
            progress=True,
        )
        self._separators[cache_key] = separator
        self.logger.info(
            "Demucs separator loaded | model=%s | elapsed=%.2fs",
            model_name,
            time.perf_counter() - load_started,
        )
        return separator

    def _finalize_sources(
        self,
        *,
        origin,
        separated: dict[str, Any],
        two_stems: str | None,
    ) -> dict[str, Any]:
        import torch

        if not two_stems:
            return separated
        if two_stems not in separated:
            raise ValueError(
                f'Stem "{two_stems}" is not available in selected model. '
                f'Available stems: {", ".join(sorted(separated.keys()))}'
            )

        accompaniment = torch.zeros_like(next(iter(separated.values())))
        for stem_name, source in separated.items():
            if stem_name != two_stems:
                accompaniment += source
        return {
            two_stems: separated[two_stems],
            f"no_{two_stems}": accompaniment,
        }

    def _save_outputs(
        self,
        separator,
        input_path: Path,
        output_root: Path,
        separated: dict[str, Any],
        options: DemucsOptions,
    ) -> None:
        from demucs.api import save_audio

        use_mp3 = self._effective_mp3(options)
        ext = "flac" if options.flac else "mp3" if use_mp3 else "wav"
        samplerate = separator.samplerate
        filename_template = options.filename_template or "{track}/{stem}.{ext}"
        track_name = input_path.stem
        track_ext = input_path.suffix.lstrip(".")
        save_kwargs = {
            "samplerate": samplerate,
            "bitrate": options.mp3_bitrate
            if options.mp3_bitrate is not None
            else self.settings.demucs_default_mp3_bitrate,
            "preset": options.mp3_preset if options.mp3_preset is not None else 2,
            "clip": options.clip_mode or "rescale",
            "as_float": options.float32,
            "bits_per_sample": 24 if options.int24 else 16,
        }

        for stem_name, source in separated.items():
            relative_name = filename_template.format(
                track=track_name,
                trackext=track_ext,
                stem=stem_name,
                ext=ext,
            )
            destination = output_root / relative_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            save_audio(source, str(destination), **save_kwargs)

    @staticmethod
    def _source_kind(
        *,
        upload: UploadFile | None,
        source_url: str | None,
        source_path: str | None,
    ) -> str:
        if upload is not None:
            return "upload"
        if source_url:
            return "source_url"
        if source_path:
            return "source_path"
        return "unknown"

    def _effective_mp3(self, options: DemucsOptions) -> bool:
        if options.flac:
            return False
        if options.mp3 is None:
            return self.settings.demucs_default_mp3
        return options.mp3
