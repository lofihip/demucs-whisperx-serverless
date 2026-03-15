from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from app.config import Settings
from app.errors import InvalidInputError
from app.models import OutputArtifact, WhisperXOptions, WhisperXResult, WhisperXSegment, WhisperXWord
from app.services.base import BaseRuntimeService
from app.utils.files import collect_artifacts, resolve_input_source, zip_directory
from app.utils.subtitles import render_srt, render_tsv, render_txt, render_vtt


class WhisperXService(BaseRuntimeService):
    def __init__(self, settings: Settings, gpu_lock: asyncio.Lock | None = None) -> None:
        super().__init__(name="whisperx", settings=settings, gpu_lock=gpu_lock)
        self.logger = logging.getLogger("app.services.whisperx")
        self._whisperx: Any | None = None
        self._asr_model: Any | None = None
        self._asr_model_name: str | None = None
        self._diarize_pipeline: Any | None = None
        self._align_cache: dict[str, tuple[Any, Any]] = {}

    async def preload(self) -> None:
        self.status.mark_starting()

        def _preload() -> None:
            started_at = time.perf_counter()
            self.logger.info(
                "WhisperX preload started | model=%s | device=%s | diarization_enabled=%s",
                self.settings.whisperx_default_model,
                self.settings.gpu_device,
                self.settings.whisperx_enable_diarization,
            )
            import whisperx
            from whisperx.diarize import DiarizationPipeline

            self._whisperx = whisperx
            self._ensure_asr_model(self.settings.whisperx_default_model)

            preload_language = self.settings.whisperx_preload_align_language.strip()
            if preload_language:
                self._ensure_align_model(preload_language)

            if self.settings.whisperx_enable_diarization and self.settings.huggingface_token:
                self._diarize_pipeline = DiarizationPipeline(
                    token=self.settings.huggingface_token,
                    device=self.settings.gpu_device,
                )
            self.logger.info("WhisperX preload finished | elapsed=%.2fs", time.perf_counter() - started_at)

        try:
            await asyncio.wait_for(asyncio.to_thread(_preload), timeout=self.settings.preload_timeout_seconds)
            self.status.mark_ready(
                model=self.settings.whisperx_default_model,
                diarization=bool(self._diarize_pipeline is not None),
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
        options: WhisperXOptions,
    ) -> tuple[Path | WhisperXResult, Path]:
        job_id = uuid.uuid4().hex

        async def _work() -> tuple[Path | WhisperXResult, Path]:
            started_at = time.perf_counter()
            job_dir = self.make_job_dir(job_id)
            source_kind = self._source_kind(upload=upload, source_url=source_url, source_path=source_path)
            self.logger.info(
                "WhisperX job started | job_id=%s | source_kind=%s | response_mode=%s | model=%s | align=%s | diarize=%s",
                job_id,
                source_kind,
                options.response_mode,
                options.model or self.settings.whisperx_default_model,
                options.align,
                options.diarize,
            )
            input_path = await resolve_input_source(
                self.settings,
                upload=upload,
                source_url=source_url,
                source_path=source_path,
                job_dir=job_dir,
            )
            self.logger.info(
                "WhisperX input prepared | job_id=%s | input_path=%s | size_bytes=%s",
                job_id,
                input_path,
                input_path.stat().st_size if input_path.exists() else "unknown",
            )
            output_dir = job_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            result = await asyncio.to_thread(self._run_pipeline, input_path, output_dir, options)
            self.logger.info(
                "WhisperX pipeline finished | job_id=%s | segments=%s | language=%s",
                job_id,
                len(result.get("segments", [])),
                result.get("language"),
            )

            if options.response_mode == "json":
                self.logger.info(
                    "WhisperX job completed | job_id=%s | response_mode=json | total_elapsed=%.2fs",
                    job_id,
                    time.perf_counter() - started_at,
                )
                return self._build_result(job_id=job_id, output_dir=output_dir, result=result), job_dir

            archive_path = await asyncio.to_thread(zip_directory, output_dir, job_dir / "whisperx-output.zip")
            self.logger.info(
                "WhisperX job completed | job_id=%s | response_mode=archive | archive_path=%s | total_elapsed=%.2fs",
                job_id,
                archive_path,
                time.perf_counter() - started_at,
            )
            return archive_path, job_dir

        return await self._run_exclusive(job_id, _work)

    def _run_pipeline(self, input_path: Path, output_dir: Path, options: WhisperXOptions) -> dict[str, Any]:
        if self._whisperx is None:
            raise InvalidInputError("WhisperX runtime was not preloaded.")

        pipeline_started = time.perf_counter()
        model_name = options.model or self.settings.whisperx_default_model
        language = (options.language or self.settings.whisperx_default_language or "").strip() or None
        batch_size = options.batch_size or self.settings.whisperx_default_batch_size
        compute_type = options.compute_type or self.settings.whisperx_compute_type

        self.logger.info(
            "WhisperX pipeline start | input_path=%s | model=%s | language=%s | batch_size=%s | compute_type=%s | align=%s | diarize=%s",
            input_path,
            model_name,
            language or "auto",
            batch_size,
            compute_type,
            options.align,
            options.diarize,
        )
        self._ensure_asr_model(model_name, compute_type=compute_type)

        audio_load_started = time.perf_counter()
        audio = self._whisperx.load_audio(str(input_path))
        self.logger.info("WhisperX audio loaded | input_path=%s | elapsed=%.2fs", input_path, time.perf_counter() - audio_load_started)

        asr_started = time.perf_counter()
        result = self._asr_model.transcribe(audio, batch_size=batch_size, language=language)
        self.logger.info(
            "WhisperX ASR finished | elapsed=%.2fs | detected_language=%s | segments=%s",
            time.perf_counter() - asr_started,
            result.get("language"),
            len(result.get("segments", [])),
        )

        if options.align:
            detected_language = result.get("language") or language
            if detected_language:
                align_started = time.perf_counter()
                model_a, metadata = self._ensure_align_model(detected_language)
                result = self._whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    self.settings.gpu_device,
                    return_char_alignments=False,
                )
                result["language"] = detected_language
                self.logger.info(
                    "WhisperX alignment finished | elapsed=%.2fs | language=%s | segments=%s",
                    time.perf_counter() - align_started,
                    detected_language,
                    len(result.get("segments", [])),
                )
            else:
                self.logger.info("WhisperX alignment skipped | reason=no_language_detected")
        else:
            self.logger.info("WhisperX alignment disabled for this request")

        if options.diarize:
            diarize_started = time.perf_counter()
            diarize_pipeline = self._ensure_diarization()
            diarize_segments = diarize_pipeline(
                audio,
                min_speakers=options.min_speakers,
                max_speakers=options.max_speakers,
            )
            result = self._whisperx.assign_word_speakers(diarize_segments, result)
            speaker_count = len({segment.get("speaker") for segment in result.get("segments", []) if segment.get("speaker")})
            self.logger.info(
                "WhisperX diarization finished | elapsed=%.2fs | speaker_count=%s",
                time.perf_counter() - diarize_started,
                speaker_count,
            )
        else:
            self.logger.info("WhisperX diarization disabled for this request")

        write_started = time.perf_counter()
        self._write_outputs(output_dir, result)
        artifacts = collect_artifacts(output_dir)
        total_size = sum(size for _, size in artifacts)
        self.logger.info(
            "WhisperX outputs written | elapsed=%.2fs | artifacts=%s | total_size_bytes=%s | output_dir=%s | total_pipeline_elapsed=%.2fs",
            time.perf_counter() - write_started,
            len(artifacts),
            total_size,
            output_dir,
            time.perf_counter() - pipeline_started,
        )
        return result

    def _write_outputs(self, output_dir: Path, result: dict[str, Any]) -> None:
        segments = result.get("segments", [])
        text = "\n".join((segment.get("text") or "").strip() for segment in segments).strip() + "\n"

        (output_dir / "transcript.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "transcript.txt").write_text(render_txt(segments), encoding="utf-8")
        (output_dir / "transcript.srt").write_text(render_srt(segments), encoding="utf-8")
        (output_dir / "transcript.vtt").write_text(render_vtt(segments), encoding="utf-8")
        (output_dir / "transcript.tsv").write_text(render_tsv(segments), encoding="utf-8")
        (output_dir / "plain.txt").write_text(text, encoding="utf-8")

    def _build_result(self, *, job_id: str, output_dir: Path, result: dict[str, Any]) -> WhisperXResult:
        artifacts = [
            OutputArtifact(
                name=path.name,
                relative_path=str(path.relative_to(output_dir)),
                size_bytes=size,
            )
            for path, size in collect_artifacts(output_dir)
        ]

        segments = [
            WhisperXSegment(
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=(segment.get("text") or "").strip(),
                speaker=segment.get("speaker"),
                words=[
                    WhisperXWord(
                        word=word.get("word"),
                        start=word.get("start"),
                        end=word.get("end"),
                        score=word.get("score"),
                        speaker=word.get("speaker"),
                    )
                    for word in segment.get("words", [])
                ],
            )
            for segment in result.get("segments", [])
        ]

        text = "\n".join(segment.text for segment in segments).strip()
        return WhisperXResult(
            job_id=job_id,
            language=result.get("language"),
            segments=segments,
            text=text,
            artifacts=artifacts,
        )

    def _ensure_asr_model(self, model_name: str, *, compute_type: str | None = None) -> Any:
        compute_type = compute_type or self.settings.whisperx_compute_type
        if self._asr_model is not None and self._asr_model_name == model_name:
            self.logger.info("WhisperX ASR model reused | model=%s | compute_type=%s", model_name, compute_type)
            return self._asr_model

        load_started = time.perf_counter()
        self.logger.info("WhisperX ASR model loading | model=%s | compute_type=%s", model_name, compute_type)
        self._asr_model = self._whisperx.load_model(
            model_name,
            self.settings.gpu_device,
            compute_type=compute_type,
            download_root=str(self.settings.model_cache_dir),
        )
        self._asr_model_name = model_name
        self.logger.info("WhisperX ASR model loaded | model=%s | elapsed=%.2fs", model_name, time.perf_counter() - load_started)
        return self._asr_model

    def _ensure_align_model(self, language: str) -> tuple[Any, Any]:
        if language in self._align_cache:
            self.logger.info("WhisperX align model reused | language=%s", language)
            return self._align_cache[language]

        load_started = time.perf_counter()
        self.logger.info("WhisperX align model loading | language=%s", language)
        model_a, metadata = self._whisperx.load_align_model(
            language_code=language,
            device=self.settings.gpu_device,
            model_dir=str(self.settings.model_cache_dir),
        )
        self._align_cache[language] = (model_a, metadata)
        self.logger.info("WhisperX align model loaded | language=%s | elapsed=%.2fs", language, time.perf_counter() - load_started)
        return model_a, metadata

    def _ensure_diarization(self) -> Any:
        if not self.settings.whisperx_enable_diarization:
            raise InvalidInputError("Diarization is disabled by configuration.")
        if not self.settings.huggingface_token:
            raise InvalidInputError("Diarization requires HUGGINGFACE_TOKEN.")
        if self._diarize_pipeline is None:
            from whisperx.diarize import DiarizationPipeline

            load_started = time.perf_counter()
            self.logger.info("WhisperX diarization pipeline loading")
            self._diarize_pipeline = DiarizationPipeline(
                token=self.settings.huggingface_token,
                device=self.settings.gpu_device,
            )
            self.logger.info("WhisperX diarization pipeline loaded | elapsed=%.2fs", time.perf_counter() - load_started)
        else:
            self.logger.info("WhisperX diarization pipeline reused")
        return self._diarize_pipeline

    @staticmethod
    def _source_kind(*, upload: UploadFile | None, source_url: str | None, source_path: str | None) -> str:
        if upload is not None:
            return "upload"
        if source_url:
            return "source_url"
        if source_path:
            return "source_path"
        return "unknown"
