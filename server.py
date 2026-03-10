import base64
import binascii
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import soundfile as sf
import torch
import whisperx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

print("[SERVER] server.py imported", flush=True)

MODEL_NAME = os.getenv("MODEL_NAME", "large-v3")
DEFAULT_TASK = os.getenv("DEFAULT_TASK", "transcribe")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "").strip() or None
WHISPER_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "8"))
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
if HF_TOKEN:
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", HF_TOKEN)

MAX_AUDIO_SECONDS = int(os.getenv("MAX_AUDIO_SECONDS", "1800"))
MAX_AUDIO_SIZE_MB = int(os.getenv("MAX_AUDIO_SIZE_MB", "100"))
MAX_BASE64_BYTES = int(os.getenv("MAX_BASE64_BYTES", str(MAX_AUDIO_SIZE_MB * 1024 * 1024)))

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/workspace/outputs"))
TMP_ROOT = Path(os.getenv("TMP_ROOT", "/workspace/workspace_tmp"))
DEMUCS_MODEL = os.getenv("DEMUCS_MODEL", "htdemucs")
DEMUCS_EXECUTABLE = os.getenv("DEMUCS_EXECUTABLE", "").strip()
ENABLE_DEMUCS_BY_DEFAULT = os.getenv("ENABLE_DEMUCS_BY_DEFAULT", "true").lower() == "true"
ENABLE_DIARIZATION_BY_DEFAULT = os.getenv("ENABLE_DIARIZATION_BY_DEFAULT", "false").lower() == "true"
STARTUP_SELF_TEST = os.getenv("STARTUP_SELF_TEST", "true").lower() == "true"
STARTUP_SELF_TEST_STRICT = os.getenv("STARTUP_SELF_TEST_STRICT", "false").lower() == "true"
SELF_TEST_AUDIO_URL = os.getenv(
    "SELF_TEST_AUDIO_URL",
    "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac",
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu" and WHISPER_COMPUTE_TYPE in {"float16", "bfloat16"}:
    EFFECTIVE_COMPUTE_TYPE = "int8"
else:
    EFFECTIVE_COMPUTE_TYPE = WHISPER_COMPUTE_TYPE

print(f"[SERVER] MODEL_NAME={MODEL_NAME}", flush=True)
print(f"[SERVER] DEVICE={DEVICE}", flush=True)
print(f"[SERVER] WHISPER_COMPUTE_TYPE={WHISPER_COMPUTE_TYPE}", flush=True)
print(f"[SERVER] EFFECTIVE_COMPUTE_TYPE={EFFECTIVE_COMPUTE_TYPE}", flush=True)
print(f"[SERVER] HF_TOKEN_PRESENT={'yes' if HF_TOKEN else 'no'}", flush=True)
print(f"[SERVER] OUTPUT_DIR={OUTPUT_DIR}", flush=True)
print(f"[SERVER] TMP_ROOT={TMP_ROOT}", flush=True)

if DEVICE == "cuda":
    try:
        print(f"[SERVER] CUDA device count={torch.cuda.device_count()}", flush=True)
        print(f"[SERVER] CUDA device name={torch.cuda.get_device_name(0)}", flush=True)
        print(f"[SERVER] CUDA capability={torch.cuda.get_device_capability(0)}", flush=True)
    except Exception as exc:
        print(f"[SERVER] failed to inspect CUDA device: {exc}", flush=True)


app = FastAPI(title="Demucs + WhisperX Server", version="1.0.0")

asr_model = None
align_model_cache: Dict[str, Tuple[Any, Any]] = {}
diarization_pipeline = None
self_test_status: Dict[str, Any] = {"ran": False, "ok": False, "detail": "not_started"}


class ProcessRequest(BaseModel):
    audio_url: Optional[str] = None
    audio_base64: Optional[str] = None
    task: str = Field(default=DEFAULT_TASK, pattern="^(transcribe|translate)$")
    language: Optional[str] = DEFAULT_LANGUAGE
    num_speakers: Optional[int] = Field(default=None, ge=1, le=20)
    enable_demucs: bool = ENABLE_DEMUCS_BY_DEFAULT
    enable_diarization: bool = ENABLE_DIARIZATION_BY_DEFAULT
    return_word_timestamps: bool = True
    return_segments: bool = True
    return_srt: bool = True
    return_vtt: bool = False
    return_base64_outputs: bool = False
    save_to_disk: bool = False

    @model_validator(mode="after")
    def validate_audio_input(self) -> "ProcessRequest":
        if not self.audio_url and not self.audio_base64:
            raise ValueError("Either audio_url or audio_base64 must be provided")
        if self.audio_url and self.audio_base64:
            raise ValueError("Provide only one of audio_url or audio_base64")
        return self


@app.on_event("startup")
def startup_event() -> None:
    global asr_model
    try:
        print("[SERVER] startup begin", flush=True)
        print("[SERVER] loading WhisperX ASR model", flush=True)
        asr_model = whisperx.load_model(
            MODEL_NAME,
            DEVICE,
            compute_type=EFFECTIVE_COMPUTE_TYPE,
        )
        if STARTUP_SELF_TEST:
            try:
                run_startup_self_test()
            except Exception as exc:
                if STARTUP_SELF_TEST_STRICT:
                    raise
                print(f"[SERVER] startup self-test non-fatal failure: {exc}", flush=True)
        print("[SERVER] Model startup finished", flush=True)
        print("[SERVER] Application startup complete.", flush=True)
    except Exception as exc:
        print(f"[SERVER] startup failed: {exc}", flush=True)
        traceback.print_exc()
        raise


@app.api_route("/health", methods=["GET", "POST"])
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "device": DEVICE,
        "model_name": MODEL_NAME,
        "compute_type": EFFECTIVE_COMPUTE_TYPE,
        "hf_token_present": bool(HF_TOKEN),
        "max_audio_seconds": MAX_AUDIO_SECONDS,
        "max_audio_size_mb": MAX_AUDIO_SIZE_MB,
        "startup_self_test_strict": STARTUP_SELF_TEST_STRICT,
        "startup_self_test": self_test_status,
    }


def run_cmd(cmd: List[str], stage: str) -> str:
    print(f"[{stage}] running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "")[-4000:]
        print(f"[{stage}] failed rc={proc.returncode} stderr={stderr_tail}", flush=True)
        raise HTTPException(status_code=400, detail=f"{stage} failed: {stderr_tail}")
    if proc.stderr:
        print(f"[{stage}] stderr: {(proc.stderr[-1000:])}", flush=True)
    return proc.stdout


def ffprobe_duration_seconds(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    out = run_cmd(cmd, "FFPROBE")
    try:
        value = json.loads(out).get("format", {}).get("duration")
        return max(0.0, float(value)) if value is not None else 0.0
    except Exception:
        return 0.0


def download_audio(url: str, dst: Path) -> None:
    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(status_code=422, detail="audio_url must be http/https")

    print(f"[SERVER] downloading audio from url={url}", flush=True)
    with requests.get(url, stream=True, timeout=(10, 120)) as resp:
        resp.raise_for_status()

        total = 0
        with dst.open("wb") as file_obj:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_AUDIO_SIZE_MB * 1024 * 1024:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Audio file is too large (> {MAX_AUDIO_SIZE_MB} MB)",
                    )
                file_obj.write(chunk)


def decode_audio_base64(audio_b64: str, dst: Path) -> None:
    try:
        raw = base64.b64decode(audio_b64, validate=True)
    except binascii.Error as exc:
        raise HTTPException(status_code=422, detail=f"Invalid audio_base64: {exc}") from exc

    if len(raw) > MAX_BASE64_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"audio_base64 is too large (> {MAX_BASE64_BYTES} bytes)",
        )

    dst.write_bytes(raw)


def convert_to_wav_mono_16k(input_path: Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    run_cmd(cmd, "FFMPEG")


def run_demucs(input_wav: Path, work_dir: Path) -> Dict[str, Path]:
    demucs_out = work_dir / "demucs_out"
    demucs_out.mkdir(parents=True, exist_ok=True)

    demucs_cmd = resolve_demucs_cmd()
    cmd = demucs_cmd + [
        "--two-stems",
        "vocals",
        "-n",
        DEMUCS_MODEL,
        "--out",
        str(demucs_out),
        str(input_wav),
    ]
    run_cmd(cmd, "DEMUCS")

    stem_dir = demucs_out / DEMUCS_MODEL / input_wav.stem
    vocals = stem_dir / "vocals.wav"
    no_vocals = stem_dir / "no_vocals.wav"

    if not vocals.exists() or not no_vocals.exists():
        raise HTTPException(status_code=500, detail="Demucs output files were not found")

    return {
        "vocals": vocals,
        "no_vocals": no_vocals,
        "cleaned_audio": vocals,
    }


def get_align_model(language: str) -> Tuple[Any, Any]:
    if language not in align_model_cache:
        print(f"[SERVER] loading alignment model for language={language}", flush=True)
        align_model_cache[language] = whisperx.load_align_model(language_code=language, device=DEVICE)
    return align_model_cache[language]


def get_diarization_pipeline() -> Any:
    global diarization_pipeline
    if diarization_pipeline is None:
        if not HF_TOKEN:
            raise HTTPException(
                status_code=400,
                detail="Diarization requires HF_TOKEN env and access to pyannote models",
            )
        print("[SERVER] loading diarization pipeline", flush=True)
        diarization_cls = getattr(whisperx, "DiarizationPipeline", None)
        if diarization_cls is None:
            try:
                from whisperx.diarize import DiarizationPipeline as diarization_cls  # type: ignore
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "WhisperX diarization API not found. "
                        "Expected whisperx.DiarizationPipeline or whisperx.diarize.DiarizationPipeline"
                    ),
                ) from exc

        init_sig = inspect.signature(diarization_cls)
        params = set(init_sig.parameters.keys())

        kwargs: Dict[str, Any] = {}
        if "device" in params:
            kwargs["device"] = DEVICE

        # WhisperX API changed between releases; support common token parameter names.
        if HF_TOKEN:
            for token_key in ("use_auth_token", "token", "hf_token", "auth_token", "huggingface_token"):
                if token_key in params:
                    kwargs[token_key] = HF_TOKEN
                    break

        try:
            diarization_pipeline = diarization_cls(**kwargs)
        except Exception as exc:
            msg = str(exc)
            if "Cannot access gated repo" in msg or "GatedRepoError" in msg or "403 Client Error" in msg:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        "Diarization model access denied on Hugging Face. "
                        "Accept terms for pyannote/speaker-diarization-community-1 "
                        "and use an authorized HF token."
                    ),
                ) from exc
            raise
    return diarization_pipeline


def format_timestamp(seconds: float, for_vtt: bool) -> str:
    ms = int(round(seconds * 1000.0))
    hours, rem = divmod(ms, 3600 * 1000)
    minutes, rem = divmod(rem, 60 * 1000)
    secs, millis = divmod(rem, 1000)
    sep = "." if for_vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def make_srt(segments: List[Dict[str, Any]]) -> str:
    rows: List[str] = []
    for idx, segment in enumerate(segments, 1):
        text = (segment.get("text") or "").strip()
        speaker = (segment.get("speaker") or "").strip()
        prefix = f"[{speaker}] " if speaker else ""
        rows.extend(
            [
                str(idx),
                f"{format_timestamp(float(segment.get('start', 0.0)), False)} --> {format_timestamp(float(segment.get('end', 0.0)), False)}",
                f"{prefix}{text}",
                "",
            ]
        )
    return "\n".join(rows)


def make_vtt(segments: List[Dict[str, Any]]) -> str:
    rows = ["WEBVTT", ""]
    for segment in segments:
        text = (segment.get("text") or "").strip()
        speaker = (segment.get("speaker") or "").strip()
        prefix = f"<{speaker}> " if speaker else ""
        rows.extend(
            [
                f"{format_timestamp(float(segment.get('start', 0.0)), True)} --> {format_timestamp(float(segment.get('end', 0.0)), True)}",
                f"{prefix}{text}",
                "",
            ]
        )
    return "\n".join(rows)


def build_speakers_summary(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    speakers: Dict[str, Dict[str, Any]] = {}
    for segment in segments:
        speaker = segment.get("speaker") or "UNKNOWN"
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        duration = max(0.0, end - start)
        text = (segment.get("text") or "").strip()

        if speaker not in speakers:
            speakers[speaker] = {"duration": 0.0, "segments": 0, "chars": 0}

        speakers[speaker]["duration"] += duration
        speakers[speaker]["segments"] += 1
        speakers[speaker]["chars"] += len(text)

    for key, value in speakers.items():
        value["duration"] = round(value["duration"], 3)
        value["speaker"] = key

    return {"count": len(speakers), "items": list(speakers.values())}


def maybe_encode_base64(path: Optional[Path], return_base64_outputs: bool) -> Optional[str]:
    if not path or not path.exists() or not return_base64_outputs:
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def safe_name(path: Path) -> str:
    return path.name.replace("..", "_")


def load_waveform_for_pyannote(path: Path) -> Dict[str, Any]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T)
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def resolve_demucs_cmd() -> List[str]:
    # Most robust path in container environments: run demucs via current interpreter.
    if not DEMUCS_EXECUTABLE:
        return [sys.executable, "-m", "demucs.separate"]

    if " " in DEMUCS_EXECUTABLE.strip():
        return DEMUCS_EXECUTABLE.strip().split()

    candidate = Path(DEMUCS_EXECUTABLE)
    if candidate.exists():
        return [str(candidate)]

    resolved = shutil.which(DEMUCS_EXECUTABLE)
    if resolved:
        return [resolved]

    raise HTTPException(status_code=500, detail=f"DEMUCS_EXECUTABLE not found: {DEMUCS_EXECUTABLE}")


def run_startup_self_test() -> None:
    global self_test_status
    print("[SERVER] startup self-test begin", flush=True)
    t0 = time.perf_counter()
    temp_dir = Path(tempfile.mkdtemp(prefix="startup_selftest_", dir=str(TMP_ROOT)))
    try:
        input_raw = temp_dir / "input_raw"
        input_wav = temp_dir / "input.wav"

        download_audio(SELF_TEST_AUDIO_URL, input_raw)
        convert_to_wav_mono_16k(input_raw, input_wav)
        duration = ffprobe_duration_seconds(input_wav)
        if duration <= 0:
            raise RuntimeError("startup self-test: invalid test audio duration")

        demucs_artifacts = run_demucs(input_wav, temp_dir)
        asr_audio_path = demucs_artifacts["cleaned_audio"]

        audio_array = whisperx.load_audio(str(asr_audio_path))
        asr_result = asr_model.transcribe(
            audio_array,
            batch_size=WHISPER_BATCH_SIZE,
            task="transcribe",
            language="en",
        )

        detected_language = asr_result.get("language") or "en"
        align_model, align_meta = get_align_model(detected_language)
        aligned = whisperx.align(
            asr_result["segments"],
            align_model,
            align_meta,
            audio_array,
            DEVICE,
            return_char_alignments=False,
        )

        pipeline = get_diarization_pipeline()
        try:
            diarize_df = pipeline(load_waveform_for_pyannote(asr_audio_path), num_speakers=1)
        except Exception:
            diarize_df = pipeline(str(asr_audio_path), num_speakers=1)
        diarized = whisperx.assign_word_speakers(diarize_df, aligned)

        seg_count = len(diarized.get("segments") or [])
        elapsed = round(time.perf_counter() - t0, 3)
        self_test_status = {
            "ran": True,
            "ok": True,
            "detail": "demucs+asr+align+diarization passed",
            "segments": seg_count,
            "elapsed_seconds": elapsed,
        }
        print(f"[SERVER] startup self-test passed in {elapsed}s segments={seg_count}", flush=True)
    except Exception as exc:
        elapsed = round(time.perf_counter() - t0, 3)
        self_test_status = {
            "ran": True,
            "ok": False,
            "detail": f"{exc}",
            "elapsed_seconds": elapsed,
        }
        print(f"[SERVER] startup self-test failed: {exc}", flush=True)
        traceback.print_exc()
        raise RuntimeError(f"startup self-test failed: {exc}") from exc
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/process/sync")
def process_sync(req: ProcessRequest) -> Dict[str, Any]:
    if asr_model is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    t_total_start = time.perf_counter()
    timings = {
        "demucs": 0.0,
        "asr": 0.0,
        "alignment": 0.0,
        "diarization": 0.0,
        "total": 0.0,
    }

    job_id = uuid.uuid4().hex
    persistent_dir = OUTPUT_DIR / job_id
    temp_dir = Path(tempfile.mkdtemp(prefix=f"job_{job_id}_", dir=str(TMP_ROOT)))

    try:
        print(f"[SERVER] /process/sync job_id={job_id}", flush=True)

        input_raw = temp_dir / "input_raw"
        input_wav = temp_dir / "input.wav"

        if req.audio_url:
            try:
                download_audio(req.audio_url, input_raw)
            except requests.HTTPError as exc:
                raise HTTPException(status_code=400, detail=f"Failed to download audio_url: {exc}") from exc
            except requests.RequestException as exc:
                raise HTTPException(status_code=400, detail=f"Network error while downloading audio_url: {exc}") from exc
        else:
            decode_audio_base64(req.audio_base64 or "", input_raw)

        convert_to_wav_mono_16k(input_raw, input_wav)

        duration = ffprobe_duration_seconds(input_wav)
        if duration <= 0:
            raise HTTPException(status_code=422, detail="Unable to detect audio duration")
        if duration > MAX_AUDIO_SECONDS:
            raise HTTPException(
                status_code=413,
                detail=f"Audio is too long ({duration:.1f}s > {MAX_AUDIO_SECONDS}s)",
            )

        asr_audio_path = input_wav
        demucs_artifacts: Dict[str, Path] = {}

        if req.enable_demucs:
            t0 = time.perf_counter()
            demucs_artifacts = run_demucs(input_wav, temp_dir)
            timings["demucs"] = round(time.perf_counter() - t0, 3)
            asr_audio_path = demucs_artifacts["cleaned_audio"]

        t_asr = time.perf_counter()
        audio_array = whisperx.load_audio(str(asr_audio_path))
        asr_result = asr_model.transcribe(
            audio_array,
            batch_size=WHISPER_BATCH_SIZE,
            task=req.task,
            language=req.language,
        )
        timings["asr"] = round(time.perf_counter() - t_asr, 3)

        detected_language = asr_result.get("language") or req.language or "unknown"

        t_align = time.perf_counter()
        aligned_result = asr_result
        try:
            align_model, align_meta = get_align_model(detected_language)
            aligned_result = whisperx.align(
                asr_result["segments"],
                align_model,
                align_meta,
                audio_array,
                DEVICE,
                return_char_alignments=False,
            )
        except Exception as exc:
            print(f"[SERVER] alignment warning: {exc}", flush=True)
            aligned_result = dict(asr_result)
        timings["alignment"] = round(time.perf_counter() - t_align, 3)

        if req.enable_diarization:
            t_dia = time.perf_counter()
            pipeline = get_diarization_pipeline()
            try:
                diarize_df = pipeline(load_waveform_for_pyannote(asr_audio_path), num_speakers=req.num_speakers)
            except Exception as exc:
                print(f"[SERVER] diarization in-memory fallback failed: {exc}; retrying with file path", flush=True)
                diarize_df = pipeline(str(asr_audio_path), num_speakers=req.num_speakers)
            aligned_result = whisperx.assign_word_speakers(diarize_df, aligned_result)
            timings["diarization"] = round(time.perf_counter() - t_dia, 3)

        raw_segments = aligned_result.get("segments") or []
        response_segments: List[Dict[str, Any]] = []
        response_words: List[Dict[str, Any]] = []

        for segment in raw_segments:
            response_segment = {
                "start": round(float(segment.get("start", 0.0)), 3),
                "end": round(float(segment.get("end", 0.0)), 3),
                "text": (segment.get("text") or "").strip(),
                "speaker": segment.get("speaker"),
            }
            response_segments.append(response_segment)

            if req.return_word_timestamps:
                for word in segment.get("words", []) or []:
                    response_words.append(
                        {
                            "word": word.get("word"),
                            "start": word.get("start"),
                            "end": word.get("end"),
                            "score": word.get("score"),
                            "speaker": word.get("speaker") or segment.get("speaker"),
                        }
                    )

        full_text = " ".join([seg["text"] for seg in response_segments if seg["text"]]).strip()

        artifacts_to_save: Dict[str, Path] = {}
        artifacts: Dict[str, Dict[str, Optional[str]]] = {}

        if demucs_artifacts:
            for key, file_path in demucs_artifacts.items():
                artifacts_to_save[key] = file_path

        if req.return_srt:
            srt_path = temp_dir / "transcript.srt"
            srt_path.write_text(make_srt(response_segments), encoding="utf-8")
            artifacts_to_save["srt"] = srt_path

        if req.return_vtt:
            vtt_path = temp_dir / "transcript.vtt"
            vtt_path.write_text(make_vtt(response_segments), encoding="utf-8")
            artifacts_to_save["vtt"] = vtt_path

        json_path = temp_dir / "result.json"
        json_payload = {
            "text": full_text,
            "segments": response_segments,
            "words": response_words if req.return_word_timestamps else [],
            "detected_language": detected_language,
        }
        json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts_to_save["json"] = json_path

        if req.save_to_disk:
            persistent_dir.mkdir(parents=True, exist_ok=True)

        for key, src_path in artifacts_to_save.items():
            stored_path: Optional[Path] = None
            if req.save_to_disk:
                dst_path = persistent_dir / safe_name(src_path)
                shutil.copy2(src_path, dst_path)
                stored_path = dst_path
            else:
                stored_path = src_path

            artifacts[key] = {
                "path": str(stored_path) if req.save_to_disk else None,
                "base64": maybe_encode_base64(stored_path, req.return_base64_outputs),
            }

        timings["total"] = round(time.perf_counter() - t_total_start, 3)

        if DEVICE == "cuda":
            try:
                print(
                    f"[SERVER] CUDA mem allocated={torch.cuda.memory_allocated() / (1024 ** 2):.1f}MB "
                    f"reserved={torch.cuda.memory_reserved() / (1024 ** 2):.1f}MB",
                    flush=True,
                )
            except Exception:
                pass

        return {
            "ok": True,
            "text": full_text,
            "segments": response_segments if req.return_segments else [],
            "words": response_words if req.return_word_timestamps else [],
            "detected_language": detected_language,
            "speakers": build_speakers_summary(response_segments),
            "artifacts": artifacts,
            "timings": timings,
            "duration_seconds": round(duration, 3),
            "job_id": job_id,
        }

    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError as exc:
        detail = f"CUDA out of memory: {exc}"
        print(f"[SERVER] {detail}", flush=True)
        raise HTTPException(status_code=507, detail=detail) from exc
    except Exception as exc:
        print(f"[SERVER] /process/sync failed: {exc}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if not req.save_to_disk:
            shutil.rmtree(temp_dir, ignore_errors=True)
