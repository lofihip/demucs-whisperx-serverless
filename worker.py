import base64
import io
import os
import traceback
import wave
from typing import Any, Dict

print("[WORKER] worker.py started", flush=True)

try:
    from vastai import (
        BenchmarkConfig,
        HandlerConfig,
        LogActionConfig,
        Worker,
        WorkerConfig,
    )
    print("[WORKER] imported vastai worker classes successfully", flush=True)
except Exception as exc:
    print(f"[WORKER] failed to import vastai worker classes: {exc}", flush=True)
    traceback.print_exc()
    raise

MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://127.0.0.1")
MODEL_SERVER_PORT = int(os.getenv("MODEL_SERVER_PORT", "8000"))
MODEL_LOG_FILE = os.getenv("MODEL_LOG_FILE", "/workspace/model.log")
DEFAULT_WORKLOAD_SECONDS = float(os.getenv("DEFAULT_WORKLOAD_SECONDS", "60"))
MAX_WORKLOAD_SECONDS = float(os.getenv("MAX_WORKLOAD_SECONDS", "3600"))

print(f"[WORKER] MODEL_SERVER_URL={MODEL_SERVER_URL}", flush=True)
print(f"[WORKER] MODEL_SERVER_PORT={MODEL_SERVER_PORT}", flush=True)
print(f"[WORKER] MODEL_LOG_FILE={MODEL_LOG_FILE}", flush=True)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _duration_from_wav_bytes(data: bytes) -> float:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            if sample_rate > 0:
                return max(0.0, n_frames / float(sample_rate))
    except Exception:
        return 0.0
    return 0.0


def workload_calculator(payload: Dict[str, Any]) -> float:
    print(f"[WORKER] workload_calculator payload keys={list(payload.keys())}", flush=True)

    duration = 0.0

    if isinstance(payload.get("duration_seconds"), (int, float)):
        duration = _safe_float(payload.get("duration_seconds"))

    if duration <= 0 and isinstance(payload.get("metadata"), dict):
        duration = _safe_float(payload["metadata"].get("duration"))

    # If caller sent wav base64, estimate actual duration from RIFF header.
    if duration <= 0 and isinstance(payload.get("audio_base64"), str):
        try:
            raw = base64.b64decode(payload["audio_base64"], validate=True)
            duration = _duration_from_wav_bytes(raw)
            if duration <= 0:
                # Fallback estimate for unknown codecs: 32KB ~= 1 second of 16kHz mono PCM16.
                duration = len(raw) / 32000.0
        except Exception:
            duration = 0.0

    if duration <= 0 and payload.get("audio_url"):
        duration = DEFAULT_WORKLOAD_SECONDS

    if duration <= 0:
        duration = DEFAULT_WORKLOAD_SECONDS

    duration = max(1.0, min(MAX_WORKLOAD_SECONDS, round(duration, 3)))
    print(f"[WORKER] workload_calculator result={duration}", flush=True)
    return duration


try:
    print("[WORKER] building WorkerConfig", flush=True)

    worker_config = WorkerConfig(
        model_server_url=MODEL_SERVER_URL,
        model_server_port=MODEL_SERVER_PORT,
        model_log_file=MODEL_LOG_FILE,
        handlers=[
            HandlerConfig(
                route="/process/sync",
                allow_parallel_requests=False,
                max_queue_time=600.0,
                workload_calculator=workload_calculator,
                benchmark_config=BenchmarkConfig(
                    generator=lambda: {
                        "audio_url": "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac",
                        "task": "transcribe",
                        "language": "en",
                        "enable_demucs": False,
                        "enable_diarization": False,
                        "return_word_timestamps": True,
                        "return_segments": True,
                        "return_srt": False,
                        "return_vtt": False,
                        "return_base64_outputs": False,
                        "save_to_disk": False,
                        "duration_seconds": 1.0,
                    },
                    runs=3,
                    concurrency=1,
                ),
            ),
            HandlerConfig(
                route="/health",
                allow_parallel_requests=True,
                max_queue_time=30.0,
                workload_calculator=lambda payload: 0.01,
            ),
        ],
        log_action_config=LogActionConfig(
            on_load=["Application startup complete.", "Model startup finished"],
            on_error=[
                "CUDA out of memory",
                "RuntimeError:",
                "HTTPException:",
                "ffmpeg error",
                "failed:",
            ],
            on_info=["[INFO]", "[BOOT]", "[SERVER]", "[WORKER]", "[DEMUCS]", "[FFMPEG]"],
        ),
    )

    print("[WORKER] WorkerConfig built successfully", flush=True)
except Exception as exc:
    print(f"[WORKER] failed to build WorkerConfig: {exc}", flush=True)
    traceback.print_exc()
    raise


if __name__ == "__main__":
    try:
        print("[WORKER] launching worker", flush=True)
        Worker(worker_config).run()
    except Exception as exc:
        print(f"[WORKER] Worker run failed: {exc}", flush=True)
        traceback.print_exc()
        raise
