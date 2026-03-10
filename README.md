# Demucs + WhisperX + Diarization (Vast.ai PyWorker)

Production-oriented serverless layout with two virtualenvs in one CUDA container:
- `model venv`: FastAPI model server (`server.py`) for Demucs + WhisperX ASR/alignment + diarization.
- `worker venv`: Vast.ai PyWorker (`worker.py`) routing `/health` and `/process/sync` to local model server.

## Project Files
- `Dockerfile`
- `start.sh`
- `server.py`
- `worker.py`
- `test.py`
- `requirements-model.txt`
- `requirements-worker.txt`
- `.env.example`
- `.gitignore`

## API
### `GET/POST /health`
Returns readiness, device, model config and limits.

### `POST /process/sync`
Input JSON:
- `audio_url` or `audio_base64` (exactly one required)
- `task`: `transcribe` or `translate`
- `language` (optional)
- `num_speakers` (optional)
- `enable_demucs` (bool)
- `enable_diarization` (bool)
- `return_word_timestamps` (bool)
- `return_segments` (bool)
- `return_srt` (bool)
- `return_vtt` (bool)
- `return_base64_outputs` (bool)
- `save_to_disk` (bool)

Output JSON:
- `ok`
- `text`
- `segments` (`start/end/text/speaker`)
- `words` (if enabled)
- `detected_language`
- `speakers` summary
- `artifacts` (paths and/or base64: cleaned audio, vocals, no_vocals, srt, vtt, json)
- `timings` (`demucs/asr/alignment/diarization/total`)

## Environment Variables
Copy `.env.example` and set values:
- `HF_TOKEN`: required for diarization (`pyannote` models access).
- `VAST_API_KEY`, `VAST_ENDPOINT_NAME`: used by `test.py` for Vast `/route` tests.
- `MODEL_NAME`: Whisper model name (`large-v3`, `medium`, etc).
- `WHISPER_COMPUTE_TYPE`: `float16`, `int8`, etc.
- `MAX_AUDIO_SECONDS`, `MAX_AUDIO_SIZE_MB`, `MAX_BASE64_BYTES`: request safety limits.
- `OUTPUT_DIR`, `TMP_ROOT`: output/temp directories.
- `DEMUCS_EXECUTABLE` (optional): explicit Demucs binary path/name if PATH is customized.
- `STARTUP_SELF_TEST`: run startup validation of Demucs + ASR + alignment + diarization.
- `STARTUP_SELF_TEST_STRICT`: if `true`, fail startup on self-test error; if `false`, continue startup and report status in `/health`.
- `SELF_TEST_AUDIO_URL`: short public audio URL used by startup self-test.

## Build
```bash
docker build -t demucs-whisperx-vast:latest .
```

## Run Local
```bash
docker run --gpus all --rm -p 8000:8000 \
  --env-file .env \
  demucs-whisperx-vast:latest
```

`start.sh` behavior:
1. Starts `uvicorn` from model venv.
2. Waits for `/health`.
3. Starts Vast `worker.py` from worker venv.

`server.py` startup behavior (if `STARTUP_SELF_TEST=true`):
1. Downloads `SELF_TEST_AUDIO_URL`.
2. Runs Demucs separation.
3. Runs WhisperX transcription + alignment.
4. Runs diarization and speaker assignment.
5. Exposes self-test status in `/health` under `startup_self_test`.

## Local cURL Examples
### Health
```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/health -H "Content-Type: application/json" -d '{}'
```

### Process
```bash
curl -X POST http://127.0.0.1:8000/process/sync \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac",
    "task": "transcribe",
    "language": "en",
    "enable_demucs": true,
    "enable_diarization": false,
    "return_word_timestamps": true,
    "return_segments": true,
    "return_srt": true,
    "return_vtt": true,
    "return_base64_outputs": false,
    "save_to_disk": true
  }'
```

## Vast End-to-End Test Client
`test.py` runs through Vast `/route` -> worker routes:
- `/health`
- basic transcription
- demucs on/off
- diarization on/off
- negative cases (empty input, broken URL, invalid format)

Artifacts are saved to `test_outputs/` and summary JSON is printed.

Run:
```bash
python test.py \
  --endpoint-name "$VAST_ENDPOINT_NAME" \
  --api-key "$VAST_API_KEY" \
  --route-url "${VAST_ROUTE_URL:-https://run.vast.ai/route/}" \
  --out-dir test_outputs
```

## Notes on Compatibility
Pinned versions in `requirements-*.txt` are selected to keep CUDA 12.1 + WhisperX + Demucs working together with Python 3.10/Ubuntu 22.04.
If upstream packages change, adjust pins together (Torch, WhisperX, Demucs, pyannote stack).

## Troubleshooting
### 1) Diarization does not work
- Ensure `HF_TOKEN` is set.
- Accept model terms on Hugging Face for pyannote diarization models with that account.
- Check `/health` -> `hf_token_present`.

### 2) CUDA OOM
- Lower input audio duration.
- Use smaller model (`MODEL_NAME=medium` or `small`).
- Set `WHISPER_COMPUTE_TYPE=int8`.
- Disable heavy stages if needed (`enable_demucs=false`, `enable_diarization=false`).

### 3) ffmpeg errors / unsupported format
- Ensure input is valid audio.
- Check server logs for `[FFMPEG]` and `[FFPROBE]` messages.
- Try sending `audio_base64` with known good WAV/MP3.

### 4) Slow startup
- First run downloads model weights.
- Keep container warm for production endpoint.

### 5) `/process/sync` returns size/duration errors
- Respect `MAX_AUDIO_SIZE_MB`, `MAX_BASE64_BYTES`, and `MAX_AUDIO_SECONDS` limits.
- Split long audio before sending.
