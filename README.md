# Demucs + WhisperX CUDA HTTP Server

CUDA-only HTTP service with two independent runtimes:

- `Demucs` for source separation
- `WhisperX` for transcription, alignment, and diarization

Both services preload on container startup, expose readiness and busy-state endpoints, and can be called independently.

## Upstream sources

- Demucs: https://github.com/facebookresearch/demucs
- WhisperX: https://github.com/m-bain/whisperX

This image installs both projects directly from their GitHub repositories, so the container uses the latest upstream code available at build time.

## API endpoints

- `GET /health`
- `GET /ready`
- `GET /status`
- `GET /status/demucs`
- `GET /status/whisperx`
- `POST /demucs/process`
- `POST /whisperx/process`

## Readiness and status

- `GET /health` returns process liveness
- `GET /ready` returns `200` only when both runtimes finished preload
- `GET /status` returns overall status plus `demucs` and `whisperx`
- `GET /status/demucs` and `GET /status/whisperx` return per-service state

Possible runtime states:

- `starting`
- `ready`
- `busy`
- `error`

## Authentication

`API_KEY` is optional.

- if `API_KEY` is empty, requests are accepted without auth
- if `API_KEY` is set, clients must send `X-API-Key`

## Input modes

Each processing endpoint accepts exactly one input source:

- multipart `file`
- `source_url`
- `source_path`

## Environment variables

Copy `.env.example` to `.env`.

Important variables:

- `APP_HOST`
- `APP_PORT`
- `APP_LOG_LEVEL`
- `API_KEY`
- `WORK_ROOT`
- `MODEL_CACHE_DIR`
- `TEMP_ROOT`
- `GPU_DEVICE`
- `GPU_SERIAL_EXECUTION`
- `PRELOAD_ON_STARTUP`
- `PRELOAD_TIMEOUT_SECONDS`

Demucs defaults:

- `DEMUCS_DEFAULT_MODEL=htdemucs`
- `DEMUCS_DEFAULT_TWO_STEMS=`
- `DEMUCS_DEFAULT_MP3=true`
- `DEMUCS_DEFAULT_MP3_BITRATE=192`

WhisperX defaults:

- `WHISPERX_DEFAULT_MODEL=large-v3`
- `WHISPERX_DEFAULT_LANGUAGE=`
- `WHISPERX_DEFAULT_BATCH_SIZE=8`
- `WHISPERX_COMPUTE_TYPE=float16`
- `WHISPERX_PRELOAD_ALIGN_LANGUAGE=en`
- `WHISPERX_ENABLE_DIARIZATION=true`
- `HUGGINGFACE_TOKEN=...`

Notes:

- `GPU_SERIAL_EXECUTION=true` is the safe default for a single GPU
- `HUGGINGFACE_TOKEN` is required for diarization
- `DEMUCS_DEFAULT_TWO_STEMS=vocals` gives `vocals` + `no_vocals`

## Build

```bash
docker build -t demucs-whisperx-serverless .
```

## Run

```bash
docker run --rm -p 8000:8000 --gpus all \
  --env-file .env \
  demucs-whisperx-serverless
```

## Wait for readiness

```bash
curl http://localhost:8000/ready
```

`/ready` returns `503` until both runtimes finish preload.

## Demucs request format

`POST /demucs/process`

Form fields:

- `file` or `source_url` or `source_path`
- `options_json`

Supported `DemucsOptions` fields:

- `model`
- `two_stems`
- `segment`
- `shifts`
- `overlap`
- `jobs`
- `clip_mode`
- `filename_template`
- `mp3`
- `mp3_bitrate`
- `mp3_preset`
- `flac`
- `int24`
- `float32`
- `repo`
- `response_mode`

`response_mode` values:

- `archive`
- `manifest`

Demucs examples:

```bash
curl -X POST "http://localhost:8000/demucs/process" \
  -H "X-API-Key: your-key" \
  -F "file=@input.mp3" \
  -F 'options_json={"model":"htdemucs","two_stems":"vocals","response_mode":"archive"}'
```

```bash
curl -X POST "http://localhost:8000/demucs/process" \
  -H "X-API-Key: your-key" \
  -F 'source_url=https://example.com/input.mp3' \
  -F 'options_json={"two_stems":"vocals","response_mode":"manifest"}'
```

## WhisperX request format

`POST /whisperx/process`

Form fields:

- `file` or `source_url` or `source_path`
- `options_json`

Supported `WhisperXOptions` fields:

- `model`
- `language`
- `batch_size`
- `compute_type`
- `align`
- `diarize`
- `min_speakers`
- `max_speakers`
- `response_mode`

`response_mode` values:

- `json`
- `archive`

WhisperX examples:

```bash
curl -X POST "http://localhost:8000/whisperx/process" \
  -H "X-API-Key: your-key" \
  -F "file=@meeting.wav" \
  -F 'options_json={"model":"large-v3","align":true,"diarize":true,"min_speakers":2,"max_speakers":4,"response_mode":"json"}'
```

```bash
curl -X POST "http://localhost:8000/whisperx/process" \
  -H "X-API-Key: your-key" \
  -F 'source_path=/workspace/input/meeting.wav' \
  -F 'options_json={"align":true,"diarize":true,"response_mode":"archive"}'
```

## Output behavior

Demucs:

- `archive` returns a zip archive with separated stems
- `manifest` returns JSON with artifact list and output directory

WhisperX:

- `json` returns structured transcript data
- `archive` returns a zip archive with generated files

WhisperX archive outputs typically include:

- `transcript.json`
- `transcript.txt`
- `transcript.srt`
- `transcript.vtt`
- `transcript.tsv`
- `plain.txt`

## Logging

The service logs:

- request acceptance
- source type
- model selection
- preload start/finish
- per-stage timings for Demucs and WhisperX
- output artifact counts and sizes
- response mode and cleanup scheduling

## Live server test

You can run a real integration test suite against a deployed server.

Windows PowerShell:

```powershell
$env:LIVE_SERVER_URL="http://142.171.48.138:31150"
$env:LIVE_SERVER_API_KEY="your-key"
$env:LIVE_TEST_SOURCE_URL="https://tmpfiles.org/dl/29079124/voice-sample.mp3"
python -m pytest -q -m live_server tests/test_live_server.py
```

Linux/macOS:

```bash
LIVE_SERVER_URL="http://142.171.48.138:31150" \
LIVE_SERVER_API_KEY="your-key" \
LIVE_TEST_SOURCE_URL="https://tmpfiles.org/dl/29079124/voice-sample.mp3" \
python -m pytest -q -m live_server tests/test_live_server.py
```

The live suite checks:

- `/health`
- `/status`
- `/ready`
- real `POST /demucs/process`
- real `POST /whisperx/process` in `json` mode
- real `POST /whisperx/process` in `archive` mode

It writes real server results into `tests/output/` by default.

Optional live-test env vars:

- `LIVE_SERVER_URL`
- `LIVE_SERVER_API_KEY`
- `LIVE_TEST_SOURCE_URL`
- `LIVE_TEST_OUTPUT_DIR`
- `LIVE_TEST_TIMEOUT_SECONDS`

## Notes

- Demucs now uses persistent in-process `demucs.api.Separator` objects and reuses loaded models between requests
- WhisperX uses persistent in-process ASR, align, and diarization objects
- the official Demucs repository is archived, but this project still installs from that upstream repository because it was explicitly requested
