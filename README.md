# Demucs + WhisperX CUDA HTTP Server

CUDA-only Docker service with two independent HTTP runtimes:

- `Demucs` for source separation
- `WhisperX` for transcription, alignment, and diarization

Both runtimes preload on container startup, download required weights, and expose readiness plus busy-state endpoints.

## Upstream sources

- Demucs: https://github.com/facebookresearch/demucs
- WhisperX: https://github.com/m-bain/whisperX

This image installs both projects directly from their GitHub repositories so the container uses the latest upstream code available at build time.

## API

- `POST /demucs/process`
- `POST /whisperx/process`
- `GET /health`
- `GET /ready`
- `GET /status`
- `GET /status/demucs`
- `GET /status/whisperx`

## Environment

Copy `.env.example` to `.env` and set:

- `HUGGINGFACE_TOKEN` for diarization
- `API_KEY` if you want request authentication

Safe default:

- `GPU_SERIAL_EXECUTION=true`

That keeps separate busy-state tracking while preventing parallel GPU inference from exhausting VRAM.

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

The endpoint returns HTTP `503` until both runtimes complete warmup and weight downloads.

## Demucs example

  ```bash
  curl -X POST "http://localhost:8000/demucs/process" \
    -H "X-API-Key: your-key" \
    -F "file=@input.mp3" \
  -F 'options_json={"model":"htdemucs","two_stems":"vocals","mp3":true,"response_mode":"archive"}'
  ```

## WhisperX example

```bash
curl -X POST "http://localhost:8000/whisperx/process" \
  -H "X-API-Key: your-key" \
  -F "file=@meeting.wav" \
  -F 'options_json={"model":"large-v3","align":true,"diarize":true,"min_speakers":2,"max_speakers":4,"response_mode":"json"}'
```

## Input modes

Each processing endpoint accepts exactly one of:

- multipart `file`
- `source_url`
- `source_path`

## Notes

- Demucs runs through the CLI to stay close to upstream behavior and expose a broad flag surface.
- WhisperX runs through the Python API so JSON output, alignment, and diarization stay structured.
- The official Demucs repository is archived, but this project still installs from that upstream repository because it was explicitly requested.

## Live server test

You can run a real integration test suite against a deployed server.

Windows PowerShell:

```powershell
$env:LIVE_SERVER_URL="http://142.171.48.138:31150"
$env:LIVE_SERVER_API_KEY="your-key"
$env:LIVE_TEST_SOURCE_URL="https://tmpfiles.org/dl/29042560/voice-sample.mp3"
python -m pytest -q -m live_server tests/test_live_server.py
```

Linux/macOS:

```bash
LIVE_SERVER_URL="http://142.171.48.138:31150" \
LIVE_SERVER_API_KEY="your-key" \
LIVE_TEST_SOURCE_URL="https://tmpfiles.org/dl/29042560/voice-sample.mp3" \
python -m pytest -q -m live_server tests/test_live_server.py
```

The live test generates tiny WAV files on the fly and checks:

- `/health`
- `/status`
- `/ready`
- real `POST /demucs/process`
- real `POST /whisperx/process` in `json` mode
- real `POST /whisperx/process` in `archive` mode

The live suite now sends processing requests through `source_url` to avoid large multipart uploads.
You can override the default file URL with `LIVE_TEST_SOURCE_URL`.
Real server responses are written into `tests/output/` by default and can be redirected with `LIVE_TEST_OUTPUT_DIR`.
