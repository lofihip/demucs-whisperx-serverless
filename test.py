import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

ROUTE_URL = "https://run.vast.ai/route/"
DEFAULT_TIMEOUT = 1800
DEFAULT_OUT_DIR = "test_outputs"


def log(msg: str) -> None:
    print(msg, flush=True)


def ok(msg: str) -> None:
    print(f"[OK] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def fail(msg: str, exit_code: int = 1) -> None:
    print(f"[FAIL] {msg}", flush=True)
    sys.exit(exit_code)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class VastServerlessClient:
    def __init__(
        self,
        endpoint_name: str,
        api_key: str,
        route_url: str,
        timeout: int,
        route_cost: float,
        max_route_wait_seconds: int,
        route_poll_interval_seconds: float,
    ):
        self.endpoint_name = endpoint_name
        self.route_url = route_url
        self.timeout = timeout
        self.route_cost = route_cost
        self.max_route_wait_seconds = max_route_wait_seconds
        self.route_poll_interval_seconds = route_poll_interval_seconds
        self.request_idx: Optional[int] = None

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
        )

    def route(self, cost: Optional[float] = None) -> Dict[str, Any]:
        start = time.time()
        while True:
            payload: Dict[str, Any] = {
                "endpoint": self.endpoint_name,
                "cost": float(self.route_cost if cost is None else cost),
            }
            if self.request_idx is not None:
                payload["request_idx"] = self.request_idx

            resp = self.session.post(self.route_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            if "request_idx" in data:
                self.request_idx = data["request_idx"]

            if {"url", "signature", "reqnum"}.issubset(data.keys()):
                return data

            elapsed = time.time() - start
            if elapsed >= self.max_route_wait_seconds:
                raise RuntimeError(
                    f"/route did not return worker url within {self.max_route_wait_seconds}s. Last={data}"
                )

            time.sleep(self.route_poll_interval_seconds)

    def call_worker(self, route_data: Dict[str, Any], route_path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        worker_url = f"{route_data['url'].rstrip('/')}{route_path}"
        body = {
            "auth_data": {
                "signature": route_data["signature"],
                "cost": route_data["cost"],
                "endpoint": route_data["endpoint"],
                "reqnum": route_data["reqnum"],
                "url": route_data["url"],
                "request_idx": route_data.get("request_idx"),
            },
            "payload": payload,
        }
        resp = self.session.post(worker_url, json=body, timeout=self.timeout)
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"Worker response is not JSON: {resp.text[:500]}")

        if resp.status_code >= 400:
            raise RuntimeError(f"Worker returned HTTP {resp.status_code}: {json.dumps(data, ensure_ascii=False)}")

        return data

    def health(self) -> Dict[str, Any]:
        return self.call_worker(self.route(cost=1.0), "/health", {})

    def process(self, payload: Dict[str, Any], cost: Optional[float] = None) -> Dict[str, Any]:
        use_cost = cost
        if use_cost is None:
            use_cost = float(max(1.0, payload.get("duration_seconds", 10.0)))
        return self.call_worker(self.route(cost=use_cost), "/process/sync", payload)


def save_artifacts_from_response(response: Dict[str, Any], out_dir: Path, test_name: str) -> Dict[str, str]:
    saved: Dict[str, str] = {}
    artifacts = response.get("artifacts") or {}

    for key, value in artifacts.items():
        if not isinstance(value, dict):
            continue

        b64 = value.get("base64")
        if b64:
            data = base64.b64decode(b64)
            ext = {
                "srt": "srt",
                "vtt": "vtt",
                "json": "json",
            }.get(key, "bin")
            out_path = out_dir / f"{test_name}_{key}.{ext}"
            out_path.write_bytes(data)
            saved[key] = str(out_path.resolve())

    return saved


def run_health_test(client: VastServerlessClient) -> Dict[str, Any]:
    log("\n=== TEST: /health via Vast route ===")
    data = client.health()
    if not data.get("ok"):
        raise RuntimeError("/health returned ok=false")
    ok("health check passed")
    return data


def run_basic_transcription(client: VastServerlessClient, sample_audio_url: str, out_dir: Path) -> Dict[str, Any]:
    log("\n=== TEST: basic transcription ===")
    payload = {
        "audio_url": sample_audio_url,
        "task": "transcribe",
        "language": "en",
        "enable_demucs": False,
        "enable_diarization": False,
        "return_word_timestamps": True,
        "return_segments": True,
        "return_srt": True,
        "return_vtt": True,
        "return_base64_outputs": True,
        "save_to_disk": False,
        "duration_seconds": 1.0,
    }
    resp = client.process(payload)
    if not resp.get("ok"):
        raise RuntimeError("basic transcription returned ok=false")
    if not resp.get("text"):
        warn("basic transcription returned empty text")

    saved = save_artifacts_from_response(resp, out_dir, "basic")
    ok("basic transcription passed")
    return {"response": resp, "saved_artifacts": saved}


def run_demucs_toggle(client: VastServerlessClient, sample_audio_url: str, enabled: bool, out_dir: Path) -> Dict[str, Any]:
    name = "demucs_on" if enabled else "demucs_off"
    log(f"\n=== TEST: {name} ===")
    payload = {
        "audio_url": sample_audio_url,
        "task": "transcribe",
        "language": "en",
        "enable_demucs": enabled,
        "enable_diarization": False,
        "return_word_timestamps": True,
        "return_segments": True,
        "return_srt": False,
        "return_vtt": False,
        "return_base64_outputs": True,
        "save_to_disk": False,
        "duration_seconds": 1.0,
    }
    resp = client.process(payload)
    if not resp.get("ok"):
        raise RuntimeError(f"{name} returned ok=false")

    if enabled and "vocals" not in (resp.get("artifacts") or {}):
        raise RuntimeError("demucs enabled but vocals artifact missing")

    saved = save_artifacts_from_response(resp, out_dir, name)
    ok(f"{name} passed")
    return {"response": resp, "saved_artifacts": saved}


def run_diarization_toggle(client: VastServerlessClient, sample_audio_url: str, enabled: bool, out_dir: Path) -> Dict[str, Any]:
    name = "diarization_on" if enabled else "diarization_off"
    log(f"\n=== TEST: {name} ===")
    payload = {
        "audio_url": sample_audio_url,
        "task": "transcribe",
        "language": "en",
        "enable_demucs": False,
        "enable_diarization": enabled,
        "num_speakers": 1,
        "return_word_timestamps": True,
        "return_segments": True,
        "return_srt": False,
        "return_vtt": False,
        "return_base64_outputs": True,
        "save_to_disk": False,
        "duration_seconds": 1.0,
    }
    resp = client.process(payload, cost=30.0 if enabled else 10.0)
    if not resp.get("ok"):
        raise RuntimeError(f"{name} returned ok=false")

    segments = resp.get("segments") or []
    if enabled:
        if not segments:
            warn("diarization enabled but no segments returned")
    saved = save_artifacts_from_response(resp, out_dir, name)
    ok(f"{name} passed")
    return {"response": resp, "saved_artifacts": saved}


def run_negative_tests(client: VastServerlessClient) -> Dict[str, str]:
    log("\n=== TEST: negative cases ===")
    results: Dict[str, str] = {}

    # empty input
    try:
        client.process({
            "task": "transcribe",
            "enable_demucs": False,
            "enable_diarization": False,
        }, cost=1.0)
        raise RuntimeError("empty input unexpectedly succeeded")
    except Exception:
        results["empty_input"] = "passed"

    # bad URL
    try:
        client.process(
            {
                "audio_url": "https://example.invalid/not_found.wav",
                "task": "transcribe",
                "enable_demucs": False,
                "enable_diarization": False,
            },
            cost=1.0,
        )
        raise RuntimeError("bad URL unexpectedly succeeded")
    except Exception:
        results["bad_url"] = "passed"

    # invalid format via base64
    invalid_blob = base64.b64encode(b"this is not audio").decode("utf-8")
    try:
        client.process(
            {
                "audio_base64": invalid_blob,
                "task": "transcribe",
                "enable_demucs": False,
                "enable_diarization": False,
            },
            cost=1.0,
        )
        raise RuntimeError("invalid format unexpectedly succeeded")
    except Exception:
        results["invalid_format"] = "passed"

    ok("negative cases passed")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Vast route test for demucs-whisperx serverless")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--route-url", default=ROUTE_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--route-cost", type=float, default=1.0)
    parser.add_argument("--max-route-wait-seconds", type=int, default=120)
    parser.add_argument("--route-poll-interval-seconds", type=float, default=2.0)
    parser.add_argument("--sample-audio-url", default="https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-diarization-on", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    client = VastServerlessClient(
        endpoint_name=args.endpoint_name,
        api_key=args.api_key,
        route_url=args.route_url,
        timeout=args.timeout,
        route_cost=args.route_cost,
        max_route_wait_seconds=args.max_route_wait_seconds,
        route_poll_interval_seconds=args.route_poll_interval_seconds,
    )

    summary: Dict[str, Any] = {
        "endpoint_name": args.endpoint_name,
        "route_url": args.route_url,
        "out_dir": str(out_dir.resolve()),
        "tests": {},
    }

    try:
        summary["tests"]["health"] = run_health_test(client)
        summary["tests"]["basic_transcription"] = run_basic_transcription(client, args.sample_audio_url, out_dir)
        summary["tests"]["demucs_off"] = run_demucs_toggle(client, args.sample_audio_url, False, out_dir)
        summary["tests"]["demucs_on"] = run_demucs_toggle(client, args.sample_audio_url, True, out_dir)
        summary["tests"]["diarization_off"] = run_diarization_toggle(client, args.sample_audio_url, False, out_dir)

        if args.skip_diarization_on:
            summary["tests"]["diarization_on"] = "skipped"
        else:
            summary["tests"]["diarization_on"] = run_diarization_toggle(client, args.sample_audio_url, True, out_dir)

        summary["tests"]["negative"] = run_negative_tests(client)

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        ok("All tests completed")
    except Exception as exc:
        summary["error"] = str(exc)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        fail(str(exc))


if __name__ == "__main__":
    main()
