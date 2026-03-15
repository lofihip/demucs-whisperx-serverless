from __future__ import annotations

import asyncio
import shutil
import uuid
import zipfile
from pathlib import Path

import aiofiles
import httpx
from fastapi import UploadFile

from app.config import Settings
from app.errors import InvalidInputError


async def save_upload(upload: UploadFile, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(destination, "wb") as output:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            await output.write(chunk)
    await upload.close()
    return destination


async def download_to_path(url: str, destination: Path, timeout_seconds: int = 3600) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            async with aiofiles.open(destination, "wb") as output:
                async for chunk in response.aiter_bytes():
                    await output.write(chunk)
    return destination


def resolve_input_source(
    settings: Settings,
    *,
    upload: UploadFile | None,
    source_url: str | None,
    source_path: str | None,
    job_dir: Path,
) -> asyncio.Task[Path]:
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    provided = [upload is not None, bool(source_url), bool(source_path)]
    if sum(provided) != 1:
        raise InvalidInputError("Provide exactly one input source: upload, source_url, or source_path.")

    if upload is not None:
        filename = upload.filename or f"upload-{uuid.uuid4().hex}"
        return asyncio.create_task(save_upload(upload, input_dir / filename))

    if source_url:
        filename = Path(source_url).name or f"download-{uuid.uuid4().hex}"
        return asyncio.create_task(download_to_path(source_url, input_dir / filename))

    path = Path(source_path or "")
    if not path.exists() or not path.is_file():
        raise InvalidInputError(f"Local source_path does not exist: {path}")
    destination = input_dir / path.name
    shutil.copy2(path, destination)
    return asyncio.create_task(asyncio.to_thread(lambda: destination))


def zip_directory(source_dir: Path, destination_zip: Path) -> Path:
    destination_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir))
    return destination_zip


def collect_artifacts(root: Path) -> list[tuple[Path, int]]:
    artifacts: list[tuple[Path, int]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            artifacts.append((path, path.stat().st_size))
    return artifacts


def remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
