from __future__ import annotations

import asyncio
from typing import Any

from app.config import Settings, get_settings
from app.models import SystemStatus
from app.services.demucs import DemucsService
from app.services.whisperx import WhisperXService


class ApplicationContext:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._gpu_lock = asyncio.Lock() if settings.gpu_serial_execution else None
        self.demucs = DemucsService(settings, gpu_lock=self._gpu_lock)
        self.whisperx = WhisperXService(settings, gpu_lock=self._gpu_lock)

    async def preload_all(self) -> None:
        await self.demucs.preload()
        await self.whisperx.preload()

    def snapshot(self) -> dict[str, Any]:
        status = SystemStatus(
            overall_ready=self.demucs.status.ready and self.whisperx.status.ready,
            gpu_device=self.settings.gpu_device,
            demucs=self.demucs.snapshot(),
            whisperx=self.whisperx.snapshot(),
        )
        return status.model_dump(mode="json")


def create_application_context() -> ApplicationContext:
    return ApplicationContext(get_settings())
