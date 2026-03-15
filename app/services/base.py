from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Awaitable, Callable, TypeVar

from app.config import Settings
from app.errors import ServiceBusyError, ServiceNotReadyError
from app.state import RuntimeStatus

T = TypeVar("T")


class BaseRuntimeService:
    def __init__(self, name: str, settings: Settings, gpu_lock: asyncio.Lock | None = None) -> None:
        self.name = name
        self.settings = settings
        self.logger = logging.getLogger(f"app.services.{name}")
        self.status = RuntimeStatus(name=name)
        self._service_lock = asyncio.Lock()
        self._gpu_lock = gpu_lock

    def snapshot(self):
        return self.status.to_model()

    def assert_ready(self) -> None:
        if not self.status.ready:
            raise ServiceNotReadyError(f"{self.name} is not ready yet.")

    async def preload(self) -> None:
        raise NotImplementedError

    async def _run_exclusive(self, job_id: str, fn: Callable[[], Awaitable[T]]) -> T:
        self.assert_ready()
        if self._service_lock.locked():
            raise ServiceBusyError(f"{self.name} is already processing another job.")

        async with self._service_lock:
            self.status.mark_busy(job_id)
            gpu_context = self._gpu_lock if self._gpu_lock is not None else _NullAsyncContext()
            async with gpu_context:
                try:
                    result = await fn()
                except Exception as exc:
                    self.status.recover_ready(str(exc))
                    raise
                else:
                    self.status.mark_ready()
                    return result

    def make_job_dir(self, job_id: str) -> Path:
        job_dir = self.settings.work_root / self.name / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir


class _NullAsyncContext:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None
