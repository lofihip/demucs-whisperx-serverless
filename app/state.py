from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.models import ServiceStatus


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class RuntimeStatus:
    name: str
    state: str = "starting"
    current_job_id: str | None = None
    warmed_up_at: datetime | None = None
    last_error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def busy(self) -> bool:
        return self.state == "busy"

    @property
    def ready(self) -> bool:
        return self.state in {"ready", "busy"}

    def mark_starting(self) -> None:
        self.state = "starting"
        self.current_job_id = None

    def mark_ready(self, **details: Any) -> None:
        self.state = "ready"
        self.current_job_id = None
        self.last_error = None
        self.warmed_up_at = utcnow()
        if details:
            self.details.update(details)

    def mark_busy(self, job_id: str, **details: Any) -> None:
        self.state = "busy"
        self.current_job_id = job_id
        if details:
            self.details.update(details)

    def mark_error(self, error: str) -> None:
        self.state = "error"
        self.current_job_id = None
        self.last_error = error

    def recover_ready(self, error: str | None = None) -> None:
        self.state = "ready"
        self.current_job_id = None
        self.last_error = error

    def to_model(self) -> ServiceStatus:
        return ServiceStatus(
            name=self.name,
            state=self.state,
            ready=self.ready,
            busy=self.busy,
            current_job_id=self.current_job_id,
            warmed_up_at=self.warmed_up_at,
            last_error=self.last_error,
            details=self.details.copy(),
        )
