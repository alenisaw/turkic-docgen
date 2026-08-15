"""
TurkicDocGen Web Panel — models.py
Data models for job tracking and run state.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class JobConfig:
    profile: str
    out_dir: str
    count: int
    seed: int
    languages: tuple[str, ...] | None = None
    layouts: tuple[str, ...] | None = None
    effects: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.languages is not None and not isinstance(self.languages, tuple):
            object.__setattr__(self, "languages", tuple(self.languages))
        if self.layouts is not None and not isinstance(self.layouts, tuple):
            object.__setattr__(self, "layouts", tuple(self.layouts))
        if self.effects is not None and not isinstance(self.effects, tuple):
            object.__setattr__(self, "effects", tuple(self.effects))


class Job:
    def __init__(self, config: JobConfig) -> None:
        self.job_id: str = uuid.uuid4().hex
        self.profile = config.profile
        self.out_dir = config.out_dir
        self.count = config.count
        self.seed = config.seed
        self.languages = list(config.languages or ())
        self.layouts = list(config.layouts or ())
        self.effects = list(config.effects or ())
        self.status: JobStatus = JobStatus.PENDING
        self.created_at: float = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.log_lines: list[str] = []
        self.progress: int = 0
        self.total: int = config.count
        self.error: str | None = None
        self._process: Any = None  # subprocess.Popen handle
        self.lock = threading.Lock()

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "profile": self.profile,
            "out_dir": self.out_dir,
            "count": self.count,
            "seed": self.seed,
            "languages": self.languages,
            "layouts": self.layouts,
            "effects": self.effects,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": self.progress,
            "total": self.total,
            "error": self.error,
        }
