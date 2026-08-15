"""
TurkicDocGen Web Panel — jobs.py
Background job management: start, stream logs, cancel.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import subprocess
import sys
import threading
import time

from .models import Job, JobConfig, JobStatus

# In-memory job registry
_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()
logger = logging.getLogger("uvicorn.error")

MAX_JOBS_IN_MEMORY = 50


def get_all_jobs() -> list[Job]:
    with _jobs_lock:
        return sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)


def get_job(job_id: str) -> Job | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def create_job(config: JobConfig) -> Job:
    job = Job(config)
    with _jobs_lock:
        # Evict oldest jobs to prevent memory leak (L-1)
        if len(_jobs) >= MAX_JOBS_IN_MEMORY:
            oldest = sorted(_jobs.values(), key=lambda j: j.created_at)
            for old_job in oldest:
                if old_job.status in (
                    JobStatus.DONE,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                ):
                    _jobs.pop(old_job.job_id, None)
                    break
        _jobs[job.job_id] = job
    return job


def start_job(job: Job) -> None:
    """Launch the generation subprocess in a background thread."""
    thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
    thread.start()


def _job_command(job: Job) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "turkicdocgen",
        "pipeline",
        "--profile",
        job.profile,
        "--out",
        job.out_dir,
        "--count",
        str(job.count),
        "--seed",
        str(job.seed),
        "--force",
    ]
    if job.languages:
        command.extend(["--language", job.languages[0]])
    if job.layouts:
        command.extend(["--layout", job.layouts[0]])
    if job.effects:
        command.extend(["--effect", job.effects[0]])
    return command


def _record_progress(job: Job, raw_line: str) -> None:
    line = raw_line.rstrip("\n\r")
    with job.lock:
        job.log_lines.append(line)
    if not line.startswith("progress:"):
        return
    try:
        progress, total = line.split(":", 1)[1].strip().split("/")
        with job.lock:
            job.progress = int(progress)
            job.total = int(total)
    except (ValueError, IndexError):
        return


def _record_process_result(job: Job, returncode: int | None) -> None:
    with job.lock:
        if job.status == JobStatus.CANCELLED:
            return
        job.status = JobStatus.DONE if returncode == 0 else JobStatus.FAILED
        if returncode != 0:
            job.error = f"Exit code {returncode}"


def _record_job_error(job: Job, exc: Exception, *, unexpected: bool) -> None:
    if unexpected:
        logger.error(
            "Unexpected error in background job %s: %s",
            job.job_id,
            exc,
            exc_info=True,
        )
    else:
        logger.exception("Job %s subprocess execution failed", job.job_id)
    with job.lock:
        job.status = JobStatus.FAILED
        prefix = "Unexpected error: " if unexpected else ""
        job.error = f"{prefix}{exc}"


def _run_job(job: Job) -> None:
    with job.lock:
        job.status = JobStatus.RUNNING
        job.started_at = time.time()

    try:
        process = subprocess.Popen(
            _job_command(job),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        with job.lock:
            job._process = process

        for raw_line in process.stdout:  # type: ignore[union-attr]
            _record_progress(job, raw_line)
        process.wait()
        _record_process_result(job, process.returncode)
    except (OSError, ValueError) as exc:
        _record_job_error(job, exc, unexpected=False)
    except Exception as exc:
        _record_job_error(job, exc, unexpected=True)
    finally:
        with job.lock:
            job.finished_at = time.time()


def cancel_job(job: Job) -> bool:
    with job.lock:
        if job.status != JobStatus.RUNNING:
            return False
        job.status = JobStatus.CANCELLED
        process = job._process
    if process is not None:
        with contextlib.suppress(OSError, ProcessLookupError):
            process.terminate()
    return True


async def log_event_generator(job: Job):
    """Async generator: yield SSE lines from job.log_lines."""
    sent = 0
    while True:
        with job.lock:
            current = list(job.log_lines)
            status = job.status

        while sent < len(current):
            line = current[sent]
            yield f"data: {line}\n\n"
            sent += 1

        if status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED):
            # Yield any remaining lines
            with job.lock:
                current = list(job.log_lines)
            while sent < len(current):
                line = current[sent]
                yield f"data: {line}\n\n"
                sent += 1
            yield "data: __DONE__\n\n"
            break

        # SSE heartbeat to prevent connection timeout (I-5)
        yield ": heartbeat\n\n"
        await asyncio.sleep(0.3)
