"""
Database-backed background job queue.

Why not Celery/RQ? The database is already the source of truth for job status
and results (the UI polls it), it needs no extra infrastructure, and it works on
Windows. Claiming uses an atomic conditional UPDATE, so any number of workers can
share one Postgres (or a single SQLite) database without taking a job twice.

Lifecycle: queued -> running -> succeeded | failed
A running job whose worker stops heart-beating (crash, redeploy) is put back in
the queue, up to JOB_MAX_ATTEMPTS, and then marked failed.
"""
import json
import logging
import os
import socket
import threading
import uuid
from datetime import timedelta

from sqlalchemy import update

import config
from database import Job, SessionLocal, utcnow

logger = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 15

# User-facing message for unexpected errors; details go to the server log only
INTERNAL_ERROR = "The job failed due to an internal error. Please try again."


def _to_dict(job: Job) -> dict:
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "params": json.loads(job.params) if job.params else {},
        "result": json.loads(job.result) if job.result else None,
        "error": job.error,
        "attempts": job.attempts,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


# ------------------------------------------------------------------ API side
def enqueue(kind: str, params: dict, owner_id: str) -> dict:
    from services.job_tasks import TASKS

    if kind not in TASKS:
        raise ValueError(f"Unknown job kind: {kind}")

    job = Job(
        id=uuid.uuid4().hex,
        owner_id=owner_id,
        kind=kind,
        status="queued",
        params=json.dumps(params),
        progress=0,
        message="Waiting for a worker",
        attempts=0,
        created_at=utcnow(),
    )
    db = SessionLocal()
    try:
        db.add(job)
        db.commit()
    finally:
        db.close()

    if config.JOB_BACKEND == "inline":
        if _claim(job.id, "inline"):
            execute_job(job.id, "inline")
        return get_job(job.id, owner_id)

    return _to_dict(job)


def get_job(job_id: str, owner_id: str) -> dict | None:
    db = SessionLocal()
    try:
        job = db.query(Job).filter_by(id=job_id, owner_id=owner_id).first()
        return _to_dict(job) if job else None
    finally:
        db.close()


def list_jobs(owner_id: str, limit: int = 50) -> list[dict]:
    db = SessionLocal()
    try:
        jobs = (
            db.query(Job).filter_by(owner_id=owner_id)
            .order_by(Job.created_at.desc()).limit(limit).all()
        )
        return [_to_dict(j) for j in jobs]
    finally:
        db.close()


# --------------------------------------------------------------- worker side
def _claim(job_id: str, worker_id: str) -> bool:
    """Atomically move one queued job to running. Returns True if we got it."""
    now = utcnow()
    db = SessionLocal()
    try:
        result = db.execute(
            update(Job)
            .where(Job.id == job_id, Job.status == "queued")
            .values(
                status="running", worker_id=worker_id, started_at=now, heartbeat_at=now,
                attempts=Job.attempts + 1, message="Starting",
            )
        )
        db.commit()
        return result.rowcount == 1
    finally:
        db.close()


def claim_next(worker_id: str) -> str | None:
    """Claim the oldest queued job, or return None if the queue is empty."""
    db = SessionLocal()
    try:
        candidates = [
            row.id for row in
            db.query(Job.id).filter_by(status="queued").order_by(Job.created_at).limit(5).all()
        ]
    finally:
        db.close()

    for job_id in candidates:  # another worker may win the race for the first one
        if _claim(job_id, worker_id):
            return job_id
    return None


def _update_running(job_id: str, worker_id: str, **values) -> None:
    """Update a job only while this worker still owns it."""
    db = SessionLocal()
    try:
        db.execute(
            update(Job)
            .where(Job.id == job_id, Job.worker_id == worker_id, Job.status == "running")
            .values(**values)
        )
        db.commit()
    finally:
        db.close()


def execute_job(job_id: str, worker_id: str) -> None:
    """Run a claimed job to completion and record the outcome."""
    from services.job_tasks import TASKS

    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        kind, owner_id, params = job.kind, job.owner_id, json.loads(job.params)
    finally:
        db.close()

    stop_heartbeat = threading.Event()

    def heartbeat():
        while not stop_heartbeat.wait(HEARTBEAT_SECONDS):
            _update_running(job_id, worker_id, heartbeat_at=utcnow())

    def progress(percent: int, message: str):
        _update_running(
            job_id, worker_id,
            progress=max(0, min(99, int(percent))), message=message[:255], heartbeat_at=utcnow(),
        )

    beat = threading.Thread(target=heartbeat, name=f"heartbeat-{job_id[:8]}", daemon=True)
    beat.start()
    logger.info("Job %s (%s) started on %s", job_id, kind, worker_id)
    try:
        result = TASKS[kind](params=params, owner_id=owner_id, progress=progress)
        _update_running(
            job_id, worker_id,
            status="succeeded", progress=100, message="Done", finished_at=utcnow(),
            result=json.dumps(result, default=str),
        )
        logger.info("Job %s succeeded", job_id)
    except ValueError as exc:
        # Expected, user-correctable problems (bad column, too few rows, ...)
        _update_running(
            job_id, worker_id,
            status="failed", error=str(exc), message="Failed", finished_at=utcnow(),
        )
        logger.info("Job %s failed: %s", job_id, exc)
    except Exception:
        logger.exception("Job %s crashed", job_id)
        _update_running(
            job_id, worker_id,
            status="failed", error=INTERNAL_ERROR, message="Failed", finished_at=utcnow(),
        )
    finally:
        stop_heartbeat.set()


def requeue_stale_jobs() -> int:
    """Recover jobs whose worker died. Returns how many jobs were touched."""
    cutoff = utcnow() - timedelta(seconds=config.JOB_STALE_SECONDS)
    db = SessionLocal()
    try:
        failed = db.execute(
            update(Job)
            .where(Job.status == "running", Job.heartbeat_at < cutoff,
                   Job.attempts >= config.JOB_MAX_ATTEMPTS)
            .values(status="failed", error="The worker stopped responding.", finished_at=utcnow())
        ).rowcount
        requeued = db.execute(
            update(Job)
            .where(Job.status == "running", Job.heartbeat_at < cutoff)
            .values(status="queued", worker_id=None, message="Retrying after a worker restart")
        ).rowcount
        db.commit()
    finally:
        db.close()

    if failed or requeued:
        logger.warning("Stale jobs: %d requeued, %d failed", requeued, failed)
    return failed + requeued


class Worker:
    """Polls the queue and runs jobs one at a time until stopped."""

    def __init__(self, name: str | None = None):
        self.worker_id = name or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.stop_event = threading.Event()

    def run(self):
        logger.info("Worker %s started", self.worker_id)
        last_reap = 0.0
        while not self.stop_event.is_set():
            try:
                now = utcnow().timestamp()
                if now - last_reap > 30:
                    requeue_stale_jobs()
                    last_reap = now

                job_id = claim_next(self.worker_id)
                if job_id:
                    execute_job(job_id, self.worker_id)
                    continue
            except Exception:
                logger.exception("Worker loop error")
            self.stop_event.wait(config.JOB_POLL_SECONDS)
        logger.info("Worker %s stopped", self.worker_id)

    def stop(self):
        self.stop_event.set()


def start_background_worker() -> Worker:
    """Run a worker in a daemon thread inside the API process (JOB_BACKEND=thread)."""
    worker = Worker()
    threading.Thread(target=worker.run, name="job-worker", daemon=True).start()
    return worker
