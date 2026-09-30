"""Job queue semantics, exercised directly (as a separate worker process would)."""
import json
import threading
from datetime import timedelta

import pytest

import config
from database import Job, SessionLocal, utcnow
from services import job_tasks, jobs


@pytest.fixture
def queue_mode(monkeypatch):
    """Enqueue without running, like the API does with JOB_BACKEND=worker."""
    monkeypatch.setattr(config, "JOB_BACKEND", "worker")


@pytest.fixture
def owner(make_user):
    from database import User

    email, _ = make_user()
    db = SessionLocal()
    user_id = db.query(User).filter_by(email=email).one().id
    db.close()
    return user_id


@pytest.fixture
def fake_tasks(monkeypatch):
    calls = []

    def ok(params, owner_id, progress):
        progress(50, "halfway")
        calls.append(params)
        return {"answer": params["x"] * 2}

    def bad_input(params, owner_id, progress):
        raise ValueError("Column 'foo' not found")

    def crash(params, owner_id, progress):
        raise RuntimeError("secret internal detail")

    monkeypatch.setitem(job_tasks.TASKS, "ok", ok)
    monkeypatch.setitem(job_tasks.TASKS, "bad_input", bad_input)
    monkeypatch.setitem(job_tasks.TASKS, "crash", crash)
    return calls


def _drain(worker_id="test-worker"):
    """Run queued jobs until the queue is empty."""
    while job_id := jobs.claim_next(worker_id):
        jobs.execute_job(job_id, worker_id)


def test_enqueue_claim_execute(queue_mode, fake_tasks, owner):
    job = jobs.enqueue("ok", {"x": 21}, owner)
    assert job["status"] == "queued"

    _drain()

    done = jobs.get_job(job["id"], owner)
    assert done["status"] == "succeeded"
    assert done["result"] == {"answer": 42}
    assert done["progress"] == 100 and done["attempts"] == 1


def test_a_job_is_claimed_only_once(queue_mode, fake_tasks, owner):
    job = jobs.enqueue("ok", {"x": 1}, owner)
    winners = []

    def claim(name):
        if jobs._claim(job["id"], name):
            winners.append(name)

    threads = [threading.Thread(target=claim, args=(f"w{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(winners) == 1


def test_user_errors_are_shown_but_crashes_are_not_leaked(queue_mode, fake_tasks, owner):
    bad = jobs.enqueue("bad_input", {}, owner)
    crash = jobs.enqueue("crash", {}, owner)
    _drain()

    assert jobs.get_job(bad["id"], owner)["error"] == "Column 'foo' not found"
    crashed = jobs.get_job(crash["id"], owner)
    assert crashed["status"] == "failed"
    assert "secret internal detail" not in crashed["error"]


def test_unknown_job_kind_rejected(owner):
    with pytest.raises(ValueError):
        jobs.enqueue("rm -rf", {}, owner)


def _make_stale(job_id, attempts):
    db = SessionLocal()
    db.query(Job).filter_by(id=job_id).update({
        "status": "running",
        "worker_id": "dead-worker",
        "attempts": attempts,
        "heartbeat_at": utcnow() - timedelta(seconds=config.JOB_STALE_SECONDS + 60),
    })
    db.commit()
    db.close()


def test_stale_job_is_requeued_then_completes(queue_mode, fake_tasks, owner):
    job = jobs.enqueue("ok", {"x": 5}, owner)
    _make_stale(job["id"], attempts=1)

    assert jobs.requeue_stale_jobs() >= 1
    assert jobs.get_job(job["id"], owner)["status"] == "queued"

    _drain()
    assert jobs.get_job(job["id"], owner)["status"] == "succeeded"


def test_stale_job_fails_after_max_attempts(queue_mode, fake_tasks, owner):
    job = jobs.enqueue("ok", {"x": 5}, owner)
    _make_stale(job["id"], attempts=config.JOB_MAX_ATTEMPTS)

    jobs.requeue_stale_jobs()
    failed = jobs.get_job(job["id"], owner)
    assert failed["status"] == "failed"
    assert "stopped responding" in failed["error"]


def test_stale_worker_cannot_overwrite_a_reclaimed_job(queue_mode, fake_tasks, owner):
    job = jobs.enqueue("ok", {"x": 1}, owner)
    _make_stale(job["id"], attempts=1)
    jobs.requeue_stale_jobs()
    assert jobs._claim(job["id"], "new-worker")

    # The old worker wakes up and tries to report: it must be ignored
    jobs._update_running(job["id"], "dead-worker", status="failed", error="zombie")
    assert jobs.get_job(job["id"], owner)["status"] == "running"


def test_background_worker_thread_processes_queue(queue_mode, fake_tasks, owner, monkeypatch):
    monkeypatch.setattr(config, "JOB_POLL_SECONDS", 0.05)
    job = jobs.enqueue("ok", {"x": 3}, owner)

    worker = jobs.start_background_worker()
    try:
        for _ in range(100):
            if jobs.get_job(job["id"], owner)["status"] == "succeeded":
                break
            threading.Event().wait(0.05)
    finally:
        worker.stop()
    assert jobs.get_job(job["id"], owner)["result"] == {"answer": 6}


def test_job_params_are_stored_as_json(queue_mode, fake_tasks, owner):
    job = jobs.enqueue("ok", {"x": 7, "nested": {"a": [1, 2]}}, owner)
    db = SessionLocal()
    row = db.get(Job, job["id"])
    db.close()
    assert json.loads(row.params) == {"x": 7, "nested": {"a": [1, 2]}}
