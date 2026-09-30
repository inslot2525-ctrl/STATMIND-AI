"""
Standalone background worker.

Run this as a separate process (or container) when the API is started with
JOB_BACKEND=worker, so training never competes with API requests for CPU:

    cd backend && python worker.py

Several workers can run at once; they share the queue safely.
"""
import logging
import os
import signal

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from database import init_db  # noqa: E402
from services.jobs import Worker  # noqa: E402


def main():
    init_db()
    worker = Worker()

    def shutdown(signum, _frame):
        logging.getLogger(__name__).info("Signal %s received; finishing current job", signum)
        worker.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    worker.run()


if __name__ == "__main__":
    main()
