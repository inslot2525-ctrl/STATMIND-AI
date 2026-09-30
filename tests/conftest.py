"""
Test isolation: point the app at a throwaway data directory *before* any backend
module is imported, so tests never write to the real statmind.db, uploads/ or models/.
"""
import os
import shutil
import sys
import tempfile

_TEST_DATA_DIR = tempfile.mkdtemp(prefix="statmind-test-")
os.environ["STATMIND_DATA_DIR"] = _TEST_DATA_DIR
os.environ.pop("DATABASE_URL", None)
os.environ.setdefault("MAX_UPLOAD_MB", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TEST_DATA_DIR, ignore_errors=True)
