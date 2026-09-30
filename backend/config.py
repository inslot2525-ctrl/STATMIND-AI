"""
Runtime configuration, read once from environment variables.

Local development needs no configuration at all: SQLite, local file storage and
an in-process job worker. Production switches each piece on with env vars
(see the Configuration table in README.md).
"""
import logging
import os
import secrets

logger = logging.getLogger(__name__)

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "true" if default else "false").lower() in ("1", "true", "yes", "on")


ENVIRONMENT = _env("STATMIND_ENV", "development")  # development | production | test
IS_PRODUCTION = ENVIRONMENT == "production"

# --- Storage --------------------------------------------------------------
DATA_DIR = os.path.abspath(_env("STATMIND_DATA_DIR", BACKEND_DIR))
STORAGE_BACKEND = _env("STORAGE_BACKEND", "local")  # local | s3
S3_BUCKET = _env("S3_BUCKET")
S3_ENDPOINT_URL = _env("S3_ENDPOINT_URL")  # set for Cloudflare R2 / MinIO; empty for AWS
S3_REGION = _env("S3_REGION", "auto" if S3_ENDPOINT_URL else "us-east-1")
S3_PREFIX = _env("S3_PREFIX", "")
MAX_UPLOAD_MB = float(_env("MAX_UPLOAD_MB", "50"))

# --- Database -------------------------------------------------------------
DATABASE_URL = _env("DATABASE_URL", f"sqlite:///{os.path.join(DATA_DIR, 'statmind.db')}")
# Railway/Heroku hand out postgres:// URLs; SQLAlchemy needs an explicit driver
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL[len("postgres://"):]
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL[len("postgresql://"):]

# --- Auth -----------------------------------------------------------------
JWT_SECRET = _env("JWT_SECRET")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_MINUTES = _env_int("ACCESS_TOKEN_MINUTES", 60 * 12)
DOWNLOAD_LINK_SECONDS = _env_int("DOWNLOAD_LINK_SECONDS", 300)
ALLOW_REGISTRATION = _env_bool("ALLOW_REGISTRATION", True)

if not JWT_SECRET:
    if IS_PRODUCTION:
        raise RuntimeError("JWT_SECRET must be set when STATMIND_ENV=production.")
    JWT_SECRET = secrets.token_urlsafe(48)
    logger.warning("JWT_SECRET not set: using a random secret. Logins will not survive a restart.")

# --- Jobs -----------------------------------------------------------------
# thread : the API process also runs a worker thread (default; single-box deploys)
# worker : the API only enqueues; run `python worker.py` as a separate process
# inline : jobs run synchronously inside the request (tests)
JOB_BACKEND = _env("JOB_BACKEND", "thread")
JOB_POLL_SECONDS = float(_env("JOB_POLL_SECONDS", "1"))
JOB_STALE_SECONDS = _env_int("JOB_STALE_SECONDS", 120)
JOB_MAX_ATTEMPTS = _env_int("JOB_MAX_ATTEMPTS", 3)
