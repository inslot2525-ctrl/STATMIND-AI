"""
Database models and connection.

Works with SQLite (default, zero setup) and PostgreSQL (set DATABASE_URL).
The schema is owned by Alembic migrations in backend/migrations; `init_db()`
upgrades the database to the latest revision on startup.
"""
import logging
import os
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint,
    create_engine, event, inspect,
)
from sqlalchemy.orm import declarative_base, sessionmaker

import config

logger = logging.getLogger(__name__)

DATABASE_URL = config.DATABASE_URL
IS_SQLITE = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30} if IS_SQLITE else {},
    pool_pre_ping=not IS_SQLITE,
    echo=False,
)

if IS_SQLITE:
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record):
        # WAL lets the job worker thread write while API requests read
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()


def utcnow() -> datetime:
    """Naive UTC timestamp (stored the same way on SQLite and Postgres)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"
    id = Column(String(32), primary_key=True)
    email = Column(String(320), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("owner_id", "content_hash", name="uq_datasets_owner_hash"),)
    id = Column(String, primary_key=True)
    owner_id = Column(String(32), ForeignKey("users.id"), nullable=True, index=True)
    content_hash = Column(String, index=True)
    original_filename = Column(String)
    stored_filename = Column(String)
    storage_key = Column(String, nullable=True)
    upload_time = Column(String)
    rows = Column(Integer, nullable=True)
    columns = Column(Integer, nullable=True)
    missing_cells = Column(Integer, nullable=True)
    duplicate_rows = Column(Integer, nullable=True)
    file_path = Column(String)  # legacy; use storage_key
    error = Column(Text, nullable=True)
    data_commit = Column(String(64), nullable=True)  # lakeFS commit containing this file


class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(String, primary_key=True)
    owner_id = Column(String(32), ForeignKey("users.id"), nullable=True, index=True)
    timestamp = Column(String)
    dataset_id = Column(String, index=True)
    dataset_info = Column(Text)  # JSON
    algorithm = Column(String)
    hyperparameters = Column(Text)  # JSON
    metrics = Column(Text)  # JSON
    model_id = Column(String, nullable=True)
    tags = Column(Text)  # JSON
    notes = Column(Text)


class ModelRegistry(Base):
    __tablename__ = "models"
    model_id = Column(String, primary_key=True)
    owner_id = Column(String(32), ForeignKey("users.id"), nullable=True, index=True)
    storage_key = Column(String, nullable=True)
    registered_at = Column(String)
    updated_at = Column(String)
    metadata_json = Column(Text)  # JSON
    versions_json = Column(Text)  # JSON


class Job(Base):
    __tablename__ = "jobs"
    id = Column(String(32), primary_key=True)
    owner_id = Column(String(32), ForeignKey("users.id"), nullable=False, index=True)
    kind = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False, index=True)  # queued|running|succeeded|failed
    params = Column(Text, nullable=False)  # JSON
    result = Column(Text, nullable=True)  # JSON
    error = Column(Text, nullable=True)
    progress = Column(Integer, nullable=False, default=0)
    message = Column(String(255), nullable=True)
    attempts = Column(Integer, nullable=False, default=0)
    worker_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    heartbeat_at = Column(DateTime, nullable=True)


class Artifact(Base):
    """A downloadable file produced for a user (prediction CSV, HTML report)."""
    __tablename__ = "artifacts"
    id = Column(String(32), primary_key=True)
    owner_id = Column(String(32), ForeignKey("users.id"), nullable=False, index=True)
    kind = Column(String(30), nullable=False)
    storage_key = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)


# ------------------------------------------------------------------ migrations
MIGRATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "migrations")

# Revision matching the schema that older versions created with create_all()
BASELINE_REVISION = "0001"


def _alembic_config():
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", MIGRATIONS_DIR)
    cfg.set_main_option("sqlalchemy.url", DATABASE_URL.replace("%", "%%"))
    return cfg


def init_db():
    """Bring the database schema up to date (safe to call repeatedly)."""
    from alembic import command

    cfg = _alembic_config()
    tables = set(inspect(engine).get_table_names())

    # Databases created before migrations existed: mark them as the baseline
    if "datasets" in tables and "alembic_version" not in tables:
        logger.info("Existing pre-migration database found; stamping baseline revision.")
        command.stamp(cfg, BASELINE_REVISION)

    command.upgrade(cfg, "head")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
