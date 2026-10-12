"""
Upgrading a database created by the pre-accounts version of StatMind.
Runs in a subprocess so it gets its own data directory and database.
"""
import os
import sqlite3
import subprocess
import sys

import pytest

from tests.conftest import BACKEND_DIR

# Schema exactly as the old create_all() produced it
LEGACY_SCHEMA = """
CREATE TABLE datasets (
    id VARCHAR PRIMARY KEY, content_hash VARCHAR, original_filename VARCHAR,
    stored_filename VARCHAR, upload_time VARCHAR, rows INTEGER, columns INTEGER,
    missing_cells INTEGER, duplicate_rows INTEGER, file_path VARCHAR, error TEXT
);
CREATE UNIQUE INDEX ix_datasets_content_hash ON datasets (content_hash);
CREATE TABLE experiments (
    id VARCHAR PRIMARY KEY, timestamp VARCHAR, dataset_id VARCHAR, dataset_info TEXT,
    algorithm VARCHAR, hyperparameters TEXT, metrics TEXT, model_id VARCHAR, tags TEXT, notes TEXT
);
CREATE INDEX ix_experiments_dataset_id ON experiments (dataset_id);
CREATE TABLE models (
    model_id VARCHAR PRIMARY KEY, registered_at VARCHAR, updated_at VARCHAR,
    metadata_json TEXT, versions_json TEXT
);
INSERT INTO datasets (id, content_hash, original_filename, stored_filename, rows, file_path)
    VALUES ('1', 'abc', 'sales.csv', 'sales.csv', 20, 'uploads/sales.csv');
INSERT INTO models (model_id, metadata_json, versions_json)
    VALUES ('random_forest_y_1234abcd.pkl', '{}', '[]');
INSERT INTO experiments (id, dataset_id, algorithm, model_id)
    VALUES ('1', '1', 'random_forest', 'random_forest_y_1234abcd.pkl');
"""


def _run(data_dir, code):
    env = {
        **os.environ,
        "STATMIND_DATA_DIR": str(data_dir),
        "JWT_SECRET": "x",
        "PYTHONPATH": BACKEND_DIR,
    }
    env.pop("DATABASE_URL", None)
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND_DIR, env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.fixture
def legacy_dir(tmp_path):
    conn = sqlite3.connect(tmp_path / "statmind.db")
    conn.executescript(LEGACY_SCHEMA)
    conn.commit()
    conn.close()
    return tmp_path


def test_legacy_database_upgrades_and_keeps_data(legacy_dir):
    _run(legacy_dir, "from database import init_db; init_db(); init_db()")

    conn = sqlite3.connect(legacy_dir / "statmind.db")
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"users", "jobs", "artifacts", "alembic_version"} <= tables
    assert conn.execute("SELECT version_num FROM alembic_version").fetchone()[0] == "0004"
    assert conn.execute("SELECT storage_key, owner_id, rows FROM datasets").fetchone() == ("uploads/sales.csv", None, 20)
    assert conn.execute("SELECT storage_key FROM models").fetchone() == ("models/random_forest_y_1234abcd.pkl",)
    conn.close()


def test_claim_legacy_data(legacy_dir):
    out = _run(legacy_dir, (
        "import sys; from database import init_db; init_db()\n"
        "from services.auth import create_user; create_user('owner@example.com', 'long-password')\n"
        "sys.argv = ['manage.py', 'claim-legacy-data', 'owner@example.com']\n"
        "import manage; manage.main()"
    ))
    assert "1 datasets" in out and "1 experiments" in out and "1 models" in out

    conn = sqlite3.connect(legacy_dir / "statmind.db")
    owner = conn.execute("SELECT id FROM users").fetchone()[0]
    assert conn.execute("SELECT owner_id FROM datasets").fetchone()[0] == owner
    assert conn.execute("SELECT owner_id FROM models").fetchone()[0] == owner
    conn.close()


def test_fresh_database_downgrade_and_upgrade(tmp_path):
    _run(tmp_path, (
        "from alembic import command\n"
        "from database import _alembic_config, init_db\n"
        "init_db(); cfg = _alembic_config()\n"
        "command.downgrade(cfg, 'base'); command.upgrade(cfg, 'head')"
    ))
