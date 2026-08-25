import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, Text
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = os.path.join(os.path.dirname(__file__), "statmind.db")
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False}, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(String, primary_key=True)
    content_hash = Column(String, unique=True, index=True)
    original_filename = Column(String)
    stored_filename = Column(String)
    upload_time = Column(String)
    rows = Column(Integer, nullable=True)
    columns = Column(Integer, nullable=True)
    missing_cells = Column(Integer, nullable=True)
    duplicate_rows = Column(Integer, nullable=True)
    file_path = Column(String)
    error = Column(Text, nullable=True)

class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(String, primary_key=True)
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
    registered_at = Column(String)
    updated_at = Column(String)
    metadata_json = Column(Text)  # JSON
    versions_json = Column(Text)  # JSON

def init_db():
    """Create tables and migrate existing JSON data if DB is empty."""
    Base.metadata.create_all(bind=engine)
    # Migrate JSON if tables are empty
    from sqlalchemy import func
    db = SessionLocal()
    try:
        ds_count = db.query(func.count(Dataset.id)).scalar()
        exp_count = db.query(func.count(Experiment.id)).scalar()
        model_count = db.query(func.count(ModelRegistry.model_id)).scalar()
        if ds_count == 0 and exp_count == 0 and model_count == 0:
            _migrate_json_to_db(db)
    finally:
        db.close()

def _migrate_json_to_db(db):
    """One-time migration from legacy JSON files."""
    import pathlib
    base = os.path.dirname(__file__)
    # datasets
    ds_path = os.path.join(base, "dataset_registry.json")
    if os.path.exists(ds_path):
        try:
            with open(ds_path, "r") as f:
                data = json.load(f)
            for did, info in data.get("datasets", {}).items():
                if db.query(Dataset).filter_by(id=did).first():
                    continue
                db.add(Dataset(
                    id=did,
                    content_hash=info.get("content_hash", ""),
                    original_filename=info.get("original_filename", ""),
                    stored_filename=info.get("stored_filename", ""),
                    upload_time=info.get("upload_time", datetime.now().isoformat()),
                    rows=info.get("rows"),
                    columns=info.get("columns"),
                    missing_cells=info.get("missing_cells"),
                    duplicate_rows=info.get("duplicate_rows"),
                    file_path=info.get("file_path", ""),
                    error=info.get("error")
                ))
            db.commit()
        except Exception as e:
            print(f"[migrate] datasets failed: {e}")
            db.rollback()
    # experiments
    exp_path = os.path.join(base, "experiment_tracking.json")
    if os.path.exists(exp_path):
        try:
            with open(exp_path, "r") as f:
                data = json.load(f)
            for eid, info in data.get("experiments", {}).items():
                if db.query(Experiment).filter_by(id=eid).first():
                    continue
                db.add(Experiment(
                    id=eid,
                    timestamp=info.get("timestamp", datetime.now().isoformat()),
                    dataset_id=info.get("dataset_id", ""),
                    dataset_info=json.dumps(info.get("dataset_info", {})),
                    algorithm=info.get("algorithm", ""),
                    hyperparameters=json.dumps(info.get("hyperparameters", {})),
                    metrics=json.dumps(info.get("metrics", {})),
                    model_id=info.get("model_id"),
                    tags=json.dumps(info.get("tags", [])),
                    notes=info.get("notes", "")
                ))
            db.commit()
        except Exception as e:
            print(f"[migrate] experiments failed: {e}")
            db.rollback()
    # models
    model_path = os.path.join(base, "model_registry.json")
    if os.path.exists(model_path):
        try:
            with open(model_path, "r") as f:
                data = json.load(f)
            for mid, info in data.get("models", {}).items():
                if db.query(ModelRegistry).filter_by(model_id=mid).first():
                    continue
                db.add(ModelRegistry(
                    model_id=mid,
                    registered_at=info.get("registered_at", datetime.now().isoformat()),
                    updated_at=info.get("updated_at", datetime.now().isoformat()),
                    metadata_json=json.dumps(info.get("metadata", {})),
                    versions_json=json.dumps(info.get("versions", []))
                ))
            db.commit()
        except Exception as e:
            print(f"[migrate] models failed: {e}")
            db.rollback()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
