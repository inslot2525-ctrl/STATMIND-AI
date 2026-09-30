"""Initial schema (as created by create_all before migrations existed)

Revision ID: 0001
Revises:
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "datasets",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("content_hash", sa.String()),
        sa.Column("original_filename", sa.String()),
        sa.Column("stored_filename", sa.String()),
        sa.Column("upload_time", sa.String()),
        sa.Column("rows", sa.Integer(), nullable=True),
        sa.Column("columns", sa.Integer(), nullable=True),
        sa.Column("missing_cells", sa.Integer(), nullable=True),
        sa.Column("duplicate_rows", sa.Integer(), nullable=True),
        sa.Column("file_path", sa.String()),
        sa.Column("error", sa.Text(), nullable=True),
    )
    op.create_index("ix_datasets_content_hash", "datasets", ["content_hash"], unique=True)

    op.create_table(
        "experiments",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("timestamp", sa.String()),
        sa.Column("dataset_id", sa.String()),
        sa.Column("dataset_info", sa.Text()),
        sa.Column("algorithm", sa.String()),
        sa.Column("hyperparameters", sa.Text()),
        sa.Column("metrics", sa.Text()),
        sa.Column("model_id", sa.String(), nullable=True),
        sa.Column("tags", sa.Text()),
        sa.Column("notes", sa.Text()),
    )
    op.create_index("ix_experiments_dataset_id", "experiments", ["dataset_id"])

    op.create_table(
        "models",
        sa.Column("model_id", sa.String(), primary_key=True),
        sa.Column("registered_at", sa.String()),
        sa.Column("updated_at", sa.String()),
        sa.Column("metadata_json", sa.Text()),
        sa.Column("versions_json", sa.Text()),
    )


def downgrade():
    op.drop_table("models")
    op.drop_index("ix_experiments_dataset_id", table_name="experiments")
    op.drop_table("experiments")
    op.drop_index("ix_datasets_content_hash", table_name="datasets")
    op.drop_table("datasets")
