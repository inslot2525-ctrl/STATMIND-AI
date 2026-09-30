"""Users, background jobs, artifacts, per-user ownership and storage keys

Revision ID: 0002
Revises: 0001
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "users",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("is_admin", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "jobs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("owner_id", sa.String(32), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("kind", sa.String(50), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("params", sa.Text(), nullable=False),
        sa.Column("result", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("progress", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("message", sa.String(255), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("worker_id", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_jobs_owner_id", "jobs", ["owner_id"])
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_index("ix_jobs_created_at", "jobs", ["created_at"])

    op.create_table(
        "artifacts",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("owner_id", sa.String(32), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("kind", sa.String(30), nullable=False),
        sa.Column("storage_key", sa.String(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_artifacts_owner_id", "artifacts", ["owner_id"])

    # Content hash is now unique per owner, not globally: two users may upload the same file
    op.drop_index("ix_datasets_content_hash", table_name="datasets")
    with op.batch_alter_table("datasets") as batch:
        batch.add_column(sa.Column("owner_id", sa.String(32), nullable=True))
        batch.add_column(sa.Column("storage_key", sa.String(), nullable=True))
        batch.create_foreign_key("fk_datasets_owner", "users", ["owner_id"], ["id"])
        batch.create_unique_constraint("uq_datasets_owner_hash", ["owner_id", "content_hash"])
    op.create_index("ix_datasets_content_hash", "datasets", ["content_hash"])
    op.create_index("ix_datasets_owner_id", "datasets", ["owner_id"])

    with op.batch_alter_table("experiments") as batch:
        batch.add_column(sa.Column("owner_id", sa.String(32), nullable=True))
        batch.create_foreign_key("fk_experiments_owner", "users", ["owner_id"], ["id"])
    op.create_index("ix_experiments_owner_id", "experiments", ["owner_id"])

    with op.batch_alter_table("models") as batch:
        batch.add_column(sa.Column("owner_id", sa.String(32), nullable=True))
        batch.add_column(sa.Column("storage_key", sa.String(), nullable=True))
        batch.create_foreign_key("fk_models_owner", "users", ["owner_id"], ["id"])
    op.create_index("ix_models_owner_id", "models", ["owner_id"])

    # Existing rows: point at their files through storage keys. They stay unowned
    # until claimed with `python manage.py claim-legacy-data <email>`.
    op.execute(
        "UPDATE datasets SET storage_key = 'uploads/' || stored_filename "
        "WHERE storage_key IS NULL AND stored_filename IS NOT NULL AND stored_filename <> ''"
    )
    op.execute("UPDATE models SET storage_key = 'models/' || model_id WHERE storage_key IS NULL")


def downgrade():
    op.drop_index("ix_models_owner_id", table_name="models")
    with op.batch_alter_table("models") as batch:
        batch.drop_constraint("fk_models_owner", type_="foreignkey")
        batch.drop_column("storage_key")
        batch.drop_column("owner_id")

    op.drop_index("ix_experiments_owner_id", table_name="experiments")
    with op.batch_alter_table("experiments") as batch:
        batch.drop_constraint("fk_experiments_owner", type_="foreignkey")
        batch.drop_column("owner_id")

    op.drop_index("ix_datasets_owner_id", table_name="datasets")
    op.drop_index("ix_datasets_content_hash", table_name="datasets")
    with op.batch_alter_table("datasets") as batch:
        batch.drop_constraint("uq_datasets_owner_hash", type_="unique")
        batch.drop_constraint("fk_datasets_owner", type_="foreignkey")
        batch.drop_column("storage_key")
        batch.drop_column("owner_id")
    op.create_index("ix_datasets_content_hash", "datasets", ["content_hash"], unique=True)

    op.drop_index("ix_artifacts_owner_id", table_name="artifacts")
    op.drop_table("artifacts")
    op.drop_index("ix_jobs_created_at", table_name="jobs")
    op.drop_index("ix_jobs_status", table_name="jobs")
    op.drop_index("ix_jobs_owner_id", table_name="jobs")
    op.drop_table("jobs")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
