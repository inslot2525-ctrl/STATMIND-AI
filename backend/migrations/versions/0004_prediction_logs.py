"""Log prediction inputs and outputs per model, for drift monitoring

Revision ID: 0004
Revises: 0003
"""
from alembic import op
import sqlalchemy as sa

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "prediction_logs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("owner_id", sa.String(32), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("rows", sa.Integer(), nullable=False),
        sa.Column("logged_rows", sa.Integer(), nullable=False),
        sa.Column("features", sa.Text(), nullable=False),
        sa.Column("predictions", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_prediction_logs_owner_model", "prediction_logs", ["owner_id", "model_id", "created_at"])


def downgrade():
    op.drop_index("ix_prediction_logs_owner_model", table_name="prediction_logs")
    op.drop_table("prediction_logs")
