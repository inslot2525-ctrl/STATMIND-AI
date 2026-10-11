"""Record the lakeFS commit that contains each dataset

Revision ID: 0003
Revises: 0002
"""
from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("datasets") as batch:
        batch.add_column(sa.Column("data_commit", sa.String(64), nullable=True))


def downgrade():
    with op.batch_alter_table("datasets") as batch:
        batch.drop_column("data_commit")
