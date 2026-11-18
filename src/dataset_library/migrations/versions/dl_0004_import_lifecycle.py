"""Imports you can leave, cancel and retry (#114): the worker's heartbeat and
when the student put an import away."""
import sqlalchemy as sa
from alembic import op

revision = "dl_0004_import_lifecycle"
down_revision = "dl_0003_limits"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("library_imports", sa.Column("heartbeat_at", sa.DateTime(), nullable=True))
    op.add_column("library_imports", sa.Column("dismissed_at", sa.DateTime(), nullable=True))


def downgrade():
    op.drop_column("library_imports", "dismissed_at")
    op.drop_column("library_imports", "heartbeat_at")
