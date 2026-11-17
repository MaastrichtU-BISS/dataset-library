"""Admin-set limits: the row cap and the download cap (#118)."""
import sqlalchemy as sa
from alembic import op

revision = "dl_0003_limits"
down_revision = "dl_0002_imports"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("library_settings", sa.Column("max_rows", sa.Integer(), nullable=True))
    op.add_column("library_settings", sa.Column("max_download_mb", sa.Integer(), nullable=True))


def downgrade():
    op.drop_column("library_settings", "max_download_mb")
    op.drop_column("library_settings", "max_rows")
