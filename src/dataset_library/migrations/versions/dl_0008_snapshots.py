"""The platform's copies of pinned datasets (#121), and which copy an import read."""
import sqlalchemy as sa
from alembic import op

revision = "dl_0008_snapshots"
down_revision = "dl_0007_citation"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("library_imports", sa.Column("source", sa.String(length=20), nullable=False, server_default="huggingface"))
    op.add_column("library_imports", sa.Column("snapshot_id", sa.String(length=36), nullable=True))
    op.create_table(
        "library_snapshots",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("pin_id", sa.String(length=36), nullable=True),
        sa.Column("provider", sa.String(length=40), nullable=False),
        sa.Column("external_id", sa.String(length=200), nullable=False),
        sa.Column("config", sa.String(length=200), nullable=True),
        sa.Column("split", sa.String(length=200), nullable=True),
        sa.Column("revision", sa.String(length=80), nullable=True),
        sa.Column("license", sa.String(length=80), nullable=True),
        sa.Column("columns", sa.JSON(), nullable=True),
        sa.Column("file_path", sa.String(length=200), nullable=True),
        sa.Column("row_count", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("error_code", sa.String(length=40), nullable=True),
        sa.Column("error_message", sa.String(length=500), nullable=True),
        sa.Column("upstream_changed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("upstream_checked_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("pin_id", name="uq_library_snapshots_pin"),
    )


def downgrade():
    op.drop_table("library_snapshots")
    op.drop_column("library_imports", "snapshot_id")
    op.drop_column("library_imports", "source")
