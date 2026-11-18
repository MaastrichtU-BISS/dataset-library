"""Curated pins: a course's starting datasets, featured or not (#119)."""
import sqlalchemy as sa
from alembic import op

revision = "dl_0006_pins"
down_revision = "dl_0005_label_balance"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "library_pins",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("provider", sa.String(length=40), nullable=False),
        sa.Column("external_id", sa.String(length=200), nullable=False),
        sa.Column("title", sa.String(length=120), nullable=True),
        sa.Column("featured", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("labelled", sa.String(length=10), nullable=False, server_default="unclear"),
        sa.Column("card", sa.JSON(), nullable=False),
        sa.Column("pinned_by", sa.String(length=255), nullable=True),
        sa.Column("pinned_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("provider", "external_id", name="uq_library_pins_dataset"),
    )


def downgrade():
    op.drop_table("library_pins")
