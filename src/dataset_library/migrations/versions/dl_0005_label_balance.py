"""Keeping the label balance when sampling (#115): the choice, the method used
and why it fell back to a random sample."""
import sqlalchemy as sa
from alembic import op

revision = "dl_0005_label_balance"
down_revision = "dl_0004_import_lifecycle"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("library_imports", sa.Column("balance", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("library_imports", sa.Column("sample_method", sa.String(length=20), nullable=True))
    op.add_column("library_imports", sa.Column("sample_note", sa.String(length=300), nullable=True))


def downgrade():
    op.drop_column("library_imports", "sample_note")
    op.drop_column("library_imports", "sample_method")
    op.drop_column("library_imports", "balance")
