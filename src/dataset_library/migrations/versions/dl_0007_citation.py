"""Citing an imported dataset (#120): the citation fields, kept with the import."""
import sqlalchemy as sa
from alembic import op

revision = "dl_0007_citation"
down_revision = "dl_0006_pins"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("library_imports", sa.Column("citation", sa.JSON(), nullable=True))


def downgrade():
    op.drop_column("library_imports", "citation")
