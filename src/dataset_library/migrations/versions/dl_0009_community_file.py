"""A Community import's Publication, Zenodo origin and file (#122).

An import of a Zenodo record is bound to the Publication it was started from, to the Zenodo host
its record is on, and to the one file it reads, so a queued import can be checked against exactly
what the student saw.
"""
import sqlalchemy as sa
from alembic import op

revision = "dl_0009_community_file"
down_revision = "dl_0008_snapshots"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("library_imports", sa.Column("publication_id", sa.String(length=36), nullable=True))
    op.add_column("library_imports", sa.Column("origin", sa.String(length=200), nullable=True))
    op.add_column("library_imports", sa.Column("file_key", sa.String(length=500), nullable=True))


def downgrade():
    op.drop_column("library_imports", "file_key")
    op.drop_column("library_imports", "origin")
    op.drop_column("library_imports", "publication_id")
