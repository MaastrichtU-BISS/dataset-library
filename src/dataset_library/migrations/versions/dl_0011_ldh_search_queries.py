"""The queries a Legal Data Hunter search import was made from (#16).

A user may import the documents they ticked across their own Legal Data Hunter searches as a Dataset. The import keeps
the queries that found them, for the Dataset's provenance; an import of a curated entry keeps none (null).
"""
import sqlalchemy as sa
from alembic import op

revision = "dl_0011_ldh_search_queries"
down_revision = "dl_0010_legal_data_hunter"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("library_imports", sa.Column("queries", sa.JSON(), nullable=True))


def downgrade():
    op.drop_column("library_imports", "queries")
