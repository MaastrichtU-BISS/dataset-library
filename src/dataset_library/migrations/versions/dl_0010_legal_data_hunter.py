"""Legal Data Hunter entries (#126): an admin's search and its frozen document ids, the documents an import has
fetched so Try again resumes, and what an import of an entry fetches and skips."""
import sqlalchemy as sa
from alembic import op

revision = "dl_0010_legal_data_hunter"
down_revision = "dl_0009_community_file"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "library_ldh_entries",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("title", sa.String(length=120), nullable=False),
        sa.Column("description", sa.String(length=2000), nullable=True),
        sa.Column("task_type", sa.String(length=40), nullable=False, server_default="classification"),
        sa.Column("query", sa.JSON(), nullable=False),
        sa.Column("documents", sa.JSON(), nullable=False),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("refreshed_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "library_ldh_documents",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("import_id", sa.String(length=36), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("outcome", sa.String(length=20), nullable=False),
        sa.Column("row", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("import_id", "position", name="uq_library_ldh_documents_position"),
    )
    op.create_index("ix_library_ldh_documents_import_id", "library_ldh_documents", ["import_id"])
    op.add_column("library_imports", sa.Column("documents", sa.JSON(), nullable=True))
    op.add_column("library_imports", sa.Column("skipped", sa.Integer(), nullable=True))
    op.add_column("library_imports", sa.Column("missing", sa.Integer(), nullable=True))
    op.add_column("library_imports", sa.Column("resume_after", sa.DateTime(), nullable=True))


def downgrade():
    op.drop_column("library_imports", "resume_after")
    op.drop_column("library_imports", "missing")
    op.drop_column("library_imports", "skipped")
    op.drop_column("library_imports", "documents")
    op.drop_index("ix_library_ldh_documents_import_id", table_name="library_ldh_documents")
    op.drop_table("library_ldh_documents")
    op.drop_table("library_ldh_entries")
