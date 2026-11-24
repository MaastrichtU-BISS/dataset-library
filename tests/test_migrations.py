"""The shipped revisions build the tables the models describe."""
import importlib.util
import pathlib

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from dataset_library.models import LibraryBase

VERSIONS = pathlib.Path(__file__).parent.parent / "src/dataset_library/migrations/versions"


def _schema(engine):
    inspector = sa.inspect(engine)
    return {
        table: (
            {c["name"]: (str(c["type"]), c["nullable"]) for c in inspector.get_columns(table)},
            sorted(tuple(u["column_names"]) for u in inspector.get_unique_constraints(table)),
        )
        for table in inspector.get_table_names()
    }


def _migrated():
    """A scratch database with every shipped revision applied, in name order."""
    migrated = sa.create_engine("sqlite://")
    for path in sorted(VERSIONS.glob("dl_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        revision = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(revision)
        with migrated.begin() as connection:
            revision.op = Operations(MigrationContext.configure(connection))
            revision.upgrade()
    return migrated


def test_the_revisions_match_the_models():
    modelled = sa.create_engine("sqlite://")
    LibraryBase.metadata.create_all(modelled)

    assert _schema(_migrated()) == _schema(modelled)


def test_an_import_keeps_its_citation_as_nullable_json():
    assert _schema(_migrated())["library_imports"][0]["citation"] == ("JSON", True)


def _revision(name):
    path = VERSIONS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legal_data_hunter_adds_its_entries_fetched_documents_and_import_columns():
    """dl_0010 (#126): the entries table, the documents an import fetched, and what an import of an entry records."""
    schema = _schema(_migrated())

    assert set(schema["library_ldh_entries"][0]) == {
        "id", "title", "description", "task_type", "query", "documents", "created_by", "created_at", "updated_at", "refreshed_at",
    }
    assert schema["library_ldh_documents"][1] == [("import_id", "position")]
    for column in ("documents", "skipped", "missing", "resume_after"):
        assert schema["library_imports"][0][column][1] is True, column
    assert _revision("dl_0010_legal_data_hunter").down_revision == "dl_0009_community_file"


def test_legal_data_hunter_downgrades_to_the_community_schema():
    migrated = _migrated()
    revision = _revision("dl_0010_legal_data_hunter")
    with migrated.begin() as connection:
        revision.op = Operations(MigrationContext.configure(connection))
        revision.downgrade()

    schema = _schema(migrated)
    assert "library_ldh_entries" not in schema and "library_ldh_documents" not in schema
    assert not {"documents", "skipped", "missing", "resume_after"} & set(schema["library_imports"][0])
