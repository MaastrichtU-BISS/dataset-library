"""A Dataset's provenance (#116): the completed import it came from, its
license and when the student confirmed that license."""
from datetime import datetime

import pytest

from conftest import ALICE, TASK_TYPES

from dataset_library import imports
from dataset_library.models import LibraryImport
from dataset_library.providers import huggingface
from test_citation import LEX_README_BIBTEX, LEXGLUE_AUTHORS, LEXGLUE_BIBTEX, SHA
from test_import import FakeCreateDataset, write_shard

BOB = {"X-User": "bob"}
LEXGLUE = {"dataset": "coastalcph/lex_glue", "config": "ledgar", "split": "train", "textColumn": "text", "labelColumn": "label", "size": 2}


@pytest.fixture
def cited_lex_glue(monkeypatch):
    """LexGLUE as the Hub answers for it, with its own BibTeX and a creation date."""
    card = huggingface._card(
        {
            "id": "coastalcph/lex_glue", "sha": SHA, "createdAt": "2021-10-17T11:42:31.000Z",
            "lastModified": "2024-01-04T14:25:27.000Z",
            "tags": ["license:cc-by-4.0", "task_categories:text-classification"],
            "cardData": {"pretty_name": "LexGLUE", "citation_bibtex": LEXGLUE_BIBTEX},
            "downloads": 1, "description": "",
        },
        TASK_TYPES,
    )
    real = huggingface.dataset
    monkeypatch.setattr(
        huggingface, "dataset",
        lambda external_id, task_types: card if external_id == "coastalcph/lex_glue" else real(external_id, task_types),
    )
    return card


def queue_lex_glue(app):
    """Queue LexGLUE through the route, as the student's Import button does."""
    response = app.test_client().post("/api/library/imports", json=LEXGLUE, headers=ALICE)
    assert response.status_code == 202, response.get_json()
    return response.get_json()


def finish(app, tmp_path, import_ref):
    """Run a queued import as the worker would; returns the dataset's provenance."""
    shard = write_shard(tmp_path / "0000.parquet", ["A clause", "Another"], [0, 1])
    with app.app_context():
        imports.run_import(
            import_ref, session=app.db.session,
            fetch_shards=lambda *_: imports.Fetched(paths=[str(shard)], revision=SHA),
            create_dataset=FakeCreateDataset(), data_dir=str(tmp_path / "data"), seed=7,
        )
    return app.test_client().get("/api/library/provenance/dataset-1", headers=ALICE).get_json()


def complete(app, tmp_path, *, license, license_ack_at=None, balance=False):
    """Run one import to completed, as the host's worker would."""
    shard = write_shard(tmp_path / "0000.parquet", ["A clause", "Another"], [0, 1])
    with app.app_context():
        session = app.db.session
        record = LibraryImport(
            owner_id="alice", provider="huggingface", external_id="coastalcph/lex_glue",
            config="ledgar", split="train", text_column="text", label_column="label",
            task_type="classification", sample_size=2, max_rows=1000, name="LexGLUE",
            license=license, license_ack_at=license_ack_at, balance=balance,
            url="https://huggingface.co/datasets/coastalcph/lex_glue", status="queued",
        )
        session.add(record)
        session.commit()

        def fetch_shards(external_id, config, split, max_bytes, dest_dir):
            return imports.Fetched(paths=[str(shard)], revision="abc123")

        imports.run_import(
            record.id, session=session, fetch_shards=fetch_shards,
            create_dataset=FakeCreateDataset(), data_dir=str(tmp_path / "data"), seed=7,
        )
        session.expunge_all()
        return session.get(LibraryImport, record.id).to_dict()


def test_a_confirmed_import_is_the_datasets_provenance(app, tmp_path):
    completed = complete(app, tmp_path, license="cc-by-nd-4.0", license_ack_at=datetime(2026, 1, 2, 3, 4, 5))
    assert completed["status"] == "completed"
    client = app.test_client()

    body = client.get("/api/library/provenance/dataset-1", headers=ALICE).get_json()

    assert body["provider"] == "huggingface"
    assert body["externalId"] == "coastalcph/lex_glue"
    assert body["config"] == "ledgar" and body["split"] == "train"
    assert body["revision"] == "abc123"
    assert body["url"] == "https://huggingface.co/datasets/coastalcph/lex_glue"
    assert body["importedAt"]
    assert body["licenseAckAt"] == "2026-01-02T03:04:05Z"
    assert body["license"]["id"] == "cc-by-nd-4.0"
    assert body["license"]["level"] == "warn"
    assert body["license"]["ack"]
    assert body["balance"] is False
    assert body["sampleMethod"] == "random"
    assert body["sampleNote"] is None


def test_provenance_says_when_the_sample_kept_the_label_balance(app, tmp_path):
    completed = complete(app, tmp_path, license="cc-by-4.0", balance=True)
    assert completed["balance"] is True
    assert completed["sampleMethod"] == "stratified"

    body = app.test_client().get("/api/library/provenance/dataset-1", headers=ALICE).get_json()

    assert body["balance"] is True
    assert body["sampleMethod"] == "stratified"
    assert body["sampleNote"] is None


def test_an_allow_licenses_import_has_no_confirmation_time(app, tmp_path):
    complete(app, tmp_path, license="cc-by-4.0")

    body = app.test_client().get("/api/library/provenance/dataset-1", headers=ALICE).get_json()

    assert body["license"]["id"] == "cc-by-4.0"
    assert body["licenseAckAt"] is None


def test_another_owner_and_an_unknown_dataset_get_not_found(app, tmp_path):
    complete(app, tmp_path, license="cc-by-4.0")
    client = app.test_client()

    assert client.get("/api/library/provenance/dataset-1", headers=BOB).status_code == 404
    assert client.get("/api/library/provenance/no-such-dataset", headers=ALICE).status_code == 404


@pytest.mark.parametrize("status", ["failed", "queued"])
def test_an_unfinished_import_is_not_a_datasets_provenance(app, status):
    with app.app_context():
        session = app.db.session
        session.add(LibraryImport(
            owner_id="alice", provider="huggingface", external_id="coastalcph/lex_glue",
            text_column="text", label_column="label", task_type="classification",
            sample_size=2, max_rows=1000, name="LexGLUE", license="cc-by-4.0",
            dataset_ref="dataset-1", status=status,
        ))
        session.commit()

    assert app.test_client().get("/api/library/provenance/dataset-1", headers=ALICE).status_code == 404


def test_the_newer_of_two_completed_imports_is_the_provenance(app, tmp_path):
    older = complete(app, tmp_path, license="cc-by-4.0")
    newer = complete(app, tmp_path, license="cc-by-4.0")
    with app.app_context():
        session = app.db.session
        session.get(LibraryImport, older["id"]).imported_at = datetime(2026, 1, 1)
        session.get(LibraryImport, newer["id"]).imported_at = datetime(2026, 2, 1)
        session.commit()

    body = app.test_client().get("/api/library/provenance/dataset-1", headers=ALICE).get_json()

    assert body["importedAt"] == "2026-02-01T00:00:00Z"


def test_provenance_reports_the_platforms_current_license_level(app, tmp_path):
    from dataset_library.models import LibrarySettings

    complete(app, tmp_path, license="cc-by-4.0")
    with app.app_context():
        app.db.session.add(LibrarySettings(id=1, license_policy={"cc-by-4.0": "warn"}))
        app.db.session.commit()

    body = app.test_client().get("/api/library/provenance/dataset-1", headers=ALICE).get_json()

    assert body["license"]["level"] == "warn"


def test_a_license_set_to_warn_without_its_own_words_gets_the_default_in_provenance(app, tmp_path):
    from dataset_library.models import LibrarySettings

    complete(app, tmp_path, license="mit", license_ack_at=datetime(2026, 1, 2, 3, 4, 5))
    with app.app_context():
        app.db.session.add(LibrarySettings(id=1, license_policy={"mit": "warn"}))
        app.db.session.commit()

    body = app.test_client().get("/api/library/provenance/dataset-1", headers=ALICE).get_json()

    assert body["license"]["level"] == "warn"
    assert body["license"]["ack"] == "I have read the MIT license and will follow its terms."


def test_an_import_cites_the_card_and_provenance_returns_that_citation(make_app, tmp_path, cited_lex_glue):
    app = make_app(enqueue=lambda import_ref: None)
    started = queue_lex_glue(app)
    with app.app_context():
        pinned = app.db.session.get(LibraryImport, started["id"]).citation
    assert pinned["revision"] == SHA and pinned["bibtex"] == LEXGLUE_BIBTEX and pinned["accessedAt"]

    body = finish(app, tmp_path, started["id"])

    assert body["citation"] == {
        "title": "LexGLUE",
        "authors": LEXGLUE_AUTHORS,
        "year": 2022,
        "url": "https://huggingface.co/datasets/coastalcph/lex_glue",
        "revision": SHA,
        # Re-pinned when the import completed: the day it was imported.
        "accessedAt": body["importedAt"],
        "bibtex": LEXGLUE_BIBTEX,
    }


def test_a_dataset_cited_only_in_its_readme_is_cited_from_that_bibtex(make_app, tmp_path):
    # LexGLUE's card has no BibTeX; its README does, at the revision the import read.
    app = make_app(enqueue=lambda import_ref: None)
    body = finish(app, tmp_path, queue_lex_glue(app)["id"])

    assert body["citation"] == {
        "title": "LexGLUE",
        "authors": LEXGLUE_AUTHORS,
        "year": 2022,
        "url": "https://huggingface.co/datasets/coastalcph/lex_glue",
        "revision": SHA,
        "accessedAt": body["importedAt"],
        "bibtex": LEX_README_BIBTEX,
    }


def test_an_import_that_cannot_know_its_revision_records_none(make_app, tmp_path, cited_lex_glue):
    # The fetch's shards were read from a revision it could not name (#168):
    # nothing is recorded, rather than the revision the import was queued at,
    # which the rows did not come from.
    app = make_app(enqueue=lambda import_ref: None)
    started = queue_lex_glue(app)
    shard = write_shard(tmp_path / "0000.parquet", ["A clause", "Another"], [0, 1])
    with app.app_context():
        imports.run_import(
            started["id"], session=app.db.session,
            fetch_shards=lambda *_: imports.Fetched(paths=[str(shard)], revision=None),
            create_dataset=FakeCreateDataset(), data_dir=str(tmp_path / "data"), seed=7,
        )
        record = app.db.session.get(LibraryImport, started["id"])
        assert record.status == "completed"
        assert record.revision is None
        assert record.citation["revision"] is None

    body = app.test_client().get("/api/library/provenance/dataset-1", headers=ALICE).get_json()
    assert body["revision"] is None
    assert body["citation"]["revision"] is None


def test_an_import_made_before_citations_existed_has_none(app, tmp_path):
    complete(app, tmp_path, license="cc-by-4.0")

    body = app.test_client().get("/api/library/provenance/dataset-1", headers=ALICE).get_json()

    assert body["citation"] is None
