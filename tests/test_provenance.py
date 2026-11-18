"""A Dataset's provenance (#116): the completed import it came from, its
license and when the student confirmed that license."""
from datetime import datetime

import pytest

from conftest import ALICE

from dataset_library import imports
from dataset_library.models import LibraryImport
from test_import import FakeCreateDataset, write_shard

BOB = {"X-User": "bob"}


def complete(app, tmp_path, *, license, license_ack_at=None):
    """Run one import to completed, as the host's worker would."""
    shard = write_shard(tmp_path / "0000.parquet", ["A clause", "Another"], [0, 1])
    with app.app_context():
        session = app.db.session
        record = LibraryImport(
            owner_id="alice", provider="huggingface", external_id="coastalcph/lex_glue",
            config="ledgar", split="train", text_column="text", label_column="label",
            task_type="classification", sample_size=2, max_rows=1000, name="LexGLUE",
            license=license, license_ack_at=license_ack_at,
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
