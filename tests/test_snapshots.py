"""Copies of pinned datasets (#121): made by the worker, kept while the license is
Allow, read when Hugging Face is down, and removed only once no import reads them.

Shards are small Parquet files; the Hub is faked at its HTTP boundary, as in the
other tests, so nothing here reaches the network."""
import errno
import os
from datetime import datetime, timedelta, timezone

import pyarrow.parquet as pq
import pytest
import sqlalchemy as sa

from dataset_library import imports, pins, snapshots
from dataset_library.errors import ProviderUnavailable
from dataset_library.models import LibraryImport, LibrarySettings, LibrarySnapshot
from dataset_library.providers import huggingface
from test_import import FakeCreateDataset, run, write_shard  # noqa: F401 - `run` is a fixture
from tests.conftest import ADMIN, ALICE

# AI-team-UoA/greek_legal_code: its revision in the recorded Hub fixture, classification, cc-by-4.0 (Allow), with rows
# for its default config and split, so a pin of it can be previewed and copied.
SHA = "de0fdb34424f07d1ac6f0ede23ee0ed44bd9f5d1"
GREEK = "AI-team-UoA/greek_legal_code"
# No license stated, so Warn: pinned, but never copied.
UNSTATED_LICENSE = "lighteval/lexglue"


def utc_now():
    """Naive UTC, as the library stores its times."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Queue:
    """Stands in for the Redis list: keeps the refs it is handed, or refuses them."""

    def __init__(self):
        self.refs, self.down = [], False

    def __call__(self, ref):
        if self.down:
            raise ConnectionError("Redis is down")
        self.refs.append(ref)


@pytest.fixture
def snapshot_dir(tmp_path):
    return str(tmp_path / "snapshots")


@pytest.fixture
def imports_queue():
    return Queue()


@pytest.fixture
def copies_queue():
    return Queue()


@pytest.fixture
def app(make_app, imports_queue, copies_queue, snapshot_dir):
    return make_app(enqueue=imports_queue, enqueue_snapshot=copies_queue, snapshot_dir=snapshot_dir, max_rows=lambda: 300)


@pytest.fixture
def client(app):
    return app.test_client()


def pin_as_admin(client, dataset=GREEK, status=201):
    resp = client.post("/api/library/pins", json={"dataset": dataset}, headers=ADMIN)
    assert resp.status_code == status, resp.get_json()
    return resp.get_json()


def take_hub_down(hub, cache):
    """Hugging Face stops answering, and the previews cached while it answered are gone
    (they last ten minutes), so an import has to ask the Hub and find it down."""
    hub.fail = 503
    cache.expire("library:")


def set_policy(app, policy):
    with app.app_context():
        settings = app.db.session.get(LibrarySettings, 1)
        if settings is None:
            app.db.session.add(LibrarySettings(id=1, license_policy=policy))
        else:
            settings.license_policy = policy
        app.db.session.commit()


def copy_row(app, pin_ref=None):
    """The copy as a dict (an instance does not outlive its session), or None."""
    with app.app_context():
        query = app.db.session.query(LibrarySnapshot)
        if pin_ref is not None:
            query = query.filter_by(pin_id=pin_ref)
        row = query.first()
        if row is None:
            return None
        return {
            name: getattr(row, name)
            for name in ("id", "pin_id", "status", "revision", "row_count", "columns", "file_path", "license",
                         "config", "split", "error_code", "error_message", "upstream_changed", "external_id")
        }


def copy_files(snapshot_dir):
    """The copies in the snapshot directory: hidden entries, such as the work area, are not copies."""
    return sorted(name for name in os.listdir(snapshot_dir) if not name.startswith("."))


def shards_of_three(tmp_path):
    return [
        write_shard(tmp_path / "0000.parquet", ["A clause", "Another"], [0, 1]),
        write_shard(tmp_path / "0001.parquet", ["Third"], [2]),
    ]


def build_copy(app, copy_id, snapshot_dir, shards, revision=SHA):
    """The worker makes a queued copy from these shards, as it would from the Hub."""

    def fetch_shards(external_id, config, split, max_bytes, dest_dir):
        return imports.Fetched(paths=[str(path) for path in shards], revision=revision)

    with app.app_context():
        snapshots.run_copy(copy_id, session=app.db.session, fetch_shards=fetch_shards, snapshot_dir=snapshot_dir, max_bytes=10**9)


def ready_copy(client, app, snapshot_dir, tmp_path):
    """A pin of greek_legal_code whose copy the worker has made, and that copy."""
    pinned = pin_as_admin(client)
    build_copy(app, copy_row(app, pinned["ref"])["id"], snapshot_dir, shards_of_three(tmp_path))
    return pinned, copy_row(app, pinned["ref"])


def offline_import(client, dataset=GREEK, size=2):
    return client.post(
        "/api/library/imports",
        json={
            "dataset": dataset, "taskType": "classification", "textColumn": "text", "labelColumn": "label",
            "size": size, "balance": False,
        },
        headers=ALICE,
    )


def work_off(app, ref, snapshot_dir, tmp_path):
    """Run a queued import as the worker would. An import from a copy must not ask the Hub."""
    create = FakeCreateDataset()

    def hub_must_not_be_asked(*args, **kwargs):
        raise AssertionError("an import from a copy asked Hugging Face")

    with app.app_context():
        session = app.db.session
        imports.run_import(
            ref, session=session, fetch_shards=hub_must_not_be_asked, create_dataset=create,
            data_dir=str(tmp_path / "data"), seed=7, heartbeat_seconds=None, snapshot_dir=snapshot_dir,
        )
        session.expunge_all()
        return session.get(LibraryImport, ref).to_dict(), create


# --- Making copies -----------------------------------------------------------------

def test_an_allow_pin_queues_a_copy_for_the_worker(client, app, copies_queue):
    body = pin_as_admin(client)

    assert body["snapshot"] == {"status": "queued", "error": None, "revision": None, "rows": None}
    copy = copy_row(app, body["ref"])
    assert copy["status"] == "queued"
    assert copy["license"] == "cc-by-4.0"
    assert (copy["config"], copy["split"]) == ("volume", "train")
    assert copies_queue.refs == [copy["id"]]


def test_a_dataset_with_no_license_stated_is_pinned_but_gets_no_copy(client, app, copies_queue):
    body = pin_as_admin(client, UNSTATED_LICENSE)

    assert body["snapshot"]["status"] == "none"
    assert copy_row(app) is None
    assert copies_queue.refs == []


def test_a_license_the_admin_set_to_warn_gets_no_copy(client, app, copies_queue):
    set_policy(app, {"cc-by-4.0": "warn"})

    body = pin_as_admin(client)

    assert body["snapshot"]["status"] == "none"
    assert copy_row(app) is None
    assert copies_queue.refs == []


def test_a_host_that_keeps_no_snapshot_directory_keeps_no_copies(make_app, copies_queue):
    client = make_app(enqueue=Queue(), enqueue_snapshot=copies_queue, max_rows=lambda: 300).test_client()

    body = pin_as_admin(client)

    assert body["snapshot"]["status"] == "none"
    assert copies_queue.refs == []


def test_the_worker_keeps_the_copy_with_its_revision_rows_license_and_columns(client, app, snapshot_dir, tmp_path):
    pinned = pin_as_admin(client)

    build_copy(app, copy_row(app, pinned["ref"])["id"], snapshot_dir, shards_of_three(tmp_path))

    copy = copy_row(app, pinned["ref"])
    assert copy["status"] == "ready"
    assert copy["revision"] == SHA
    assert copy["row_count"] == 3
    assert copy["license"] == "cc-by-4.0"
    assert copy["columns"] == [
        {"name": "text", "type": "string"}, {"name": "label", "type": "int64"}, {"name": "extra", "type": "string"},
    ]
    assert pq.read_table(os.path.join(snapshot_dir, copy["file_path"])).column("text").to_pylist() == ["A clause", "Another", "Third"]
    # Only the finished file is kept: the work area it was built in is empty.
    assert copy_files(snapshot_dir) == [copy["file_path"]]
    assert os.listdir(os.path.join(snapshot_dir, snapshots.INCOMING)) == []
    listed = client.get("/api/library/pins", headers=ADMIN).get_json()["pins"][0]
    assert listed["snapshot"] == {"status": "ready", "error": None, "revision": SHA, "rows": 3}


@pytest.mark.parametrize(
    "error, code, message",
    [
        (errno.ENOSPC, "storage_full", snapshots.FULL),
        (errno.EACCES, "write_failed", snapshots.UNEXPECTED),
    ],
)
def test_a_failed_write_records_the_copy_as_failed(client, app, snapshot_dir, tmp_path, monkeypatch, error, code, message):
    pinned = pin_as_admin(client)

    def refuse_to_keep_it(source, destination):
        raise OSError(error, os.strerror(error))

    monkeypatch.setattr(snapshots.os, "replace", refuse_to_keep_it)
    build_copy(app, copy_row(app, pinned["ref"])["id"], snapshot_dir, shards_of_three(tmp_path))

    copy = copy_row(app, pinned["ref"])
    assert copy["status"] == "failed"
    assert (copy["error_code"], copy["error_message"]) == (code, message)
    assert copy_files(snapshot_dir) == []
    listed = client.get("/api/library/pins", headers=ADMIN).get_json()["pins"][0]
    assert listed["snapshot"]["status"] == "failed"


def test_a_hub_that_names_no_revision_fails_the_copy(client, app, snapshot_dir, tmp_path):
    pinned = pin_as_admin(client)

    build_copy(app, copy_row(app, pinned["ref"])["id"], snapshot_dir, shards_of_three(tmp_path), revision=None)

    copy = copy_row(app, pinned["ref"])
    assert (copy["status"], copy["error_code"]) == ("failed", "no_revision")


def test_a_queue_that_refuses_the_copy_fails_it_at_once(make_app, snapshot_dir):
    copies = Queue()
    copies.down = True
    client = make_app(enqueue=Queue(), enqueue_snapshot=copies, snapshot_dir=snapshot_dir, max_rows=lambda: 300).test_client()

    body = pin_as_admin(client)

    assert body["snapshot"] == {"status": "failed", "error": snapshots.NOT_STARTED, "revision": None, "rows": None}


def test_an_admin_takes_a_copy_for_a_pin_that_has_none_once_its_license_allows(client, app, copies_queue):
    set_policy(app, {"cc-by-4.0": "warn"})
    pinned = pin_as_admin(client)
    refused = client.post(f"/api/library/pins/{pinned['ref']}/snapshot", headers=ADMIN)
    assert (refused.status_code, refused.get_json()["code"]) == (409, "not_allowed")

    set_policy(app, {})
    resp = client.post(f"/api/library/pins/{pinned['ref']}/snapshot", headers=ADMIN)

    assert resp.status_code == 202
    assert resp.get_json()["snapshot"]["status"] == "queued"
    assert copies_queue.refs == [copy_row(app, pinned["ref"])["id"]]
    assert client.post(f"/api/library/pins/{pinned['ref']}/snapshot", headers=ALICE).status_code == 403


def test_taking_a_copy_is_refused_on_a_host_that_keeps_none(make_app):
    client = make_app(enqueue=Queue(), max_rows=lambda: 300).test_client()
    pinned = pin_as_admin(client)

    resp = client.post(f"/api/library/pins/{pinned['ref']}/snapshot", headers=ADMIN)

    assert (resp.status_code, resp.get_json()["code"]) == (404, "not_offered")


# --- Serving while Hugging Face is down -------------------------------------------

def test_with_hugging_face_down_search_returns_the_pins_with_the_unavailable_status(client, app, hub, snapshot_dir, tmp_path):
    pinned, _ = ready_copy(client, app, snapshot_dir, tmp_path)
    pin_as_admin(client, UNSTATED_LICENSE)
    hub.fail = 503

    body = client.get("/api/library/search", headers=ALICE).get_json()

    assert body["status"] == "unavailable"
    by_id = {card["externalId"]: card for card in body["results"]}
    assert sorted(by_id) == [GREEK, UNSTATED_LICENSE]
    assert by_id[GREEK]["pinRef"] == pinned["ref"]
    assert by_id[GREEK]["snapshot"]["status"] == "ready"
    assert by_id[UNSTATED_LICENSE]["snapshot"]["status"] == "none"


def test_the_pins_a_search_returns_while_hugging_face_is_down_are_filtered_as_the_search_is(client, app, hub, snapshot_dir, tmp_path):
    ready_copy(client, app, snapshot_dir, tmp_path)
    pin_as_admin(client, UNSTATED_LICENSE)
    hub.fail = 503

    def ids(**params):
        resp = client.get("/api/library/search", query_string=params, headers=ALICE).get_json()
        return sorted(card["externalId"] for card in resp["results"])

    assert ids(q="greek") == [GREEK]
    assert ids(task="classification") == [GREEK]
    assert ids(license="cc-by-4.0") == [GREEK]
    assert ids(q="nothing-matches") == []


def test_with_hugging_face_down_an_import_completes_from_the_copy(client, app, hub, cache, imports_queue, snapshot_dir, tmp_path):
    pinned, _ = ready_copy(client, app, snapshot_dir, tmp_path)
    take_hub_down(hub, cache)

    resp = offline_import(client)

    assert resp.status_code == 202, resp.get_json()
    started = resp.get_json()
    assert (started["source"], started["revision"]) == ("snapshot", SHA)
    assert imports_queue.refs == [started["id"]]

    result, create = work_off(app, started["id"], snapshot_dir, tmp_path)

    assert result["status"] == "completed"
    assert result["rows"] == 2
    assert len(create.calls[0]["rows"]) == 2
    provenance = client.get(f"/api/library/provenance/{result['datasetRef']}", headers=ALICE).get_json()
    assert (provenance["source"], provenance["revision"]) == ("snapshot", SHA)


def test_with_hugging_face_down_and_no_copy_an_import_is_refused_as_unavailable(client, hub, cache, imports_queue):
    pin_as_admin(client, UNSTATED_LICENSE)
    take_hub_down(hub, cache)

    resp = offline_import(client, dataset=UNSTATED_LICENSE)

    assert (resp.status_code, resp.get_json()["code"]) == (503, "provider_unavailable")
    assert imports_queue.refs == []


def test_an_import_that_loses_hugging_face_mid_run_reads_the_copy_of_the_same_part(app, run, client, snapshot_dir, tmp_path):
    _, copy = ready_copy(client, app, snapshot_dir, tmp_path)
    assert (copy["config"], copy["split"]) == ("volume", "train")

    result, create = run(
        [write_shard(tmp_path / "unused.parquet", ["x"], [0])], size=2,
        external_id=GREEK, config="volume", fail_fetch=ProviderUnavailable("Hugging Face 503"), snapshot_dir=snapshot_dir,
    )

    assert result["status"] == "completed"
    assert (result["source"], result["revision"]) == ("snapshot", SHA)
    assert len(create.calls[0]["rows"]) == 2


def test_a_copy_of_another_part_is_not_read_in_place_of_a_download(app, run, client, snapshot_dir, tmp_path):
    _, copy = ready_copy(client, app, snapshot_dir, tmp_path)
    with app.app_context():
        app.db.session.execute(sa.text("UPDATE library_snapshots SET split = 'test' WHERE id = :id"), {"id": copy["id"]})
        app.db.session.commit()

    result, _ = run(
        [write_shard(tmp_path / "unused.parquet", ["x"], [0])], size=2,
        external_id=GREEK, config="volume", fail_fetch=ProviderUnavailable("Hugging Face 503"), snapshot_dir=snapshot_dir,
    )

    assert (result["status"], result["code"]) == ("failed", "provider_busy")


# --- Upstream revisions -------------------------------------------------------------

def test_a_revision_the_hub_no_longer_has_keeps_the_copy_and_flags_the_pin(client, app, hub, snapshot_dir, tmp_path):
    pinned, copy = ready_copy(client, app, snapshot_dir, tmp_path)
    hub.gone_revisions.add((GREEK, SHA))

    with app.app_context():
        assert snapshots.check_upstream(app.db.session) == 1

    kept = copy_row(app, pinned["ref"])
    assert (kept["status"], kept["upstream_changed"]) == ("ready", True)
    assert os.path.exists(os.path.join(snapshot_dir, copy["file_path"]))
    listed = client.get("/api/library/pins", headers=ADMIN).get_json()["pins"][0]
    assert listed["upstreamChanged"] is True
    assert listed["snapshot"]["status"] == "ready"


def test_the_hub_is_asked_about_each_copy_once_an_hour_and_a_check_it_cannot_make_is_tried_later(client, app, hub, snapshot_dir, tmp_path):
    ready_copy(client, app, snapshot_dir, tmp_path)
    now = utc_now()
    with app.app_context():
        session = app.db.session
        hub.fail = 503
        assert snapshots.check_upstream(session, now=now) == 0
        assert session.query(LibrarySnapshot).one().upstream_checked_at is None
        hub.fail = None
        assert snapshots.check_upstream(session, now=now) == 1
        assert snapshots.check_upstream(session, now=now + timedelta(minutes=30)) == 0
        assert snapshots.check_upstream(session, now=now + timedelta(hours=2)) == 1


def test_the_hub_says_whether_a_revision_exists(hub):
    assert huggingface.revision_exists(GREEK, SHA) is True
    assert huggingface.revision_exists(GREEK, "0" * 40) is False
    hub.fail = 503
    with pytest.raises(ProviderUnavailable):
        huggingface.revision_exists(GREEK, SHA)


# --- Removing copies -----------------------------------------------------------------

def test_unpinning_deletes_the_copy_and_its_file_when_no_import_reads_it(client, app, snapshot_dir, tmp_path):
    pinned, copy = ready_copy(client, app, snapshot_dir, tmp_path)
    path = os.path.join(snapshot_dir, copy["file_path"])

    assert client.delete(f"/api/library/pins/{pinned['ref']}", headers=ADMIN).status_code == 204

    assert copy_row(app) is None
    assert not os.path.exists(path)


def test_unpinning_keeps_the_copy_for_a_running_import_until_that_import_finishes(app, run, client, snapshot_dir, tmp_path):
    pinned, copy = ready_copy(client, app, snapshot_dir, tmp_path)
    path = os.path.join(snapshot_dir, copy["file_path"])
    seen = {}

    def unpin_while_saving(app):
        # The import is saving its Dataset, so it is running and still reads the copy.
        pins.remove(app.extensions["dataset_library"], pinned["ref"])
        seen["status"] = copy_row(app)["status"]
        seen["file_kept"] = os.path.exists(path)

    result, create = run(
        [write_shard(tmp_path / "unused.parquet", ["x"], [0])], size=2,
        external_id=GREEK, config="volume", while_saving=unpin_while_saving, snapshot_dir=snapshot_dir, source="snapshot", snapshot_id=copy["id"],
    )

    assert seen == {"status": "removing", "file_kept": True}
    assert result["status"] == "completed"
    assert len(create.calls) == 1
    assert copy_row(app) is None
    assert not os.path.exists(path)


def test_an_import_queued_from_a_copy_counts_as_reading_it_and_a_removed_copy_is_not_served(client, app, hub, cache, snapshot_dir, tmp_path):
    pinned, copy = ready_copy(client, app, snapshot_dir, tmp_path)
    path = os.path.join(snapshot_dir, copy["file_path"])
    take_hub_down(hub, cache)
    started = offline_import(client).get_json()

    with app.app_context():
        pins.remove(app.extensions["dataset_library"], pinned["ref"])
    assert os.path.exists(path)
    cache.expire("library:")
    assert offline_import(client).status_code == 503

    result, _ = work_off(app, started["id"], snapshot_dir, tmp_path)

    assert result["status"] == "completed"
    assert copy_row(app) is None
    assert not os.path.exists(path)


def test_a_license_the_admin_sets_to_warn_removes_its_copy(client, app, snapshot_dir, tmp_path):
    pinned, copy = ready_copy(client, app, snapshot_dir, tmp_path)
    path = os.path.join(snapshot_dir, copy["file_path"])

    resp = client.put(
        "/api/library/admin/settings",
        json={"sources": [], "license_policy": {"cc-by-4.0": "warn"}},
        headers=ADMIN,
    )

    assert resp.status_code == 200
    assert copy_row(app) is None
    assert not os.path.exists(path)
    assert client.get("/api/library/pins", headers=ADMIN).get_json()["pins"][0]["snapshot"]["status"] == "none"


def test_the_sweep_fails_a_copy_a_worker_left_building(client, app, snapshot_dir):
    pinned = pin_as_admin(client)
    copy_id = copy_row(app, pinned["ref"])["id"]
    with app.app_context():
        # A copy a worker started two hours ago and never finished.
        copy = app.db.session.get(LibrarySnapshot, copy_id)
        copy.status, copy.updated_at = "building", utc_now() - timedelta(hours=2)
        app.db.session.commit()

    with app.app_context():
        assert snapshots.sweep(app.db.session, snapshot_dir) == [copy_id]

    copy = copy_row(app, pinned["ref"])
    assert (copy["status"], copy["error_code"]) == ("failed", "stopped")
