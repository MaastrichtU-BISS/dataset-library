"""Copies of pinned datasets (#121): made by the worker, kept while the license is
Allow, read when Hugging Face is down, and removed only once no import reads them.

Shards are small Parquet files; the Hub is faked at its HTTP boundary, as in the
other tests, so nothing here reaches the network."""
import errno
import os
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
import sqlalchemy as sa

from dataset_library import imports, pins, preview as preview_module, snapshots
from dataset_library.errors import ProviderUnavailable, ProviderRateLimited
from dataset_library.models import LibraryImport, LibraryPin, LibrarySettings, LibrarySnapshot
from dataset_library.providers import huggingface
from test_import import FakeCreateDataset, run, write_ner_shard, write_shard, write_summarization_shard  # noqa: F401 - `run` is a fixture
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


def test_a_copy_is_made_with_the_range_read_fetch_from_shards_over_the_cap(client, app, snapshot_dir, tmp_path, monkeypatch):
    # #178: the built-in fetch reads row groups, not whole shards, and writes
    # them as local Parquet files a copy is made from as before (#121). Every
    # shard here is over the copy's cap.
    from dataset_library.providers import parquet
    from test_fetch_shards import FakeHuggingFace, parquet_shard

    pinned = pin_as_admin(client)
    copy = copy_row(app, pinned["ref"])
    fake = FakeHuggingFace()
    shards = [parquet_shard(s, groups=6, rows_per_group=20, text_bytes=200) for s in range(3)]
    fake.add_dataset(GREEK, copy["config"], copy["split"], shards, revision=SHA)
    cap = 20_000
    assert min(len(shard) for shard in shards) > cap

    def fetch_shards(external_id, config, split, max_bytes, dest_dir, **kwargs):
        return parquet.fetch_shards(external_id, config, split, max_bytes, dest_dir, sleep=lambda s: None, **kwargs)

    with monkeypatch.context() as hub:
        hub.setattr(parquet.requests, "get", fake.get)
        hub.setattr(parquet.requests, "post", fake.post)
        with app.app_context():
            snapshots.run_copy(copy["id"], session=app.db.session, fetch_shards=fetch_shards, snapshot_dir=snapshot_dir, max_bytes=cap)

    copy = copy_row(app, pinned["ref"])
    assert copy["status"] == "ready", copy
    assert copy["revision"] == SHA
    texts = pq.read_table(os.path.join(snapshot_dir, copy["file_path"])).column("text").to_pylist()
    assert copy["row_count"] == len(texts) > 0
    assert {text.split("-", 1)[0] for text in texts} == {"s0", "s1", "s2"}
    assert copy["columns"] == [{"name": "text", "type": "string"}, {"name": "label", "type": "int64"}]
    assert 0 < fake.served <= cap


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
    # The Dataset's provenance says the same: the copy was read, at its revision.
    provenance = client.get(f"/api/library/provenance/{result['datasetRef']}", headers=ALICE).get_json()
    assert (provenance["source"], provenance["revision"]) == ("snapshot", SHA)


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
        # A check the Hub could not answer goes to the back of the order, its flag unchanged.
        copy = session.query(LibrarySnapshot).one()
        assert copy.upstream_changed is False and copy.upstream_checked_at is not None
        hub.fail = None
        assert snapshots.check_upstream(session, now=now + timedelta(minutes=30)) == 0
        assert snapshots.check_upstream(session, now=now + timedelta(hours=2)) == 1
        assert snapshots.check_upstream(session, now=now + timedelta(hours=2, minutes=30)) == 0
        assert snapshots.check_upstream(session, now=now + timedelta(hours=4)) == 1


def test_the_hub_says_whether_a_revision_exists(hub):
    assert huggingface.revision_exists(GREEK, SHA) is True
    assert huggingface.revision_exists(GREEK, "0" * 40) is False
    # A dataset that is gone or no longer public: the Hub answers 401, not 404.
    assert huggingface.revision_exists("someone/deleted-dataset", SHA) is False
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


# --- Opening a dataset while Hugging Face is down ----------------------------------

def test_a_dataset_opens_from_its_copy_while_hugging_face_is_down(client, app, hub, cache, snapshot_dir, tmp_path):
    ready_copy(client, app, snapshot_dir, tmp_path)
    take_hub_down(hub, cache)

    resp = client.get(f"/api/library/datasets/huggingface/{GREEK}", headers=ALICE)

    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    assert (body["preview"], body["offline"], body["config"], body["split"]) == ("ok", True, "volume", "train")
    # The label is a ClassLabel in the copy's Hugging Face features, so it shows as one, as the live preview does.
    assert [(column["name"], column["type"]) for column in body["columns"]] == [
        ("text", "text"), ("label", "class label, 3 classes"), ("extra", "text"),
    ]
    # The rows are the copy's first rows, with class labels by name, as the live preview shows them.
    assert body["rows"] == [
        ["A clause", "Terminations", "x"], ["Another", "Governing Laws", "x"], ["Third", "Notices", "x"],
    ]
    assert (body["rowCount"], body["revision"]) == (3, SHA)


def test_a_dataset_whose_copy_file_cannot_be_read_still_opens_without_rows(client, app, hub, cache, snapshot_dir, tmp_path):
    _, copy = ready_copy(client, app, snapshot_dir, tmp_path)
    os.remove(os.path.join(snapshot_dir, copy["file_path"]))
    take_hub_down(hub, cache)

    body = client.get(f"/api/library/datasets/huggingface/{GREEK}", headers=ALICE).get_json()

    assert (body["preview"], body["rows"], body["offline"]) == ("ok", [], True)
    assert [column["name"] for column in body["columns"]] == ["text", "label", "extra"]


def test_a_copy_whose_license_is_now_hidden_is_not_opened_offline(client, app, hub, cache, snapshot_dir, tmp_path):
    ready_copy(client, app, snapshot_dir, tmp_path)
    set_policy(app, {"cc-by-4.0": "hide"})
    take_hub_down(hub, cache)

    resp = client.get(f"/api/library/datasets/huggingface/{GREEK}", headers=ALICE)

    assert (resp.status_code, resp.get_json()["code"]) == (404, "not_found")


def drop_pin_row(app, pin_ref):
    """A pin removed underneath its copy, as a removal that has not reached the copy would leave it."""
    with app.app_context():
        app.db.session.execute(sa.text("DELETE FROM library_pins WHERE id = :id"), {"id": pin_ref})
        app.db.session.commit()


# --- A viewer that is down while the Hub answers (#121) -------------------------------

def take_viewer_down(hub, cache):
    """datasets-server stops answering while the Hub still does; previews cached while it answered are gone."""
    hub.viewer_fail = 503
    cache.expire("library:")


def test_with_only_the_viewer_down_a_dataset_opens_from_its_copy_and_imports_from_it(client, app, hub, cache, snapshot_dir, tmp_path):
    ready_copy(client, app, snapshot_dir, tmp_path)
    take_viewer_down(hub, cache)

    body = client.get(f"/api/library/datasets/huggingface/{GREEK}", headers=ALICE).get_json()

    assert (body["preview"], body["offline"], body["config"], body["split"]) == ("ok", True, "volume", "train")
    assert body["rows"][0] == ["A clause", "Terminations", "x"]
    resp = offline_import(client)
    assert resp.status_code == 202, resp.get_json()
    started = resp.get_json()
    assert (started["source"], started["revision"]) == ("snapshot", SHA)

    result, create = work_off(app, started["id"], snapshot_dir, tmp_path)

    assert result["status"] == "completed"
    assert len(create.calls[0]["rows"]) == 2
    provenance = client.get(f"/api/library/provenance/{result['datasetRef']}", headers=ALICE).get_json()
    assert (provenance["source"], provenance["revision"]) == ("snapshot", SHA)


def test_a_preview_built_while_the_viewer_is_down_is_not_kept_for_the_next_open(client, hub, cache):
    take_viewer_down(hub, cache)
    first = client.get(f"/api/library/datasets/huggingface/{GREEK}", headers=ALICE).get_json()
    assert (first["preview"], first["viewerDown"]) == ("unavailable", True)

    hub.viewer_fail = None
    second = client.get(f"/api/library/datasets/huggingface/{GREEK}", headers=ALICE).get_json()

    assert second["preview"] == "ok"
    assert "viewerDown" not in second


def test_with_the_viewer_down_and_no_copy_an_import_is_still_refused_as_not_importable(client, hub, cache, imports_queue):
    take_viewer_down(hub, cache)

    resp = offline_import(client)

    assert (resp.status_code, resp.get_json()["code"]) == (409, "not_importable")
    assert imports_queue.refs == []


def test_a_pin_made_while_the_viewer_is_down_says_its_copy_was_busy_and_a_retry_makes_one(client, app, hub, cache, copies_queue):
    take_viewer_down(hub, cache)
    pinned = pin_as_admin(client)
    assert (pinned["snapshot"]["status"], pinned["snapshot"]["error"]) == ("failed", snapshots.BUSY)

    hub.viewer_fail = None
    resp = client.post(f"/api/library/pins/{pinned['ref']}/snapshot", headers=ADMIN)

    assert resp.status_code == 202, resp.get_json()
    assert resp.get_json()["snapshot"]["status"] == "queued"
    assert copies_queue.refs == [copy_row(app, pinned["ref"])["id"]]


@pytest.mark.parametrize("status", [501, 404, 401])
def test_a_viewer_that_says_no_for_good_is_not_an_outage(client, hub, cache, copies_queue, status):
    # 501: the viewer cannot read this dataset; 404 and 401: not there, or gated. None is an outage.
    hub.viewer_fail = status
    cache.expire("library:")
    pinned = pin_as_admin(client)

    assert (pinned["snapshot"]["status"], pinned["snapshot"]["error"]) == ("failed", snapshots.NO_ROWS)
    body = client.get(f"/api/library/datasets/huggingface/{GREEK}", headers=ADMIN).get_json()
    assert "viewerDown" not in body
    assert copies_queue.refs == []


def test_a_viewer_server_error_is_an_outage():
    from dataset_library.providers import viewer

    class Answer:
        def __init__(self, status):
            self.status_code = status

        def json(self):
            raise ValueError("not JSON")

    import unittest.mock as mock

    for status, kind in [(503, viewer.ViewerDown), (500, viewer.ViewerDown), (501, viewer.ViewerUnavailable)]:
        with mock.patch("dataset_library.providers.viewer.requests.get", return_value=Answer(status)):
            with pytest.raises(viewer.ViewerUnavailable) as raised:
                viewer.get("splits", dataset=GREEK)
        assert (type(raised.value) is viewer.ViewerDown) == (kind is viewer.ViewerDown), status
    with mock.patch("dataset_library.providers.viewer.requests.get", return_value=Answer(200)):
        with pytest.raises(viewer.ViewerDown):
            viewer.get("splits", dataset=GREEK)


# --- Crashed builds (#121) ------------------------------------------------------------

def leave_killed_build(app, copy_id, snapshot_dir):
    """What a worker killed mid-build leaves: its row building and long stale, a part file in
    its work area, and its staged file already renamed into the snapshot directory."""
    work = os.path.join(snapshot_dir, snapshots.INCOMING, copy_id)
    os.makedirs(work)
    with open(os.path.join(work, "0000.parquet"), "wb") as part:
        part.write(b"part")
    kept = os.path.join(snapshot_dir, f"{copy_id}.parquet")
    with open(kept, "wb") as staged:
        staged.write(b"copy")
    with app.app_context():
        copy = app.db.session.get(LibrarySnapshot, copy_id)
        copy.status, copy.updated_at = "building", utc_now() - timedelta(hours=2)
        app.db.session.commit()
    return work, kept


def test_the_sweep_fails_a_killed_build_and_removes_what_it_left(client, app, snapshot_dir):
    pinned = pin_as_admin(client)
    copy_id = copy_row(app, pinned["ref"])["id"]
    work, kept = leave_killed_build(app, copy_id, snapshot_dir)

    with app.app_context():
        assert snapshots.sweep(app.db.session, snapshot_dir) == [copy_id]

    copy = copy_row(app, pinned["ref"])
    assert (copy["status"], copy["error_code"]) == ("failed", "stopped")
    assert not os.path.exists(work)
    assert not os.path.exists(kept)


def test_unpinning_a_failed_copy_deletes_a_stray_file_and_work_area_under_its_id(client, app, snapshot_dir):
    pinned = pin_as_admin(client)
    copy_id = copy_row(app, pinned["ref"])["id"]
    work, kept = leave_killed_build(app, copy_id, snapshot_dir)
    with app.app_context():
        copy = app.db.session.get(LibrarySnapshot, copy_id)
        # Failed, and with a path the row does not hold: the files are found by the copy's id, not its path.
        copy.status, copy.file_path = "failed", "unrecorded.parquet"
        app.db.session.commit()

    assert client.delete(f"/api/library/pins/{pinned['ref']}", headers=ADMIN).status_code == 204

    assert copy_row(app) is None
    assert not os.path.exists(kept)
    assert not os.path.exists(work)


def test_the_sweep_removes_files_and_work_areas_that_no_kept_copy_owns(client, app, snapshot_dir, tmp_path):
    pinned, copy = ready_copy(client, app, snapshot_dir, tmp_path)
    ghost = "5f1d0b8e-6c2a-4f3e-9a7b-2d4c6e8f0a1b"  # a copy with no row at all
    ghost_work = os.path.join(snapshot_dir, snapshots.INCOMING, ghost)
    ghost_file = os.path.join(snapshot_dir, f"{ghost}.parquet")
    os.makedirs(ghost_work)
    open(ghost_file, "wb").close()
    kept_work = os.path.join(snapshot_dir, snapshots.INCOMING, copy["id"])
    os.makedirs(kept_work)

    with app.app_context():
        snapshots.sweep(app.db.session, snapshot_dir)

    assert not os.path.exists(ghost_work)
    assert not os.path.exists(ghost_file)
    assert not os.path.exists(kept_work)
    assert copy_files(snapshot_dir) == [copy["file_path"]]


# --- Attempts and heartbeats (#121) ---------------------------------------------------

def test_a_stale_build_is_retried_as_a_new_copy_and_the_old_try_cannot_finish_or_delete_it(client, app, snapshot_dir, tmp_path):
    pinned = pin_as_admin(client)
    old_id = copy_row(app, pinned["ref"])["id"]
    shards = shards_of_three(tmp_path)
    host = app.extensions["dataset_library"]

    def fetch_until_the_copy_is_retried(external_id, config, split, max_bytes, dest_dir):
        # The worker of the first try goes quiet: the sweep fails it, an admin asks again, and the new try is made.
        with app.app_context():
            assert snapshots.sweep(app.db.session, snapshot_dir, now=utc_now() + timedelta(hours=1)) == [old_id]
            assert pins.take_copy(host, pinned["ref"])["snapshot"]["status"] == "queued"
        build_copy(app, copy_row(app, pinned["ref"])["id"], snapshot_dir, shards)
        # Then the first try finishes, late, with what it fetched.
        return imports.Fetched(paths=[str(path) for path in shards], revision=SHA)

    with app.app_context():
        snapshots.run_copy(
            old_id, session=app.db.session, fetch_shards=fetch_until_the_copy_is_retried,
            snapshot_dir=snapshot_dir, max_bytes=10**9, heartbeat_seconds=None,
        )

    copy = copy_row(app, pinned["ref"])
    assert copy["id"] != old_id
    assert (copy["status"], copy["row_count"]) == ("ready", 3)
    assert copy_files(snapshot_dir) == [copy["file_path"]]
    assert os.listdir(os.path.join(snapshot_dir, snapshots.INCOMING)) == []
    with app.app_context():
        assert app.db.session.get(LibrarySnapshot, old_id) is None


def test_a_building_copy_refreshes_its_updated_at_while_it_runs(make_app, hub, tmp_path, snapshot_dir):
    """On a file database, so the heartbeat's own connection sees the copy."""
    app = make_app(
        enqueue=Queue(), enqueue_snapshot=Queue(), snapshot_dir=snapshot_dir, max_rows=lambda: 300,
        config={"SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'db.sqlite'}"},
    )
    pinned = pin_as_admin(app.test_client())
    copy_id = copy_row(app, pinned["ref"])["id"]
    shards = shards_of_three(tmp_path)
    beats = []

    def slow_fetch(external_id, config, split, max_bytes, dest_dir):
        with app.db.engine.connect() as connection:
            first = connection.execute(sa.text("SELECT updated_at FROM library_snapshots")).scalar_one()
        time.sleep(0.3)
        with app.db.engine.connect() as connection:
            beats.append((first, connection.execute(sa.text("SELECT updated_at FROM library_snapshots")).scalar_one()))
        return imports.Fetched(paths=[str(path) for path in shards], revision=SHA)

    with app.app_context():
        snapshots.run_copy(
            copy_id, session=app.db.session, fetch_shards=slow_fetch,
            snapshot_dir=snapshot_dir, max_bytes=10**9, heartbeat_seconds=0.05,
        )

    [(first, later)] = beats
    assert later > first
    assert copy_row(app, pinned["ref"])["status"] == "ready"


def test_a_worker_that_loses_the_claim_leaves_the_work_of_the_worker_that_won(client, app, snapshot_dir, tmp_path, monkeypatch):
    """Between this worker's look at the copy and its claim, another worker claims it and writes."""
    pinned = pin_as_admin(client)
    copy_id = copy_row(app, pinned["ref"])["id"]
    work = os.path.join(snapshot_dir, snapshots.INCOMING, copy_id)
    winners_part = os.path.join(work, "winners.part")
    fetched = []

    def the_other_worker_claims_it(session):
        os.makedirs(work)
        with open(winners_part, "wb") as part:
            part.write(b"the other worker's part")
        with app.app_context():
            app.db.session.execute(sa.text("UPDATE library_snapshots SET status = 'building' WHERE id = :id"), {"id": copy_id})
            app.db.session.commit()
        return None

    monkeypatch.setattr(snapshots, "download_cap_bytes", the_other_worker_claims_it)

    def fetch_shards(external_id, config, split, max_bytes, dest_dir):
        fetched.append(external_id)
        return imports.Fetched(paths=[], revision=SHA)

    with app.app_context():
        snapshots.run_copy(copy_id, session=app.db.session, fetch_shards=fetch_shards, snapshot_dir=snapshot_dir, max_bytes=10**9, heartbeat_seconds=None)

    assert fetched == []
    assert os.path.exists(winners_part)
    assert copy_row(app, pinned["ref"])["status"] == "building"


# --- Pins and licenses that change while a copy is made (#121) ------------------------

def test_no_copy_is_queued_for_a_pin_removed_meanwhile(client, app, copies_queue):
    pinned = pin_as_admin(client)
    host = app.extensions["dataset_library"]
    queued = list(copies_queue.refs)
    with app.app_context():
        facts = preview_module.preview(host, GREEK)
        pins.remove(host, pinned["ref"])
        # The request still holds the pin it found before the removal.
        stale = SimpleNamespace(id=pinned["ref"], provider="huggingface", external_id=GREEK)
        assert snapshots.request_copy(host, stale, facts) is None

    assert copy_row(app) is None
    assert copies_queue.refs == queued


def test_a_request_that_finds_a_copy_queued_meanwhile_returns_that_copy(client, app, copies_queue, monkeypatch):
    pinned = pin_as_admin(client)
    existing = copy_row(app, pinned["ref"])["id"]
    host = app.extensions["dataset_library"]
    real_for_pin = snapshots.for_pin
    looks = []

    def first_look_finds_nothing(session, pin_id):
        # The other request has not committed its copy when this one first looks.
        looks.append(pin_id)
        return None if len(looks) == 1 else real_for_pin(session, pin_id)

    monkeypatch.setattr(snapshots, "for_pin", first_look_finds_nothing)
    stale = SimpleNamespace(id=pinned["ref"], provider="huggingface", external_id=GREEK)
    with app.app_context():
        facts = preview_module.preview(host, GREEK)
        returned = snapshots.request_copy(host, stale, facts).id

    assert returned == existing
    assert copies_queue.refs == [existing]


def test_a_pin_removed_while_its_copy_is_made_gets_no_copy_and_no_file(client, app, snapshot_dir, tmp_path):
    pinned = pin_as_admin(client)
    copy_id = copy_row(app, pinned["ref"])["id"]
    shards = shards_of_three(tmp_path)

    def fetch_and_unpin(external_id, config, split, max_bytes, dest_dir):
        drop_pin_row(app, pinned["ref"])
        return imports.Fetched(paths=[str(path) for path in shards], revision=SHA)

    with app.app_context():
        snapshots.run_copy(
            copy_id, session=app.db.session, fetch_shards=fetch_and_unpin,
            snapshot_dir=snapshot_dir, max_bytes=10**9, heartbeat_seconds=None,
        )

    assert copy_row(app) is None
    assert copy_files(snapshot_dir) == []
    assert os.listdir(os.path.join(snapshot_dir, snapshots.INCOMING)) == []


def test_a_license_changed_to_warn_while_its_copy_is_made_leaves_no_copy_and_no_file(client, app, snapshot_dir, tmp_path):
    pinned = pin_as_admin(client)
    copy_id = copy_row(app, pinned["ref"])["id"]
    shards = shards_of_three(tmp_path)

    def fetch_and_restrict(external_id, config, split, max_bytes, dest_dir):
        set_policy(app, {"cc-by-4.0": "warn"})
        return imports.Fetched(paths=[str(path) for path in shards], revision=SHA)

    with app.app_context():
        snapshots.run_copy(
            copy_id, session=app.db.session, fetch_shards=fetch_and_restrict,
            snapshot_dir=snapshot_dir, max_bytes=10**9, heartbeat_seconds=None,
        )

    assert copy_row(app) is None
    assert copy_files(snapshot_dir) == []
    assert os.listdir(os.path.join(snapshot_dir, snapshots.INCOMING)) == []


@pytest.mark.parametrize("change", ["license", "pin"])
def test_a_copy_is_not_offered_once_its_license_is_not_allow_or_its_pin_is_gone(client, app, hub, cache, snapshot_dir, tmp_path, change):
    pinned, _ = ready_copy(client, app, snapshot_dir, tmp_path)
    if change == "license":
        set_policy(app, {"cc-by-4.0": "warn"})
    else:
        drop_pin_row(app, pinned["ref"])
    take_hub_down(hub, cache)

    resp = client.get(f"/api/library/datasets/huggingface/{GREEK}", headers=ALICE)

    assert (resp.status_code, resp.get_json()["code"]) == (503, "provider_unavailable")


@pytest.mark.parametrize("change", ["license", "pin"])
def test_a_running_import_reads_no_copy_once_its_license_is_not_allow_or_its_pin_is_gone(app, run, client, snapshot_dir, tmp_path, change):
    pinned, _ = ready_copy(client, app, snapshot_dir, tmp_path)
    if change == "license":
        set_policy(app, {"cc-by-4.0": "warn"})
    else:
        drop_pin_row(app, pinned["ref"])

    result, create = run(
        [write_shard(tmp_path / "unused.parquet", ["x"], [0])], size=2,
        external_id=GREEK, config="volume", fail_fetch=ProviderUnavailable("Hugging Face 503"), snapshot_dir=snapshot_dir,
    )

    assert (result["status"], result["code"]) == ("failed", "provider_busy")
    assert create.calls == []


# --- Upstream checks (#121) -----------------------------------------------------------

def add_ready_copies(app, count):
    """Ready copies of `count` pins, made directly: what these tests are about is the Hub's answers."""
    external_ids = [f"coastalcph/kept-{n}" for n in range(count)]
    with app.app_context():
        session = app.db.session
        for external_id in external_ids:
            pin = LibraryPin(provider="huggingface", external_id=external_id, card={"license": {"id": "cc-by-4.0"}}, labelled="unclear")
            session.add(pin)
            session.flush()
            session.add(LibrarySnapshot(
                pin_id=pin.id, provider="huggingface", external_id=external_id, config="ledgar", split="train",
                revision=SHA, license="cc-by-4.0", status="ready", row_count=1,
            ))
        session.commit()
    return external_ids


def test_a_hub_outage_ends_the_upstream_round_after_two_failed_checks_in_a_row(app):
    add_ready_copies(app, 3)
    asked = []

    def hub_down(external_id, revision):
        asked.append(external_id)
        raise ProviderUnavailable("Hugging Face 503")

    with app.app_context():
        assert snapshots.check_upstream(app.db.session, revision_exists=hub_down) == 0
        assert app.db.session.query(LibrarySnapshot).filter(LibrarySnapshot.upstream_checked_at.is_(None)).count() == 1
        assert app.db.session.query(LibrarySnapshot).filter(LibrarySnapshot.upstream_changed.is_(True)).count() == 0
    assert len(asked) == 2


def test_a_rate_limit_ends_the_upstream_round_at_once(app):
    add_ready_copies(app, 3)
    asked = []

    def limited(external_id, revision):
        asked.append(external_id)
        raise ProviderRateLimited("Hugging Face is limiting requests")

    with app.app_context():
        assert snapshots.check_upstream(app.db.session, revision_exists=limited) == 0
    assert len(asked) == 1


def test_one_copy_the_hub_always_refuses_does_not_stop_the_others_being_checked(app):
    add_ready_copies(app, 3)
    with app.app_context():
        session = app.db.session
        refused = session.query(LibrarySnapshot).order_by(LibrarySnapshot.id).first().external_id

    def one_refused(external_id, revision):
        if external_id == refused:
            raise ProviderUnavailable("Hugging Face answered 403")
        return True

    with app.app_context():
        session = app.db.session
        now = utc_now()
        for hours in range(5):
            snapshots.check_upstream(session, revision_exists=one_refused, now=now + timedelta(hours=2 * hours))
        healthy = session.query(LibrarySnapshot).filter(LibrarySnapshot.external_id != refused).all()
        assert len(healthy) == 2 and all(copy.upstream_checked_at is not None for copy in healthy)


def test_a_copy_unpinned_during_the_upstream_round_is_a_no_op_and_the_rest_are_still_checked(app):
    external_ids = add_ready_copies(app, 3)
    gone = external_ids[0]

    def check(external_id, revision):
        if external_id == gone:
            # The copy is unpinned while the Hub is asked about it.
            with app.app_context():
                app.db.session.execute(sa.text("DELETE FROM library_snapshots WHERE external_id = :id"), {"id": gone})
                app.db.session.commit()
            return False
        return True

    with app.app_context():
        assert snapshots.check_upstream(app.db.session, revision_exists=check) == 2
        rows = {copy.external_id: copy for copy in app.db.session.query(LibrarySnapshot)}
    assert gone not in rows
    assert {rows[external_id].upstream_checked_at is not None for external_id in external_ids[1:]} == {True}
    assert not any(rows[external_id].upstream_changed for external_id in external_ids[1:])


# --- Opening a copy offline: types, proposals, and what the rows are (#121) -----------

NER = "coastalcph/legal_ner"
SUMMARIES = "coastalcph/legal_summaries"


def test_an_offline_open_proposes_the_task_the_live_preview_would_for_its_columns(client, app, hub, cache, snapshot_dir, tmp_path):
    pinned = pin_as_admin(client, NER)
    shard = write_ner_shard(tmp_path / "ner.parquet", [["Clause", "one"], ["Another"]], [[1, 0], [0]])
    build_copy(app, copy_row(app, pinned["ref"])["id"], snapshot_dir, [shard])
    take_hub_down(hub, cache)

    body = client.get(f"/api/library/datasets/huggingface/{NER}", headers=ALICE).get_json()

    assert body["offline"] is True
    assert [column["name"] for column in body["columns"]] == ["tokens", "ner_tags"]
    assert body["proposed"] == {"taskType": "ner", "textColumn": "tokens", "labelColumn": "ner_tags"}
    assert body["rowCount"] == 2


def test_an_offline_open_proposes_the_summary_columns_for_a_summarization_pin(client, app, hub, cache, snapshot_dir, tmp_path):
    pinned = pin_as_admin(client, SUMMARIES)
    shard = write_summarization_shard(tmp_path / "summaries.parquet", ["A long document.", "Another"], ["Short.", "Brief."])
    build_copy(app, copy_row(app, pinned["ref"])["id"], snapshot_dir, [shard])
    take_hub_down(hub, cache)

    body = client.get(f"/api/library/datasets/huggingface/{SUMMARIES}", headers=ALICE).get_json()

    assert body["proposed"] == {"taskType": "summarization", "textColumn": "document", "labelColumn": "summary"}
    assert body["rowCount"] == 2


def test_a_corrupt_copy_opens_without_rows_instead_of_failing(client, app, hub, cache, snapshot_dir, tmp_path):
    _, copy = ready_copy(client, app, snapshot_dir, tmp_path)
    with open(os.path.join(snapshot_dir, copy["file_path"]), "wb") as corrupt:
        corrupt.write(b"not a Parquet file at all" * 10)
    take_hub_down(hub, cache)

    resp = client.get(f"/api/library/datasets/huggingface/{GREEK}", headers=ALICE)

    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    assert (body["preview"], body["rows"], body["offline"]) == ("ok", [], True)
    assert [column["name"] for column in body["columns"]] == ["text", "label", "extra"]


def test_an_offline_search_names_each_pin_by_its_title_then_its_cards_then_its_id(client, app, hub, cache):
    pin_as_admin(client, GREEK)
    named = pin_as_admin(client, "coastalcph/lex_glue")
    assert client.patch(f"/api/library/pins/{named['ref']}", json={"title": "Our LexGLUE"}, headers=ADMIN).status_code == 200
    untitled = pin_as_admin(client, NER)
    with app.app_context():
        pin = app.db.session.get(LibraryPin, untitled["ref"])
        pin.card = {**pin.card, "title": None}
        app.db.session.commit()
    take_hub_down(hub, cache)

    body = client.get("/api/library/search", headers=ALICE).get_json()

    assert body["status"] == "unavailable"
    titles = {card["externalId"]: card["title"] for card in body["results"]}
    assert titles == {GREEK: "Greek Legal Code", "coastalcph/lex_glue": "Our LexGLUE", NER: "legal_ner"}


def test_a_queued_copy_waiting_behind_imports_is_not_failed_until_hours_have_passed(app, snapshot_dir):
    add_ready_copies(app, 1)
    with app.app_context():
        session = app.db.session
        copy = session.query(LibrarySnapshot).one()
        copy.status, copy.updated_at = "queued", utc_now() - timedelta(hours=2)
        session.commit()
        assert snapshots.sweep(session, snapshot_dir) == []
        assert session.query(LibrarySnapshot).one().status == "queued"
        assert snapshots.sweep(session, snapshot_dir, now=utc_now() + timedelta(hours=5)) == [copy.id]


def test_the_sweep_leaves_files_it_did_not_name(app, snapshot_dir):
    os.makedirs(os.path.join(snapshot_dir, snapshots.INCOMING, "someone-elses"), exist_ok=True)
    foreign = os.path.join(snapshot_dir, "notes.parquet")
    orphan = os.path.join(snapshot_dir, "0b1d6a3e-8f2c-4c1a-9d7e-5a6b7c8d9e0f.parquet")
    for path in (foreign, orphan):
        with open(path, "wb") as handle:
            handle.write(b"x")
    with app.app_context():
        snapshots.sweep(app.db.session, snapshot_dir)
    assert os.path.exists(foreign)
    assert os.path.isdir(os.path.join(snapshot_dir, snapshots.INCOMING, "someone-elses"))
    assert not os.path.exists(orphan)


def test_an_offline_open_of_a_copy_whose_pin_just_went_is_not_an_error(app):
    from dataset_library.preview import LicenseHidden

    add_ready_copies(app, 1)
    with app.app_context():
        session = app.db.session
        copy = session.query(LibrarySnapshot).one()
        session.delete(session.get(LibraryPin, copy.pin_id))
        session.commit()
        host = type("Host", (), {"db": app.db, "snapshot_dir": None})()
        with pytest.raises(LicenseHidden):
            snapshots.offline_detail(host, copy)
