"""A queued Community import, as the worker runs it (#122, fixes 4 to 8): it checks what the student agreed to
again before it reads, reads exactly the file it was started on, keeps to the host's cap and the admin's, and
says Zenodo's words when Zenodo is the trouble."""
import copy
import csv
import threading

import pytest

from conftest import ADMIN, ALICE
from dataset_library import imports
from dataset_library.models import LibraryImport
from dataset_library.providers import community as provider
from fake_zenodo import FakeZenodo, ORIGIN

SANDBOX = "https://sandbox.zenodo.org"
PUBLICATION_101 = "5b1e8a6c-3f0d-4a7e-9c11-000000000101"


class Queue:
    def __init__(self):
        self.refs = []

    def __call__(self, import_ref):
        self.refs.append(import_ref)


class Sink:
    """create_dataset, as the host's would store what it was given."""

    def __init__(self):
        self.calls = []

    def __call__(self, owner_id, file_path, name, description, task_type):
        with open(file_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.calls.append({"owner": owner_id, "name": name, "description": description, "task_type": task_type, "rows": rows})
        return "dataset-1"


def forbidden(*args, **kwargs):
    raise AssertionError("a Community import reads its record on Zenodo, never Hugging Face shards")


@pytest.fixture
def zenodo():
    return FakeZenodo()


@pytest.fixture
def queue():
    return Queue()


@pytest.fixture
def app(make_app, zenodo, queue):
    return make_app(
        community_datasets=zenodo.datasets,
        community_record=zenodo.record,
        community_download=zenodo.download,
        community_base_url=zenodo.base_url,
        enqueue=queue,
        max_rows=lambda: 300,
    )


def switch(client, enabled=True, policy=None, limits=None):
    body = {"sources": [], "license_policy": policy or {}, "community": {"enabled": enabled}}
    if limits is not None:
        body["limits"] = limits
    response = client.put("/api/library/admin/settings", json=body, headers=ADMIN)
    assert response.status_code == 200, response.get_json()


def start(client, record="101", status=202, **body):
    payload = {"provider": "community", "dataset": record, "taskType": "classification",
               "textColumn": "text", "labelColumn": "label", "size": 5, **body}
    response = client.post("/api/library/imports", json=payload, headers=ALICE)
    assert response.status_code == status, response.get_json()
    return response.get_json()


def preview(client, record="101", status=200):
    response = client.get(f"/api/library/datasets/community/{record}", headers=ALICE)
    assert response.status_code == status, response.get_json()
    return response.get_json()


def work(app, tmp_path, zenodo, ref, *, max_bytes=None):
    """Run one queued import as the worker does, with the Community seams and no Hugging Face."""
    sink = Sink()
    extra = {} if max_bytes is None else {"max_bytes": max_bytes}
    with app.app_context():
        session = app.db.session
        imports.run_import(
            ref,
            session=session,
            fetch_shards=forbidden,
            create_dataset=sink,
            data_dir=str(tmp_path / "data"),
            seed=7,
            heartbeat_seconds=None,
            community_datasets=zenodo.datasets,
            community_base_url=zenodo.base_url,
            community_record=zenodo.record,
            community_download=zenodo.download,
            **extra,
        )
        session.expunge_all()
        return session.get(LibraryImport, ref).to_dict(), sink


def row_of(app, ref):
    with app.app_context():
        session = app.db.session
        found = session.get(LibraryImport, ref)
        return {"publication_id": found.publication_id, "origin": found.origin, "file_key": found.file_key,
                "revision": found.revision, "citation": found.citation}


def clear_reads(zenodo):
    zenodo.calls.clear()


def expire(cache):
    """Forget the Community preview cache, as its record and preview lifetimes do once they run out."""
    for key in [key for key in cache.values if key.startswith("library:community:")]:
        del cache.values[key]


def reads(zenodo):
    return [call for call in zenodo.calls if call[0] in ("record", "download")]


# --- Eligibility is checked again before anything is read (fix 4) -------------------------------

def test_a_publication_deleted_while_its_import_is_queued_fails_before_anything_is_read(client, app, zenodo, tmp_path):
    switch(client)
    ref = start(client)["id"]
    zenodo.listed = []
    clear_reads(zenodo)

    finished, sink = work(app, tmp_path, zenodo, ref)

    assert finished["status"] == "failed"
    assert finished["code"] == "not_in_library"
    assert finished["error"] == provider.NOT_LISTED
    assert reads(zenodo) == [] and sink.calls == []


def test_a_license_moved_to_hide_while_queued_fails_the_import(client, app, zenodo, tmp_path):
    switch(client)
    ref = start(client)["id"]
    switch(client, policy={"cc-by-4.0": "hide"})
    clear_reads(zenodo)

    finished, sink = work(app, tmp_path, zenodo, ref)

    assert finished["status"] == "failed"
    assert finished["code"] == "license_hidden"
    assert finished["error"] == provider.LICENSE_HIDDEN
    assert reads(zenodo) == [] and sink.calls == []


def test_the_source_switched_off_while_queued_fails_the_import(client, app, zenodo, tmp_path):
    switch(client)
    ref = start(client)["id"]
    switch(client, enabled=False)
    clear_reads(zenodo)

    finished, sink = work(app, tmp_path, zenodo, ref)

    assert finished["status"] == "failed"
    assert finished["code"] == "source_off"
    assert finished["error"] == provider.SOURCE_OFF
    assert reads(zenodo) == [] and sink.calls == []


def test_a_license_moved_to_warn_without_a_confirmation_on_record_fails_the_import(client, app, zenodo, tmp_path):
    switch(client)
    ref = start(client)["id"]  # cc-by-4.0 allows, so no confirmation was recorded
    switch(client, policy={"cc-by-4.0": "warn"})

    finished, sink = work(app, tmp_path, zenodo, ref)

    assert finished["status"] == "failed"
    assert finished["code"] == "needs_ack"
    assert sink.calls == []


def test_a_warn_license_confirmed_when_queued_still_imports(client, app, zenodo, tmp_path):
    switch(client)
    ref = start(client, record="102", licenseAck=True)["id"]  # cc-by-nc-4.0 warns; the student confirmed it
    switch(client, policy={"cc-by-nc-4.0": "warn"})

    finished, sink = work(app, tmp_path, zenodo, ref)

    assert finished["status"] == "completed"
    assert len(sink.calls) == 1


def test_a_publication_moved_to_another_zenodo_host_fails_the_import(client, app, zenodo, tmp_path):
    switch(client)
    ref = start(client)["id"]
    zenodo.listed[1]["origin"] = SANDBOX
    clear_reads(zenodo)

    finished, _ = work(app, tmp_path, zenodo, ref)

    assert finished["code"] == "not_in_library"
    assert reads(zenodo) == []


def test_the_worker_asks_zenodo_for_the_record_now_not_from_the_preview_cache(client, app, zenodo, tmp_path):
    switch(client)
    ref = start(client)["id"]  # the preview cached the record as open
    zenodo.records["101"] = copy.deepcopy(zenodo.records["104"])  # now restricted on Zenodo
    zenodo.records["101"]["record_id"] = "101"

    finished, _ = work(app, tmp_path, zenodo, ref)

    assert finished["code"] == "not_downloadable"
    assert finished["error"] == "This record is no longer downloadable"


# --- The file is pinned (fix 5) -----------------------------------------------------------------

def test_the_import_is_bound_to_its_publication_origin_and_file(client, app, zenodo):
    switch(client)
    ref = start(client)["id"]

    bound = row_of(app, ref)

    assert bound["publication_id"] == PUBLICATION_101
    assert bound["origin"] == ORIGIN
    assert bound["file_key"] == "data.csv"


def test_the_pinned_file_missing_from_the_record_fails_and_no_other_table_is_read(client, app, zenodo, tmp_path):
    switch(client)
    ref = start(client)["id"]
    # The record now lists a different table and no data.csv: the import must not pick notes.csv.
    zenodo.records["101"]["files"] = [{**zenodo.records["101"]["files"][0], "key": "notes.csv"}]
    clear_reads(zenodo)

    finished, sink = work(app, tmp_path, zenodo, ref)

    assert finished["status"] == "failed"
    assert finished["code"] == "not_downloadable"
    assert finished["error"] == "This record is no longer downloadable"
    assert [call[2] for call in zenodo.downloads()] == [] and sink.calls == []


def test_a_changed_file_says_the_community_words(client, app, zenodo, tmp_path):
    switch(client)
    ref = start(client)["id"]
    zenodo.records["101"]["files"][0]["checksum"] = "md5:0000"

    finished, _ = work(app, tmp_path, zenodo, ref)

    assert finished["code"] == "changed"
    assert finished["error"] == provider.CHANGED


def test_a_changed_file_at_start_says_the_community_words(client):
    switch(client)
    first = start(client)

    refused = start(client, status=409, revision="md5:0000")

    assert refused["code"] == "changed"
    assert refused["error"] == provider.CHANGED
    assert first["revision"] != "md5:0000"


# --- The caps are the same in the API and the worker (fix 6) -------------------------------------

@pytest.fixture
def capped(make_app, zenodo, queue):
    """A host whose download limit is 4 MB, with the admin's cap at 8 MB: the lower one holds."""
    return make_app(
        community_datasets=zenodo.datasets,
        community_record=zenodo.record,
        community_download=zenodo.download,
        community_base_url=zenodo.base_url,
        enqueue=queue,
        max_rows=lambda: 300,
        max_fetch_bytes=4_000_000,
    )


def test_a_file_between_the_host_limit_and_the_admin_cap_is_refused_by_the_api_and_the_worker_alike(capped, zenodo, tmp_path, cache):
    client = capped.test_client()
    switch(client, limits={"maxDownloadMb": 8})
    ref = start(client)["id"]  # a small file, as it was when the student started
    zenodo.records["101"]["files"][0]["size"] = 6_000_000  # now 6 MB: over the host's 4 MB, under the admin's 8 MB
    expire(cache)

    previewed = preview(client)
    finished, sink = work(capped, tmp_path, zenodo, ref, max_bytes=4_000_000)

    assert previewed["importable"] is False and previewed["reasonCode"] == "too_large"
    assert finished["status"] == "failed" and finished["code"] == "too_large"
    assert finished["error"] == "Too large to sample here (limit 4 MB)"
    assert sink.calls == []


def test_the_admin_cap_lowers_the_limit_and_never_raises_it(capped, zenodo, tmp_path):
    client = capped.test_client()
    switch(client, limits={"maxDownloadMb": 1})
    ref = start(client)["id"]

    finished, sink = work(capped, tmp_path, zenodo, ref, max_bytes=4_000_000)

    assert finished["status"] == "completed"  # 973 bytes is under the admin's 1 MB, which is under the host's 4 MB
    assert len(sink.calls) == 1


# --- Start and retry say Zenodo, and bind the citation (fixes 7 and 8) -------------------------------

def test_a_zenodo_outage_at_start_says_zenodo(client, zenodo):
    switch(client)
    zenodo.busy = True

    refused = start(client, status=503)

    assert refused["code"] == "provider_busy"
    assert refused["error"] == "Zenodo is busy. Try again in a few minutes."


def test_a_zenodo_outage_at_retry_says_zenodo(client, app, zenodo, cache):
    switch(client)
    first = start(client)
    with app.app_context():
        session = app.db.session
        session.get(LibraryImport, first["id"]).status = imports.FAILED
        session.commit()
    expire(cache)  # the retry asks Zenodo again, as it would once the preview cache had run out
    zenodo.busy = True

    retried = client.post(f"/api/library/imports/{first['id']}/retry", headers=ALICE)

    assert retried.status_code == 503
    assert retried.get_json()["error"] == "Zenodo is busy. Try again in a few minutes."
    assert retried.get_json()["code"] == "provider_busy"


def test_a_community_citation_has_no_revision_when_it_is_started(client, app):
    switch(client)
    ref = start(client)["id"]

    assert row_of(app, ref)["citation"]["revision"] is None
    assert row_of(app, ref)["revision"] is not None  # the checksum stays on the import, for the change check


def test_a_start_while_the_preview_is_still_being_prepared_asks_for_a_retry(make_app, zenodo, queue):
    zenodo.hold = threading.Event()  # Zenodo is slow: the preview has not answered when the student starts
    client = make_app(
        community_datasets=zenodo.datasets,
        community_record=zenodo.record,
        community_download=zenodo.download,
        community_base_url=zenodo.base_url,
        community_preview_seconds=0.05,
        enqueue=queue,
        max_rows=lambda: 300,
    ).test_client()
    switch(client)
    try:
        refused = start(client, status=503)
    finally:
        zenodo.hold.set()

    assert refused["code"] == "preparing"
    assert refused["error"] == provider.PREPARING
    assert queue.refs == []
