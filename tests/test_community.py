"""The Community source (#122): Zenodo records of open datasets BAIGLE users published, found, previewed, imported.

The host's three seams are faked from recorded Zenodo answers (fake_zenodo.py). No test reaches the network,
and a Community import never reads a Hugging Face shard or a local Dataset."""
import copy
import csv

import pytest

from conftest import ADMIN, ALICE
from dataset_library import imports
from dataset_library.licenses import resolve_license
from dataset_library.models import LibraryImport
from dataset_library.providers import community as provider
from fake_zenodo import RECORDS, FakeZenodo


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


def switch(client, enabled=True, policy=None):
    body = {"sources": [], "license_policy": policy or {}, "community": {"enabled": enabled}}
    response = client.put("/api/library/admin/settings", json=body, headers=ADMIN)
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def cards(client, **params):
    response = client.get("/api/library/search", query_string={"source": "community", **params}, headers=ALICE)
    assert response.status_code == 200, response.get_json()
    return response.get_json()


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


def answer(zenodo, record_id, like):
    """Make Zenodo answer for `record_id` as it answers for `like`: the way a record was closed."""
    zenodo.records[record_id] = copy.deepcopy(RECORDS[like])
    zenodo.records[record_id]["record_id"] = record_id


def checksum_of(record_id):
    return RECORDS[record_id]["files"][0]["checksum"]


# --- The provider, as pure functions ---------------------------------------------------------

def test_a_creator_is_read_as_family_then_given_and_anything_else_as_literal():
    assert provider.author("Chakravarthy, Shashank") == {"family": "Chakravarthy", "given": "Shashank"}
    assert provider.author("  de la Cruz,  Ana ") == {"family": "de la Cruz", "given": "Ana"}
    assert provider.author("Legal Blocks Lab") == {"literal": "Legal Blocks Lab"}
    assert provider.author("Shashank Chakravarthy") == {"literal": "Shashank Chakravarthy"}
    assert provider.author("Family,") == {"literal": "Family"}
    assert provider.author("  ") is None


def test_the_year_is_the_publication_years_and_nothing_else():
    assert provider.year("2026-11-10T09:30:00+00:00") == 2026
    assert provider.year("") is None
    assert provider.year("20x6-01-01") is None


def test_the_table_file_is_the_data_file_BAIGLE_publishes_or_else_the_first_table():
    assert provider.table_file([{"key": "README.txt"}, {"key": "data.csv"}])["key"] == "data.csv"
    assert provider.table_file([{"key": "notes.csv"}, {"key": "data.xlsx"}])["key"] == "data.xlsx"
    assert provider.table_file([{"key": "notes.csv"}])["key"] == "notes.csv"
    assert provider.table_file([{"key": "README.txt"}]) is None


def test_a_record_that_cannot_be_imported_says_why():
    open_record = {"access": {"record": "public", "files": "public"}, "downloadable": True,
                   "files": [{"key": "data.csv"}]}
    assert provider.refusal(open_record) is None
    assert provider.refusal({**open_record, "access": {"record": "restricted", "files": "restricted"}, "downloadable": False}) == (
        "restricted", provider.RESTRICTED)
    assert provider.refusal({**open_record, "access": {"record": "public", "files": "restricted"}, "downloadable": False}) == (
        "metadata_only", provider.METADATA_ONLY)
    assert provider.refusal({**open_record, "files": [{"key": "README.txt"}]}) == ("no_table", provider.NO_TABLE)


def test_the_message_for_a_record_no_longer_downloadable_is_exact():
    assert provider.NOT_DOWNLOADABLE == "This record is no longer downloadable"
    assert provider.too_large(200_000_000) == "Too large to sample here (limit 200 MB)"


@pytest.mark.parametrize("tag", ["cc-by-4.0", "cc-by-sa-4.0", "cc-by-nc-4.0", "cc0-1.0", "mit", "apache-2.0"])
def test_every_license_fair_publishing_offers_resolves_to_its_own_entry(tag):
    # fair-publishing 0.2.0 config.DEFAULT_LICENSES: each must be in the library's table, or it would count as unknown.
    assert resolve_license(tag).id == tag


def test_a_card_is_shaped_for_the_library(zenodo):
    card = provider.card(zenodo.listed[1])

    assert card["id"] == "community:101"
    assert card["provider"] == "community"
    assert card["externalId"] == "101"
    assert card["author"] == "Chakravarthy, Shashank"
    assert card["url"] == "https://doi.org/10.5281/zenodo.101"
    assert card["doi"] == "10.5281/zenodo.101"
    assert card["tasks"] == []
    assert card["license"] == "cc-by-4.0"
    assert card["citation"] == {
        "title": "Termination and governing-law clauses",
        "authors": [{"family": "Chakravarthy", "given": "Shashank"}, {"literal": "Legal Blocks Lab"}],
        "year": 2026,
        "url": "https://doi.org/10.5281/zenodo.101",
        "revision": None,
        "accessedAt": None,
        "bibtex": None,
    }


# --- Listing and search ----------------------------------------------------------------------

def test_open_datasets_are_listed_as_community_cards_with_their_license_resolved(client, zenodo):
    switch(client)

    body = cards(client)

    assert [card["externalId"] for card in body["results"]] == ["102", "101", "103"]
    by_record = {card["externalId"]: card for card in body["results"]}
    assert by_record["101"]["license"] == resolve_license("cc-by-4.0").at_level("allow")
    assert by_record["102"]["license"]["level"] == "warn"  # non-commercial: a student confirms it
    assert by_record["103"]["license"]["id"] == "cc0-1.0"
    assert by_record["103"]["author"] is None  # a record from before 0.2.0 lists no creators
    assert by_record["102"]["citation"]["authors"] == [{"family": "de la Cruz", "given": "Ana"}]
    assert ("datasets",) in zenodo.calls


def test_no_library_route_takes_an_id_parameter_a_host_would_refuse(app):
    # A host may refuse any `*_id` URL parameter that is not its own UUID (BAIGLE does), so the library uses `*_ref`.
    parameters = {argument for rule in app.url_map.iter_rules() for argument in rule.arguments}

    assert not {name for name in parameters if name.endswith("_id")}


def test_the_community_source_is_off_until_an_admin_switches_it_on(client):
    everything = client.get("/api/library/search", headers=ALICE).get_json()
    assert all(card["provider"] != "community" for card in everything["results"])
    options = client.get("/api/library/options", headers=ALICE).get_json()
    assert all(source["provider"] != "community" for source in options["sources"])
    refused = client.get("/api/library/search", query_string={"source": "community"}, headers=ALICE)
    assert refused.status_code == 400


def test_switched_on_the_source_is_offered_with_its_name(client):
    switch(client)

    options = client.get("/api/library/options", headers=ALICE).get_json()

    assert {"value": "community", "label": "Community", "provider": "community"} in options["sources"]


def test_without_the_host_seams_the_source_is_not_offered_at_all(make_app):
    client = make_app().test_client()

    settings = switch_refused(client, enabled=True)

    assert settings["fields"] == {"community": "The Community source is not set up on this platform."}
    current = client.get("/api/library/admin/settings", headers=ADMIN).get_json()
    assert current["community"] == {"available": False, "enabled": False}
    everything = client.get("/api/library/search", headers=ALICE).get_json()
    assert all(card["provider"] != "community" for card in everything["results"])


def switch_refused(client, enabled):
    response = client.put(
        "/api/library/admin/settings",
        json={"sources": [], "license_policy": {}, "community": {"enabled": enabled}},
        headers=ADMIN,
    )
    assert response.status_code == 400
    return response.get_json()


def test_a_query_matches_the_title_or_a_creators_name(client):
    switch(client)

    assert [c["externalId"] for c in cards(client, q="cruz")["results"]] == ["102"]
    assert [c["externalId"] for c in cards(client, q="termination")["results"]] == ["101"]
    assert cards(client, q="nothing like this")["results"] == []


def test_a_license_filter_finds_only_that_license(client):
    switch(client)

    assert [c["externalId"] for c in cards(client, license="cc-by-nc-4.0")["results"]] == ["102"]
    assert cards(client, license="unknown")["results"] == []


def test_a_publication_carries_no_task_so_a_task_filter_finds_none(client):
    switch(client)

    assert cards(client, task="classification")["results"] == []


def test_a_hidden_license_removes_its_cards(client):
    switch(client, policy={"cc-by-nc-4.0": "hide"})

    assert [c["externalId"] for c in cards(client)["results"]] == ["101", "103"]


def test_a_hub_outage_leaves_the_community_cards_in_place(client, hub):
    switch(client)
    hub.fail = 503

    body = client.get("/api/library/search", query_string={"q": "termination"}, headers=ALICE).get_json()

    assert body["status"] == "unavailable"
    assert [card["externalId"] for card in body["results"] if card["provider"] == "community"] == ["101"]


def test_with_no_hugging_face_organisation_an_empty_search_asks_the_hub_for_nothing(client, hub):
    # With no organisation and no query the Hub would list its whole catalogue, so nothing is asked for (#122, fix 9).
    switch(client)

    body = client.get("/api/library/search", query_string={"q": ""}, headers=ALICE).get_json()

    assert body["status"] == "ok"
    assert hub.calls == []
    assert sorted(card["externalId"] for card in body["results"]) == ["101", "102", "103"]


def test_a_community_search_never_asks_the_hub(client, hub):
    switch(client)

    cards(client)

    assert hub.calls == []


def test_an_organisation_cannot_take_the_community_source_name(client):
    response = client.put(
        "/api/library/admin/settings",
        json={"sources": [{"ref": "community", "enabled": True}], "license_policy": {}},
        headers=ADMIN,
    )

    assert response.status_code == 400
    assert "sources.0" in response.get_json()["fields"]


def test_switching_the_source_on_and_off_is_audited(client, audits):
    switch(client)
    switch(client, enabled=False)

    community_saves = [a for a in audits if a["event"] == "library.settings_saved" and "community_changed" in a]
    assert [a["community_changed"] for a in community_saves] == [True, True]
    assert client.get("/api/library/admin/settings", headers=ADMIN).get_json()["community"]["enabled"] is False


# --- Preview: the record as Zenodo says it is now --------------------------------------------

def test_an_open_record_is_importable_with_its_table_and_a_proposal(client, zenodo):
    switch(client)

    body = preview(client)

    assert body["importable"] is True
    assert body["reasonCode"] is None
    assert body["preview"] == "ok"
    assert [column["name"] for column in body["columns"]] == ["text", "label", "label_source"]
    assert body["rowCount"] == 12
    assert len(body["rows"]) == 8
    assert body["rows"][0][0] == "The Supplier may terminate this Agreement by giving thirty days' notice."
    assert body["proposed"] == {"taskType": "classification", "textColumn": "text", "labelColumn": "label"}
    assert body["file"] == {"key": "data.csv", "size": RECORDS["101"]["files"][0]["size"]}
    assert body["revision"] == checksum_of("101")
    assert body["url"] == "https://doi.org/10.5281/zenodo.101"
    assert body["license"]["level"] == "allow"


def test_a_record_closed_since_it_was_listed_shows_as_restricted(client, zenodo):
    switch(client)
    answer(zenodo, "101", like="104")

    body = preview(client)

    assert body["importable"] is False
    assert body["reasonCode"] == "restricted"
    assert body["reason"] == provider.RESTRICTED


def test_a_metadata_only_record_shows_as_not_importable_with_its_reason(client, zenodo):
    switch(client)
    answer(zenodo, "101", like="105")

    body = preview(client)

    assert body["importable"] is False
    assert body["reasonCode"] == "metadata_only"
    assert body["reason"] == provider.METADATA_ONLY


def test_a_record_zenodo_no_longer_serves_is_not_downloadable(client, zenodo):
    switch(client)
    zenodo.gone.add("101")

    body = preview(client)

    assert body["importable"] is False
    assert body["reason"] == "This record is no longer downloadable"


def test_a_record_with_no_table_file_says_which_files_it_can_read(client, zenodo):
    switch(client)
    answer(zenodo, "101", like="107")

    body = preview(client)

    assert body["reasonCode"] == "no_table"


def test_a_file_over_the_download_cap_is_not_importable(client, zenodo):
    switch(client)
    zenodo.records["101"]["files"][0]["size"] = 300_000_000

    body = preview(client)

    assert body["reasonCode"] == "too_large"
    assert body["reason"] == "Too large to sample here (limit 200 MB)"
    assert not zenodo.downloads()


def test_a_file_that_is_not_a_table_is_not_importable_with_a_plain_reason(client, zenodo):
    switch(client)
    zenodo.contents[("101", "data.csv")] = b"\x81\x8d\x8f\x90"  # not UTF-8, and undefined in Windows-1252 too

    body = preview(client)

    assert body["reasonCode"] == "unreadable"
    assert body["reason"].startswith("We can't read this record's data file as a table")


def test_the_preview_reads_the_file_once_for_its_checksum_and_cap(client, zenodo):
    switch(client)

    preview(client)
    preview(client)

    assert len(zenodo.downloads()) == 1


def test_a_license_the_admin_hid_is_not_found_and_nothing_is_downloaded(client, zenodo):
    switch(client, policy={"cc-by-4.0": "hide"})

    preview(client, status=404)

    assert zenodo.downloads() == []
    assert not [call for call in zenodo.calls if call[0] == "record"]


def test_a_record_that_is_not_listed_is_not_in_the_library(client):
    switch(client)

    preview(client, record="999", status=404)


def test_a_record_id_that_is_not_one_is_refused(client):
    switch(client)

    assert client.get("/api/library/datasets/community/not-a-record", headers=ALICE).status_code == 400


def test_zenodo_unavailable_is_a_503_not_a_500(client, zenodo):
    switch(client)
    zenodo.busy = True

    body = preview(client, status=503)

    assert body["code"] == "provider_unavailable"


# --- Starting an import ----------------------------------------------------------------------

def test_an_import_is_queued_with_its_record_doi_file_checksum_and_citation_source(client, queue):
    switch(client)

    started = start(client)

    assert started["status"] == "queued"
    assert started["provider"] == "community"
    assert started["externalId"] == "101"
    assert started["url"] == "https://doi.org/10.5281/zenodo.101"
    assert started["revision"] == checksum_of("101")
    assert started["source"] == "zenodo"
    assert started["name"] == "Termination and governing-law clauses"
    assert queue.refs == [started["id"]]


def test_a_restricted_record_is_refused_before_anything_is_queued(client, zenodo, queue):
    switch(client)
    answer(zenodo, "101", like="104")

    refused = start(client, status=409)

    # Closed since it was listed: at start it is refused in the worker's words; the preview keeps "restricted" (#122).
    assert refused["code"] == "not_downloadable"
    assert refused["error"] == "This record is no longer downloadable"
    assert queue.refs == []
    assert preview(client)["reasonCode"] == "restricted"


def test_a_non_commercial_record_needs_the_students_confirmation(client):
    switch(client)

    assert start(client, record="102", status=409)["code"] == "needs_ack"
    confirmed = start(client, record="102", licenseAck=True)

    assert confirmed["license"] == "cc-by-nc-4.0"
    assert confirmed["licenseAckAt"] is not None


def test_a_hidden_license_is_refused_and_says_so(client):
    switch(client, policy={"cc-by-nc-4.0": "hide"})

    assert start(client, record="102", status=403)["code"] == "license_hidden"


def test_a_community_import_needs_a_record_id(client):
    switch(client)

    assert start(client, record="coastalcph/lex_glue", status=400)["code"] == "bad_id"


def test_a_retry_of_a_failed_community_import_stays_a_community_import(client, app, queue):
    switch(client)
    first = start(client)
    with app.app_context():
        session = app.db.session
        session.get(LibraryImport, first["id"]).status = imports.FAILED
        session.commit()

    retried = client.post(f"/api/library/imports/{first['id']}/retry", headers=ALICE)

    assert retried.status_code == 202
    assert retried.get_json()["provider"] == "community"
    assert retried.get_json()["externalId"] == "101"


# --- The worker reads the record's own file --------------------------------------------------

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
        found = session.get(LibraryImport, ref)
        return found.to_dict(), sink


def test_a_community_import_samples_the_record_file_and_never_a_hugging_face_shard(client, app, zenodo, tmp_path):
    switch(client)
    ref = start(client, size=5)["id"]
    zenodo.calls.clear()  # count only the worker's reads, not the preview at start

    finished, sink = work(app, tmp_path, zenodo, ref)

    assert finished["status"] == "completed"
    assert finished["rows"] == 5
    (created,) = sink.calls
    assert created["task_type"] == "classification"
    # A labelled classification import keeps the label balance by default, and its description says so.
    assert created["description"] == "A sample that keeps the label balance, from Zenodo: https://doi.org/10.5281/zenodo.101"
    assert all(row["text"] and row["label"] in {"Terminations", "Governing Laws", "Notices"} for row in created["rows"])
    # The import is checked against the listing, the record is asked again, then its one table file is read at the
    # default cap, and nothing else is read.
    assert [call[:2] for call in zenodo.calls] == [("datasets",), ("record", "101"), ("download", "101")]
    assert zenodo.downloads()[0][2:] == ("data.csv", 200_000_000)


def test_a_balanced_sample_of_a_community_xlsx_keeps_its_labels(client, app, zenodo, tmp_path):
    switch(client)
    ref = start(client, record="102", size=4, licenseAck=True)["id"]

    finished, sink = work(app, tmp_path, zenodo, ref)

    assert finished["status"] == "completed"
    assert {row["label"] for row in sink.calls[0]["rows"]} <= {"Granted", "Dismissed"}
    assert len(sink.calls[0]["rows"]) == 4


def test_a_json_record_with_numeric_labels_keeps_them_as_written(client, app, zenodo, tmp_path):
    switch(client)
    ref = start(client, record="103", size=4, textColumn="text", labelColumn="label")["id"]

    finished, sink = work(app, tmp_path, zenodo, ref)

    assert finished["status"] == "completed"
    assert {row["label"] for row in sink.calls[0]["rows"]} <= {"0", "1"}


def test_a_record_gone_mid_import_fails_with_the_zenodo_words(client, app, zenodo, tmp_path):
    switch(client)
    ref = start(client)["id"]
    zenodo.gone.add("101")

    finished, sink = work(app, tmp_path, zenodo, ref)

    assert finished["status"] == "failed"
    assert finished["code"] == "not_downloadable"
    assert finished["error"] == "This record is no longer downloadable"
    assert sink.calls == []


def test_a_record_closed_mid_import_fails_with_the_same_words(client, app, zenodo, tmp_path):
    switch(client)
    ref = start(client)["id"]
    answer(zenodo, "101", like="104")

    finished, _ = work(app, tmp_path, zenodo, ref)

    assert finished["status"] == "failed"
    assert finished["error"] == "This record is no longer downloadable"


def test_a_file_changed_since_the_preview_is_not_imported(client, app, zenodo, tmp_path):
    switch(client)
    ref = start(client)["id"]
    zenodo.records["101"]["files"][0]["checksum"] = "md5:0000"

    finished, sink = work(app, tmp_path, zenodo, ref)

    assert finished["status"] == "failed"
    assert finished["code"] == "changed"
    assert sink.calls == []


def test_a_file_over_the_worker_cap_fails_with_the_limit(client, app, zenodo, tmp_path):
    switch(client)
    ref = start(client)["id"]
    zenodo.records["101"]["files"][0]["size"] = 2_000_000

    finished, _ = work(app, tmp_path, zenodo, ref, max_bytes=1_000_000)

    assert finished["status"] == "failed"
    assert finished["code"] == "too_large"
    assert finished["error"] == "Too large to sample here (limit 1 MB)"


def test_zenodo_busy_during_an_import_fails_with_its_own_words(client, app, zenodo, tmp_path):
    switch(client)
    ref = start(client)["id"]
    zenodo.busy = True

    finished, _ = work(app, tmp_path, zenodo, ref)

    assert finished["status"] == "failed"
    assert finished["code"] == "provider_busy"
    assert finished["error"] == "Zenodo is busy. Try again in a few minutes."


def test_a_finished_community_import_keeps_its_doi_record_and_citation_as_provenance(client, app, zenodo, tmp_path):
    switch(client)
    ref = start(client)["id"]
    work(app, tmp_path, zenodo, ref)

    response = client.get("/api/library/provenance/dataset-1", headers=ALICE)

    assert response.status_code == 200
    record = response.get_json()
    assert record["provider"] == "community"
    assert record["externalId"] == "101"
    assert record["doi"] == "10.5281/zenodo.101"
    assert record["source"] == "zenodo"
    assert record["revision"] == checksum_of("101")
    # A Community citation names its DOI, not a revision: the checksum is the import's, and only the import's (#122).
    assert record["citation"]["revision"] is None
    assert record["citation"]["accessedAt"] is not None
    assert record["citation"]["url"] == "https://doi.org/10.5281/zenodo.101"
    assert record["citation"]["bibtex"] is None


def test_a_worker_without_the_community_seams_fails_the_import_rather_than_reading_anything_else(client, app, zenodo, tmp_path):
    switch(client)
    ref = start(client)["id"]
    zenodo.calls.clear()  # the preview at start read the file; the worker must read nothing
    sink = Sink()
    with app.app_context():
        session = app.db.session
        imports.run_import(
            ref, session=session, fetch_shards=forbidden, create_dataset=sink,
            data_dir=str(tmp_path / "data"), heartbeat_seconds=None,
        )
        found = session.get(LibraryImport, ref).to_dict()

    assert found["status"] == "failed"
    assert found["code"] == "not_offered"
    assert sink.calls == []
    assert zenodo.downloads() == []
