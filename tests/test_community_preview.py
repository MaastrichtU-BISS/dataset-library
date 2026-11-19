"""A Community preview (#122, fixes 1 and 2): the preview budget, one fetch per key at a time, the record
cache and the deadline, and the Zenodo origin a Publication's record is on.

Zenodo is faked (fake_zenodo.py). A slow repository is one whose `hold` event is not yet set."""
import threading
import time

import pytest

from conftest import ADMIN, ALICE
from dataset_library import community, limits
from dataset_library.providers import community as provider
from fake_zenodo import FakeZenodo

SANDBOX = "https://sandbox.zenodo.org"


@pytest.fixture
def zenodo():
    return FakeZenodo()


def make(make_app, zenodo, **options):
    return make_app(
        community_datasets=zenodo.datasets,
        community_record=zenodo.record,
        community_download=zenodo.download,
        community_base_url=zenodo.base_url,
        enqueue=lambda import_ref: None,
        max_rows=lambda: 300,
        **options,
    ).test_client()


def switch(client, enabled=True, limits=None):
    body = {"sources": [], "license_policy": {}, "community": {"enabled": enabled}}
    if limits is not None:
        body["limits"] = limits
    response = client.put("/api/library/admin/settings", json=body, headers=ADMIN)
    assert response.status_code == 200, response.get_json()


def preview(client, record="101", status=200):
    response = client.get(f"/api/library/datasets/community/{record}", headers=ALICE)
    assert response.status_code == status, response.get_json()
    return response.get_json()


def until_ready(client, record="101"):
    """Ask again, as the page does, until the preview is no longer being prepared."""
    for _ in range(200):
        body = preview(client, record)
        if not body["preparing"]:
            return body
        time.sleep(0.02)
    raise AssertionError("the preview was still being prepared")


def record_calls(zenodo):
    return [call for call in zenodo.calls if call[0] == "record"]


def cards_for(client):
    response = client.get("/api/library/search", query_string={"source": "community"}, headers=ALICE)
    return sorted(card["externalId"] for card in response.get_json()["results"])


# --- The preview budget (fix 1) ----------------------------------------------------------------

def test_a_file_over_the_preview_budget_is_not_downloaded_and_stays_importable(make_app, zenodo):
    client = make(make_app, zenodo, community_preview_max_bytes=500)
    switch(client)

    body = preview(client)

    assert body["preview"] == "too_large"
    assert body["importable"] is True
    assert body["reasonCode"] is None
    assert body["file"] == {"key": "data.csv", "size": 973}  # the card, its files and the size all stand
    assert body["rows"] == [] and body["columns"] == []
    assert zenodo.downloads() == []


def test_a_file_over_the_budget_can_still_be_queued_and_the_worker_checks_its_columns(make_app, zenodo):
    client = make(make_app, zenodo, community_preview_max_bytes=500)
    switch(client)

    response = client.post(
        "/api/library/imports",
        json={"provider": "community", "dataset": "101", "taskType": "classification", "textColumn": "text",
              "labelColumn": "label", "size": 5},
        headers=ALICE,
    )

    assert response.status_code == 202, response.get_json()
    assert response.get_json()["status"] == "queued"


def test_a_file_within_the_budget_is_read_for_its_rows_at_no_more_than_the_budget(make_app, zenodo):
    client = make(make_app, zenodo, community_preview_max_bytes=2_000)
    switch(client)

    body = preview(client)

    assert body["preview"] == "ok"
    assert len(body["rows"]) == 8
    assert [call[3] for call in zenodo.downloads()] == [2_000]


def test_a_file_of_unknown_size_is_refused_at_the_budget_and_shown_without_rows(make_app, zenodo):
    zenodo.records["101"]["files"][0]["size"] = None
    client = make(make_app, zenodo, community_preview_max_bytes=500)
    switch(client)

    body = preview(client)

    assert body["preview"] == "too_large"
    assert body["importable"] is True
    assert [call[3] for call in zenodo.downloads()] == [500]


def test_a_file_within_the_budget_but_over_the_import_cap_is_not_importable(make_app, zenodo):
    client = make(make_app, zenodo, max_fetch_bytes=500)
    switch(client)

    body = preview(client)

    assert body["importable"] is False
    assert body["reasonCode"] == "too_large"
    assert zenodo.downloads() == []


# --- One fetch per key, a deadline, and the record cache (fix 1) ----------------------------------

def test_a_request_that_finds_a_fetch_running_is_told_the_preview_is_being_prepared(make_app, zenodo):
    zenodo.hold = threading.Event()  # Zenodo is slow: nothing answers until this is set
    client = make(make_app, zenodo, community_preview_seconds=0.05)
    switch(client)

    started = time.monotonic()
    first = preview(client)  # its fetch runs on a helper thread that the request waits for only briefly
    waited = time.monotonic() - started
    second = preview(client)  # the lock is taken: this request asks Zenodo for nothing

    assert first["preparing"] is True and first["preview"] == "preparing" and first["importable"] is False
    assert second["preparing"] is True and second["importable"] is False
    assert waited < 5
    assert record_calls(zenodo) == [("record", "101")]
    assert zenodo.downloads() == []

    zenodo.hold.set()
    body = until_ready(client)

    assert body["preview"] == "ok"
    assert len(zenodo.downloads()) == 1


def test_the_fetch_lock_is_released_when_the_fetch_fails(make_app, zenodo, cache):
    client = make(make_app, zenodo)
    switch(client)
    zenodo.busy = True

    preview(client, status=503)

    assert not [key for key in cache.values if key.endswith(":lock")]
    zenodo.busy = False
    assert preview(client)["preview"] == "ok"


def test_a_record_is_kept_for_a_minute_and_a_preview_for_ten_minutes(make_app, zenodo, cache):
    client = make(make_app, zenodo)
    switch(client)

    preview(client)
    preview(client)

    assert len(record_calls(zenodo)) == 1
    assert [cache.ttls[key] for key in cache.values if ":record:" in key] == [60]
    assert [cache.ttls[key] for key in cache.values if ":head:" in key] == [600]


def test_the_preview_deadline_and_budget_are_well_inside_the_web_workers_limit():
    assert community.PREVIEW_SECONDS < 120
    assert community.PREVIEW_MAX_BYTES == 16_000_000
    assert community.Seams.__dataclass_fields__["preview_seconds"].default == community.PREVIEW_SECONDS
    assert community.Seams.__dataclass_fields__["preview_max_bytes"].default == community.PREVIEW_MAX_BYTES


# --- Record identity (fix 2) -------------------------------------------------------------------

def test_a_publication_on_another_zenodo_host_is_not_previewed_as_the_configured_record(make_app, zenodo):
    zenodo.listed[1]["origin"] = SANDBOX  # record 101 is a sandbox record; this platform reads production
    client = make(make_app, zenodo)
    switch(client)

    body = preview(client, status=404)

    assert body["code"] == "not_found"
    assert record_calls(zenodo) == []
    assert zenodo.downloads() == []


def test_only_publications_on_the_configured_host_are_listed(make_app, zenodo):
    zenodo.listed[1]["origin"] = SANDBOX
    client = make(make_app, zenodo)
    switch(client)

    assert cards_for(client) == ["102", "103"]


def test_a_publication_with_no_origin_is_not_listed(make_app, zenodo):
    zenodo.listed[1].pop("origin")
    client = make(make_app, zenodo)
    switch(client)

    assert cards_for(client) == ["102", "103"]


def test_the_fetch_cache_is_bound_to_the_zenodo_origin(make_app, zenodo, cache):
    client = make(make_app, zenodo)
    switch(client)
    preview(client)
    assert any(key.startswith("library:community:https://zenodo.org:") for key in cache.values)

    # The platform moves to the sandbox: nothing fetched from production answers for it.
    zenodo.configured = SANDBOX
    for row in zenodo.listed:
        row["origin"] = SANDBOX
    preview(client)

    assert len(record_calls(zenodo)) == 2
    assert any(key.startswith(f"library:community:{SANDBOX}:") for key in cache.values)


def test_a_zenodo_origin_is_scheme_and_host_with_the_default_port_left_out():
    assert provider.origin_of("https://zenodo.org/records/101") == "https://zenodo.org"
    assert provider.origin_of("HTTPS://Sandbox.Zenodo.org:443/api") == "https://sandbox.zenodo.org"
    assert provider.origin_of("https://zenodo.org:8443/x") == "https://zenodo.org:8443"
    assert provider.origin_of("ftp://zenodo.org/x") is None
    assert provider.origin_of("https://user@zenodo.org/x") is None
    assert provider.origin_of("") is None


def test_a_publication_is_on_the_origin_its_record_link_names_or_its_doi_prefix_names():
    assert provider.origin_for("https://zenodo.org/records/101", "10.5281/zenodo.101") == "https://zenodo.org"
    assert provider.origin_for("", "10.5072/zenodo.101") == "https://sandbox.zenodo.org"
    assert provider.origin_for("https://sandbox.zenodo.org/records/101", "") == "https://sandbox.zenodo.org"


def test_a_publication_whose_link_and_doi_disagree_is_on_no_origin():
    # A sandbox DOI with a production link is not the production record, even with the same id (#122, fix 2).
    assert provider.origin_for("https://zenodo.org/records/101", "10.5072/zenodo.101") is None
    assert provider.origin_for("", "10.1000/zenodo.101") is None


def test_the_cap_an_import_meets_is_the_admins_but_never_above_the_host_limit():
    class Settings:
        max_download_mb = None

    class Session:
        def __init__(self, admin_mb):
            self.admin_mb = admin_mb

        def get(self, model, key):
            if self.admin_mb is None:
                return None
            settings = Settings()
            settings.max_download_mb = self.admin_mb
            return settings

    assert limits.import_cap_bytes(Session(None), 4_000_000) == 4_000_000
    assert limits.import_cap_bytes(Session(2), 4_000_000) == 2_000_000
    assert limits.import_cap_bytes(Session(8), 4_000_000) == 4_000_000


def test_a_host_with_only_some_of_the_community_seams_does_not_offer_the_source(make_app, zenodo, caplog):
    app = make_app(community_datasets=zenodo.datasets, community_record=zenodo.record, community_download=zenodo.download)
    client = app.test_client()

    settings = client.get("/api/library/admin/settings", headers=ADMIN).get_json()

    assert settings["community"] == {"available": False, "enabled": False}
    assert "needs all four seams" in caplog.text
