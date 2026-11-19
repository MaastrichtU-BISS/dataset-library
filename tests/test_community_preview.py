"""A Community preview is read by the worker, never by a request (#122, fix 1 of the second rework): a request re-checks the
record, answers from the preview's cache, and queues one job per cache key when there is none. The worker reads the file
under the preview budget, checks its bytes against the checksum the request saw, and caches the head or the refusal.

Zenodo is faked (fake_zenodo.py). The queued jobs run as the worker's loop runs them: `PreviewQueue.run`."""
import pytest

from conftest import ADMIN, ALICE
from dataset_library import community, limits, tables
from dataset_library.providers import community as provider
from fake_zenodo import FakeZenodo, PreviewQueue

SANDBOX = "https://sandbox.zenodo.org"


@pytest.fixture
def zenodo():
    return FakeZenodo()


@pytest.fixture
def previews():
    return PreviewQueue()


def make(make_app, zenodo, previews, cache, **options):
    app = make_app(
        community_datasets=zenodo.datasets,
        community_record=zenodo.record,
        community_download=zenodo.download,
        community_base_url=zenodo.base_url,
        community_enqueue_preview=previews,
        enqueue=lambda import_ref: None,
        max_rows=lambda: 300,
        **options,
    )
    app.previews, app.zenodo, app.cache = previews, zenodo, cache
    return app.test_client()


def switch(client, enabled=True, limits=None):
    body = {"sources": [], "license_policy": {}, "community": {"enabled": enabled}}
    if limits is not None:
        body["limits"] = limits
    response = client.put("/api/library/admin/settings", json=body, headers=ADMIN)
    assert response.status_code == 200, response.get_json()


def get(client, record="101"):
    return client.get(f"/api/library/datasets/community/{record}", headers=ALICE)


def preview(client, record="101", status=200):
    """The dataset as the page shows it: asked again once the worker has read its preview, as the page's polling does."""
    response = get(client, record)
    assert response.status_code == status, response.get_json()
    if status == 200 and response.get_json()["preparing"]:
        settle(client)
        response = get(client, record)
        assert response.status_code == status, response.get_json()
    return response.get_json()


def settle(client):
    """Run the queued preview jobs, as the worker's loop does."""
    app = client.application
    app.previews.run(app.cache, app.zenodo)


def downloads_of(zenodo):
    return zenodo.downloads()


def cards_for(client):
    response = client.get("/api/library/search", query_string={"source": "community"}, headers=ALICE)
    return sorted(card["externalId"] for card in response.get_json()["results"])


def expire(cache):
    """Forget the preview cache, as its lifetime does once it runs out."""
    for key in [key for key in cache.values if key.startswith("library:community:")]:
        del cache.values[key]


# --- The request reads nothing: it re-checks the record and queues the worker's read (fix 1) --------

def test_a_request_downloads_and_parses_nothing_it_queues_the_read_for_the_worker(make_app, zenodo, previews, cache, monkeypatch):
    client = make(make_app, zenodo, previews, cache)
    switch(client)

    def parsed(*args, **kwargs):
        raise AssertionError("the request parsed a Community file")

    monkeypatch.setattr(tables, "read", parsed)
    monkeypatch.setattr(tables, "head", parsed)

    body = get(client).get_json()

    assert body["preparing"] is True
    assert body["preview"] == "preparing"
    assert zenodo.downloads() == []
    assert len(previews.jobs) == 1


def test_a_request_while_the_first_is_queued_queues_nothing_more(make_app, zenodo, previews, cache):
    client = make(make_app, zenodo, previews, cache)
    switch(client)

    first = get(client).get_json()
    second = get(client).get_json()

    assert first["preparing"] and second["preparing"]
    assert len(previews.jobs) == 1  # one job per cache key, however many times the page asks


def test_the_worker_fills_the_cache_and_the_next_request_shows_the_head(make_app, zenodo, previews, cache):
    client = make(make_app, zenodo, previews, cache)
    switch(client)
    get(client)

    settle(client)
    body = get(client).get_json()

    assert body["preparing"] is False
    assert body["preview"] == "ok"
    assert body["importable"] is True
    assert len(body["rows"]) == 8
    assert [call[0:2] for call in downloads_of(zenodo)] == [("download", "101")]  # read once, by the worker
    assert previews.jobs == []


def test_a_request_for_a_cached_preview_queues_nothing(make_app, zenodo, previews, cache):
    client = make(make_app, zenodo, previews, cache)
    switch(client)
    preview(client)

    body = get(client).get_json()

    assert body["preview"] == "ok"
    assert previews.jobs == []


def test_a_queue_that_fails_answers_unavailable_and_releases_its_lock(make_app, zenodo, cache):
    def refused(job):
        raise ConnectionError("Redis is down")

    client = make(make_app, zenodo, refused, cache)
    switch(client)

    response = get(client)

    assert response.status_code == 503
    assert response.get_json()["code"] == "provider_unavailable"
    assert not [key for key in cache.values if key.endswith(":lock")]


# --- The worker: the head, a refusal, too large, Zenodo's failure (fix 1) -------------------------

def test_a_file_over_the_preview_budget_is_not_downloaded_and_stays_importable(make_app, zenodo, previews, cache):
    client = make(make_app, zenodo, previews, cache, community_preview_max_bytes=500)
    switch(client)

    body = get(client).get_json()

    assert body["preview"] == "too_large"
    assert body["importable"] is True
    assert body["reasonCode"] is None
    assert body["file"] == {"key": "data.csv", "size": 973}  # the card, its files and the size all stand
    assert body["rows"] == [] and body["columns"] == []
    assert previews.jobs == []  # not even queued: nothing is read
    assert zenodo.downloads() == []


def test_a_file_over_the_budget_can_still_be_queued_and_the_worker_checks_its_columns(make_app, zenodo, previews, cache):
    client = make(make_app, zenodo, previews, cache, community_preview_max_bytes=500)
    switch(client)

    response = client.post(
        "/api/library/imports",
        json={"provider": "community", "dataset": "101", "taskType": "classification", "textColumn": "text",
              "labelColumn": "label", "size": 5},
        headers=ALICE,
    )

    assert response.status_code == 202, response.get_json()
    assert response.get_json()["status"] == "queued"


def test_a_file_within_the_budget_is_read_by_the_worker_at_no_more_than_the_budget(make_app, zenodo, previews, cache):
    client = make(make_app, zenodo, previews, cache, community_preview_max_bytes=2_000)
    switch(client)

    get(client)
    assert zenodo.downloads() == []  # the request reads nothing
    settle(client)

    assert [call[3] for call in downloads_of(zenodo)] == [2_000]
    assert preview(client)["preview"] == "ok"


def test_a_file_of_unknown_size_is_refused_at_the_budget_and_shown_without_rows(make_app, zenodo, previews, cache):
    zenodo.records["101"]["files"][0]["size"] = None
    client = make(make_app, zenodo, previews, cache, community_preview_max_bytes=500)
    switch(client)

    body = preview(client)

    assert body["preview"] == "too_large"
    assert body["importable"] is True
    assert [call[3] for call in downloads_of(zenodo)] == [500]


def test_a_file_within_the_budget_but_over_the_import_cap_is_not_importable(make_app, zenodo, previews, cache):
    client = make(make_app, zenodo, previews, cache, max_fetch_bytes=500)
    switch(client)

    body = get(client).get_json()

    assert body["importable"] is False
    assert body["reasonCode"] == "too_large"
    assert previews.jobs == [] and zenodo.downloads() == []


def test_a_refusal_is_cached_so_a_bad_file_is_not_fetched_again(make_app, zenodo, previews, cache):
    zenodo.replace_file("101", "data.csv", b"\x81\x8d\x8f\x90")  # not UTF-8, and undefined in Windows-1252 too
    client = make(make_app, zenodo, previews, cache)
    switch(client)

    body = preview(client)
    again = get(client).get_json()

    assert body["reasonCode"] == "unreadable"
    assert again["reasonCode"] == "unreadable"
    assert len(downloads_of(zenodo)) == 1  # the worker read it once; the next open read the refusal from the cache
    assert previews.jobs == []
    head_keys = [key for key in cache.values if ":head:" in key]
    assert [cache.ttls[key] for key in head_keys] == [600]


def test_the_worker_refuses_bytes_that_do_not_match_the_checksum_the_request_saw(make_app, zenodo, previews, cache):
    zenodo.contents[("101", "data.csv")] = b"text,label\nA,x\n"  # the record still lists the old checksum
    client = make(make_app, zenodo, previews, cache)
    switch(client)

    body = preview(client)

    assert body["reasonCode"] == "changed"
    assert body["reason"] == provider.CHANGED


def test_zenodo_failing_during_the_read_is_kept_for_a_minute_and_answers_unavailable(make_app, zenodo, previews, cache):
    client = make(make_app, zenodo, previews, cache)
    switch(client)
    get(client)
    zenodo.busy = True

    settle(client)
    response = get(client)

    assert response.status_code == 503
    assert response.get_json()["code"] == "provider_unavailable"
    busy = [key for key in cache.values if ":head:" in key]
    assert [cache.ttls[key] for key in busy] == [community.BUSY_SECONDS]


def test_the_worker_releases_its_lock_so_a_later_open_after_the_busy_minute_queues_again(make_app, zenodo, previews, cache):
    client = make(make_app, zenodo, previews, cache)
    switch(client)
    get(client)
    zenodo.busy = True
    settle(client)
    zenodo.busy = False
    expire(cache)

    body = get(client).get_json()

    assert body["preparing"] is True
    assert len(previews.jobs) == 1


# --- Timing, the lock and the record cache (fix 1) --------------------------------------------------

def test_a_record_is_kept_for_a_minute_and_a_preview_for_ten_minutes(make_app, zenodo, previews, cache):
    client = make(make_app, zenodo, previews, cache)
    switch(client)

    preview(client)
    get(client)

    assert len([call for call in zenodo.calls if call[0] == "record"]) == 1  # the second open checked nothing live
    assert [cache.ttls[key] for key in cache.values if ":record:" in key] == [60]
    assert [cache.ttls[key] for key in cache.values if ":head:" in key] == [600]


def test_the_lock_outlives_fair_publishings_read_deadline_and_the_budget_is_16_mb():
    assert community.LOCK_SECONDS == 660
    assert community.LOCK_SECONDS > 600
    assert community.PREVIEW_MAX_BYTES == 16_000_000
    assert community.Seams.__dataclass_fields__["preview_max_bytes"].default == community.PREVIEW_MAX_BYTES


def test_a_lock_is_released_only_by_the_token_that_took_it(cache):
    assert cache.add("library:community:x:lock", "token-a", 660)

    cache.delete_if("library:community:x:lock", "token-b")
    assert not cache.add("library:community:x:lock", "token-c", 660)  # still held by token-a

    cache.delete_if("library:community:x:lock", "token-a")
    assert cache.add("library:community:x:lock", "token-c", 660)


# --- Record identity (fix 2) -------------------------------------------------------------------

def test_a_publication_on_another_zenodo_host_is_not_previewed_as_the_configured_record(make_app, zenodo, previews, cache):
    zenodo.listed[1]["origin"] = SANDBOX  # record 101 is a sandbox record; this platform reads production
    client = make(make_app, zenodo, previews, cache)
    switch(client)

    body = preview(client, status=404)

    assert body["code"] == "not_found"
    assert not [call for call in zenodo.calls if call[0] == "record"]
    assert zenodo.downloads() == []


def test_only_publications_on_the_configured_host_are_listed(make_app, zenodo, previews, cache):
    zenodo.listed[1]["origin"] = SANDBOX
    client = make(make_app, zenodo, previews, cache)
    switch(client)

    assert cards_for(client) == ["102", "103"]


def test_a_publication_with_no_origin_is_not_listed(make_app, zenodo, previews, cache):
    zenodo.listed[1].pop("origin")
    client = make(make_app, zenodo, previews, cache)
    switch(client)

    assert cards_for(client) == ["102", "103"]


def test_the_preview_cache_is_bound_to_the_zenodo_origin(make_app, zenodo, previews, cache):
    client = make(make_app, zenodo, previews, cache)
    switch(client)
    preview(client)
    assert any(key.startswith("library:community:https://zenodo.org:") for key in cache.values)

    # The platform moves to the sandbox: nothing read from production answers for it.
    zenodo.configured = SANDBOX
    for row in zenodo.listed:
        row["origin"] = SANDBOX
    body = preview(client)

    assert body["preview"] == "ok"
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
    assert "needs all five seams" in caplog.text
