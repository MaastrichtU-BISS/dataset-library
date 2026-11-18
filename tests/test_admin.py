import pytest

from tests.conftest import ADMIN, ALICE


def settings(client, headers=ADMIN):
    resp = client.get("/api/library/admin/settings", headers=headers)
    return resp


def save(client, body, headers=ADMIN):
    return client.put("/api/library/admin/settings", json=body, headers=headers)


def test_only_admins_reach_the_settings(client):
    assert settings(client, ALICE).status_code == 403
    assert save(client, {"sources": [], "license_policy": {}}, ALICE).status_code == 403
    assert client.post("/api/library/admin/sources/check", json={"ref": "coastalcph"}, headers=ALICE).status_code == 403


def test_options_say_whether_the_user_is_an_admin(client):
    assert client.get("/api/library/options", headers=ALICE).get_json()["admin"] is False
    assert client.get("/api/library/options", headers=ADMIN).get_json()["admin"] is True


def test_fresh_settings_have_no_sources_and_every_license_at_its_default(client):
    body = settings(client).get_json()

    assert body["sources"] == []
    licenses = {lic["id"]: lic for lic in body["licenses"]}
    assert licenses["cc-by-4.0"]["level"] == "allow" and licenses["cc-by-4.0"]["default"] == "allow"
    assert licenses["cc-by-nd-4.0"]["level"] == "warn"
    assert body["groups"][0] == "Open"


def test_settings_are_saved_whole_and_apply_to_search(client, hub, audits):
    resp = save(
        client,
        {
            "sources": [{"ref": "coastalcph", "enabled": True}, {"ref": "nguha", "enabled": False}],
            "license_policy": {"cc-by-4.0": "warn", "mit": "hide"},
        },
    )

    assert resp.status_code == 200, resp.get_json()
    body = settings(client).get_json()
    assert [(s["ref"], s["enabled"]) for s in body["sources"]] == [("coastalcph", True), ("nguha", False)]
    assert {lic["id"]: lic["level"] for lic in body["licenses"]}["mit"] == "hide"
    assert client.get("/api/library/options", headers=ALICE).get_json()["sources"] == [
        {"value": "coastalcph", "label": "coastalcph", "provider": "huggingface"}
    ]
    assert audits == [
        {"event": "library.settings_saved", "sources_added": 2, "sources_removed": 0, "licenses_changed": ["cc-by-4.0", "mit"], "limits_changed": []}
    ]

    # A second save replaces the first: a source left out is removed.
    assert save(client, {"sources": [{"ref": "nguha", "enabled": True}], "license_policy": {}}).status_code == 200
    body = settings(client).get_json()
    assert [s["ref"] for s in body["sources"]] == ["nguha"]
    assert {lic["id"]: lic["level"] for lic in body["licenses"]}["mit"] == "allow"
    assert audits[-1] == {"event": "library.settings_saved", "sources_added": 0, "sources_removed": 1, "licenses_changed": ["cc-by-4.0", "mit"], "limits_changed": []}


@pytest.mark.parametrize(
    "body, field",
    [
        ({"sources": [], "license_policy": {"beerware": "allow"}}, "license_policy.beerware"),
        ({"sources": [], "license_policy": {"mit": "maybe"}}, "license_policy.mit"),
        ({"sources": [{"ref": "../etc", "enabled": True}], "license_policy": {}}, "sources.0"),
        ({"sources": [{"ref": "nobody-at-all", "enabled": True}], "license_policy": {}}, "sources.0"),
        ({"sources": [{"ref": "coastalcph", "enabled": True}, {"ref": "coastalcph", "enabled": True}], "license_policy": {}}, "sources.1"),
    ],
)
def test_an_invalid_field_is_named_and_nothing_is_saved(client, audits, body, field):
    resp = save(client, body)

    assert resp.status_code == 400
    assert field in resp.get_json()["fields"]
    assert settings(client).get_json()["sources"] == []
    assert audits == []


def test_a_new_source_can_be_checked_before_saving(client):
    ok = client.post("/api/library/admin/sources/check", json={"ref": "coastalcph"}, headers=ADMIN)
    missing = client.post("/api/library/admin/sources/check", json={"ref": "nobody-at-all"}, headers=ADMIN)

    assert ok.status_code == 200 and ok.get_json() == {"ref": "coastalcph", "ok": True}
    assert missing.status_code == 200
    assert missing.get_json() == {
        "ref": "nobody-at-all",
        "ok": False,
        "error": "No public datasets from “nobody-at-all” on Hugging Face. Check the organisation’s name.",
    }


def test_checking_a_source_while_hugging_face_is_down_says_so(client, hub):
    hub.fail = 503

    resp = client.post("/api/library/admin/sources/check", json={"ref": "coastalcph"}, headers=ADMIN)

    assert resp.status_code == 503


def test_fresh_limits_are_the_hosts_defaults_and_the_hard_maximum(client):
    limits = settings(client).get_json()["limits"]

    assert limits == {
        "maxRows": None, "maxRowsDefault": 1000, "maxRowsHardMax": 5000,
        "maxDownloadMb": None, "maxDownloadMbDefault": 200, "maxDownloadMbHardMax": 1000,
    }


def test_an_admins_row_cap_applies_to_what_students_are_offered(client, audits):
    resp = save(client, {"sources": [], "license_policy": {}, "limits": {"maxRows": 2500, "maxDownloadMb": 50}})

    assert resp.status_code == 200, resp.get_json()
    assert resp.get_json()["limits"]["maxRows"] == 2500
    assert resp.get_json()["limits"]["maxDownloadMb"] == 50
    assert client.get("/api/library/options", headers=ALICE).get_json()["maxRows"] == 2500
    assert audits[-1]["limits_changed"] == ["max_download_mb", "max_rows"]

    # null goes back to the default; leaving limits out keeps them.
    assert save(client, {"sources": [], "license_policy": {}}).get_json()["limits"]["maxRows"] == 2500
    assert save(client, {"sources": [], "license_policy": {}, "limits": {"maxRows": None}}).get_json()["limits"]["maxRows"] is None
    assert client.get("/api/library/options", headers=ALICE).get_json()["maxRows"] == 1000


@pytest.mark.parametrize(
    "limits, field",
    [
        ({"maxRows": 0}, "limits.maxRows"),
        ({"maxRows": 5001}, "limits.maxRows"),
        ({"maxRows": "lots"}, "limits.maxRows"),
        ({"maxRows": True}, "limits.maxRows"),
        ({"maxDownloadMb": 0}, "limits.maxDownloadMb"),
        ({"maxDownloadMb": 1001}, "limits.maxDownloadMb"),
    ],
)
def test_a_cap_out_of_range_is_named_and_nothing_is_saved(client, audits, limits, field):
    resp = save(client, {"sources": [], "license_policy": {}, "limits": limits})

    assert resp.status_code == 400
    assert field in resp.get_json()["fields"]
    assert settings(client).get_json()["limits"]["maxRows"] is None
    assert audits == []
