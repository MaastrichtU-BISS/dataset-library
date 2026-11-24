"""Pins: what an admin curates, and what any signed-in user sees of it."""
import pytest

from tests.conftest import ADMIN, ALICE
from test_citation import LEX_README_BIBTEX, SHA


def pin(client, dataset, status=201, headers=ADMIN):
    resp = client.post("/api/library/pins", json={"dataset": dataset}, headers=headers)
    assert resp.status_code == status, resp.get_json()
    return resp.get_json()


def patch(client, ref, body, status=200, headers=ADMIN):
    resp = client.patch(f"/api/library/pins/{ref}", json=body, headers=headers)
    assert resp.status_code == status, resp.get_json()
    return resp.get_json() if status != 204 else None


def delete(client, ref, status=204, headers=ADMIN):
    resp = client.delete(f"/api/library/pins/{ref}", headers=headers)
    assert resp.status_code == status, resp.get_json()
    return resp


def card_title(client, dataset):
    return client.get(f"/api/library/datasets/huggingface/{dataset}", headers=ALICE).get_json()["title"]


def test_an_admin_pins_a_dataset_and_its_card_is_stored(client, audits):
    title = card_title(client, "coastalcph/lex_glue")
    body = pin(client, "coastalcph/lex_glue")

    assert body["provider"] == "huggingface"
    assert body["externalId"] == "coastalcph/lex_glue"
    assert body["title"] == title and body["titleOverride"] is None
    assert body["featured"] is False
    assert body["ref"]
    assert body["pinnedAt"].endswith("Z")
    assert body["card"]["id"] == "huggingface:coastalcph/lex_glue"
    assert body["card"]["title"] == title
    assert body["card"]["license"]["id"] == "cc-by-4.0" and body["card"]["license"]["level"] == "allow"
    assert body["card"]["tasks"] and body["card"]["description"]
    assert audits == [
        {"event": "library.pin_added", "provider": "huggingface", "external_id": "coastalcph/lex_glue"}
    ]


def test_the_same_dataset_pins_once(client):
    first = pin(client, "coastalcph/lex_glue")

    again = pin(client, "coastalcph/lex_glue", status=200)

    assert again["ref"] == first["ref"]
    assert again["featured"] is False


def test_a_pinned_dataset_keeps_the_labels_verdict_it_was_pinned_with(client, audits):
    yes = pin(client, "AI-team-UoA/greek_legal_code")
    unclear = pin(client, "nguha/legalbench")

    assert yes["labelled"] == "yes"
    assert unclear["labelled"] == "unclear"
    assert [a["event"] for a in audits] == ["library.pin_added", "library.pin_added"]


@pytest.mark.parametrize("body", [{"dataset": "no-dataset"}, {"dataset": "../etc/passwd"}, {}])
def test_a_bad_id_is_refused_before_the_hub_is_asked(client, hub, body):
    pin(client, body.get("dataset", ""), status=400)

    assert hub.calls == []


def test_a_dataset_outside_the_enabled_sources_is_not_found(app, client):
    from dataset_library.models import LibrarySource

    pin(client, "nguha/legalbench", status=201)
    with app.app_context():
        app.db.session.add(LibrarySource(provider="huggingface", kind="org", ref="coastalcph"))
        app.db.session.commit()

    pin(client, "nguha/legalbench", status=404)


def test_a_hidden_license_is_refused(app, client):
    from dataset_library.models import LibrarySettings

    with app.app_context():
        app.db.session.add(LibrarySettings(id=1, license_policy={"cc-by-4.0": "hide"}))
        app.db.session.commit()

    refused = client.post("/api/library/pins", json={"dataset": "coastalcph/lex_glue"}, headers=ADMIN)

    assert refused.status_code == 403
    assert refused.get_json()["code"] == "license_hidden"


def test_a_private_dataset_is_just_not_found(client):
    pin(client, "private-law/restricted_corpus", status=404)


def test_while_hugging_face_is_down_pinning_says_so(client, hub):
    hub.fail = 503

    resp = client.post("/api/library/pins", json={"dataset": "coastalcph/lex_glue"}, headers=ADMIN)

    assert resp.status_code == 503
    assert resp.get_json()["code"] == "provider_unavailable"


def test_three_can_be_featured_and_a_fourth_waits(client, audits):
    refs = [pin(client, d)["ref"] for d in ["coastalcph/lex_glue", "nguha/legalbench", "jfrenz/legalglue", "lighteval/lexglue"]]

    for ref in refs[:3]:
        assert patch(client, ref, {"featured": True})["featured"] is True
    full = client.patch(f"/api/library/pins/{refs[3]}", json={"featured": True}, headers=ADMIN)
    assert full.status_code == 409
    assert full.get_json() == {
        "error": "At most 3 datasets can be featured. Unfeature one first.",
        "code": "too_many_featured",
    }

    # Making room works: the fourth joins, in pin order, not feature order.
    patch(client, refs[0], {"featured": False})
    assert patch(client, refs[3], {"featured": True})["featured"] is True
    assert [p["externalId"] for p in client.get("/api/library/featured", headers=ALICE).get_json()["featured"]] == [
        "nguha/legalbench", "jfrenz/legalglue", "lighteval/lexglue",
    ]


def test_a_display_title_is_set_cleared_and_length_checked(client, audits):
    ref = pin(client, "coastalcph/lex_glue")["ref"]

    titled = patch(client, ref, {"title": "  Legal GLUE  "})
    assert titled["title"] == "Legal GLUE" and titled["titleOverride"] == "Legal GLUE"
    assert audits[-1] == {"event": "library.pin_updated", "external_id": "coastalcph/lex_glue", "title_changed": True}

    for cleared in ("", "   ", None):
        cleared_pin = patch(client, ref, {"title": cleared})
        assert cleared_pin["titleOverride"] is None
        assert cleared_pin["title"] == cleared_pin["card"]["title"]

    too_long = client.patch(f"/api/library/pins/{ref}", json={"title": "x" * 121}, headers=ADMIN)
    assert too_long.status_code == 400
    assert too_long.get_json() == {"error": "Keep the title under 121 characters.", "code": "bad_title"}
    assert patch(client, ref, {"title": "x" * 120})["title"] == "x" * 120


def test_a_featured_that_is_not_a_boolean_is_refused(client):
    ref = pin(client, "coastalcph/lex_glue")["ref"]

    resp = client.patch(f"/api/library/pins/{ref}", json={"featured": "yes"}, headers=ADMIN)

    assert resp.status_code == 400
    assert resp.get_json()["code"] == "bad_featured"


def test_an_unknown_pin_ref_is_not_found(client):
    pin(client, "coastalcph/lex_glue")

    resp = client.patch("/api/library/pins/no-such-ref", json={"featured": True}, headers=ADMIN)
    assert resp.status_code == 404 and resp.get_json()["code"] == "not_found"
    assert delete(client, "no-such-ref", status=404)


def test_unpinning_removes_the_pin_for_good(client, audits):
    ref = pin(client, "coastalcph/lex_glue")["ref"]

    delete(client, ref)

    assert client.get("/api/library/pins", headers=ADMIN).get_json()["pins"] == []
    assert delete(client, ref, status=404)
    assert audits[-1] == {"event": "library.pin_removed", "external_id": "coastalcph/lex_glue"}


def test_only_admins_manage_pins(client):
    ref = pin(client, "coastalcph/lex_glue")["ref"]
    not_admin = client.get("/api/library/pins", headers=ALICE).get_json()["code"]

    assert not_admin == "not_admin"
    assert client.post("/api/library/pins", json={"dataset": "jfrenz/legalglue"}, headers=ALICE).get_json()["code"] == "not_admin"
    assert client.patch(f"/api/library/pins/{ref}", json={"featured": True}, headers=ALICE).get_json()["code"] == "not_admin"
    assert client.delete(f"/api/library/pins/{ref}", headers=ALICE).get_json()["code"] == "not_admin"
    # Nothing changed: still one pin, still not featured.
    pins = client.get("/api/library/pins", headers=ADMIN).get_json()["pins"]
    assert [p["featured"] for p in pins] == [False]


def test_a_featured_pin_keeps_its_badge_exact_and_its_level_current(client, audits):
    ref = pin(client, "AI-team-UoA/greek_legal_code")["ref"]
    patch(client, ref, {"featured": True, "title": "Greek law"})

    featured = client.get("/api/library/featured", headers=ALICE).get_json()["featured"]
    assert len(featured) == 1
    assert featured[0]["labelled"] == "yes"
    assert featured[0]["title"] == "Greek law"
    assert featured[0]["card"]["license"]["level"] == "allow"

    with client.application.app_context():
        from dataset_library.models import LibrarySettings

        client.application.db.session.add(LibrarySettings(id=1, license_policy={"cc-by-4.0": "warn"}))
        client.application.db.session.commit()

    featured = client.get("/api/library/featured", headers=ALICE).get_json()["featured"]
    assert featured[0]["card"]["license"]["level"] == "warn"


def test_the_featured_row_leaves_out_what_the_library_now_refuses(app, client):
    from dataset_library.models import LibrarySettings, LibrarySource

    pin(client, "coastalcph/lex_glue")
    pin(client, "nguha/legalbench")
    for ref in client.get("/api/library/pins", headers=ADMIN).get_json()["pins"]:
        patch(client, ref["ref"], {"featured": True})

    def set_rules(**rows):
        with app.app_context():
            session = app.db.session
            session.query(LibrarySource).delete()
            for source in rows.get("sources", []):
                session.add(LibrarySource(provider="huggingface", kind="org", ref=source))
            settings = session.get(LibrarySettings, 1) or LibrarySettings(id=1, license_policy={})
            settings.license_policy = rows.get("license_policy", {})
            session.add(settings)
            session.commit()

    shown = lambda: [
        p["externalId"] for p in client.get("/api/library/featured", headers=ALICE).get_json()["featured"]
    ]

    assert shown() == ["coastalcph/lex_glue", "nguha/legalbench"]
    set_rules(sources=["coastalcph"])
    assert shown() == ["coastalcph/lex_glue"]  # nguha is outside the sources now
    set_rules(sources=["coastalcph"], license_policy={"cc-by-4.0": "hide"})
    assert shown() == []  # and the last one's license is hidden


def test_featured_is_oldest_pin_first(client):
    refs = [pin(client, d)["ref"] for d in ["coastalcph/lex_glue", "nguha/legalbench", "jfrenz/legalglue"]]
    for ref in reversed(refs):  # feature them in the opposite order
        patch(client, ref, {"featured": True})

    assert [p["ref"] for p in client.get("/api/library/featured", headers=ALICE).get_json()["featured"]] == refs


def test_search_results_carry_the_ref_of_their_pin(client):
    unpinneable = [r["externalId"] for r in client.get("/api/library/search?q=lex_glue", headers=ALICE).get_json()["results"]]
    assert all(r["pinRef"] is None for r in client.get("/api/library/search?q=lex_glue", headers=ALICE).get_json()["results"])

    ref = pin(client, "coastalcph/lex_glue")["ref"]

    results = client.get("/api/library/search?q=lex_glue", headers=ALICE).get_json()["results"]
    assert "coastalcph/lex_glue" in unpinneable
    assert [r["pinRef"] for r in results if r["externalId"] == "coastalcph/lex_glue"] == [ref]
    assert all(r["pinRef"] is None for r in results if r["externalId"] != "coastalcph/lex_glue")


def test_the_featured_row_is_empty_for_a_fresh_course(client):
    assert client.get("/api/library/featured", headers=ALICE).get_json() == {"featured": [], "legalDataHunter": []}


def test_a_pin_inserted_between_the_lookup_and_the_insert_is_returned(client, monkeypatch):
    from dataset_library import pins
    from dataset_library.models import LibraryPin

    def insert_first(cache, ids):
        # Another request wins the race after our lookup but before our insert.
        session = client.application.db.session
        session.add(LibraryPin(provider="huggingface", external_id="coastalcph/lex_glue", card={}, labelled="unclear"))
        session.commit()
        return {external_id: "unclear" for external_id in ids}

    monkeypatch.setattr(pins.labels_module, "for_ids", insert_first)

    body = pin(client, "coastalcph/lex_glue", status=200)

    assert body["externalId"] == "coastalcph/lex_glue"
    with client.application.app_context():
        assert client.application.db.session.query(LibraryPin).count() == 1


def test_a_pin_uses_the_hubs_canonical_id(client, monkeypatch):
    from dataset_library import pins
    from dataset_library.models import LibraryPin

    real_preview = pins.preview_module.preview

    def renamed(host, external_id, *args, **kwargs):
        body = real_preview(host, external_id, *args, **kwargs)
        return {**body, "externalId": "canonical/name", "id": "huggingface:canonical/name"}

    monkeypatch.setattr(pins.preview_module, "preview", renamed)

    body = pin(client, "coastalcph/lex_glue")

    assert body["externalId"] == "canonical/name"
    with client.application.app_context():
        stored = client.application.db.session.query(LibraryPin).one()
        assert stored.external_id == "canonical/name"


def test_the_audit_never_carries_the_title_and_an_empty_patch_is_not_audited(client, audits):
    ref = pin(client, "coastalcph/lex_glue")["ref"]
    audits.clear()

    patch(client, ref, {"title": "Secret title"})
    assert audits == [
        {"event": "library.pin_updated", "external_id": "coastalcph/lex_glue", "title_changed": True}
    ]

    audits.clear()
    patch(client, ref, {})
    patch(client, ref, {"title": "Secret title"})
    patch(client, ref, {"featured": False})
    assert audits == []


@pytest.mark.parametrize("body, code", [([], "bad_request"), ({"dataset": 123}, "bad_id")])
def test_a_malformed_pin_body_is_refused(client, body, code):
    resp = client.post("/api/library/pins", json=body, headers=ADMIN)

    assert resp.status_code == 400
    assert resp.get_json()["code"] == code


def test_a_patch_body_that_is_not_an_object_is_refused(client):
    ref = pin(client, "coastalcph/lex_glue")["ref"]

    resp = client.patch(f"/api/library/pins/{ref}", json=[], headers=ADMIN)

    assert resp.status_code == 400
    assert resp.get_json()["code"] == "bad_request"


def test_a_patch_with_null_or_broken_json_is_refused(client):
    ref = pin(client, "coastalcph/lex_glue")["ref"]

    for data in ("null", "{bad"):
        resp = client.patch(f"/api/library/pins/{ref}", data=data, content_type="application/json", headers=ADMIN)
        assert resp.status_code == 400
        assert resp.get_json()["code"] == "bad_request"


def test_a_pin_made_before_a_rename_is_found_by_its_old_id(client, monkeypatch):
    from dataset_library import pins
    from dataset_library.models import LibraryPin

    first = pin(client, "coastalcph/lex_glue")
    real_preview = pins.preview_module.preview

    def renamed(host, external_id, *args, **kwargs):
        body = real_preview(host, external_id, *args, **kwargs)
        return {**body, "externalId": "coastalcph/lex_glue_v2", "id": "huggingface:coastalcph/lex_glue_v2"}

    monkeypatch.setattr(pins.preview_module, "preview", renamed)

    again = client.post("/api/library/pins", json={"dataset": "coastalcph/lex_glue"}, headers=ADMIN)

    assert again.status_code == 200
    assert again.get_json()["ref"] == first["ref"]
    with client.application.app_context():
        assert client.application.db.session.query(LibraryPin).count() == 1


def test_a_featured_card_keeps_the_citation_it_was_pinned_with(client):
    # LexGLUE's BibTeX is in its README, read when the dataset was pinned (#120).
    body = pin(client, "coastalcph/lex_glue")
    featured = patch(client, body["ref"], {"featured": True})

    citation = featured["card"]["citation"]
    assert citation["title"] == "LexGLUE"
    assert citation["revision"] == SHA
    assert citation["bibtex"] == LEX_README_BIBTEX
    assert [row["card"]["citation"]["bibtex"] for row in client.get("/api/library/featured", headers=ALICE).get_json()["featured"]] == [LEX_README_BIBTEX]
