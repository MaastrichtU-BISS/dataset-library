"""Legal Data Hunter entries (#126): what an admin curates with their own key, the featured cards a student sees,
the confirmation every import needs, and the worker fetching each document with the student's key, pausing when
the credits run out and resuming without fetching anything twice. Legal Data Hunter is faked from recorded
responses (fake_legal_data_hunter.py); no test reaches the network."""
import csv
import json
import logging
from datetime import timedelta

import pytest
import requests
import sqlalchemy as sa

from conftest import ADMIN, ALICE
from dataset_library import imports
from dataset_library import legal_data_hunter as ldh
from dataset_library import policy
from dataset_library.licenses import LEGAL_DATA_HUNTER_TERMS, LICENSES, UNKNOWN, resolve_license
from dataset_library.models import LibraryImport, LibraryLegalDataHunterDocument, LibraryLegalDataHunterEntry
from dataset_library.providers import legal_data_hunter as provider
from fake_legal_data_hunter import ADMIN_KEY, DOCUMENTS, STUDENT_KEY, FakeLegalDataHunter

BOB = {"X-User": "bob"}  # a student with no key
QUERY = {"q": "unfair contract terms consumer", "corpus": "case_law", "country": ["nl", "FR", "DE"], "dateFrom": "2020-01-01", "dateTo": "2024-12-31"}
NEEDS_KEY = "Needs your Legal Data Hunter key — add it in Settings"
CONFIRMATION = (
    "Legal Data Hunter's terms forbid storing full text or training models unless your plan allows it. "
    "The free plan does not."
)
CHECKBOX = "My Legal Data Hunter plan allows storing these documents and training models on them."


class Queue:
    def __init__(self):
        self.refs = []
        self.fail = False

    def __call__(self, ref):
        if self.fail:
            raise RuntimeError("the queue is down")
        self.refs.append(ref)


class Sink:
    def __init__(self):
        self.calls = []

    def __call__(self, owner_id, file_path, name, description, task_type):
        with open(file_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            header = reader.fieldnames
        self.calls.append({"owner": owner_id, "name": name, "description": description, "task_type": task_type,
                           "rows": rows, "header": header})
        return f"dataset-{len(self.calls)}"


def forbidden(*args, **kwargs):
    raise AssertionError("a Legal Data Hunter import fetches its documents, never Hugging Face shards")


@pytest.fixture
def hunter(monkeypatch):
    fake = FakeLegalDataHunter()
    monkeypatch.setattr(provider.requests, "request", fake.request)
    return fake


@pytest.fixture
def keys():
    return {"ada": ADMIN_KEY, "alice": STUDENT_KEY}


@pytest.fixture
def queue():
    return Queue()


@pytest.fixture
def app(make_app, hunter, keys, queue):
    app = make_app(legal_data_hunter_key=lambda owner: keys.get(owner), enqueue=queue, max_rows=lambda: 300)
    app.hunter = hunter
    return app


def create(client, status=201, **body):
    payload = {"title": "Unfair terms in consumer contracts", "description": "Five decisions to classify.",
               "query": QUERY, **body}
    response = client.post("/api/library/admin/legal-data-hunter/entries", json=payload, headers=ADMIN)
    assert response.status_code == status, response.get_json()
    return response.get_json()


def start(client, entry, status=202, headers=ALICE, **body):
    payload = {"provider": "legal_data_hunter", "dataset": entry["ref"], "licenseAck": True, **body}
    response = client.post("/api/library/imports", json=payload, headers=headers)
    assert response.status_code == status, response.get_json()
    return response.get_json()


def work(app, tmp_path, ref, keys=None, sink=None, seam=..., **options):
    sink = sink or Sink()
    keys = {"alice": STUDENT_KEY} if keys is None else keys
    seam = keys.get if seam is ... else seam
    with app.app_context():
        session = app.db.session
        imports.run_import(
            ref, session=session, fetch_shards=forbidden, create_dataset=sink, data_dir=str(tmp_path / "data"),
            heartbeat_seconds=None, legal_data_hunter_key=seam, **options,
        )
        session.expunge_all()
        return session.get(LibraryImport, ref).to_dict(), sink


def kept(app, ref):
    with app.app_context():
        return app.db.session.query(LibraryLegalDataHunterDocument).filter_by(import_id=ref).count()


# --- The admin's entries ----------------------------------------------------------------------------------------


def test_a_preview_searches_with_the_admins_own_key_and_the_filters(client, hunter):
    response = client.post("/api/library/admin/legal-data-hunter/preview", json={**QUERY, "source": ["rechtspraak"]}, headers=ADMIN)

    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    [call] = hunter.calls
    assert call["origin"] == "https://legaldatahunter.com" and call["path"] == "/v1/search"
    assert call["key"] == ADMIN_KEY
    assert call["json"] == {
        "q": "unfair contract terms consumer", "namespace": "case_law", "top_k": 100, "result_detail": "snippet",
        "country": ["NL", "FR", "DE"], "source": ["rechtspraak"], "date_start": "2020-01-01", "date_end": "2024-12-31",
    }
    # The hit without a source id is left out; a source named with its country is split.
    assert [hit["sourceId"] for hit in body["hits"]] == [
        "ECLI:NL:HR:2023:1234", "6543a1b2c3d4e5f6", "1 BvR 123/20", "ECLI:NL:RBAMS:2022:999", "ECLI:NL:GHARL:2021:42",
    ]
    assert body["hits"][1]["country"] == "FR" and body["hits"][1]["source"] == "Judilibre"
    assert body["total"] == 1287 and body["max"] == 100
    assert ADMIN_KEY not in response.get_data(as_text=True)


@pytest.mark.parametrize("corpus", ["case_law", "legislation", "doctrine"])
def test_a_preview_looks_in_one_corpus(client, hunter, corpus):
    response = client.post("/api/library/admin/legal-data-hunter/preview", json={"q": "data", "corpus": corpus}, headers=ADMIN)

    assert response.status_code == 200
    assert hunter.calls[-1]["json"]["namespace"] == corpus


@pytest.mark.parametrize("body, field", [
    ({"q": ""}, "query.q"),
    ({"q": "x", "corpus": "patents"}, "query.corpus"),
    ({"q": "x", "corpus": ["case_law"]}, "query.corpus"),
    ({"q": "x", "corpus": {"case_law": True}}, "query.corpus"),
    ({"q": "x", "dateFrom": "2024-13-01"}, "query.dateFrom"),
    ({"q": "x", "dateFrom": "2024-02-01", "dateTo": "2024-01-01"}, "query.dateTo"),
    ({"q": "x", "limit": 101}, "query.limit"),
    ({"q": "x", "country": ["N L"]}, "query.country"),
])
def test_a_preview_with_a_bad_query_is_refused_without_a_call(client, hunter, body, field):
    response = client.post("/api/library/admin/legal-data-hunter/preview", json=body, headers=ADMIN)

    assert response.status_code == 400
    assert field in response.get_json()["fields"]
    assert hunter.calls == []


def test_an_admin_without_a_key_cannot_preview(client, keys, hunter):
    del keys["ada"]

    response = client.post("/api/library/admin/legal-data-hunter/preview", json=QUERY, headers=ADMIN)

    assert response.status_code == 409
    assert response.get_json() == {"error": NEEDS_KEY, "code": "needs_key"}
    assert hunter.calls == []


def test_a_refused_key_and_spent_credits_say_so(client, hunter, keys):
    keys["ada"] = "sk-revoked"
    refused = client.post("/api/library/admin/legal-data-hunter/preview", json=QUERY, headers=ADMIN)
    keys["ada"] = ADMIN_KEY
    hunter.credits = 0
    spent = client.post("/api/library/admin/legal-data-hunter/preview", json=QUERY, headers=ADMIN)

    assert refused.status_code == 409 and refused.get_json()["code"] == "bad_key"
    assert "sk-revoked" not in refused.get_data(as_text=True)
    assert spent.status_code == 429
    assert spent.get_json() == {
        "error": "Your Legal Data Hunter credits ran out. Try again when they reset.", "code": "out_of_credits", "retryAfter": 3600,
    }


def test_only_an_admin_manages_entries(client):
    for method, path in [
        ("post", "/api/library/admin/legal-data-hunter/preview"),
        ("get", "/api/library/admin/legal-data-hunter/entries"),
        ("post", "/api/library/admin/legal-data-hunter/entries"),
        ("get", "/api/library/admin/legal-data-hunter/entries/x"),
        ("patch", "/api/library/admin/legal-data-hunter/entries/x"),
        ("post", "/api/library/admin/legal-data-hunter/entries/x/refresh"),
        ("delete", "/api/library/admin/legal-data-hunter/entries/x"),
    ]:
        assert getattr(client, method)(path, json={}, headers=ALICE).status_code == 403, path


def test_an_entry_is_created_listed_changed_and_deleted(client, audits):
    hits = client.post("/api/library/admin/legal-data-hunter/preview", json=QUERY, headers=ADMIN).get_json()["hits"]

    entry = create(client, documents=hits[:3])

    assert entry["taskType"] == "classification"  # the default
    assert entry["documentCount"] == 3
    assert entry["documents"][0] == {"country": "NL", "source": "rechtspraak", "sourceId": "ECLI:NL:HR:2023:1234",
                                     "title": "Hoge Raad 14 april 2023, oneerlijk beding"}
    # The admin's choice is the entry's size, so a Refresh keeps it.
    assert entry["query"] == {**QUERY, "country": ["NL", "FR", "DE"], "source": [], "limit": 3}
    listed = client.get("/api/library/admin/legal-data-hunter/entries", headers=ADMIN).get_json()
    assert [e["ref"] for e in listed["entries"]] == [entry["ref"]] and listed["max"] == 100

    changed = client.patch(f"/api/library/admin/legal-data-hunter/entries/{entry['ref']}",
                           json={"title": "Unfair terms", "taskType": "summarization", "description": ""}, headers=ADMIN)
    assert changed.status_code == 200
    assert (changed.get_json()["title"], changed.get_json()["taskType"], changed.get_json()["description"]) == ("Unfair terms", "summarization", None)
    assert changed.get_json()["documents"] == entry["documents"]

    assert client.delete(f"/api/library/admin/legal-data-hunter/entries/{entry['ref']}", headers=ADMIN).status_code == 204
    assert client.get("/api/library/admin/legal-data-hunter/entries", headers=ADMIN).get_json()["entries"] == []
    assert client.delete(f"/api/library/admin/legal-data-hunter/entries/{entry['ref']}", headers=ADMIN).status_code == 404
    assert [a["event"] for a in audits] == ["library.ldh_entry_created", "library.ldh_entry_updated", "library.ldh_entry_removed"]


def test_an_entry_saved_without_documents_keeps_what_its_query_finds(client, hunter):
    entry = create(client, query={**QUERY, "limit": 4})

    assert [d["sourceId"] for d in entry["documents"]] == [
        "ECLI:NL:HR:2023:1234", "6543a1b2c3d4e5f6", "1 BvR 123/20", "ECLI:NL:RBAMS:2022:999",
    ]
    assert hunter.calls[-1]["key"] == ADMIN_KEY and hunter.calls[-1]["json"]["top_k"] == 4


def test_an_entry_holds_at_most_100_documents(client, hunter):
    documents = [{"country": "NL", "source": "rechtspraak", "sourceId": f"ECLI:NL:HR:2023:{n}"} for n in range(101)]

    refused = create(client, status=400, documents=documents)
    accepted = create(client, documents=documents[:100])

    assert refused["fields"] == {"documents": "An entry holds at most 100 documents."}
    assert accepted["documentCount"] == 100
    assert hunter.calls == []


@pytest.mark.parametrize("document", [
    {"country": "NL", "source": "..", "sourceId": "x"},
    {"country": "NL", "source": "rechtspraak", "sourceId": ""},
    {"country": "NL", "source": "rechtspraak", "sourceId": "a\nb"},
    {"country": "N1", "source": "rechtspraak", "sourceId": "x"},
    "NL/rechtspraak/x",
])
def test_an_entry_refuses_what_is_not_a_document_id(client, document):
    assert "documents.0" in create(client, status=400, documents=[document])["fields"]


def test_an_entry_needs_a_title_and_a_known_task(client):
    body = create(client, status=400, title=" ", taskType="astrology", documents=[{"country": "NL", "source": "r", "sourceId": "x"}])

    assert set(body["fields"]) == {"title", "taskType"}


def test_refresh_replaces_the_ids_and_leaves_started_imports_alone(client, app, hunter, tmp_path):
    entry = create(client, documents=[
        {"country": "EU", "source": "eurlex", "sourceId": "32016R0679"},
        {"country": "NL", "source": "rechtspraak", "sourceId": "ECLI:NL:HR:2023:1234"},
    ])
    first = start(client, entry)
    done, _ = work(app, tmp_path, first["id"])

    refreshed = client.post(f"/api/library/admin/legal-data-hunter/entries/{entry['ref']}/refresh", headers=ADMIN)

    assert refreshed.status_code == 200
    body = refreshed.get_json()
    # The query runs again at the entry's size, two, not the query's 100.
    assert [d["sourceId"] for d in body["documents"]] == ["ECLI:NL:HR:2023:1234", "6543a1b2c3d4e5f6"]
    assert hunter.calls[-1]["json"]["top_k"] == 2
    assert body["refreshedAt"] >= entry["refreshedAt"]
    assert hunter.calls[-1]["key"] == ADMIN_KEY
    # The import made before the refresh keeps the documents it copied, and so does its dataset's provenance.
    with app.app_context():
        record = app.db.session.get(LibraryImport, first["id"])
        assert record.documents[0] == {"country": "EU", "source": "eurlex", "sourceId": "32016R0679", "title": None}
    provenance = client.get(f"/api/library/provenance/{done['datasetRef']}", headers=ALICE).get_json()
    assert provenance["legalDataHunter"]["documents"] == 2
    # A new import takes the refreshed ids.
    assert start(client, entry)["documentCount"] == 2


# --- What a student sees ----------------------------------------------------------------------------------------


def test_entries_are_featured_cards_labelled_legal_data_hunter_with_their_credit_cost(client):
    entry = create(client)

    featured = client.get("/api/library/featured", headers=ALICE).get_json()

    [card] = featured["legalDataHunter"]
    assert card["provider"] == "legal_data_hunter" and card["providerLabel"] == "Legal Data Hunter"
    assert card["featured"] is True and card["externalId"] == entry["ref"]
    assert card["documentCount"] == 5 and card["credits"] == 5
    assert card["creditCost"] == "uses 5 of your Legal Data Hunter credits"
    assert card["taskType"] == "classification"
    assert card["license"]["id"] == "legal-data-hunter-terms" and card["license"]["level"] == "warn"
    assert card["license"]["levels"] == ["warn", "hide"]
    assert card["confirmation"] == {"text": CONFIRMATION, "checkbox": CHECKBOX}
    assert card["importable"] is True and card["hasKey"] is True and card["reason"] is None
    assert "documents" not in card  # the ids stay on the server
    assert STUDENT_KEY not in json.dumps(featured)


def test_a_card_costs_100_credits_for_a_full_entry(client):
    create(client, documents=[{"country": "NL", "source": "rechtspraak", "sourceId": f"ECLI:NL:HR:2023:{n}"} for n in range(100)])

    [card] = client.get("/api/library/legal-data-hunter/entries", headers=ALICE).get_json()["entries"]

    assert card["creditCost"] == "uses 100 of your Legal Data Hunter credits"


def test_a_student_without_a_key_sees_import_disabled_and_why(client):
    entry = create(client)

    listing = client.get("/api/library/legal-data-hunter/entries", headers=BOB).get_json()
    options = client.get("/api/library/options", headers=BOB).get_json()["legalDataHunter"]
    refused = start(client, entry, status=409, headers=BOB)

    [card] = listing["entries"]
    assert listing["hasKey"] is False and listing["needsKey"] == NEEDS_KEY
    assert card["importable"] is False and card["reasonCode"] == "needs_key" and card["reason"] == NEEDS_KEY
    assert options["offered"] is True and options["hasKey"] is False and options["needsKey"] == NEEDS_KEY
    assert refused == {"error": NEEDS_KEY, "code": "needs_key"}


def test_the_entry_page_shows_the_confirmation(client):
    entry = create(client)

    response = client.get(f"/api/library/datasets/legal_data_hunter/{entry['ref']}", headers=ALICE)
    missing = client.get("/api/library/datasets/legal_data_hunter/nope", headers=ALICE)

    assert response.status_code == 200
    assert response.get_json()["confirmation"] == {"text": CONFIRMATION, "checkbox": CHECKBOX}
    assert missing.status_code == 404


# --- The license table ------------------------------------------------------------------------------------------


def settings(client, policy_body):
    return client.put("/api/library/admin/settings", json={"sources": [], "license_policy": policy_body}, headers=ADMIN)


def test_legal_data_hunter_terms_are_warn_or_hide_never_allow(client):
    listed = client.get("/api/library/admin/settings", headers=ADMIN).get_json()
    entry = next(lic for lic in listed["licenses"] if lic["id"] == "legal-data-hunter-terms")
    refused = settings(client, {"legal-data-hunter-terms": "allow"})
    hidden = settings(client, {"legal-data-hunter-terms": "hide"})

    assert entry["name"] == "Legal Data Hunter terms"
    assert (entry["level"], entry["default"], entry["levels"]) == ("warn", "warn", ["warn", "hide"])
    assert entry["condition"] == CONFIRMATION and entry["ack"] == CHECKBOX
    assert listed["legalDataHunter"] == {"available": True}
    assert refused.status_code == 400
    assert refused.get_json()["fields"] == {"license_policy.legal-data-hunter-terms": "Choose Warn or Hide."}
    assert hidden.status_code == 200
    assert next(lic for lic in hidden.get_json()["licenses"] if lic["id"] == "legal-data-hunter-terms")["level"] == "hide"
    # Other licenses may still be Allow.
    assert LICENSES["cc-by-nc-4.0"].levels == ["allow", "warn", "hide"]


def test_an_allow_stored_by_hand_still_warns():
    assert policy.level(LEGAL_DATA_HUNTER_TERMS, {"legal-data-hunter-terms": "allow"}) == "warn"
    assert policy.level(LEGAL_DATA_HUNTER_TERMS, {"legal-data-hunter-terms": "hide"}) == "hide"


def test_hidden_terms_take_the_entries_out_of_the_library(client, app, tmp_path):
    entry = create(client)
    queued = start(client, entry)
    settings(client, {"legal-data-hunter-terms": "hide"})

    assert client.get("/api/library/featured", headers=ALICE).get_json()["legalDataHunter"] == []
    assert client.get(f"/api/library/datasets/legal_data_hunter/{entry['ref']}", headers=ALICE).status_code == 404
    assert start(client, entry, status=403)["code"] == "license_hidden"
    # One queued before the admin hid them fails in the worker, before any document is fetched.
    done, sink = work(app, tmp_path, queued["id"])
    assert done["status"] == "failed" and done["code"] == "license_hidden"
    assert app.hunter.documents_fetched() == [] and sink.calls == []


# --- Importing --------------------------------------------------------------------------------------------------


def test_an_import_needs_the_confirmation_every_time(client, queue):
    entry = create(client)

    for ack in (None, False, "yes"):
        body = {"provider": "legal_data_hunter", "dataset": entry["ref"]}
        if ack is not None:
            body["licenseAck"] = ack
        response = client.post("/api/library/imports", json=body, headers=ALICE)
        assert response.status_code == 409 and response.get_json()["code"] == "needs_ack"
    assert queue.refs == []


def test_an_import_fetches_each_document_with_the_students_key_and_skips_those_without_text(client, app, hunter, tmp_path):
    entry = create(client)
    queued = start(client, entry)
    hunter.calls.clear()

    done, sink = work(app, tmp_path, queued["id"])

    assert done["status"] == "completed", done
    assert done["documentCount"] == 5 and done["rows"] == 3
    assert {call["key"] for call in hunter.calls} == {STUDENT_KEY}
    assert {call["origin"] for call in hunter.calls} == {"https://legaldatahunter.com"}
    assert hunter.documents_fetched() == [
        "NL/rechtspraak/ECLI:NL:HR:2023:1234", "FR/Judilibre/6543a1b2c3d4e5f6", "DE/bverfg/1 BvR 123/20",
        "NL/rechtspraak/ECLI:NL:RBAMS:2022:999", "NL/rechtspraak/ECLI:NL:GHARL:2021:42",
    ]
    [made] = sink.calls
    assert made["header"] == ["text", "court", "date", "country", "language", "ecli", "eli", "source_url", "text_truncated"]
    assert made["task_type"] == "classification" and made["owner"] == "alice"
    assert made["description"] == "5 documents from Legal Data Hunter, unlabelled"
    assert made["rows"][0] == {
        "text": "De Hoge Raad oordeelt dat het beding in de algemene voorwaarden oneerlijk is in de zin van richtlijn 93/13/EEG.",
        "court": "Hoge Raad", "date": "2023-04-14", "country": "NL", "language": "nl", "ecli": "ECLI:NL:HR:2023:1234",
        "eli": "", "source_url": "https://uitspraken.rechtspraak.nl/details?id=ECLI:NL:HR:2023:1234", "text_truncated": "false",
    }
    assert made["rows"][1]["text_truncated"] == "true"
    assert [row["ecli"] for row in made["rows"]] == [
        "ECLI:NL:HR:2023:1234", "ECLI:FR:CCASS:2023:C100123", "ECLI:DE:BVerfG:2022:rk20220202.1bvr012320",
    ]
    assert kept(app, queued["id"]) == 0  # the dataset holds the rows now


def test_skipped_and_missing_documents_are_counted_on_the_dataset(client, app, hunter, tmp_path):
    entry = create(client)
    queued = start(client, entry)

    done, sink = work(app, tmp_path, queued["id"])
    provenance = client.get(f"/api/library/provenance/{done['datasetRef']}", headers=ALICE).get_json()

    # NL:RBAMS has no text; NL:GHARL is not one Legal Data Hunter still has (404).
    assert (done["skipped"], done["missing"], done["rows"]) == (1, 1, 3)
    assert len(sink.calls[0]["rows"]) == 3
    assert provenance["legalDataHunter"] == {
        "entry": entry["ref"], "documents": 5, "rows": 3, "skipped": 1, "missing": 1,
        "termsConfirmedAt": provenance["licenseAckAt"],
    }


def test_the_confirmation_time_is_the_datasets_provenance_and_it_cannot_be_published(client, app, tmp_path):
    entry = create(client)
    queued = start(client, entry)
    done, _ = work(app, tmp_path, queued["id"])

    provenance = client.get(f"/api/library/provenance/{done['datasetRef']}", headers=ALICE).get_json()

    assert queued["licenseAckAt"] is not None
    assert provenance["licenseAckAt"] == queued["licenseAckAt"]
    assert provenance["provider"] == "legal_data_hunter" and provenance["source"] == "legal_data_hunter"
    assert provenance["license"]["id"] == "legal-data-hunter-terms"
    assert provenance["publishable"] is False
    assert provenance["publishRefusal"] == provider.DATASET_PUBLISH_REFUSED
    with app.app_context():
        session = app.db.session
        assert ldh.publish_refusal(session, done["datasetRef"]) == ldh.DATASET_PUBLISH_REFUSED
        assert ldh.publish_refusal(session, done["datasetRef"], model=True) == ldh.MODEL_PUBLISH_REFUSED
        assert ldh.publish_refusal(session, "someone-elses-upload") is None


def test_a_legislation_entry_keeps_the_eli(client, app, tmp_path):
    entry = create(client, query={"q": "data protection", "corpus": "legislation"})
    done, sink = work(app, tmp_path, start(client, entry)["id"])

    assert done["status"] == "completed"
    assert sink.calls[0]["rows"][0]["eli"] == "http://data.europa.eu/eli/reg/2016/679/oj"


def test_out_of_credits_pauses_the_import_and_try_again_resumes_without_refetching(client, app, hunter, tmp_path):
    entry = create(client)
    queued = start(client, entry)
    hunter.calls.clear()
    hunter.credits = 2

    paused, sink = work(app, tmp_path, queued["id"])

    assert paused["status"] == "failed" and paused["code"] == "out_of_credits"
    assert paused["error"] == "Fetched 2 of 5 — your Legal Data Hunter credits ran out. Try again when they reset."
    assert paused["resumeAfter"] is not None
    assert sink.calls == []
    assert kept(app, queued["id"]) == 2  # what it fetched is kept
    first_fetches = hunter.documents_fetched()
    assert first_fetches[:2] == ["NL/rechtspraak/ECLI:NL:HR:2023:1234", "FR/Judilibre/6543a1b2c3d4e5f6"]

    hunter.calls.clear()
    hunter.credits = None  # the credits reset
    retried = client.post(f"/api/library/imports/{queued['id']}/retry", headers=ALICE)
    assert retried.status_code == 202
    again = retried.get_json()
    assert again["licenseAckAt"] == queued["licenseAckAt"]  # the confirmation it was started with
    assert kept(app, queued["id"]) == 0 and kept(app, again["id"]) == 2
    done, sink = work(app, tmp_path, again["id"])

    assert done["status"] == "completed", done
    assert hunter.documents_fetched() == [
        "DE/bverfg/1 BvR 123/20", "NL/rechtspraak/ECLI:NL:RBAMS:2022:999", "NL/rechtspraak/ECLI:NL:GHARL:2021:42",
    ]
    assert [row["country"] for row in sink.calls[0]["rows"]] == ["NL", "FR", "DE"]  # in the entry's order
    assert (done["skipped"], done["missing"]) == (1, 1)


def test_a_retry_keeps_the_documents_it_started_with_after_a_refresh(client, app, hunter, tmp_path):
    entry = create(client, documents=[{"country": "EU", "source": "eurlex", "sourceId": "32016R0679"}])
    queued = start(client, entry)
    hunter.credits = 0
    paused, _ = work(app, tmp_path, queued["id"])
    assert paused["error"] == "Fetched 0 of 1 — your Legal Data Hunter credits ran out. Try again when they reset."
    hunter.credits = None
    client.post(f"/api/library/admin/legal-data-hunter/entries/{entry['ref']}/refresh", headers=ADMIN)

    again = client.post(f"/api/library/imports/{queued['id']}/retry", headers=ALICE).get_json()

    assert again["documentCount"] == 1


def test_a_paused_import_keeps_what_it_fetched_until_it_is_put_away(client, app, hunter, tmp_path):
    entry = create(client)
    queued = start(client, entry)
    hunter.credits = 3
    work(app, tmp_path, queued["id"])
    assert kept(app, queued["id"]) == 3

    dismissed = client.post(f"/api/library/imports/{queued['id']}/dismiss", headers=ALICE)

    assert dismissed.status_code == 200
    assert kept(app, queued["id"]) == 0


def test_a_cancel_stops_the_fetch_before_the_next_document(client, app, hunter, tmp_path, monkeypatch):
    entry = create(client)
    queued = start(client, entry)
    real = provider.document

    def cancel_after_first(key, ref):
        found = real(key, ref)
        with app.app_context():
            imports.cancel(app.extensions["dataset_library"], "alice", queued["id"])
        return found

    monkeypatch.setattr(provider, "document", cancel_after_first)
    done, sink = work(app, tmp_path, queued["id"])

    assert done["status"] == "cancelled"
    assert len(hunter.documents_fetched()) == 1 and sink.calls == []
    # The document answered after the cancel is never written.
    assert kept(app, queued["id"]) == 0


def test_the_sweep_deletes_fetched_documents_nobody_will_resume(client, app, hunter, tmp_path):
    entry = create(client)
    queued = start(client, entry)
    hunter.credits = 1
    work(app, tmp_path, queued["id"])
    with app.app_context():
        session = app.db.session
        imports.sweep(session)
        assert kept(app, queued["id"]) == 1  # still resumable
        imports.sweep(session, now=imports._now() + ldh.KEEP_FETCHED + timedelta(minutes=1))
    assert kept(app, queued["id"]) == 0


def test_erasing_a_user_deletes_what_their_imports_fetched(client, app, hunter, tmp_path):
    entry = create(client)
    queued = start(client, entry)
    hunter.credits = 2
    work(app, tmp_path, queued["id"])

    with app.app_context():
        assert ldh.erase(app.db.session, "alice") == 2
    assert kept(app, queued["id"]) == 0


def test_a_refused_key_fails_the_import_with_words_to_act_on(client, app, tmp_path):
    entry = create(client)
    queued = start(client, entry)

    done, _ = work(app, tmp_path, queued["id"], keys={"alice": "sk-revoked"})
    gone, _ = work(app, tmp_path, start(client, entry)["id"], keys={})

    assert (done["code"], done["error"]) == ("bad_key", "Legal Data Hunter refused your key. Check it in Settings.")
    assert (gone["code"], gone["error"]) == ("needs_key", NEEDS_KEY)


def test_a_busy_legal_data_hunter_fails_the_import_as_busy(client, app, hunter, tmp_path):
    entry = create(client)
    queued = start(client, entry)
    hunter.busy = True

    done, _ = work(app, tmp_path, queued["id"])

    assert (done["code"], done["error"]) == ("provider_busy", "Legal Data Hunter is busy. Try again in a few minutes.")


def test_a_double_click_returns_the_running_import(client):
    entry = create(client)
    first = start(client, entry)

    second = client.post("/api/library/imports", json={"provider": "legal_data_hunter", "dataset": entry["ref"], "licenseAck": True}, headers=ALICE)

    assert second.status_code == 200 and second.get_json()["id"] == first["id"]


def test_an_import_fetches_no_more_documents_than_the_row_cap(client, app, hunter, tmp_path):
    entry = create(client)
    client.put("/api/library/admin/settings", json={"sources": [], "license_policy": {}, "limits": {"maxRows": 2}}, headers=ADMIN)

    card = client.get("/api/library/legal-data-hunter/entries", headers=ALICE).get_json()["entries"][0]
    queued = start(client, entry)
    hunter.calls.clear()
    work(app, tmp_path, queued["id"])

    assert card["documentCount"] == 5 and card["creditCost"] == "uses 2 of your Legal Data Hunter credits"
    assert queued["documentCount"] == 2
    assert len(hunter.documents_fetched()) == 2


# --- Only legaldatahunter.com -----------------------------------------------------------------------------------


def test_a_redirect_to_another_host_is_refused_before_anything_is_sent_there(client, hunter):
    hunter.redirects["/v1/search"] = "https://evil.example/v1/search"

    response = client.post("/api/library/admin/legal-data-hunter/preview", json=QUERY, headers=ADMIN)

    assert response.status_code == 503 and response.get_json()["code"] == "provider_unavailable"
    assert {call["origin"] for call in hunter.calls} == {"https://legaldatahunter.com"}


def test_same_origin_redirects_are_followed_and_others_refused_for_documents(hunter):
    ref = {"country": "NL", "source": "rechtspraak", "sourceId": "ECLI:NL:HR:2023:1234"}
    hunter.redirects["/v1/documents/NL/old"] = "/v1/documents/NL/rechtspraak?source_id=ECLI:NL:HR:2023:1234"

    found = provider.document(STUDENT_KEY, {**ref, "source": "old"})
    hunter.redirects["/v1/documents/NL/rechtspraak"] = "http://legaldatahunter.com/v1/documents/NL/rechtspraak"
    with pytest.raises(provider.OffOrigin):
        provider.document(STUDENT_KEY, ref)

    assert found["ecli"] == "ECLI:NL:HR:2023:1234"
    assert {call["origin"] for call in hunter.calls} == {"https://legaldatahunter.com"}


def test_a_redirect_off_origin_during_an_import_fails_it_without_leaving_the_origin(client, app, hunter, tmp_path):
    entry = create(client)
    queued = start(client, entry)
    hunter.redirects["/v1/documents/NL/rechtspraak"] = "https://legaldatahunter.com.evil.example/steal"

    done, _ = work(app, tmp_path, queued["id"])

    assert done["code"] == "provider_busy"
    assert {call["origin"] for call in hunter.calls} == {"https://legaldatahunter.com"}


def test_the_source_id_goes_in_the_query_not_the_path(hunter):
    provider.document(STUDENT_KEY, {"country": "DE", "source": "bverfg", "sourceId": "1 BvR 123/20"})

    [call] = hunter.calls
    assert call["path"] == "/v1/documents/DE/bverfg" and call["params"] == {"source_id": "1 BvR 123/20"}


def test_retry_after_reads_seconds_and_dates():
    from datetime import datetime, timezone

    now = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)
    assert provider.retry_after("120") == 120
    assert provider.retry_after("Tue, 24 Nov 2026 13:00:00 GMT", now=now) == 3600
    assert provider.retry_after("soon") is None and provider.retry_after(None) is None
    # ASCII digits only, and never more than 31 days.
    assert provider.retry_after("\u0661\u0662\u0660") is None
    assert provider.retry_after("99999999999") == 31 * 86_400
    assert provider.retry_after("Tue, 24 Nov 2099 13:00:00 GMT", now=now) == 31 * 86_400


# --- Not offered ------------------------------------------------------------------------------------------------


def test_without_the_key_seam_legal_data_hunter_is_not_offered(make_app, hunter):
    app = make_app(enqueue=Queue())
    client = app.test_client()
    with app.app_context():
        session = app.db.session
        session.add(LibraryLegalDataHunterEntry(title="t", query={"q": "x"}, documents=[{"country": "NL", "source": "r", "sourceId": "x"}]))
        session.commit()
        ref = session.query(LibraryLegalDataHunterEntry).one().id

    assert client.post("/api/library/admin/legal-data-hunter/preview", json=QUERY, headers=ADMIN).status_code == 404
    assert client.get("/api/library/featured", headers=ALICE).get_json()["legalDataHunter"] == []
    assert client.get("/api/library/options", headers=ALICE).get_json()["legalDataHunter"]["offered"] is False
    refused = client.post("/api/library/imports", json={"provider": "legal_data_hunter", "dataset": ref, "licenseAck": True}, headers=ALICE)
    assert refused.status_code == 404 and refused.get_json()["code"] == "not_offered"
    assert hunter.calls == []


def test_a_source_filter_may_name_its_country_as_discovery_does(client, hunter):
    response = client.post(
        "/api/library/admin/legal-data-hunter/preview", json={**QUERY, "source": ["FR/CNIL", "rechtspraak"]}, headers=ADMIN,
    )
    assert response.status_code == 200, response.get_json()
    assert hunter.calls[-1]["json"]["source"] == ["FR/CNIL", "rechtspraak"]
    bad = client.post("/api/library/admin/legal-data-hunter/preview", json={**QUERY, "source": ["FR/CNIL/x"]}, headers=ADMIN)
    assert bad.status_code == 400 and "query.source" in bad.get_json()["fields"]


def test_the_per_minute_cap_is_waited_out_not_shown_as_credits_running_out(client, app, hunter, tmp_path, monkeypatch):
    waited = []
    monkeypatch.setattr(ldh, "_sleep", waited.append)
    entry = create(client)
    queued = start(client, entry)
    hunter.calls.clear()
    hunter.minute_limited = 2

    done, sink = work(app, tmp_path, queued["id"])

    assert done["status"] == "completed", done
    assert waited == [7, 7]  # Retry-After, twice
    fetched = hunter.documents_fetched()
    assert fetched[0] == fetched[1] == fetched[2]  # the same document asked for again, not skipped
    assert len(set(fetched)) == 5


def test_a_per_minute_cap_that_does_not_lift_stops_the_import_as_busy_and_keeps_what_it_fetched(client, app, hunter, tmp_path, monkeypatch):
    monkeypatch.setattr(ldh, "_sleep", lambda seconds: None)
    entry = create(client)
    queued = start(client, entry)
    hunter.calls.clear()
    hunter.credits = 2  # two documents go through, then only per-minute refusals
    original = hunter.request

    def minute_after_two(*args, **kwargs):
        if hunter.credits == 0:
            hunter.minute_limited = 1
        return original(*args, **kwargs)

    monkeypatch.setattr(provider.requests, "request", minute_after_two)

    stopped, _ = work(app, tmp_path, queued["id"])

    assert stopped["status"] == "failed" and stopped["code"] == "provider_busy"
    assert stopped["error"] == provider.SLOW_DOWN
    assert kept(app, queued["id"]) == 2


def test_an_admin_preview_over_the_per_minute_cap_is_asked_to_slow_down(client, hunter):
    hunter.minute_limited = 1
    response = client.post("/api/library/admin/legal-data-hunter/preview", json=QUERY, headers=ADMIN)
    assert response.status_code == 429
    assert response.get_json()["code"] == "slow_down"


# --- Council fixes (#126 review) --------------------------------------------------------------------------------


def library(app):
    return app.extensions["dataset_library"]


def pause(client, app, hunter, tmp_path, credits=2):
    """An entry, and an import of it paused as out of credits after `credits` documents."""
    entry = create(client)
    queued = start(client, entry)
    hunter.credits = credits
    paused, _ = work(app, tmp_path, queued["id"])
    assert paused["code"] == "out_of_credits", paused
    hunter.credits = None
    return entry, queued


def test_a_dataset_saved_before_the_worker_died_is_still_refused_publishing(client, app, hunter, tmp_path, monkeypatch):
    entry = create(client)
    queued = start(client, entry)
    real = imports._advance

    def die_before_completing(session, record, state, updates=None):
        if state == imports.COMPLETED:
            raise KeyboardInterrupt  # the worker is killed: nothing after the save runs
        return real(session, record, state, updates)

    monkeypatch.setattr(imports, "_advance", die_before_completing)
    with pytest.raises(KeyboardInterrupt):
        work(app, tmp_path, queued["id"])

    with app.app_context():
        session = app.db.session
        assert session.get(LibraryImport, queued["id"]).status == "saving"
        assert imports.sweep(session, now=imports._now() + timedelta(minutes=5)) == [queued["id"]]
        swept = session.get(LibraryImport, queued["id"])
        assert (swept.status, swept.dataset_ref) == ("failed", "dataset-1")
        assert ldh.publish_refusal(session, "dataset-1") == ldh.DATASET_PUBLISH_REFUSED
        assert ldh.publish_refusal(session, "dataset-1", model=True) == ldh.MODEL_PUBLISH_REFUSED


def test_cleanup_after_completion_never_deletes_the_datasets_csv(client, app, hunter, tmp_path, monkeypatch, caplog):
    entry = create(client)
    queued = start(client, entry)

    def broken(session, import_ref):
        raise RuntimeError("the database went away")

    monkeypatch.setattr(ldh, "discard", broken)
    with caplog.at_level(logging.WARNING, logger="dataset_library"):
        done, sink = work(app, tmp_path, queued["id"])

    assert done["status"] == "completed" and done["datasetRef"] == "dataset-1"
    assert (tmp_path / "data" / f"library-{queued['id']}.csv").exists()
    assert "RuntimeError" in caplog.text and "the database went away" not in caplog.text
    assert kept(app, queued["id"]) == 5
    with app.app_context():
        imports.sweep(app.db.session)  # the purge deletes them later
    assert kept(app, queued["id"]) == 0


@pytest.mark.parametrize("source", ["legal_data_hunter", "huggingface"])
def test_of_two_workers_handed_one_import_only_the_one_that_claims_it_runs_it(client, app, hunter, tmp_path, monkeypatch, source):
    if source == "legal_data_hunter":
        ref = start(client, create(client))["id"]
    else:
        with app.app_context():
            record = LibraryImport(
                owner_id="alice", provider="huggingface", external_id="coastalcph/lex_glue", config="ledgar",
                split="train", text_column="text", label_column="label", task_type="classification", sample_size=10,
                max_rows=1000, name="LexGLUE", license="cc-by-4.0", status="queued",
            )
            app.db.session.add(record)
            app.db.session.commit()
            ref = record.id
    real = imports.import_cap_bytes

    def the_other_worker_claims_it_first(session, cap):
        # Both workers read `queued`; the other one's claim lands between this one's read and its claim.
        session.execute(sa.update(LibraryImport).where(LibraryImport.id == ref).values(status="fetching"))
        session.commit()
        return real(session, cap)

    monkeypatch.setattr(imports, "import_cap_bytes", the_other_worker_claims_it_first)
    hunter.calls.clear()

    done, sink = work(app, tmp_path, ref)

    # This worker did nothing: no fetch (Hugging Face's would have failed the import), no dataset, no failure.
    assert done["status"] == "fetching" and done["code"] is None
    assert hunter.calls == [] and sink.calls == []


def test_a_cancel_mid_fetch_discards_what_the_import_fetched(client, app, hunter, tmp_path, monkeypatch):
    entry = create(client)
    queued = start(client, entry)
    real = provider.document
    answered = []

    def cancel_at_the_third(key, ref):
        found = real(key, ref)
        answered.append(ref)
        if len(answered) == 3:
            assert kept(app, queued["id"]) == 2
            with app.app_context():
                imports.cancel(library(app), "alice", queued["id"])
        return found

    monkeypatch.setattr(provider, "document", cancel_at_the_third)
    done, sink = work(app, tmp_path, queued["id"])

    assert done["status"] == "cancelled" and sink.calls == []
    assert kept(app, queued["id"]) == 0  # the two before the cancel are discarded, the third never written


def test_a_document_answered_after_the_import_was_erased_is_never_written(client, app, hunter, tmp_path, monkeypatch):
    entry = create(client)
    queued = start(client, entry)
    real = provider.document

    def erase_after_first(key, ref):
        found = real(key, ref)
        with app.app_context():
            app.db.session.query(LibraryImport).filter_by(id=queued["id"]).delete()
            app.db.session.commit()
        return found

    monkeypatch.setattr(provider, "document", erase_after_first)
    sink = Sink()
    with app.app_context():
        imports.run_import(
            queued["id"], session=app.db.session, fetch_shards=forbidden, create_dataset=sink,
            data_dir=str(tmp_path / "data"), heartbeat_seconds=None, legal_data_hunter_key={"alice": STUDENT_KEY}.get,
        )

    assert kept(app, queued["id"]) == 0 and sink.calls == []


def test_an_import_that_cannot_run_here_discards_what_it_fetched(client, app, hunter, tmp_path):
    _, queued = pause(client, app, hunter, tmp_path)
    again = client.post(f"/api/library/imports/{queued['id']}/retry", headers=ALICE).get_json()
    assert kept(app, again["id"]) == 2

    done, _ = work(app, tmp_path, again["id"], seam=None)

    assert done["code"] == "not_offered"
    assert kept(app, again["id"]) == 0


def test_hiding_the_terms_discards_what_imports_not_running_fetched(client, app, hunter, tmp_path):
    entry, queued = pause(client, app, hunter, tmp_path)
    running = start(client, entry)
    with app.app_context():
        app.db.session.add(LibraryLegalDataHunterDocument(import_id=running["id"], position=0, outcome="missing"))
        app.db.session.commit()
    assert kept(app, queued["id"]) == 2

    assert settings(client, {"legal-data-hunter-terms": "hide"}).status_code == 200

    assert kept(app, queued["id"]) == 0
    assert kept(app, running["id"]) == 1  # a running import's worker stops itself


def test_the_sweep_keeps_documents_only_for_a_failure_try_again_resumes(client, app, hunter, tmp_path):
    entry, paused = pause(client, app, hunter, tmp_path)
    with app.app_context():
        session = app.db.session
        for status, code in (("cancelled", None), ("failed", "invalid")):
            record = LibraryImport(
                owner_id="alice", provider="legal_data_hunter", external_id=entry["ref"], text_column="text",
                task_type="classification", sample_size=1, max_rows=300, name="x", license="legal-data-hunter-terms",
                status=status, error_code=code, documents=[{"country": "NL", "source": "r", "sourceId": "x"}],
            )
            session.add(record)
            session.flush()
            session.add(LibraryLegalDataHunterDocument(import_id=record.id, position=0, outcome="missing"))
        session.commit()
        imports.sweep(session)
        assert session.query(LibraryLegalDataHunterDocument).count() == 2
    assert kept(app, paused["id"]) == 2


def test_erasing_a_user_after_the_host_deleted_their_imports_still_deletes_what_they_fetched(client, app, hunter, tmp_path):
    entry, queued = pause(client, app, hunter, tmp_path)

    with app.app_context():
        session = app.db.session
        session.query(LibraryImport).filter_by(owner_id="alice").delete()
        session.commit()
        assert ldh.erase(session, "alice") == 2
        assert ldh.erase(session, "ada") == 0
        assert session.get(LibraryLegalDataHunterEntry, entry["ref"]).created_by is None
    assert kept(app, queued["id"]) == 0


def test_a_retry_resumes_its_own_documents_after_the_entry_is_deleted_and_the_cap_lowered(client, app, hunter, tmp_path):
    entry, queued = pause(client, app, hunter, tmp_path)
    assert client.delete(f"/api/library/admin/legal-data-hunter/entries/{entry['ref']}", headers=ADMIN).status_code == 204
    client.put("/api/library/admin/settings", json={"sources": [], "license_policy": {}, "limits": {"maxRows": 2}}, headers=ADMIN)
    hunter.calls.clear()

    retried = client.post(f"/api/library/imports/{queued['id']}/retry", headers=ALICE)

    assert retried.status_code == 202, retried.get_json()
    again = retried.get_json()
    assert again["documentCount"] == 5 and kept(app, again["id"]) == 2
    done, sink = work(app, tmp_path, again["id"])
    assert done["status"] == "completed", done
    assert len(hunter.documents_fetched()) == 3  # only the rest
    assert len(sink.calls[0]["rows"]) == 3


def test_a_retry_the_queue_refuses_leaves_the_paused_import_as_it_was(client, app, hunter, tmp_path, queue):
    _, queued = pause(client, app, hunter, tmp_path)
    queue.fail = True

    refused = client.post(f"/api/library/imports/{queued['id']}/retry", headers=ALICE)

    assert refused.status_code == 503 and refused.get_json()["code"] == "not_started"
    with app.app_context():
        session = app.db.session
        [new] = session.query(LibraryImport).filter(LibraryImport.id != queued["id"]).all()
        assert new.dismissed_at is not None
        assert session.get(LibraryImport, queued["id"]).dismissed_at is None
        listed = [record["id"] for record in imports.current(library(app), "alice")]
        new_ref = new.id
    assert kept(app, queued["id"]) == 2 and kept(app, new_ref) == 0
    assert listed == [queued["id"]]
    queue.fail = False
    again = client.post(f"/api/library/imports/{queued['id']}/retry", headers=ALICE)
    assert again.status_code == 202 and kept(app, again.get_json()["id"]) == 2


def test_a_retry_while_another_import_of_the_entry_runs_is_refused(client, app, hunter, tmp_path):
    entry, queued = pause(client, app, hunter, tmp_path)
    running = start(client, entry)

    refused = client.post(f"/api/library/imports/{queued['id']}/retry", headers=ALICE)

    assert refused.status_code == 409 and refused.get_json()["code"] == "already_running"
    assert refused.get_json()["error"] == provider.ALREADY_RUNNING
    with app.app_context():
        assert app.db.session.get(LibraryImport, queued["id"]).dismissed_at is None
    assert kept(app, queued["id"]) == 2 and kept(app, running["id"]) == 0


def test_a_put_away_import_cannot_be_tried_again(client, app, hunter, tmp_path, queue):
    _, queued = pause(client, app, hunter, tmp_path)
    client.post(f"/api/library/imports/{queued['id']}/dismiss", headers=ALICE)
    queue.refs.clear()

    refused = client.post(f"/api/library/imports/{queued['id']}/retry", headers=ALICE)

    assert refused.status_code == 409 and refused.get_json()["code"] == "not_retryable"
    assert queue.refs == []


def test_a_pause_stays_listed_and_resumable_until_a_day_after_the_credits_reset(client, app, hunter, tmp_path):
    hunter.retry_after = str(10 * 86_400)
    _, queued = pause(client, app, hunter, tmp_path)

    with app.app_context():
        session = app.db.session
        now = imports._now()

        def listed(days):
            return queued["id"] in [r["id"] for r in imports.current(library(app), "alice", now=now + timedelta(days=days))]

        assert listed(2) and listed(9)  # past a day, and past the week: the credits reset on day 10
        assert not listed(12)
        ldh.purge(session, now=now + timedelta(days=9))
        assert kept(app, queued["id"]) == 2
        ldh.purge(session, now=now + timedelta(days=12))
    assert kept(app, queued["id"]) == 0


def test_a_council_of_europe_document_keeps_the_case_legal_data_hunter_gives_it(client, app, hunter, tmp_path):
    preview = client.post(
        "/api/library/admin/legal-data-hunter/preview",
        json={"q": "data protection", "corpus": "legislation", "country": ["eu", "CoE", "INTL"]}, headers=ADMIN,
    )
    assert hunter.calls[-1]["json"]["country"] == ["EU", "CoE", "INTL"]  # only two-letter codes upper-cased
    coe = preview.get_json()["hits"][1]
    assert (coe["country"], coe["source"], coe["sourceId"]) == ("CoE", "conventions", "ETS-108")
    entry = create(client, query={"q": "data protection", "corpus": "legislation"}, documents=[coe])
    hunter.calls.clear()

    done, sink = work(app, tmp_path, start(client, entry)["id"])

    assert done["status"] == "completed", done
    assert [call["path"] for call in hunter.calls] == ["/v1/documents/CoE/conventions"]
    assert sink.calls[0]["rows"][0]["country"] == "CoE"


def test_a_source_prefix_that_disagrees_with_the_country_is_no_document_id():
    assert provider.document_ref({"country": "NL", "source": "FR/Judilibre", "sourceId": "x"}) is None
    assert provider.document_ref({"country": "FR", "source": "FR/Judilibre", "sourceId": "x"})["source"] == "Judilibre"
    assert provider.document_ref({"source": "CoE/conventions", "sourceId": "x"})["country"] == "CoE"


def test_a_refresh_that_finds_nothing_leaves_the_entry_as_it_was(client, hunter):
    documents = [{"country": "NL", "source": "rechtspraak", "sourceId": "ECLI:NL:HR:2023:1234"}]
    entry = create(client, query={"q": "anything", "corpus": "doctrine"}, documents=documents)

    refreshed = client.post(f"/api/library/admin/legal-data-hunter/entries/{entry['ref']}/refresh", headers=ADMIN)

    assert refreshed.status_code == 400
    assert refreshed.get_json()["fields"] == {"documents": "This search found no documents."}
    after = client.get(f"/api/library/admin/legal-data-hunter/entries/{entry['ref']}", headers=ADMIN).get_json()
    assert after["documents"] == entry["documents"] and after["refreshedAt"] == entry["refreshedAt"]


def test_a_document_legal_data_hunter_rejects_is_counted_missing_not_busy(client, app, hunter, tmp_path):
    entry = create(client)
    queued = start(client, entry)
    hunter.rejected["FR/Judilibre/6543a1b2c3d4e5f6"] = 400
    hunter.rejected["DE/bverfg/1 BvR 123/20"] = 422

    done, sink = work(app, tmp_path, queued["id"])

    assert done["status"] == "completed", done
    assert (done["rows"], done["skipped"], done["missing"]) == (1, 1, 3)


def test_an_import_stops_at_the_download_cap_and_keeps_what_it_fetched(client, app, hunter, tmp_path):
    entry = create(client)
    queued = start(client, entry)
    first = len(json.dumps(DOCUMENTS["NL/rechtspraak/ECLI:NL:HR:2023:1234"]).encode("utf-8"))

    done, sink = work(app, tmp_path, queued["id"], max_bytes=first + 10)

    assert done["status"] == "failed" and done["code"] == "too_large"
    assert done["error"] == provider.too_large(first + 10)
    assert kept(app, queued["id"]) == 1 and sink.calls == []


def test_a_netrc_entry_cannot_replace_the_users_key(tmp_path, monkeypatch):
    netrc = tmp_path / "netrc"
    netrc.write_text("machine legaldatahunter.com login intruder password hunter2\n")
    netrc.chmod(0o600)
    monkeypatch.setenv("NETRC", str(netrc))
    assert requests.utils.get_netrc_auth(provider.ORIGIN + "/v1/search") == ("intruder", "hunter2")
    sent = []

    def send(self, request, **kwargs):
        sent.append(request)
        response = requests.models.Response()
        response.status_code, response._content, response.url, response.request = 200, b'{"hits": []}', request.url, request
        return response

    monkeypatch.setattr(requests.Session, "send", send)

    provider.search(STUDENT_KEY, provider.check_query({"q": "x"}))

    [request] = sent
    assert request.headers["Authorization"] == f"Bearer {STUDENT_KEY}"


def test_a_key_seam_that_raises_is_no_key_and_never_breaks_a_page(make_app, hunter, caplog):
    asked = []

    def seam(owner):
        asked.append(owner)
        if owner == "alice":
            raise RuntimeError("the vault said sk-secret-9999")
        return ADMIN_KEY

    app = make_app(legal_data_hunter_key=seam, enqueue=Queue(), max_rows=lambda: 300)
    client = app.test_client()
    create(client)
    with caplog.at_level(logging.WARNING, logger="dataset_library"):
        featured = client.get("/api/library/featured", headers=ALICE)
        options = client.get("/api/library/options", headers=ALICE)
        asked.clear()
        listing = client.get("/api/library/legal-data-hunter/entries", headers=ALICE)

    assert asked == ["alice"]  # once per request
    assert featured.status_code == options.status_code == listing.status_code == 200
    assert featured.get_json()["legalDataHunter"][0]["reasonCode"] == "needs_key"
    assert options.get_json()["legalDataHunter"]["hasKey"] is False
    assert listing.get_json()["hasKey"] is False and listing.get_json()["entries"][0]["importable"] is False
    assert "RuntimeError" in caplog.text and "sk-secret" not in caplog.text


def test_a_per_minute_retry_after_of_zero_waits_a_second(client, app, hunter, tmp_path, monkeypatch):
    waited = []
    monkeypatch.setattr(ldh, "_sleep", waited.append)
    queued = start(client, create(client))
    hunter.minute_limited, hunter.minute_retry_after = 1, "0"

    done, _ = work(app, tmp_path, queued["id"])

    assert done["status"] == "completed" and waited == [1]


@pytest.mark.parametrize("location", ["", None])
def test_a_redirect_with_nowhere_to_go_is_refused_at_once(client, hunter, location):
    hunter.redirects["/v1/search"] = location

    response = client.post("/api/library/admin/legal-data-hunter/preview", json=QUERY, headers=ADMIN)

    assert response.status_code == 503 and response.get_json()["code"] == "provider_unavailable"
    assert len(hunter.calls) == 1


def test_the_providers_terms_are_not_a_license_a_student_filters_by(client, hub):
    options = client.get("/api/library/options", headers=ALICE).get_json()
    admin = client.get("/api/library/admin/settings", headers=ADMIN).get_json()
    hub.calls.clear()

    found = client.get("/api/library/search?license=legal-data-hunter-terms", headers=ALICE)

    assert "legal-data-hunter-terms" not in {lic["id"] for lic in options["licenses"]}
    assert "Provider terms" not in options["groups"]
    assert "legal-data-hunter-terms" in {lic["id"] for lic in admin["licenses"]} and "Provider terms" in admin["groups"]
    assert found.status_code == 200 and found.get_json()["results"] == [] and hub.calls == []
    assert resolve_license("legal-data-hunter-terms") is UNKNOWN
    assert resolve_license("license:Legal-Data-Hunter-Terms") is UNKNOWN
