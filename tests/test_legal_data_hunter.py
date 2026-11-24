"""Legal Data Hunter entries (#126): what an admin curates with their own key, the featured cards a student sees,
the confirmation every import needs, and the worker fetching each document with the student's key, pausing when
the credits run out and resuming without fetching anything twice. Legal Data Hunter is faked from recorded
responses (fake_legal_data_hunter.py); no test reaches the network."""
import csv
import json
from datetime import timedelta

import pytest

from conftest import ADMIN, ALICE
from dataset_library import imports
from dataset_library import legal_data_hunter as ldh
from dataset_library import policy
from dataset_library.licenses import LEGAL_DATA_HUNTER_TERMS, LICENSES
from dataset_library.models import LibraryImport, LibraryLegalDataHunterDocument, LibraryLegalDataHunterEntry
from dataset_library.providers import legal_data_hunter as provider
from fake_legal_data_hunter import ADMIN_KEY, STUDENT_KEY, FakeLegalDataHunter

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

    def __call__(self, ref):
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


def work(app, tmp_path, ref, keys=None):
    sink = Sink()
    keys = {"alice": STUDENT_KEY} if keys is None else keys
    seam = keys.get
    with app.app_context():
        session = app.db.session
        imports.run_import(
            ref, session=session, fetch_shards=forbidden, create_dataset=sink, data_dir=str(tmp_path / "data"),
            heartbeat_seconds=None, legal_data_hunter_key=seam,
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
    assert entry["query"] == {**QUERY, "country": ["NL", "FR", "DE"], "source": [], "limit": 100}
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
    entry = create(client, documents=[{"country": "EU", "source": "eurlex", "sourceId": "32016R0679"}])
    first = start(client, entry)
    done, _ = work(app, tmp_path, first["id"])

    refreshed = client.post(f"/api/library/admin/legal-data-hunter/entries/{entry['ref']}/refresh", headers=ADMIN)

    assert refreshed.status_code == 200
    body = refreshed.get_json()
    assert [d["sourceId"] for d in body["documents"]] == [
        "ECLI:NL:HR:2023:1234", "6543a1b2c3d4e5f6", "1 BvR 123/20", "ECLI:NL:RBAMS:2022:999", "ECLI:NL:GHARL:2021:42",
    ]
    assert body["refreshedAt"] >= entry["refreshedAt"]
    assert hunter.calls[-1]["key"] == ADMIN_KEY
    # The import made before the refresh keeps the documents it copied, and so does its dataset's provenance.
    with app.app_context():
        record = app.db.session.get(LibraryImport, first["id"])
        assert record.documents == [{"country": "EU", "source": "eurlex", "sourceId": "32016R0679", "title": None}]
    provenance = client.get(f"/api/library/provenance/{done['datasetRef']}", headers=ALICE).get_json()
    assert provenance["legalDataHunter"]["documents"] == 1
    # A new import takes the refreshed ids.
    assert start(client, entry)["documentCount"] == 5


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


def test_a_cancelled_import_keeps_what_it_fetched_until_it_is_put_away(client, app, hunter, tmp_path):
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
