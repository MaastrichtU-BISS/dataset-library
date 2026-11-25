"""Importing a Legal Data Hunter search as a Dataset (#16): `POST /api/library/imports` with the documents a user ticked
across one or more searches (up to 100), the queries that found them, a name, a task type and the terms confirmation.
It runs through the curated-entry import end to end: one credit per document with the user's own key, rows unlabelled,
documents without text skipped and counted, a pause when the credits run out and Try again resuming, and no
publishing. Legal Data Hunter is faked from recorded responses (fake_legal_data_hunter.py); no test reaches the
network."""
import pytest

from conftest import ADMIN, ALICE
from dataset_library import legal_data_hunter as ldh
from dataset_library.models import LibraryImport
from dataset_library.providers import legal_data_hunter as provider
from fake_legal_data_hunter import STUDENT_KEY
# The fixtures and helpers of the curated-entry tests: the faked Legal Data Hunter, the users' keys, the app with the
# key seam, the queue, and running the worker.
from test_legal_data_hunter import (  # noqa: F401
    BOB, QUERY, Queue, app, create, hunter, kept, keys, queue, settings, work,
)

IMPORTS = "/api/library/imports"
SEARCH = "/api/library/legal-data-hunter/search"
# The five documents the recorded case-law search finds: NL:RBAMS has no text, NL:GHARL Legal Data Hunter no longer has.
FIVE = [
    "NL/rechtspraak/ECLI:NL:HR:2023:1234", "FR/Judilibre/6543a1b2c3d4e5f6", "DE/bverfg/1 BvR 123/20",
    "NL/rechtspraak/ECLI:NL:RBAMS:2022:999", "NL/rechtspraak/ECLI:NL:GHARL:2021:42",
]
EU = {"country": "EU", "source": "eurlex", "sourceId": "32016R0679", "title": "GDPR"}


def searched(client, body=None):
    """A user's search through the route: its hits and the query it ran."""
    response = client.post(SEARCH, json=QUERY if body is None else body, headers=ALICE)
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def ids(hits):
    return [{key: hit[key] for key in ("country", "source", "sourceId", "title")} for hit in hits]


def post(client, headers=ALICE, **body):
    return client.post(IMPORTS, json={"provider": "legal_data_hunter", **body}, headers=headers)


def start_search(client, documents, queries=None, status=202, headers=ALICE, **body):
    payload = {"documents": documents, "queries": [QUERY] if queries is None else queries, "licenseAck": True,
               "name": "Unfair terms, my pick", **body}
    response = post(client, headers=headers, **payload)
    assert response.status_code == status, response.get_json()
    return response.get_json()


def provenance_of(client, dataset_ref):
    response = client.get(f"/api/library/provenance/{dataset_ref}", headers=ALICE)
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def many(count):
    return [{"country": "NL", "source": "rechtspraak", "sourceId": f"ECLI:NL:HR:2023:{n}"} for n in range(count)]


# --- What a search import fetches -------------------------------------------------------------------------------


def test_a_search_import_fetches_the_ticked_documents_with_the_users_own_key(client, app, hunter, tmp_path, queue):
    found = searched(client)
    queued = start_search(client, ids(found["hits"]), queries=[found["query"]])
    hunter.calls.clear()

    done, sink = work(app, tmp_path, queued["id"])

    assert queue.refs == [queued["id"]]
    assert queued["provider"] == "legal_data_hunter" and queued["documentCount"] == 5
    assert queued["name"] == "Unfair terms, my pick" and queued["taskType"] == "classification"
    assert done["status"] == "completed", done
    # One request, so one credit, per document, each with the user's own key, on Legal Data Hunter only.
    assert hunter.documents_fetched() == FIVE
    assert {call["key"] for call in hunter.calls} == {STUDENT_KEY}
    assert {call["origin"] for call in hunter.calls} == {"https://legaldatahunter.com"}
    # Rows as for a curated entry, unlabelled; no text skipped, gone counted missing.
    assert (done["rows"], done["skipped"], done["missing"]) == (3, 1, 1)
    [made] = sink.calls
    assert made["header"] == list(provider.ROW_COLUMNS)
    assert made["name"] == "Unfair terms, my pick" and made["owner"] == "alice" and made["task_type"] == "classification"
    assert made["rows"][0] == {
        "text": "De Hoge Raad oordeelt dat het beding in de algemene voorwaarden oneerlijk is in de zin van richtlijn 93/13/EEG.",
        "court": "Hoge Raad", "date": "2023-04-14", "country": "NL", "language": "nl", "ecli": "ECLI:NL:HR:2023:1234",
        "eli": "", "source_url": "https://uitspraken.rechtspraak.nl/details?id=ECLI:NL:HR:2023:1234", "text_truncated": "false",
    }
    assert made["rows"][1]["text_truncated"] == "true"
    assert kept(app, queued["id"]) == 0  # the dataset holds the rows now


def test_documents_ticked_across_searches_are_imported_once_each_in_the_order_ticked(client, app, hunter, tmp_path):
    case_law = searched(client)
    legislation = searched(client, {"q": "data protection", "corpus": "legislation"})
    ticked = ids(case_law["hits"][:2]) + [EU] + ids(case_law["hits"][:1]) + [{**EU, "title": "Another title"}]

    queued = start_search(client, ticked, queries=[case_law["query"], legislation["query"]])
    hunter.calls.clear()
    done, sink = work(app, tmp_path, queued["id"])

    # The same document is never paid for twice, whichever search found it.
    assert queued["documentCount"] == 3
    assert hunter.documents_fetched() == [
        "NL/rechtspraak/ECLI:NL:HR:2023:1234", "FR/Judilibre/6543a1b2c3d4e5f6", "EU/eurlex/32016R0679",
    ]
    assert done["status"] == "completed" and done["rows"] == 3
    assert sink.calls[0]["rows"][2]["eli"] == "http://data.europa.eu/eli/reg/2016/679/oj"


def test_a_search_import_takes_the_task_type_chosen(client, app, tmp_path):
    queued = start_search(client, [EU], taskType="ner")
    done, sink = work(app, tmp_path, queued["id"])

    assert queued["taskType"] == "ner"
    assert done["status"] == "completed" and sink.calls[0]["task_type"] == "ner"


def test_a_search_import_without_a_name_is_named_after_its_first_query(client):
    queued = start_search(client, [EU], name=None)
    blank = start_search(client, [EU, *many(1)], name="   ")

    assert queued["name"] == "Legal Data Hunter: unfair contract terms consumer"
    assert blank["name"] == "Legal Data Hunter: unfair contract terms consumer"


def test_a_search_import_is_cut_to_the_row_cap_as_an_entry_is(make_app, hunter, keys, tmp_path):
    app = make_app(legal_data_hunter_key=lambda owner: keys.get(owner), enqueue=Queue(), max_rows=lambda: 2)
    client = app.test_client()

    queued = start_search(client, many(5))
    # The same first 2 with others past the cap: the same import, not a second one paying for them again.
    again = start_search(client, many(2) + many(7)[5:], status=200)

    assert queued["documentCount"] == 2
    assert again["id"] == queued["id"]


# --- Refusals ---------------------------------------------------------------------------------------------------


def test_more_than_100_documents_are_refused_after_duplicates_are_dropped(client, hunter, queue):
    hundred = start_search(client, many(100) + many(100))
    refused = post(client, documents=many(101), queries=[QUERY], licenseAck=True)

    assert hundred["documentCount"] == 100
    assert refused.status_code == 400
    assert refused.get_json() == {"error": "Import at most 100 Legal Data Hunter documents at once.", "code": "too_many_documents"}
    assert queue.refs == [hundred["id"]]
    assert hunter.calls == []


@pytest.mark.parametrize("documents, code", [
    ([], "no_documents"),
    (None, "no_documents"),
    ("NL/rechtspraak/ECLI:NL:HR:2023:1234", "no_documents"),
    ([{"country": "NL", "source": "rechtspraak"}], "bad_document"),
    ([EU, {"country": "N L", "source": "rechtspraak", "sourceId": "x"}], "bad_document"),
    ([EU, "EU/eurlex/32016R0679"], "bad_document"),
    ([{"country": "FR", "source": "NL/rechtspraak", "sourceId": "x"}], "bad_document"),
])
def test_malformed_documents_are_refused_with_a_code_and_nothing_queued(client, hunter, queue, documents, code):
    response = post(client, documents=documents, queries=[QUERY], licenseAck=True)

    assert response.status_code == 400, response.get_json()
    assert response.get_json()["code"] == code
    assert queue.refs == [] and hunter.calls == []


def test_a_malformed_document_is_named_by_its_place(client):
    response = post(client, documents=[EU, {"country": "NL"}], queries=[QUERY], licenseAck=True)

    assert response.get_json() == {"error": "Document 2 is not a Legal Data Hunter document id.", "code": "bad_document"}


@pytest.mark.parametrize("queries", [None, [], "unfair terms", [{"q": ""}], [{"q": "x", "corpus": "patents"}], [3], [""],
                                     [{"q": "x"}] * 101])
def test_the_queries_that_found_the_documents_are_required_and_checked(client, queue, queries):
    response = post(client, documents=[EU], queries=queries, licenseAck=True)

    assert response.status_code == 400, response.get_json()
    assert response.get_json()["code"] == "bad_queries"
    assert queue.refs == []


def test_an_entry_and_documents_together_are_refused(client, queue):
    entry = create(client)

    response = post(client, dataset=entry["ref"], documents=[EU], queries=[QUERY], licenseAck=True)

    assert response.status_code == 400
    assert response.get_json()["code"] == "bad_request"
    assert queue.refs == []


def test_an_entry_import_with_documents_null_is_still_an_entry_import(client, queue):
    entry = create(client)

    response = post(client, dataset=entry["ref"], documents=None, licenseAck=True)

    assert response.status_code == 202, response.get_json()
    assert not response.get_json()["externalId"].startswith(ldh.SEARCH_PREFIX)


@pytest.mark.parametrize("ack", [None, False, "yes", 1])
def test_a_search_import_needs_the_terms_confirmation(client, queue, ack):
    body = {"documents": [EU], "queries": [QUERY]}
    if ack is not None:
        body["licenseAck"] = ack

    response = post(client, **body)

    assert response.status_code == 409
    assert response.get_json() == {"error": provider.NEEDS_CONFIRMATION, "code": "needs_ack"}
    assert queue.refs == []


def test_a_user_without_a_key_cannot_import_a_search(client, queue):
    response = post(client, headers=BOB, documents=[EU], queries=[QUERY], licenseAck=True)

    assert response.status_code == 409 and response.get_json()["code"] == "needs_key"
    assert queue.refs == []


@pytest.mark.parametrize("body, code", [
    ({"taskType": "astrology"}, "bad_task"),
    ({"name": "x" * 256}, "bad_name"),
])
def test_an_unknown_task_type_or_a_long_name_is_refused(client, queue, body, code):
    response = post(client, documents=[EU], queries=[QUERY], licenseAck=True, **body)

    assert response.status_code == 400 and response.get_json()["code"] == code
    assert queue.refs == []


def test_hidden_terms_refuse_a_search_import_and_fail_one_already_queued(client, app, hunter, tmp_path, queue):
    queued = start_search(client, [EU])
    settings(client, {"legal-data-hunter-terms": "hide"})

    refused = post(client, documents=[EU], queries=[QUERY], licenseAck=True)
    done, sink = work(app, tmp_path, queued["id"])

    assert refused.status_code == 403
    assert refused.get_json() == {"error": ldh.SEARCH_HIDDEN, "code": "license_hidden"}
    assert queue.refs == [queued["id"]]
    assert done["status"] == "failed" and done["code"] == "license_hidden"
    assert done["error"] == ldh.SEARCH_HIDDEN  # in the words of what the user chose: documents, not an entry
    assert hunter.documents_fetched() == [] and sink.calls == []


def test_without_the_key_seam_a_search_import_is_not_offered(make_app, hunter):
    app = make_app(enqueue=Queue())

    response = app.test_client().post(
        IMPORTS, json={"provider": "legal_data_hunter", "documents": [EU], "queries": [QUERY], "licenseAck": True},
        headers=ALICE,
    )

    assert response.status_code == 404 and response.get_json()["code"] == "not_offered"
    assert hunter.calls == []


# --- One user's imports side by side ----------------------------------------------------------------------------


def test_a_double_click_returns_the_running_import_and_other_selections_start_their_own(client, queue):
    first = start_search(client, [EU, *many(2)])
    again = start_search(client, [*many(2), EU], status=200)  # the same documents, ticked in another order
    other = start_search(client, many(3))
    single = start_search(client, [EU])

    assert again["id"] == first["id"]
    assert len({first["id"], other["id"], single["id"]}) == 3
    assert queue.refs == [first["id"], other["id"], single["id"]]


def test_a_search_import_never_collides_with_an_entry_import(client, queue):
    entry = create(client)
    from_entry = client.post(IMPORTS, json={"provider": "legal_data_hunter", "dataset": entry["ref"], "licenseAck": True},
                             headers=ALICE).get_json()

    from_search = start_search(client, ids(searched(client)["hits"]))

    assert from_search["id"] != from_entry["id"]
    assert queue.refs == [from_entry["id"], from_search["id"]]


# --- Out of credits, Try again ----------------------------------------------------------------------------------


def test_out_of_credits_pauses_a_search_import_and_try_again_resumes_without_refetching(client, app, hunter, tmp_path):
    found = searched(client)
    queued = start_search(client, ids(found["hits"]), queries=[found["query"]])
    hunter.calls.clear()
    hunter.credits = 2

    paused, sink = work(app, tmp_path, queued["id"])

    assert paused["status"] == "failed" and paused["code"] == "out_of_credits"
    assert paused["error"] == "Fetched 2 of 5 — your Legal Data Hunter credits ran out. Try again when they reset."
    assert sink.calls == [] and kept(app, queued["id"]) == 2

    hunter.calls.clear()
    hunter.credits = None
    retried = client.post(f"{IMPORTS}/{queued['id']}/retry", headers=ALICE)
    assert retried.status_code == 202, retried.get_json()
    again = retried.get_json()
    assert again["licenseAckAt"] == queued["licenseAckAt"]
    assert (again["documentCount"], again["name"], again["taskType"]) == (5, queued["name"], queued["taskType"])
    assert kept(app, queued["id"]) == 0 and kept(app, again["id"]) == 2
    done, sink = work(app, tmp_path, again["id"])

    assert done["status"] == "completed", done
    assert hunter.documents_fetched() == FIVE[2:]
    assert [row["country"] for row in sink.calls[0]["rows"]] == ["NL", "FR", "DE"]
    # The resumed import still says where it came from.
    record = provenance_of(client, done["datasetRef"])["legalDataHunter"]
    assert record["origin"] == "legal_data_hunter_search" and record["queries"] == [found["query"]]


# --- Provenance and publishing ----------------------------------------------------------------------------------


def test_provenance_records_a_search_its_queries_its_documents_and_the_confirmation(client, app, tmp_path):
    case_law = searched(client)
    legislation = searched(client, {"q": "data protection", "corpus": "legislation"})
    ticked = ids(case_law["hits"][:2]) + [EU]
    queued = start_search(client, ticked, queries=[case_law["query"], legislation["query"], "consumer credit"])
    done, _ = work(app, tmp_path, queued["id"])

    found = provenance_of(client, done["datasetRef"])

    assert found["provider"] == "legal_data_hunter" and found["source"] == "legal_data_hunter"
    assert found["license"]["id"] == "legal-data-hunter-terms"
    assert found["licenseAckAt"] == queued["licenseAckAt"] is not None
    assert found["legalDataHunter"] == {
        "origin": "legal_data_hunter_search",
        "originLabel": "Legal Data Hunter search",
        "entry": None,
        "queries": [case_law["query"], legislation["query"], {"q": "consumer credit"}],
        "documentIds": ticked,
        "documents": 3, "rows": 3, "skipped": 0, "missing": 0,
        "termsConfirmedAt": queued["licenseAckAt"],
    }


def test_a_curated_entrys_provenance_says_it_came_from_an_entry(client, app, tmp_path):
    entry = create(client)
    queued = client.post(IMPORTS, json={"provider": "legal_data_hunter", "dataset": entry["ref"], "licenseAck": True},
                         headers=ALICE).get_json()
    done, _ = work(app, tmp_path, queued["id"])

    found = provenance_of(client, done["datasetRef"])["legalDataHunter"]

    assert (found["origin"], found["originLabel"], found["entry"]) == ("legal_data_hunter_entry", "Curated entry", entry["ref"])
    assert "queries" not in found


def test_a_dataset_from_a_search_and_models_trained_on_it_cannot_be_published(client, app, tmp_path):
    done, _ = work(app, tmp_path, start_search(client, [EU])["id"])

    found = provenance_of(client, done["datasetRef"])

    assert found["publishable"] is False and found["publishRefusal"] == provider.DATASET_PUBLISH_REFUSED
    with app.app_context():
        session = app.db.session
        assert ldh.publish_refusal(session, done["datasetRef"]) == provider.DATASET_PUBLISH_REFUSED
        assert ldh.publish_refusal(session, done["datasetRef"], model=True) == provider.MODEL_PUBLISH_REFUSED


@pytest.mark.parametrize("sent", [7, 500, "abc"])
def test_a_query_is_stored_with_the_searchs_limit_whatever_limit_is_sent(client, app, sent):
    queued = start_search(client, [EU], queries=[{**QUERY, "limit": sent}])

    with app.app_context():
        assert app.db.session.get(LibraryImport, queued["id"]).queries[0]["limit"] == ldh.SEARCH_HITS


def test_the_queries_are_kept_on_the_import(client, app):
    queued = start_search(client, [EU], queries=[QUERY, QUERY, "x"])

    with app.app_context():
        record = app.db.session.get(LibraryImport, queued["id"])
        # Each once, with the limit the search route asked for (50), whatever limit was sent.
        assert record.queries == [provider.check_query({**QUERY, "limit": ldh.SEARCH_HITS}), {"q": "x"}]
        assert record.queries[0]["limit"] == 50
        assert record.documents == [EU]


# --- A row's source link (#15 review) ---------------------------------------------------------------------------


@pytest.mark.parametrize("url, kept_as", [
    ("javascript:alert(document.cookie)", ""),
    ("data:text/html,<script>alert(1)</script>", ""),
    ("//evil.example/x", ""),
    ("https://uitspraken.rechtspraak.nl/details?id=1", "https://uitspraken.rechtspraak.nl/details?id=1"),
])
def test_a_rows_source_url_is_kept_only_when_it_is_a_web_link(url, kept_as):
    assert provider.row({"text": "Some text.", "url": url})["source_url"] == kept_as


def test_an_imported_document_with_a_script_link_has_no_source_url(client, app, hunter, tmp_path, monkeypatch):
    import fake_legal_data_hunter

    documents = {**fake_legal_data_hunter.DOCUMENTS}
    documents["EU/eurlex/32016R0679"] = {**documents["EU/eurlex/32016R0679"], "url": "javascript:alert(1)"}
    monkeypatch.setattr(fake_legal_data_hunter, "DOCUMENTS", documents)

    done, sink = work(app, tmp_path, start_search(client, [EU])["id"])

    assert done["status"] == "completed"
    assert sink.calls[0]["rows"][0]["source_url"] == ""


def test_admin_can_import_a_search_with_their_own_key_like_any_user(client, app, hunter, tmp_path):
    queued = start_search(client, [EU], headers=ADMIN)
    hunter.calls.clear()

    done, _ = work(app, tmp_path, queued["id"], keys={"ada": "sk-admin-0000000000000000"})

    assert done["status"] == "completed"
    assert {call["key"] for call in hunter.calls} == {"sk-admin-0000000000000000"}
