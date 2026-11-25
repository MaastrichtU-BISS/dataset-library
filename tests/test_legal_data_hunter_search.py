"""Legal Data Hunter search for every signed-in user (#15): `POST /api/library/legal-data-hunter/search` searches with
the caller's own key, asks for 50 hits, and turns every refusal into a stable code. Legal Data Hunter is faked from
recorded responses (fake_legal_data_hunter.py); no test reaches the network."""
import copy

import pytest

import fake_legal_data_hunter
from conftest import ADMIN, ALICE
from dataset_library.providers.legal_data_hunter import MAX_RETRY_AFTER
from fake_legal_data_hunter import ADMIN_KEY, STUDENT_KEY
# The fixtures the curated-entry tests use: the faked Legal Data Hunter, the users' keys and an app with the key seam.
from test_legal_data_hunter import BOB, QUERY, app, hunter, keys, queue, settings  # noqa: F401

SEARCH = "/api/library/legal-data-hunter/search"
CAROL = {"X-User": "carol"}  # a student with her own key


def search(client, body=None, headers=ALICE):
    return client.post(SEARCH, json=QUERY if body is None else body, headers=headers)


def test_a_student_searches_with_their_own_key_and_the_filters(client, hunter):
    response = search(client, {**QUERY, "source": ["rechtspraak", "FR/Judilibre"]})

    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    [call] = hunter.calls
    assert call["origin"] == "https://legaldatahunter.com" and call["path"] == "/v1/search"
    assert call["key"] == STUDENT_KEY
    assert call["json"] == {
        "q": "unfair contract terms consumer", "namespace": "case_law", "top_k": 50, "result_detail": "snippet",
        "country": ["NL", "FR", "DE"], "source": ["rechtspraak", "FR/Judilibre"],
        "date_start": "2020-01-01", "date_end": "2024-12-31",
    }
    assert [hit["sourceId"] for hit in body["hits"]] == [
        "ECLI:NL:HR:2023:1234", "6543a1b2c3d4e5f6", "1 BvR 123/20", "ECLI:NL:RBAMS:2022:999", "ECLI:NL:GHARL:2021:42",
    ]
    assert set(body["hits"][0]) == {
        "country", "source", "sourceId", "title", "court", "date", "ecli", "language", "url", "snippet",
    }
    assert body["total"] == 1287 and body["max"] == 50
    assert body["query"]["q"] == "unfair contract terms consumer" and body["query"]["limit"] == 50
    assert STUDENT_KEY not in response.get_data(as_text=True)


def test_each_user_searches_with_their_own_key_never_anothers(client, hunter, keys):
    keys["carol"] = "sk-carol-2222222222222222"
    hunter.keys.add("sk-carol-2222222222222222")

    assert search(client, headers=CAROL).status_code == 200
    assert search(client, headers=ADMIN).status_code == 200
    assert search(client, headers=BOB).get_json()["code"] == "needs_key"

    assert [call["key"] for call in hunter.calls] == ["sk-carol-2222222222222222", ADMIN_KEY]


def test_signing_in_is_required(client, hunter):
    response = client.post(SEARCH, json=QUERY)

    assert response.status_code == 401
    assert hunter.calls == []


def test_at_most_50_hits_are_asked_for_whatever_the_body_says(client, hunter, monkeypatch):
    many = copy.deepcopy(fake_legal_data_hunter.SEARCHES)
    first = many["case_law"]["hits"][0]
    many["case_law"]["hits"] = [{**first, "source_id": f"ECLI:NL:HR:2023:{n}"} for n in range(80)]
    monkeypatch.setattr(fake_legal_data_hunter, "SEARCHES", many)

    response = search(client, {**QUERY, "limit": 100})

    assert response.status_code == 200, response.get_json()
    assert hunter.calls[-1]["json"]["top_k"] == 50
    assert len(response.get_json()["hits"]) == 50


@pytest.mark.parametrize("corpus", ["case_law", "legislation", "doctrine"])
def test_a_search_looks_in_one_corpus(client, hunter, corpus):
    response = search(client, {"q": "data", "corpus": corpus})

    assert response.status_code == 200
    assert hunter.calls[-1]["json"]["namespace"] == corpus


@pytest.mark.parametrize("body, field", [
    ({"q": ""}, "query.q"),
    ({"q": "x", "corpus": "patents"}, "query.corpus"),
    ({"q": "x", "dateFrom": "2024-13-01"}, "query.dateFrom"),
    ({"q": "x", "dateFrom": "2024-02-01", "dateTo": "2024-01-01"}, "query.dateTo"),
    ({"q": "x", "country": ["N L"]}, "query.country"),
    ({"q": "x", "source": ["FR/CNIL/x"]}, "query.source"),
])
def test_a_bad_filter_is_refused_without_a_call(client, hunter, body, field):
    response = search(client, body)

    assert response.status_code == 400
    assert response.get_json()["code"] == "invalid"
    assert field in response.get_json()["fields"]
    assert hunter.calls == []


def test_a_body_that_is_not_an_object_is_refused(client, hunter):
    response = client.post(SEARCH, json=["unfair terms"], headers=ALICE)

    assert response.status_code == 400 and response.get_json()["code"] == "bad_request"
    assert hunter.calls == []


def test_a_user_without_a_key_is_told_to_add_one(client, hunter):
    response = search(client, headers=BOB)

    assert response.status_code == 409
    assert response.get_json() == {"error": "Needs your Legal Data Hunter key — add it in Settings", "code": "needs_key"}
    assert hunter.calls == []


def test_a_rejected_key_is_its_own_refusal_and_never_echoed(client, hunter, keys):
    keys["alice"] = "sk-revoked-3333333333333333"

    response = search(client)

    assert response.status_code == 409
    assert response.get_json() == {"error": "Legal Data Hunter refused your key. Check it in Settings.", "code": "bad_key"}
    assert "sk-revoked" not in response.get_data(as_text=True)


def test_an_unreachable_legal_data_hunter_is_its_own_refusal(client, hunter):
    hunter.busy = True

    response = search(client)

    assert response.status_code == 503
    assert response.get_json() == {"error": "Legal Data Hunter is busy. Try again in a few minutes.", "code": "provider_unavailable"}
    assert STUDENT_KEY not in response.get_data(as_text=True)


def test_a_redirect_off_origin_is_unreachable_and_nothing_goes_there(client, hunter):
    hunter.redirects["/v1/search"] = "https://evil.example/v1/search"

    response = search(client)

    assert response.status_code == 503 and response.get_json()["code"] == "provider_unavailable"
    assert {call["origin"] for call in hunter.calls} == {"https://legaldatahunter.com"}


def test_spent_credits_pass_the_retry_after_through(client, hunter):
    hunter.credits = 0

    response = search(client)

    assert response.status_code == 429
    assert response.get_json() == {
        "error": "Your Legal Data Hunter credits ran out. Try again when they reset.", "code": "out_of_credits",
        "retryAfter": 3600,
    }
    assert response.headers["Retry-After"] == "3600"


def test_the_per_minute_cap_asks_to_slow_down_with_its_retry_after(client, hunter):
    hunter.minute_limited = 1

    response = search(client)

    assert response.status_code == 429
    assert response.get_json() == {
        "error": "Legal Data Hunter asked us to slow down. Try again in a minute.", "code": "slow_down", "retryAfter": 7,
    }
    assert response.headers["Retry-After"] == "7"


def test_a_429_without_a_retry_after_says_so_with_null(client, hunter):
    hunter.credits, hunter.retry_after = 0, ""

    response = search(client)

    assert response.status_code == 429
    assert response.get_json()["retryAfter"] is None
    assert "Retry-After" not in response.headers


def test_hidden_terms_refuse_the_search_as_if_legal_data_hunter_were_not_there(client, hunter, make_app):
    assert settings(client, {"legal-data-hunter-terms": "hide"}).status_code == 200

    hidden = search(client)
    hidden_bad_body = client.post(SEARCH, json=["x"], headers=ALICE)
    hidden_no_key = search(client, headers=BOB)
    absent = make_app().test_client().post(SEARCH, json=QUERY, headers=ALICE)

    for response in (hidden, hidden_bad_body, hidden_no_key, absent):
        assert response.status_code == 404
        assert response.get_json() == {"error": "Legal Data Hunter is not set up on this platform.", "code": "not_offered"}
    assert hunter.calls == []


def test_hidden_terms_leave_the_admin_preview_working(client, hunter):
    settings(client, {"legal-data-hunter-terms": "hide"})

    response = client.post("/api/library/admin/legal-data-hunter/preview", json=QUERY, headers=ADMIN)

    assert response.status_code == 200, response.get_json()


@pytest.mark.parametrize("given, header", [
    ("9" * 5000, str(MAX_RETRY_AFTER)),
    ("Wed, 21 Oct 2099 07:28:00 GMT", str(MAX_RETRY_AFTER)),
    ("Wed, 21 Oct 2015 07:28:00 GMT", "0"),
])
def test_the_retry_after_header_is_the_parsed_capped_seconds_never_the_raw_value(client, hunter, given, header):
    hunter.credits, hunter.retry_after = 0, given

    response = search(client)

    assert response.status_code == 429
    assert response.headers["Retry-After"] == header
    assert response.get_json()["retryAfter"] == int(header)


@pytest.mark.parametrize("url", [
    "javascript:fetch('/api/x',{credentials:'include'})", "JaVaScRiPt:alert(1)", "data:text/html,<script>x</script>",
    "//evil.example/x", "vbscript:x", "http://[bad",
])
def test_a_hit_link_that_is_not_http_or_https_is_dropped(client, hunter, monkeypatch, url):
    tampered = copy.deepcopy(fake_legal_data_hunter.SEARCHES)
    tampered["case_law"]["hits"][0]["url"] = url
    monkeypatch.setattr(fake_legal_data_hunter, "SEARCHES", tampered)

    hits = search(client).get_json()["hits"]

    assert hits[0]["url"] is None
    assert hits[1]["url"].startswith("https://")


def test_the_search_works_again_once_the_terms_are_warn(client, hunter):
    settings(client, {"legal-data-hunter-terms": "hide"})
    settings(client, {"legal-data-hunter-terms": "warn"})

    assert search(client).status_code == 200
