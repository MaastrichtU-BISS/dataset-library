"""Legal Data Hunter's REST API (#126): search, and one document by its id.

Only ``https://legaldatahunter.com`` is ever called. A redirect is followed only on that origin; one to any other
host is refused before anything is sent there, so the user's key never leaves it. The key is the caller's: it is
sent as ``Authorization: Bearer sk-…`` and never logged, stored or put in an error.

Search is ``POST /v1/search`` (one corpus, its ``namespace``, per call; at most 100 hits), a document is
``GET /v1/documents/{country}/{source}?source_id=…`` (the query form, since a source id may hold slashes). Each
request costs one call of the user's plan; over the limit Legal Data Hunter answers 429 with ``Retry-After``.
"""
import re
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote, urljoin

import requests

from ..errors import ProviderError, ProviderRateLimited, ProviderUnavailable
from .community import origin_of

PROVIDER = "legal_data_hunter"
LABEL = "Legal Data Hunter"
ORIGIN = "https://legaldatahunter.com"
TIMEOUT = 30
MAX_REDIRECTS = 3
# An entry holds at most this many documents (decision L8): each costs one credit per import.
MAX_DOCUMENTS = 100
# The corpora a search may look in (decision L15), as Legal Data Hunter names them (`namespace`).
CORPORA = {"case_law": "Case law", "legislation": "Legislation", "doctrine": "Doctrine"}
DEFAULT_CORPUS = "case_law"
MAX_QUERY = 1000
MAX_FILTERS = 10
COUNTRY = re.compile(r"^[A-Za-z]{2,8}$")
SOURCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\- ]{0,99}$")
SOURCE_ID_MAX = 500
TITLE_MAX = 300

# What a student and an admin read (user-facing strings, exactly as #126 words them).
NEEDS_KEY = "Needs your Legal Data Hunter key — add it in Settings"
CONFIRMATION = (
    "Legal Data Hunter's terms forbid storing full text or training models unless your plan allows it. "
    "The free plan does not."
)
CONFIRMATION_CHECKBOX = "My Legal Data Hunter plan allows storing these documents and training models on them."
NEEDS_CONFIRMATION = "Tick the box to confirm your Legal Data Hunter plan allows storing these documents and training models on them."
BAD_KEY = "Legal Data Hunter refused your key. Check it in Settings."
BUSY = "Legal Data Hunter is busy. Try again in a few minutes."
CREDITS_RAN_OUT = "Your Legal Data Hunter credits ran out. Try again when they reset."
NOT_OFFERED = "Legal Data Hunter is not set up on this platform."
NOT_IN_LIBRARY = "That dataset is not in the library."
LICENSE_HIDDEN = "Legal Data Hunter's terms are not allowed on this platform, so this entry cannot be imported."
NO_DOCUMENTS = "This entry has no documents to import."
NO_TEXT = "None of these documents had any text, so there is nothing to import."
# Publishing to Zenodo is refused for a dataset built from Legal Data Hunter and for a model trained on one (L10).
DATASET_PUBLISH_REFUSED = (
    "This dataset was built from Legal Data Hunter, whose terms forbid redistributing its documents, "
    "so it cannot be published."
)
MODEL_PUBLISH_REFUSED = (
    "This model was trained on a dataset built from Legal Data Hunter, whose terms forbid redistributing its documents "
    "and models trained on them, so it cannot be published."
)

# The columns of an imported row, in the order the dataset's CSV has them (decision L6).
ROW_COLUMNS = ("text", "court", "date", "country", "language", "ecli", "eli", "source_url", "text_truncated")


def credits(count):
    """What an import of `count` documents costs, as the card says it."""
    return f"uses {count} of your Legal Data Hunter credits"


def out_of_credits(fetched, total):
    """Why an import paused when the user's credits ran out, after `fetched` of its `total` documents (L12)."""
    return f"Fetched {fetched} of {total} — your Legal Data Hunter credits ran out. Try again when they reset."


class BadKey(ProviderError):
    """Legal Data Hunter refused the key (401 or 403)."""


class OutOfCredits(ProviderRateLimited):
    """The key's calls ran out (429). `retry_after` is how many seconds Legal Data Hunter said to wait, or None."""

    def __init__(self, retry_after=None):
        super().__init__("Legal Data Hunter's limit was reached")
        self.retry_after = retry_after


class OffOrigin(ProviderUnavailable):
    """Legal Data Hunter redirected to another host, which is never called."""


class DocumentNotFound(ProviderError):
    """Legal Data Hunter no longer has this document (404): its source may have been removed."""


def _request(method, path, key, *, params=None, body=None):
    """One call to Legal Data Hunter, as JSON. Redirects are followed by hand, and only on its origin."""
    if not isinstance(key, str) or not key.strip():
        raise BadKey("No key")
    url = ORIGIN + path
    headers = {"Authorization": f"Bearer {key.strip()}", "Accept": "application/json"}
    for _ in range(MAX_REDIRECTS + 1):
        try:
            response = requests.request(
                method, url, params=params, json=body, headers=headers, timeout=TIMEOUT, allow_redirects=False,
            )
        except requests.RequestException as failed:
            raise ProviderUnavailable(f"Legal Data Hunter did not answer: {type(failed).__name__}") from None
        status = response.status_code
        if status in (301, 302, 303, 307, 308):
            target = urljoin(url, response.headers.get("Location") or "")
            if origin_of(target) != ORIGIN:
                raise OffOrigin("Legal Data Hunter redirected to another host")
            if status == 303 or (status in (301, 302) and method != "GET"):
                method, body = "GET", None
            url, params = target, None
            continue
        if status in (401, 403):
            raise BadKey(f"Legal Data Hunter answered {status}")
        if status == 404:
            raise DocumentNotFound("Legal Data Hunter answered 404")
        if status == 429:
            raise OutOfCredits(retry_after(response.headers.get("Retry-After")))
        if status != 200:
            raise ProviderUnavailable(f"Legal Data Hunter answered {status}")
        try:
            found = response.json()
        except ValueError:
            raise ProviderUnavailable("Legal Data Hunter answered with something unreadable") from None
        if not isinstance(found, dict):
            raise ProviderUnavailable("Legal Data Hunter answered with something unreadable")
        return found
    raise ProviderUnavailable("Legal Data Hunter redirected too many times")


def retry_after(value, now=None):
    """Seconds to wait from a Retry-After header: a number of seconds or an HTTP date. None when it says neither."""
    text = str(value or "").strip()
    if text.isdigit():
        return int(text)
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0, int((when - (now or datetime.now(timezone.utc))).total_seconds()))


class InvalidQuery(Exception):
    """A query with fields to fix: `fields` maps each field to what is wrong with it."""

    def __init__(self, fields):
        super().__init__("Invalid query")
        self.fields = fields


def _list_of(value, pattern, name, fields, upper=False):
    if value in (None, "", []):
        return []
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or len(items) > MAX_FILTERS or not all(isinstance(i, str) and pattern.match(i.strip()) for i in items):
        fields[name] = f"Give up to {MAX_FILTERS} {name} codes."
        return []
    cleaned = [i.strip().upper() if upper else i.strip() for i in items]
    return list(dict.fromkeys(cleaned))


def _day(value, name, fields):
    if value in (None, ""):
        return None
    try:
        if isinstance(value, str) and len(value) == 10:
            return date.fromisoformat(value).isoformat()
    except ValueError:
        pass
    fields[name] = "Give a date as YYYY-MM-DD."
    return None


def check_query(body):
    """The search an admin asked for, cleaned: {q, corpus, country, source, dateFrom, dateTo, limit}.
    Raises InvalidQuery with the fields to fix."""
    if not isinstance(body, dict):
        raise InvalidQuery({"query": "Send the search as an object."})
    fields = {}
    q = body.get("q")
    q = q.strip() if isinstance(q, str) else ""
    if not q or len(q) > MAX_QUERY:
        fields["q"] = f"Type what to search for, up to {MAX_QUERY} characters."
    corpus = body.get("corpus") or DEFAULT_CORPUS
    if corpus not in CORPORA:
        fields["corpus"] = "Choose case law, legislation or doctrine."
    country = _list_of(body.get("country"), COUNTRY, "country", fields, upper=True)
    source = _list_of(body.get("source"), SOURCE, "source", fields)
    date_from = _day(body.get("dateFrom"), "dateFrom", fields)
    date_to = _day(body.get("dateTo"), "dateTo", fields)
    if date_from and date_to and date_from > date_to:
        fields["dateTo"] = "The end date is before the start date."
    limit = body.get("limit", MAX_DOCUMENTS)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_DOCUMENTS:
        fields["limit"] = f"Choose from 1 to {MAX_DOCUMENTS} documents."
    if fields:
        raise InvalidQuery(fields)
    return {"q": q, "corpus": corpus, "country": country, "source": source, "dateFrom": date_from, "dateTo": date_to, "limit": limit}


def search(key, query):
    """The hits of a checked query (`check_query`), at most its limit, as {hits, total}. One call of the key's plan."""
    body = {"q": query["q"], "namespace": query["corpus"], "top_k": query["limit"], "result_detail": "snippet"}
    if query.get("country"):
        body["country"] = query["country"]
    if query.get("source"):
        body["source"] = query["source"]
    if query.get("dateFrom"):
        body["date_start"] = query["dateFrom"]
    if query.get("dateTo"):
        body["date_end"] = query["dateTo"]
    found = _request("POST", "/v1/search", key, body=body)
    raw = found.get("hits")
    if not isinstance(raw, list):
        raise ProviderUnavailable("Legal Data Hunter answered with something unreadable")
    hits, seen = [], set()
    for item in raw:
        hit = _hit(item)
        if hit is None or (hit["country"], hit["source"], hit["sourceId"]) in seen:
            continue
        seen.add((hit["country"], hit["source"], hit["sourceId"]))
        hits.append(hit)
        if len(hits) == query["limit"]:
            break
    total = found.get("total_hits")
    return {"hits": hits, "total": total if isinstance(total, int) and not isinstance(total, bool) else len(hits)}


def _text(value, limit=None):
    if value is None or isinstance(value, (dict, list)):
        return None
    text = str(value).strip()
    return (text[:limit] if limit else text) or None


def _hit(item):
    """A search hit as the library shows it, or None when it names no document that can be fetched."""
    if not isinstance(item, dict):
        return None
    ref = document_ref({"country": item.get("country"), "source": item.get("source"), "sourceId": item.get("source_id")})
    if ref is None:
        return None
    return {
        **ref,
        "title": _text(item.get("title"), TITLE_MAX),
        "court": _text(item.get("court")),
        "date": _text(item.get("date")),
        "ecli": _text(item.get("ecli")),
        "language": _text(item.get("language")),
        "url": _text(item.get("url")),
        "snippet": _text(item.get("snippet"), 600),
    }


def document_ref(value):
    """A document's id as an entry keeps it, {country, source, sourceId, title}, or None when it is not one.

    Legal Data Hunter may name a source with its country (`FR/Judilibre`); the country is then taken from it."""
    if not isinstance(value, dict):
        return None
    country, source, source_id = value.get("country"), value.get("source"), value.get("sourceId")
    if isinstance(source, str) and "/" in source:
        prefix, _, source = source.partition("/")
        country = country or prefix
    if not (isinstance(country, str) and COUNTRY.match(country.strip())):
        return None
    if not (isinstance(source, str) and SOURCE.match(source.strip()) and source.strip() not in (".", "..")):
        return None
    if isinstance(source_id, int) and not isinstance(source_id, bool):
        source_id = str(source_id)
    if not isinstance(source_id, str) or not source_id.strip() or len(source_id) > SOURCE_ID_MAX:
        return None
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in source_id):
        return None
    ref = {"country": country.strip().upper(), "source": source.strip(), "sourceId": source_id.strip()}
    title = value.get("title")
    ref["title"] = _text(title, TITLE_MAX) if isinstance(title, str) else None
    return ref


def document(key, ref):
    """One document's JSON, by its id. Raises DocumentNotFound when Legal Data Hunter no longer has it."""
    path = f"/v1/documents/{quote(ref['country'], safe='')}/{quote(ref['source'], safe='')}"
    return _request("GET", path, key, params={"source_id": ref["sourceId"]})


def row(found, ref=None):
    """A dataset row from a document (decision L6): its text and where it is from. None when it has no text."""
    text = found.get("text")
    if found.get("text_unavailable") or not isinstance(text, str) or not text.strip():
        return None
    meta = found.get("meta") if isinstance(found.get("meta"), dict) else {}
    eli = found.get("eli_uri") or found.get("eli") or meta.get("eli_uri") or meta.get("eli")
    return {
        "text": text,
        "court": _text(found.get("court")) or "",
        "date": _text(found.get("date")) or "",
        "country": _text(found.get("country")) or (ref or {}).get("country") or "",
        "language": _text(found.get("language")) or "",
        "ecli": _text(found.get("ecli")) or "",
        "eli": _text(eli) or "",
        "source_url": _text(found.get("url")) or "",
        "text_truncated": bool(found.get("text_truncated")),
    }
