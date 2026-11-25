"""Legal Data Hunter entries for a host (#126): what a course admin curates, the featured cards students see,
and importing an entry with the student's own key.

Legal Data Hunter has documents, not datasets. An admin searches it with their own key, previews the hits, and
saves up to 100 of them as an entry: the query and the frozen document ids. Refresh runs the query again and
replaces the ids; an import copies the ids when it starts, so a Refresh never changes an import or a dataset.

Its terms forbid storing full text, training models and redistribution unless the user's plan allows it, and
the free plan does not. So an import always needs the student's confirmation, whose time is its provenance; the
"Legal Data Hunter terms" entry of the license table is Warn or Hide, never Allow; and a dataset built from it,
or a model trained on one, is never published (the host refuses, with `DATASET_PUBLISH_REFUSED` or
`MODEL_PUBLISH_REFUSED`).

The package never keeps a key. The host gives one seam, ``legal_data_hunter_key(owner_id)``, which returns that
user's key or None; it is asked when the key is needed, by the request or by the worker.
"""
import hashlib
import logging
import json
import time
import uuid
from collections import namedtuple
from datetime import timedelta

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from . import imports as imports_module
from . import limits as limits_module
from . import policy as policy_module
from .errors import ProviderError
from .licenses import LEGAL_DATA_HUNTER_TERMS
from .models import LibraryImport, LibraryLegalDataHunterDocument, LibraryLegalDataHunterEntry, LibrarySettings
from .providers import legal_data_hunter as provider
from .providers.legal_data_hunter import (  # noqa: F401 - the host's names for these come from here
    BAD_KEY,
    CONFIRMATION,
    CONFIRMATION_CHECKBOX,
    DATASET_PUBLISH_REFUSED,
    LABEL,
    MAX_DOCUMENTS,
    MODEL_PUBLISH_REFUSED,
    NEEDS_KEY,
    SEARCH_HIDDEN,
    PROVIDER,
    BadKey,
    DocumentNotFound,
    DocumentRejected,
    InvalidQuery,
    OutOfCredits,
    credits,
    out_of_credits,
)

logger = logging.getLogger("dataset_library")

NO_HITS = "This search found no documents."
MAX_TITLE = 120
MAX_DESCRIPTION = 2000
# A paused import's fetched documents are kept this long for Try again; the free plan's day limit resets well within it.
# An import paused as out of credits is also kept until a day after its credits reset (`resume_after`).
KEEP_FETCHED = imports_module.KEEP_FETCHED
# The failures Try again resumes from: their fetched documents are kept. Any other failure, and a cancel, discards them.
RESUMABLE = imports_module.RESUMABLE
# Every plan has a per-minute cap (60 a minute on the free plan), which a 100-document import can reach. A 429 for it
# is waited out, never shown as credits running out; these bound the wait so an import cannot hang on it.
MINUTE_WAIT_SECONDS = 60
MINUTE_WAITS = 3
FETCHED, NO_TEXT, MISSING = "fetched", "no_text", "missing"
TERMS = LEGAL_DATA_HUNTER_TERMS
# A document an import kept: its outcome, its row (None unless fetched) and its response's size in bytes.
_Kept = namedtuple("_Kept", "outcome row nbytes")


class NotOffered(Exception):
    """The host gave no key seam, so Legal Data Hunter is not offered."""


class NeedsKey(Exception):
    """The signed-in user has no Legal Data Hunter key."""


class NotFound(Exception):
    pass


class Invalid(Exception):
    def __init__(self, fields):
        super().__init__("Invalid entry")
        self.fields = fields


def offered(host):
    return getattr(host, "legal_data_hunter_key", None) is not None


def key_for(host, owner_id):
    """The user's key from the host's seam, or None. Never stored, logged or returned to a browser."""
    if not offered(host) or owner_id is None:
        return None
    try:
        key = host.legal_data_hunter_key(owner_id)
    except Exception as failed:  # noqa: BLE001 - a broken seam is no key, never a broken page
        # Only the exception's type: its message could hold the key.
        logger.warning("library.ldh.key_seam_failed error=%s", type(failed).__name__)
        return None
    return key.strip() if isinstance(key, str) and key.strip() else None


def has_key(host, owner_id):
    return key_for(host, owner_id) is not None


def _level(host):
    return policy_module.level(TERMS, policy_module.overrides(host.db))


def _admin_key(host):
    if not offered(host):
        raise NotOffered()
    key = key_for(host, host.current_owner())
    if key is None:
        raise NeedsKey()
    return key


# --- What the admin curates -------------------------------------------------------------------------------------


def preview(host, body):
    """The hits of the admin's search, with the admin's own key: {query, hits, total, max}. Raises InvalidQuery,
    NeedsKey, NotOffered, or the provider's errors (BadKey, OutOfCredits, ProviderError)."""
    key = _admin_key(host)
    query = provider.check_query(body)
    found = provider.search(key, query)
    return {"query": query, "hits": found["hits"], "total": found["total"], "max": MAX_DOCUMENTS}


# --- Legal Data Hunter search, for any signed-in user (#15) -------------------------------------------------------

# A user's search asks for this many hits, whatever the request says: one page, one call of the user's plan.
SEARCH_HITS = 50


def searchable(host):
    """Whether a user may search Legal Data Hunter here: the host gives the key seam and the admin has not set the
    terms to Hide. When not, the search is refused as if Legal Data Hunter were not there."""
    return offered(host) and _level(host) != "hide"


def search(host, body):
    """The hits of a user's search, with that user's own key: {query, hits, total, max}. Any `limit` in the body is
    ignored: SEARCH_HITS are asked for. Raises NotOffered (also when the terms are Hide), NeedsKey, InvalidQuery, or
    the provider's errors (BadKey, OutOfCredits, ProviderError)."""
    if not searchable(host):
        raise NotOffered()
    key = key_for(host, host.current_owner())
    if key is None:
        raise NeedsKey()
    query = provider.check_query({**body, "limit": SEARCH_HITS})
    found = provider.search(key, query)
    return {"query": query, "hits": found["hits"], "total": found["total"], "max": SEARCH_HITS}


def list_all(host):
    """Every entry, newest first, for the admin's settings section, with its documents."""
    found = host.db.session.query(LibraryLegalDataHunterEntry).order_by(
        LibraryLegalDataHunterEntry.created_at.desc(), LibraryLegalDataHunterEntry.id
    )
    return [to_dict(entry) for entry in found]


def get(host, entry_ref):
    entry = host.db.session.get(LibraryLegalDataHunterEntry, entry_ref)
    if entry is None:
        raise NotFound()
    return to_dict(entry)


def _fields(host, body, *, creating):
    """The entry's own fields from a body: title, description and task type. Field errors go in `fields`."""
    fields, values = {}, {}
    if creating or "title" in body:
        title = body.get("title")
        title = title.strip() if isinstance(title, str) else ""
        if not title or len(title) > MAX_TITLE:
            fields["title"] = f"Give the entry a title of up to {MAX_TITLE} characters."
        values["title"] = title
    if creating or "description" in body:
        description = body.get("description")
        if description is not None and not isinstance(description, str):
            fields["description"] = "The description must be text."
        else:
            description = (description or "").strip() or None
            if description and len(description) > MAX_DESCRIPTION:
                fields["description"] = f"Keep the description under {MAX_DESCRIPTION + 1} characters."
            values["description"] = description
    if creating or "taskType" in body:
        task_type = body.get("taskType") or "classification"
        if task_type not in host.task_types:
            fields["taskType"] = f"Unknown task type: {task_type}"
        values["task_type"] = task_type
    return fields, values


def _documents(value, fields):
    """The admin's chosen documents, checked: at most 100 well-formed ids, each once."""
    if not isinstance(value, list):
        fields["documents"] = "Send the documents as a list."
        return []
    if len(value) > MAX_DOCUMENTS:
        fields["documents"] = f"An entry holds at most {MAX_DOCUMENTS} documents."
        return []
    documents, seen = [], set()
    for index, item in enumerate(value):
        ref = provider.document_ref(item)
        if ref is None:
            fields[f"documents.{index}"] = "Not a Legal Data Hunter document id."
            continue
        identity = (ref["country"], ref["source"], ref["sourceId"])
        if identity not in seen:
            seen.add(identity)
            documents.append(ref)
    if not documents and "documents" not in fields and not any(k.startswith("documents.") for k in fields):
        fields["documents"] = "Choose at least one document."
    return documents


def _refs(hits):
    return [{key: hit[key] for key in ("country", "source", "sourceId", "title")} for hit in hits][:MAX_DOCUMENTS]


def create(host, body):
    """Save an entry: its title, description, task type (classification by default), query, and documents.

    `documents` are the hits the admin chose from the preview, at most 100. Without them, the query is run with the
    admin's key and its hits are kept. Returns the entry."""
    if not offered(host):
        raise NotOffered()
    if not isinstance(body, dict):
        raise Invalid({"body": "Send a JSON object."})
    fields, values = _fields(host, body, creating=True)
    try:
        query = provider.check_query(body.get("query"))
    except InvalidQuery as invalid:
        fields.update({f"query.{name}": message for name, message in invalid.fields.items()})
        query = None
    documents = _documents(body["documents"], fields) if body.get("documents") is not None else None
    if fields:
        raise Invalid(fields)
    if documents is None:
        documents = _refs(provider.search(_admin_key(host), query)["hits"])
        if not documents:
            raise Invalid({"documents": NO_HITS})
    else:
        # The admin's choice is the entry's size: a Refresh keeps it, never growing it to the query's 100.
        query = {**query, "limit": len(documents)}
    entry = LibraryLegalDataHunterEntry(
        **values, query=query, documents=documents, created_by=host.current_owner(),
    )
    host.db.session.add(entry)
    host.db.session.commit()
    host.audit("library.ldh_entry_created", documents=len(documents))
    return to_dict(entry)


def update(host, entry_ref, body):
    """Change an entry's title, description or task type. Its query and documents change only by Refresh."""
    if not isinstance(body, dict):
        raise Invalid({"body": "Send a JSON object."})
    entry = host.db.session.get(LibraryLegalDataHunterEntry, entry_ref)
    if entry is None:
        raise NotFound()
    fields, values = _fields(host, body, creating=False)
    if fields:
        raise Invalid(fields)
    changed = sorted(name for name, value in values.items() if getattr(entry, name) != value)
    for name in changed:
        setattr(entry, name, values[name])
    if changed:
        host.db.session.commit()
        host.audit("library.ldh_entry_updated", fields=changed)
    return to_dict(entry)


def refresh(host, entry_ref):
    """Run the entry's query again with the admin's key and replace its documents. Imports already started, and the
    datasets they made, keep the ids they copied."""
    entry = host.db.session.get(LibraryLegalDataHunterEntry, entry_ref)
    if entry is None:
        raise NotFound()
    documents = _refs(provider.search(_admin_key(host), provider.check_query(entry.query))["hits"])
    if not documents:
        # An empty search never empties an entry: it keeps the documents it has.
        raise Invalid({"documents": NO_HITS})
    before = len(entry.documents or [])
    entry.documents = documents
    entry.refreshed_at = imports_module._now()
    host.db.session.commit()
    host.audit("library.ldh_entry_refreshed", documents=len(documents), before=before)
    return to_dict(entry)


def remove(host, entry_ref):
    entry = host.db.session.get(LibraryLegalDataHunterEntry, entry_ref)
    if entry is None:
        raise NotFound()
    host.db.session.delete(entry)
    host.db.session.commit()
    host.audit("library.ldh_entry_removed", documents=len(entry.documents or []))


def to_dict(entry):
    """An entry as the admin's settings section shows it."""
    return {
        "ref": entry.id,
        "provider": PROVIDER,
        "title": entry.title,
        "description": entry.description,
        "taskType": entry.task_type,
        "query": entry.query,
        "documents": entry.documents,
        "documentCount": len(entry.documents or []),
        "createdAt": imports_module._iso(entry.created_at),
        "updatedAt": imports_module._iso(entry.updated_at),
        "refreshedAt": imports_module._iso(entry.refreshed_at),
    }


# --- What a student sees ----------------------------------------------------------------------------------------


def cards(host, owner_id, *, keyed=None):
    """The entries as featured library cards, oldest first; none when Legal Data Hunter is not offered or an admin
    hid its terms. Each says whether the signed-in user can import it, and what it costs them. `keyed` is whether
    the user has a key, when the caller already asked."""
    if not offered(host):
        return []
    level = _level(host)
    if level == "hide":
        return []
    keyed = has_key(host, owner_id) if keyed is None else keyed
    cap = host.row_cap()
    found = host.db.session.query(LibraryLegalDataHunterEntry).order_by(
        LibraryLegalDataHunterEntry.created_at, LibraryLegalDataHunterEntry.id
    )
    return [card(entry, level=level, keyed=keyed, cap=cap) for entry in found]


def view(host, owner_id, entry_ref):
    """One entry's card, for the import's steps. Raises NotFound when it is not in the library."""
    if not offered(host):
        raise NotFound()
    level = _level(host)
    entry = host.db.session.get(LibraryLegalDataHunterEntry, entry_ref)
    if entry is None or level == "hide":
        raise NotFound()
    return card(entry, level=level, keyed=has_key(host, owner_id), cap=host.row_cap())


def card(entry, *, level, keyed, cap):
    """A featured card: labelled Legal Data Hunter, with its document count and the credits an import uses (one per
    document, up to the row cap). Without a key, Import is disabled and `reason` says why."""
    count = len(entry.documents or [])
    cost = min(count, cap) if cap else count
    reason_code, reason = None, None
    if not keyed:
        reason_code, reason = "needs_key", NEEDS_KEY
    elif count == 0:
        reason_code, reason = "no_documents", provider.NO_DOCUMENTS
    return {
        "id": f"{PROVIDER}:{entry.id}",
        "provider": PROVIDER,
        "providerLabel": LABEL,
        "externalId": entry.id,
        "title": entry.title,
        "description": entry.description,
        "taskType": entry.task_type,
        "tasks": [entry.task_type],
        "featured": True,
        "url": provider.ORIGIN,
        "corpus": (entry.query or {}).get("corpus"),
        "documentCount": count,
        "credits": cost,
        "creditCost": credits(cost),
        "refreshedAt": imports_module._iso(entry.refreshed_at),
        "license": TERMS.at_level(level),
        "hasKey": keyed,
        "importable": reason is None,
        "reasonCode": reason_code,
        "reason": reason,
        "confirmation": {"text": CONFIRMATION, "checkbox": CONFIRMATION_CHECKBOX},
        "publishable": False,
        "publishRefusal": DATASET_PUBLISH_REFUSED,
        "citation": _citation(entry.title),
    }


def _citation(title):
    return {
        "title": title, "authors": [{"literal": LABEL}], "year": None, "url": provider.ORIGIN,
        "revision": None, "accessedAt": None, "bibtex": None,
    }


# --- Importing --------------------------------------------------------------------------------------------------


# A search import (#16) is not of an entry: its `external_id` is this prefix and a digest of its documents.
SEARCH_PREFIX = "search:"
# Where a Legal Data Hunter import's documents were chosen, as its provenance says.
ENTRY_ORIGIN, SEARCH_ORIGIN = "legal_data_hunter_entry", "legal_data_hunter_search"
ORIGIN_LABELS = {ENTRY_ORIGIN: "Curated entry", SEARCH_ORIGIN: "Legal Data Hunter search"}
# The most document ids a search import's request may carry, duplicates included; at most MAX_DOCUMENTS are unique.
MAX_SENT = 10 * MAX_DOCUMENTS


def is_search(record):
    """Whether an import is of a user's own Legal Data Hunter search (#16), not of a curated entry."""
    return (record.external_id or "").startswith(SEARCH_PREFIX)


def hidden_message(record):
    """Why an import is refused while the terms are Hide, in the words of what the user chose."""
    return provider.SEARCH_HIDDEN if is_search(record) else provider.LICENSE_HIDDEN


def origin(record):
    return SEARCH_ORIGIN if is_search(record) else ENTRY_ORIGIN


def start(host, owner_id, body, *, ack_at=None, previous=None):
    """Queue an import of an entry, or of the documents a user ticked across their own searches (#16): one row per
    document, fetched by the worker with the user's key.

    An entry import names the entry (`dataset`). A search import sends `documents` instead (each a document id,
    {country, source, sourceId, title?}; duplicates are dropped, keeping the first, and at most 100 may remain), the
    `queries` that found them (1 to 100, each a checked query or its text), and a `taskType` (classification by
    default); its name defaults to its first query's text. Sending both is refused.

    The user must confirm (`licenseAck: true`) every time; a retry carries its import's confirmation time (`ack_at`).
    The import copies the entry's (or the request's) documents, up to the row cap; a retry (`previous`) keeps its
    import's, and takes over the documents it already fetched, so Try again fetches none of them again. A retry needs
    no entry: it resumes from its import's own documents even when the admin has since deleted the entry, and never
    cuts them to a lowered row cap. A retry of an import that stopped at the download cap is refused until the cap is
    higher.

    At most one import of an entry (or of one set of documents) per owner is queued or running: the new import holds
    `active_key` (unique). A search import's `external_id` is `search:` and a digest of its documents, so the same
    documents imported twice at once (a double click) make one import, and other selections their own. A retry claims
    its paused import (puts it away, if no one has) in the same transaction that inserts the new import and moves the
    documents, so of two retries of one import only one starts. Returns the import as JSON and whether it is new;
    raises imports.Refused."""
    Refused = imports_module.Refused
    if not offered(host):
        raise Refused(404, "not_offered", provider.NOT_OFFERED)
    searching = previous is not None and is_search(previous)
    entry, queries = None, None
    # `documents: null` beside an entry is an entry import, as before #16; without one it is a search import with none.
    if previous is None and "documents" in body and (body["documents"] is not None or body.get("dataset") is None):
        searching = True
        if body.get("dataset") is not None:
            raise Refused(400, "bad_request", provider.BOTH_KINDS)
        if _level(host) == "hide":
            raise Refused(403, "license_hidden", provider.SEARCH_HIDDEN)
        documents, queries, task_type = _chosen(host, body)
        entry_ref = SEARCH_PREFIX + _digest(documents)
    else:
        entry_ref = body.get("dataset")
        entry = (
            host.db.session.get(LibraryLegalDataHunterEntry, entry_ref)
            if previous is None and isinstance(entry_ref, str) else None
        )
        if previous is not None:
            entry_ref = previous.external_id
        elif entry is None:
            raise Refused(404, "not_found", provider.NOT_IN_LIBRARY)
        if _level(host) == "hide":
            raise Refused(403, "license_hidden", provider.SEARCH_HIDDEN if searching else provider.LICENSE_HIDDEN)
    if not has_key(host, owner_id):
        raise Refused(409, "needs_key", NEEDS_KEY)
    if body.get("licenseAck") is not True:
        raise Refused(409, "needs_ack", provider.NEEDS_CONFIRMATION)
    name = body.get("name")
    name = name.strip() if isinstance(name, str) else ""
    if len(name) > 255:
        raise Refused(400, "bad_name", "Keep the name under 256 characters.")
    session = host.db.session
    cap = host.row_cap()
    if previous is not None:
        # The paused import's own list, whole: it was paid for at the cap it started under.
        documents = list(previous.documents or [])
        max_rows = previous.max_rows if previous.max_rows is not None else cap
        task_type = previous.task_type
        queries = previous.queries
    elif entry is not None:
        documents = list(entry.documents or [])[:cap]
        max_rows = cap
        task_type = entry.task_type
    else:
        documents = documents[:cap]
        max_rows = cap
    if not documents:
        raise Refused(409, "no_documents", provider.CHOOSE_DOCUMENTS if searching else provider.NO_DOCUMENTS)
    if previous is not None and (previous.saved_path or previous.dataset_ref):
        # It stopped once it reached saving (a sweep or restart then): its Dataset may exist, or be being made, so Try
        # again could make a second one. The claim below checks this again, in the same statement.
        raise Refused(409, "already_saved", provider.ALREADY_SAVED)
    if previous is not None and previous.error_code == "too_large" and previous.failed_cap is not None:
        # Try again would pay for the same documents and stop at the same place.
        if limits_module.import_cap_bytes(session, host.max_fetch_bytes) <= previous.failed_cap:
            raise Refused(409, "cap_unchanged", provider.CAP_UNCHANGED)
    running = (
        session.query(LibraryImport)
        .filter(
            LibraryImport.owner_id == owner_id,
            LibraryImport.provider == PROVIDER,
            LibraryImport.external_id == entry_ref,
            LibraryImport.status.in_(imports_module.RUNNING),
        )
        .order_by(LibraryImport.created_at.desc())
        .first()
    )
    already_running = provider.SEARCH_RUNNING if searching else provider.ALREADY_RUNNING
    if running is not None:
        if previous is not None:
            # Never hand back the other import and put the paused one (and what it fetched) away.
            raise Refused(409, "already_running", already_running)
        imports_module._log(running, "repeated")
        return running.to_dict(), False
    now = imports_module._now()
    if previous is not None:
        record_name = name or previous.name
        citation = imports_module._pinned(previous.citation or _citation(previous.name), None, now)
    elif entry is not None:
        earlier = session.query(LibraryImport).filter_by(owner_id=owner_id, provider=PROVIDER, external_id=entry.id).count()
        record_name = name or (entry.title if not earlier else f"{entry.title} ({earlier + 1})")
        citation = imports_module._pinned(_citation(entry.title), None, now)
    else:
        record_name = name or f"{LABEL}: {queries[0]['q']}"[:255].strip()
        citation = imports_module._pinned(_citation(record_name), None, now)
    previous_ref = previous.id if previous is not None else None
    if previous_ref is not None and not _put_away(session, previous_ref, now):
        # Another Try again (or a Put away) got there first, or a worker reached saving since the check above.
        session.rollback()
        session.refresh(previous)
        if previous.saved_path or previous.dataset_ref:
            raise Refused(409, "already_saved", provider.ALREADY_SAVED)
        raise Refused(409, "not_retryable", provider.PUT_AWAY)
    record = LibraryImport(
        owner_id=owner_id, provider=PROVIDER, external_id=entry_ref, config=None, split=None, revision=None,
        url=provider.ORIGIN, license=TERMS.id, license_ack_at=ack_at or now,
        text_column="text", label_column=None, task_type=task_type,
        sample_size=len(documents), max_rows=max_rows, balance=False,
        name=record_name, citation=citation,
        source=PROVIDER, documents=documents, queries=queries, skipped=0, missing=0,
        status=imports_module.QUEUED, active_key=active_key(owner_id, entry_ref),
    )
    session.add(record)
    try:
        session.flush()
    except IntegrityError:
        # Another import of the entry (or of these documents) was queued between the check above and this insert.
        session.rollback()
        raise Refused(409, "already_running", already_running) from None
    if previous_ref is not None:
        # Try again: the documents already fetched move to the new import, so they are not fetched (or paid for) again.
        _move_documents(session, previous_ref, record.id)
    session.commit()
    imports_module._log(record, imports_module.QUEUED, size=record.sample_size)
    restore = None
    if previous_ref is not None:
        record_ref = record.id

        def restore():
            # Nothing will run the new import: in the same commit that fails it, it is put away, and the paused one
            # gets its documents back and is no longer put away, so Try again still resumes from them.
            record.dismissed_at = imports_module._now()
            _move_documents(session, record_ref, previous_ref)
            session.execute(
                sa.update(LibraryImport)
                .where(LibraryImport.id == previous_ref)
                .values(dismissed_at=None, updated_at=LibraryImport.updated_at)
                .execution_options(synchronize_session=False)
            )

    imports_module._enqueue(host, record, restore)
    return record.to_dict(), True


def _chosen(host, body):
    """A search import's documents (checked, each once, in the order sent), queries and task type. Raises
    imports.Refused with the first thing wrong."""
    Refused = imports_module.Refused
    sent = body.get("documents")
    if not isinstance(sent, list) or not sent:
        raise Refused(400, "no_documents", provider.CHOOSE_DOCUMENTS)
    if len(sent) > MAX_SENT:
        raise Refused(400, "too_many_documents", provider.TOO_MANY_DOCUMENTS)
    documents, seen = [], set()
    for place, item in enumerate(sent, start=1):
        ref = provider.document_ref(item)
        if ref is None:
            raise Refused(400, "bad_document", provider.bad_document(place))
        identity = (ref["country"], ref["source"], ref["sourceId"])
        if identity in seen:
            continue
        seen.add(identity)
        documents.append(ref)
    if len(documents) > MAX_DOCUMENTS:
        raise Refused(400, "too_many_documents", provider.TOO_MANY_DOCUMENTS)
    queries = _queries(body.get("queries"))
    task_type = body.get("taskType") or "classification"
    if not isinstance(task_type, str) or task_type not in host.task_types:
        raise Refused(400, "bad_task", f"Unknown task type: {task_type}")
    return documents, queries, task_type


def _queries(sent):
    """The searches that found a search import's documents, each once: a checked query ({q, corpus, …}, as the search
    answered it), or just its text ({q}). Raises imports.Refused (`bad_queries`)."""
    refused = imports_module.Refused(400, "bad_queries", provider.NO_QUERIES)
    if not isinstance(sent, list) or not 1 <= len(sent) <= MAX_DOCUMENTS:
        raise refused
    queries = []
    for item in sent:
        if isinstance(item, str):
            text = item.strip()
            if not text or len(text) > provider.MAX_QUERY:
                raise refused
            query = {"q": text}
        elif isinstance(item, dict):
            try:
                # As the search route asked it: SEARCH_HITS, whatever limit (or none) is sent.
                query = provider.check_query({**item, "limit": SEARCH_HITS})
            except InvalidQuery:
                raise refused from None
        else:
            raise refused
        if query not in queries:
            queries.append(query)
    return queries


def _digest(documents):
    """The same documents, in any order, give the same digest."""
    identities = sorted([ref["country"], ref["source"], ref["sourceId"]] for ref in documents)
    return hashlib.sha256(json.dumps(identities, ensure_ascii=False).encode("utf-8")).hexdigest()


def active_key(owner_id, entry_ref):
    """What an active (queued or running) import of an entry holds, unique per owner and entry."""
    return f"{owner_id}:{entry_ref}"


def _put_away(session, import_ref, now):
    """Claim a paused import for Try again: put it away if it is still failed or cancelled, nobody has, and it never
    reached saving (a save that commits first wins). One conditional update, not committed; returns whether it
    matched. Its `updated_at` is kept, so its pause is not lengthened if the retry is undone."""
    return bool(session.execute(
        sa.update(LibraryImport)
        .where(
            LibraryImport.id == import_ref,
            LibraryImport.dismissed_at.is_(None),
            LibraryImport.status.in_((imports_module.FAILED, imports_module.CANCELLED)),
            LibraryImport.saved_path.is_(None),
            LibraryImport.dataset_ref.is_(None),
        )
        .values(dismissed_at=now, updated_at=LibraryImport.updated_at)
        .execution_options(synchronize_session=False)
    ).rowcount)


def _move_documents(session, from_ref, to_ref):
    session.execute(
        sa.update(LibraryLegalDataHunterDocument)
        .where(LibraryLegalDataHunterDocument.import_id == from_ref)
        .values(import_id=to_ref)
        .execution_options(synchronize_session=False)
    )


def fetch(session, record, key_for_owner, max_bytes=None):
    """The worker's share (decision L6, L12): fetch every document of the import not fetched yet, with the student's
    key, keeping each as it comes, and return the rows in the entry's order with the counts skipped.

    A document with no text is skipped and counted (`skipped`); one Legal Data Hunter no longer has (404), or refuses
    as malformed (400, 422), is too (`missing`). When the credits run out (429) the import fails as `out_of_credits`
    with "Fetched N of M…", keeping what it fetched for Try again. `max_bytes` is the worker's download cap: the
    documents' responses together may not pass it (`too_large`, keeping those fetched and recording the cap as
    `failed_cap`); each response is read only as far as what is left of it. A key seam that raises is no key
    (`needs_key`, resumable). Terms set to Hide while it runs fail it as `license_hidden`, discarding what it fetched.
    Raises imports.ImportFailed, or imports._Stopped when the import ended elsewhere (cancelled, put away, or its row
    erased), even between the last check and a document's write."""
    ImportFailed = imports_module.ImportFailed
    level = policy_module.level(TERMS, policy_module.for_session(session))
    refusal = None
    if level == "hide":
        refusal = ImportFailed("license_hidden", hidden_message(record))
    elif record.license_ack_at is None:
        refusal = ImportFailed("needs_ack", provider.NEEDS_CONFIRMATION)
    elif key_for_owner is None:
        refusal = ImportFailed("not_offered", provider.NOT_OFFERED)
    if refusal is not None:
        # Nothing will resume from what an earlier run fetched: the import cannot run here.
        discard(session, record.id)
        raise refusal
    key = _worker_key(key_for_owner, record)
    if key is None:
        # Resumable: the documents kept are not lost to a missing key, or a seam that broke.
        raise ImportFailed("needs_key", NEEDS_KEY)
    documents = list(record.documents or [])
    done = {
        found.position: _Kept(found.outcome, found.row, found.nbytes)
        for found in session.query(LibraryLegalDataHunterDocument).filter_by(import_id=record.id)
    }
    # The bytes the kept documents' responses measured, so a resumed import cannot step past the cap a run at a time.
    total = sum(_paid(found) for found in done.values())
    if max_bytes is not None and total > max_bytes:
        # The cap was lowered since they were kept: even with nothing more to fetch, the import is over it.
        raise ImportFailed("too_large", provider.too_large(max_bytes), failed_cap=max_bytes)
    waits = 0
    position = 0
    while position < len(documents):
        if position in done:
            position += 1
            continue
        ref = documents[position]
        _still_running(session, record)
        # Before every request, a repeat after a per-minute wait too: under Hide nothing more is asked for.
        still_allowed(session, record)
        # What is left of the cap: the response is read only this far.
        allowance = None if max_bytes is None else max(max_bytes - total, 0)
        try:
            found = provider.document(key, ref, max_bytes=allowance)
        except provider.TooLarge:
            raise ImportFailed("too_large", provider.too_large(max_bytes), failed_cap=max_bytes) from None
        except (DocumentNotFound, DocumentRejected) as gone:
            outcome, row, size = MISSING, None, getattr(gone, "nbytes", 0) or 0
        except OutOfCredits as ran_out:
            if ran_out.per_minute and waits < MINUTE_WAITS:
                # Only the per-minute cap: wait it out and ask for the same document again.
                waits += 1
                wait = MINUTE_WAIT_SECONDS if ran_out.retry_after is None else max(ran_out.retry_after, 1)
                _sleep(min(wait, MINUTE_WAIT_SECONDS))
                continue
            if ran_out.per_minute:
                raise ImportFailed("provider_busy", provider.SLOW_DOWN) from None
            resume = None
            if ran_out.retry_after is not None:
                resume = imports_module._now() + timedelta(seconds=ran_out.retry_after)
            raise ImportFailed("out_of_credits", out_of_credits(len(done), len(documents)), resume_after=resume) from None
        except BadKey:
            raise ImportFailed("bad_key", BAD_KEY) from None
        except ProviderError:
            raise ImportFailed("provider_busy", provider.BUSY) from None
        else:
            size = getattr(found, "nbytes", None)
            size = _weight(found) if size is None else size
            row = provider.row(found, ref)
            outcome = FETCHED if row is not None else NO_TEXT
        if max_bytes is not None and total + size > max_bytes:
            raise ImportFailed("too_large", provider.too_large(max_bytes), failed_cap=max_bytes)
        total += size
        # A cancel, a Put away, an erasure or Hide while Legal Data Hunter answered: its text is never written.
        _still_wanted(session, record)
        # And one that lands after that check: the write itself checks the import again, in the same statement.
        if not _write(session, record.id, position, outcome, row, size):
            raise imports_module._Stopped()
        done[position] = _Kept(outcome, row, size)
        waits = 0
        position += 1
    rows = [done[p].row for p in sorted(done) if done[p].outcome == FETCHED and p < len(documents)]
    skipped = sum(1 for p in done if done[p].outcome == NO_TEXT and p < len(documents))
    missing = sum(1 for p in done if done[p].outcome == MISSING and p < len(documents))
    return rows, skipped, missing


def _worker_key(key_for_owner, record):
    """The owner's key from the worker's seam, or None. A seam that raises is no key: only the exception's type is
    logged, never its message, which could hold the key."""
    try:
        key = key_for_owner(record.owner_id)
    except Exception as failed:  # noqa: BLE001
        broken = type(failed).__name__
    else:
        return key.strip() if isinstance(key, str) and key.strip() else None
    logger.warning("library.ldh.key_seam_failed import=%s error=%s", record.id, broken)
    return None


def _paid(kept):
    """The bytes a kept document's response measured; for a row kept without that, its own size."""
    if kept.nbytes is not None:
        return kept.nbytes
    return _weight(kept.row) if kept.row is not None else 0


def _write(session, import_ref, position, outcome, row, nbytes):
    """Keep one document, in one statement that writes it only while the import is running and not put away (an
    INSERT … SELECT … WHERE EXISTS), and commit. Returns whether it was written: a cancel, Put away or erasure that
    committed after the worker's last check leaves nothing behind."""
    table = LibraryLegalDataHunterDocument
    value = sa.null() if row is None else sa.literal(row, sa.JSON)
    if row is not None and session.get_bind().dialect.name != "sqlite":
        # A bare parameter in a select list is text to PostgreSQL; SQLite would read CAST(… AS JSON) as a number.
        value = sa.cast(value, sa.JSON)
    alive = sa.exists().where(
        LibraryImport.id == import_ref,
        LibraryImport.status.in_(imports_module.RUNNING),
        LibraryImport.dismissed_at.is_(None),
    )
    chosen = sa.select(
        sa.literal(str(uuid.uuid4()), sa.String),
        sa.literal(import_ref, sa.String),
        sa.literal(position, sa.Integer),
        sa.literal(outcome, sa.String),
        value,
        sa.literal(nbytes, sa.Integer),
        sa.literal(imports_module._now(), sa.DateTime),
    ).where(alive)
    written = session.execute(
        sa.insert(table).from_select(
            [table.id, table.import_id, table.position, table.outcome, table.row, table.nbytes, table.created_at], chosen,
        )
    ).rowcount
    session.commit()
    return written == 1


def _sleep(seconds):
    """Wait out Legal Data Hunter's per-minute cap. Tests replace it."""
    time.sleep(seconds)


def _weight(value):
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def _still_wanted(session, record):
    """Stop as `_still_running` does, and also when the import was put away, or fail it when the terms are now Hide."""
    _still_running(session, record)
    if record.dismissed_at is not None:
        raise imports_module._Stopped()
    still_allowed(session, record)


def still_allowed(session, record):
    """Fail the import as `license_hidden`, discarding what it fetched, once an admin set the terms to Hide."""
    if policy_module.level(TERMS, policy_module.for_session(session)) == "hide":
        discard(session, record.id)
        raise imports_module.ImportFailed("license_hidden", hidden_message(record))


def _still_running(session, record):
    """Stop when the import ended elsewhere (a cancel, a sweep) or its row is gone."""
    try:
        session.refresh(record)
    except Exception:  # noqa: BLE001 - a deleted row (account erasure) stops it as a cancel does
        raise imports_module._Stopped() from None
    if record.status in imports_module.TERMINAL:
        raise imports_module._Stopped()


def discard(session, import_ref):
    """Delete the documents an import fetched (it completed, or the student put it away), and a CSV it wrote but never
    handed to the host."""
    session.execute(
        sa.delete(LibraryLegalDataHunterDocument)
        .where(LibraryLegalDataHunterDocument.import_id == import_ref)
        .execution_options(synchronize_session=False)
    )
    session.commit()
    unstage(session, LibraryImport.id == import_ref)


def unstage(session, *where):
    """Delete the CSVs of the imports matching `where` that were written (`staging_path`) but never handed to the host
    (`saved_path` null): the worker died, or was stopped, before saving. Each is claimed first, by setting its
    `staging_path` to null while `saved_path` is still null, and committed, so a worker can no longer move to saving
    with it (that move needs the CSV still staged); only then is the file deleted. A file whose import reached saving
    is never deleted: it may be the host's Dataset's. Returns how many went."""
    found = session.execute(
        sa.select(LibraryImport.id, LibraryImport.staging_path).where(
            LibraryImport.staging_path.is_not(None), LibraryImport.saved_path.is_(None), *where,
        )
    ).all()
    claimed = []
    for import_ref, path in found:
        if session.execute(
            sa.update(LibraryImport)
            .where(
                LibraryImport.id == import_ref,
                LibraryImport.staging_path == path,
                LibraryImport.saved_path.is_(None),
            )
            .values(staging_path=None, updated_at=LibraryImport.updated_at)
            .execution_options(synchronize_session=False)
        ).rowcount:
            claimed.append(path)
    session.commit()
    for path in claimed:
        imports_module._remove(path)
    return len(claimed)


def terms_not_hidden():
    """A SQL condition: the admin has not set the terms to Hide, read from the policy row in the statement it is part
    of (the move to saving), so a Hide committed first wins."""
    chosen = LibrarySettings.license_policy[TERMS.id].as_string()
    return ~sa.exists().where(LibrarySettings.id == 1, chosen == "hide")


def not_saved(session, record):
    """The move to saving did not happen: stop when the import ended elsewhere or its CSV was claimed (an erasure);
    otherwise the terms were Hide, and it fails as `license_hidden`, discarding what it fetched."""
    try:
        session.refresh(record)
    except Exception:  # noqa: BLE001 - a deleted row (account erasure) stops it as a cancel does
        raise imports_module._Stopped() from None
    if record.status in imports_module.TERMINAL or record.staging_path is None:
        raise imports_module._Stopped()
    discard(session, record.id)
    raise imports_module.ImportFailed("license_hidden", hidden_message(record))


def purge(session, *, now=None, keep=KEEP_FETCHED):
    """Delete fetched documents no import will resume from. Kept: those of a running import, and of a failed one not
    put away whose failure Try again resumes from (RESUMABLE), for `keep` after it failed, or, out of credits, until a
    day after its credits reset if that is later. Everything else goes: completed, cancelled, put away, other
    failures, and documents whose import is gone. While the terms are Hide, only a running import's are kept: its worker
    stops itself. Run by `imports.sweep`. Returns how many went."""
    now = now or imports_module._now()
    if policy_module.level(TERMS, policy_module.for_session(session)) == "hide":
        alive = sa.select(LibraryImport.id).where(LibraryImport.status.in_(imports_module.RUNNING))
    else:
        alive = _resumable_or_running(now, keep)
    gone = session.execute(
        sa.delete(LibraryLegalDataHunterDocument)
        .where(LibraryLegalDataHunterDocument.import_id.not_in(alive))
        .execution_options(synchronize_session=False)
    ).rowcount
    session.commit()
    # And the CSVs of those imports never handed to the host.
    unstage(session, LibraryImport.provider == PROVIDER, LibraryImport.id.not_in(alive))
    return gone


def _resumable_or_running(now, keep):
    return sa.select(LibraryImport.id).where(
        sa.or_(
            LibraryImport.status.in_(imports_module.RUNNING),
            sa.and_(
                LibraryImport.status == imports_module.FAILED,
                LibraryImport.dismissed_at.is_(None),
                imports_module.resumable_clause(),
                imports_module.paused_until_clause(now, keep),
            ),
        )
    )


def erase(session, owner_id):
    """For a host erasing a user: delete the documents that user's imports fetched, and the CSVs they wrote but never
    handed to the host, and set `created_by` to null on the entries they made (entries are course data, and stay).
    The imports are the host's to delete with the rest of the user's data, after this call (their CSVs are found from
    them; a CSV that became a Dataset's file goes with the user's Datasets): documents whose import is already gone are
    deleted too. Returns how many documents went."""
    unstage(session, LibraryImport.owner_id == owner_id)
    owned = sa.select(LibraryImport.id).where(LibraryImport.owner_id == owner_id)
    every = sa.select(LibraryImport.id)
    gone = session.execute(
        sa.delete(LibraryLegalDataHunterDocument)
        .where(
            sa.or_(
                LibraryLegalDataHunterDocument.import_id.in_(owned),
                LibraryLegalDataHunterDocument.import_id.not_in(every),
            )
        )
        .execution_options(synchronize_session=False)
    ).rowcount
    session.execute(
        sa.update(LibraryLegalDataHunterEntry)
        .where(LibraryLegalDataHunterEntry.created_by == owner_id)
        .values(created_by=None)
        .execution_options(synchronize_session=False)
    )
    session.commit()
    return gone


def hidden(session):
    """For an admin's settings save: once the terms are Hide, delete the documents of every Legal Data Hunter import
    not running. No import may resume while they are hidden. Returns how many went."""
    if policy_module.level(TERMS, policy_module.for_session(session)) != "hide":
        return 0
    running = sa.select(LibraryImport.id).where(LibraryImport.status.in_(imports_module.RUNNING))
    gone = session.execute(
        sa.delete(LibraryLegalDataHunterDocument)
        .where(LibraryLegalDataHunterDocument.import_id.not_in(running))
        .execution_options(synchronize_session=False)
    ).rowcount
    session.commit()
    return gone


def write_csv(path, rows):
    """The dataset's file: one row per document, its text and where it is from (ROW_COLUMNS), unlabelled."""
    import csv
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(provider.ROW_COLUMNS)
        for row in rows:
            writer.writerow([_cell(row.get(column)) for column in provider.ROW_COLUMNS])


def _cell(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else value


def from_legal_data_hunter(session, dataset_ref, file_path=None):
    """Whether an import of Legal Data Hunter made this Dataset, whatever its status. It matches the Dataset's ref, or
    its `file_path`: the import saves the CSV's path before `create_dataset` is called, so a Dataset whose worker died
    before its ref was recorded is still caught by its file. The host checks it before publishing the dataset, or a
    model trained on it, to Zenodo (decision L10), and should pass the Dataset's `file_path`."""
    matches = []
    if dataset_ref is not None:
        matches.append(LibraryImport.dataset_ref == str(dataset_ref))
    if file_path is not None:
        matches.append(LibraryImport.saved_path == str(file_path))
    if not matches:
        return False
    return session.query(sa.exists().where(LibraryImport.provider == PROVIDER, sa.or_(*matches))).scalar()


def publish_refusal(session, dataset_ref, *, model=False, file_path=None):
    """Why a dataset, or a model trained on it (`model=True`), cannot be published, or None when Legal Data Hunter's
    terms do not stand in the way. Pass the Dataset's `file_path` too: it is matched before the ref is recorded."""
    if not from_legal_data_hunter(session, dataset_ref, file_path):
        return None
    return MODEL_PUBLISH_REFUSED if model else DATASET_PUBLISH_REFUSED
