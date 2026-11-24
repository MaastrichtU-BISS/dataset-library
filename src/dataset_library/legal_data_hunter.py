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
import logging
import time
from datetime import timedelta

import sqlalchemy as sa

from . import imports as imports_module
from . import policy as policy_module
from .errors import ProviderError
from .licenses import LEGAL_DATA_HUNTER_TERMS
from .models import LibraryImport, LibraryLegalDataHunterDocument, LibraryLegalDataHunterEntry
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
    PROVIDER,
    BadKey,
    DocumentNotFound,
    InvalidQuery,
    OutOfCredits,
    credits,
    out_of_credits,
)

logger = logging.getLogger("dataset_library")

MAX_TITLE = 120
MAX_DESCRIPTION = 2000
# A paused import's fetched documents are kept this long for Try again; the free plan's day limit resets well within it.
KEEP_FETCHED = timedelta(days=7)
# Every plan has a per-minute cap (60 a minute on the free plan), which a 100-document import can reach. A 429 for it
# is waited out, never shown as credits running out; these bound the wait so an import cannot hang on it.
MINUTE_WAIT_SECONDS = 60
MINUTE_WAITS = 3
FETCHED, NO_TEXT, MISSING = "fetched", "no_text", "missing"
TERMS = LEGAL_DATA_HUNTER_TERMS


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
    key = host.legal_data_hunter_key(owner_id)
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
            raise Invalid({"documents": "This search found no documents."})
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


def cards(host, owner_id):
    """The entries as featured library cards, oldest first; none when Legal Data Hunter is not offered or an admin
    hid its terms. Each says whether the signed-in user can import it, and what it costs them."""
    if not offered(host):
        return []
    level = _level(host)
    if level == "hide":
        return []
    keyed = has_key(host, owner_id)
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


def start(host, owner_id, body, *, ack_at=None, previous=None):
    """Queue an import of an entry: one row per document, fetched by the worker with the student's key.

    The student must confirm (`licenseAck: true`) every time; a retry carries its import's confirmation time
    (`ack_at`). The import copies the entry's documents, up to the row cap; a retry (`previous`) keeps its import's,
    and takes over the documents it already fetched, so Try again fetches none of them again. Returns the import as
    JSON and whether it is new; raises imports.Refused."""
    Refused = imports_module.Refused
    if not offered(host):
        raise Refused(404, "not_offered", provider.NOT_OFFERED)
    entry_ref = body.get("dataset")
    entry = host.db.session.get(LibraryLegalDataHunterEntry, entry_ref) if isinstance(entry_ref, str) else None
    if entry is None:
        raise Refused(404, "not_found", provider.NOT_IN_LIBRARY)
    if _level(host) == "hide":
        raise Refused(403, "license_hidden", provider.LICENSE_HIDDEN)
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
    documents = list((previous.documents if previous is not None else entry.documents) or [])[:cap]
    if not documents:
        raise Refused(409, "no_documents", provider.NO_DOCUMENTS)
    task_type = previous.task_type if previous is not None else entry.task_type
    running = (
        session.query(LibraryImport)
        .filter(
            LibraryImport.owner_id == owner_id,
            LibraryImport.provider == PROVIDER,
            LibraryImport.external_id == entry.id,
            LibraryImport.status.in_(imports_module.RUNNING),
        )
        .order_by(LibraryImport.created_at.desc())
        .first()
    )
    if running is not None:
        imports_module._log(running, "repeated")
        return running.to_dict(), False
    earlier = session.query(LibraryImport).filter_by(owner_id=owner_id, provider=PROVIDER, external_id=entry.id).count()
    now = imports_module._now()
    record = LibraryImport(
        owner_id=owner_id, provider=PROVIDER, external_id=entry.id, config=None, split=None, revision=None,
        url=provider.ORIGIN, license=TERMS.id, license_ack_at=ack_at or now,
        text_column="text", label_column=None, task_type=task_type,
        sample_size=len(documents), max_rows=cap, balance=False,
        name=name or (entry.title if not earlier else f"{entry.title} ({earlier + 1})"),
        citation=imports_module._pinned(_citation(entry.title), None, now),
        source=PROVIDER, documents=documents, skipped=0, missing=0,
        status=imports_module.QUEUED,
    )
    session.add(record)
    session.flush()
    if previous is not None:
        # Try again: the documents already fetched move to the new import, so they are not fetched (or paid for) again.
        session.execute(
            sa.update(LibraryLegalDataHunterDocument)
            .where(LibraryLegalDataHunterDocument.import_id == previous.id)
            .values(import_id=record.id)
            .execution_options(synchronize_session=False)
        )
        session.execute(
            sa.delete(LibraryLegalDataHunterDocument)
            .where(LibraryLegalDataHunterDocument.import_id == record.id, LibraryLegalDataHunterDocument.position >= len(documents))
            .execution_options(synchronize_session=False)
        )
    session.commit()
    imports_module._log(record, imports_module.QUEUED, size=record.sample_size)
    imports_module._enqueue(host, record)
    return record.to_dict(), True


def fetch(session, record, key_for_owner):
    """The worker's share (decision L6, L12): fetch every document of the import not fetched yet, with the student's
    key, keeping each as it comes, and return the rows in the entry's order with the counts skipped.

    A document with no text is skipped and counted (`skipped`); one Legal Data Hunter no longer has is too (`missing`).
    When the credits run out (429) the import fails as `out_of_credits` with "Fetched N of M…", keeping what it fetched
    for Try again. Raises imports.ImportFailed, or imports._Stopped when the import ended elsewhere (cancelled)."""
    ImportFailed = imports_module.ImportFailed
    level = policy_module.level(TERMS, policy_module.for_session(session))
    if level == "hide":
        raise ImportFailed("license_hidden", provider.LICENSE_HIDDEN)
    if record.license_ack_at is None:
        raise ImportFailed("needs_ack", provider.NEEDS_CONFIRMATION)
    if key_for_owner is None:
        raise ImportFailed("not_offered", provider.NOT_OFFERED)
    key = key_for_owner(record.owner_id)
    key = key.strip() if isinstance(key, str) and key.strip() else None
    if key is None:
        raise ImportFailed("needs_key", NEEDS_KEY)
    documents = list(record.documents or [])
    done = {
        found.position: found
        for found in session.query(LibraryLegalDataHunterDocument).filter_by(import_id=record.id)
    }
    waits = 0
    position = 0
    while position < len(documents):
        if position in done:
            position += 1
            continue
        ref = documents[position]
        _still_running(session, record)
        try:
            found = provider.document(key, ref)
        except DocumentNotFound:
            outcome, row = MISSING, None
        except OutOfCredits as ran_out:
            if ran_out.per_minute and waits < MINUTE_WAITS:
                # Only the per-minute cap: wait it out and ask for the same document again.
                waits += 1
                _sleep(min(max(ran_out.retry_after or MINUTE_WAIT_SECONDS, 1), MINUTE_WAIT_SECONDS))
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
            row = provider.row(found, ref)
            outcome = FETCHED if row is not None else NO_TEXT
        kept = LibraryLegalDataHunterDocument(import_id=record.id, position=position, outcome=outcome, row=row)
        session.add(kept)
        session.commit()
        done[position] = kept
        waits = 0
        position += 1
    rows = [done[p].row for p in sorted(done) if done[p].outcome == FETCHED and p < len(documents)]
    skipped = sum(1 for p in done if done[p].outcome == NO_TEXT and p < len(documents))
    missing = sum(1 for p in done if done[p].outcome == MISSING and p < len(documents))
    return rows, skipped, missing


def _sleep(seconds):
    """Wait out Legal Data Hunter's per-minute cap. Tests replace it."""
    time.sleep(seconds)


def _still_running(session, record):
    """Stop when the import ended elsewhere (a cancel, a sweep) or its row is gone."""
    try:
        session.refresh(record)
    except Exception:  # noqa: BLE001 - a deleted row (account erasure) stops it as a cancel does
        raise imports_module._Stopped() from None
    if record.status in imports_module.TERMINAL:
        raise imports_module._Stopped()


def discard(session, import_ref):
    """Delete the documents an import fetched: it completed, or the student put it away."""
    session.execute(
        sa.delete(LibraryLegalDataHunterDocument)
        .where(LibraryLegalDataHunterDocument.import_id == import_ref)
        .execution_options(synchronize_session=False)
    )
    session.commit()


def purge(session, *, now=None, keep=KEEP_FETCHED):
    """Delete fetched documents no import will resume from: those of an import that completed, was put away, or ended
    more than `keep` ago, and those whose import is gone. Run by `imports.sweep`. Returns how many went."""
    now = now or imports_module._now()
    alive = sa.select(LibraryImport.id).where(
        sa.or_(
            LibraryImport.status.in_(imports_module.RUNNING),
            sa.and_(
                LibraryImport.status.in_({imports_module.FAILED, imports_module.CANCELLED}),
                LibraryImport.dismissed_at.is_(None),
                LibraryImport.updated_at >= now - keep,
            ),
        )
    )
    gone = session.execute(
        sa.delete(LibraryLegalDataHunterDocument)
        .where(LibraryLegalDataHunterDocument.import_id.not_in(alive))
        .execution_options(synchronize_session=False)
    ).rowcount
    session.commit()
    return gone


def erase(session, owner_id):
    """For a host erasing a user: delete the documents that user's imports fetched. The imports are the host's to
    delete with the rest of the user's data. Returns how many went."""
    owned = sa.select(LibraryImport.id).where(LibraryImport.owner_id == owner_id)
    gone = session.execute(
        sa.delete(LibraryLegalDataHunterDocument)
        .where(LibraryLegalDataHunterDocument.import_id.in_(owned))
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


def from_legal_data_hunter(session, dataset_ref):
    """Whether a completed import of Legal Data Hunter made this Dataset. The host checks it before publishing the
    dataset, or a model trained on it, to Zenodo (decision L10)."""
    return session.query(
        sa.exists().where(
            LibraryImport.dataset_ref == str(dataset_ref),
            LibraryImport.provider == PROVIDER,
            LibraryImport.status == imports_module.COMPLETED,
        )
    ).scalar()


def publish_refusal(session, dataset_ref, *, model=False):
    """Why a dataset, or a model trained on it (`model=True`), cannot be published, or None when Legal Data Hunter's
    terms do not stand in the way."""
    if not from_legal_data_hunter(session, dataset_ref):
        return None
    return MODEL_PUBLISH_REFUSED if model else DATASET_PUBLISH_REFUSED
