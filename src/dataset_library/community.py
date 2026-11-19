"""The Community source for a host (#122): the datasets it lists, what the library shows for one, and the
switch an admin sets. It reads only through the host's seams (ADR 0005), so a host that gave no seams does
not offer it at all.

A Community preview is read by the ML worker, never by a request (#122, fix 1). A request re-checks the record
with Zenodo, a metadata call only, and answers from the preview's cache. A preview that is not cached yet is
asked for by queueing a job, once per cache key: a request that finds the job queued or running says the preview
is being prepared, and the page asks again. The worker reads at most the preview budget, under the same parser
limits as an import, and caches what it found: the head of the table, or the reason it cannot be previewed.
"""
import json
import secrets
from dataclasses import dataclass
from typing import Callable

from . import limits as limits_module
from . import policy as policy_module
from .errors import ProviderError, ProviderUnavailable
from .licenses import LICENSES, UNKNOWN
from .models import LibrarySource
from .preview import LicenseHidden, NotInLibrary
from .providers import community as provider
from .providers.community import NotDownloadable, TooLarge, origin_of

CACHE_SECONDS = 600
# A record as Zenodo said it, kept for display only: a start or a retry asks again before it downloads (imports.py).
RECORD_SECONDS = 60
# The most of a file a preview reads: a larger one is not previewed, and stays importable within the import cap.
PREVIEW_MAX_BYTES = 16_000_000
# How long a preview job's lock lasts if its worker dies. Longer than fair-publishing's 600 s read deadline, so the
# lock does not lapse under a job that is still reading.
LOCK_SECONDS = 660
# How long Zenodo's failure to answer a preview is kept: short, so the next open asks again soon.
BUSY_SECONDS = 60
ROWS = 8
CELL_CHARACTERS = 400
NO_PROPOSAL = {"taskType": None, "textColumn": None, "labelColumn": None}
PREPARING = {"preparing": True, "preview": "preparing"}


class NotImportable(Exception):
    """The dataset is in the library but cannot be imported now; `code` and `message` say why."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


@dataclass(frozen=True)
class Pin:
    """The Publication, the Zenodo origin and the file a Community import is bound to (#122, fix 3). A new import takes
    them as the library lists them now; a retry keeps the ones its import was started on."""

    publication_id: str
    origin: str
    file_key: str | None = None


@dataclass(frozen=True)
class Seams:
    """The host's functions (see `init_dataset_library`), and the Community settings it gives.

    datasets() -> the open dataset Publications, as fair-publishing's open_datasets returns them, each with the
    Zenodo origin its record is on (`origin`).
    record(record_id) -> the record's access, files and downloadability, as record_files returns them.
    download(record_id, key, max_bytes) -> one public file's bytes. Only the worker calls it, for a preview or an import.
    base_url() -> the Zenodo base URL this platform reads records from. Only Publications on this origin are listed.
    enqueue_preview(job) -> hands a preview job (a JSON string) to the worker, which calls `run_preview` (the API only).
    Each function may raise NotDownloadable, TooLarge, or ProviderError."""

    datasets: Callable[[], list]
    record: Callable[[str], dict]
    download: Callable[[str, str, int], bytes]
    base_url: Callable[[], str]
    # Only the request that answers a preview queues one: the worker's seams (run_import) leave it out.
    enqueue_preview: Callable[[str], None] | None = None
    preview_max_bytes: int = PREVIEW_MAX_BYTES


def enabled(host):
    """Whether the library offers Community datasets: the host gave its seams and an admin switched the source on."""
    return host.community is not None and switched_on(host.db.session)


def switched_on(session):
    row = _row(session)
    return row is not None and row.enabled


def set_enabled(session, enabled_now):
    row = _row(session)
    if row is None:
        session.add(LibrarySource(provider=provider.PROVIDER, kind=provider.SOURCE_KIND, ref=provider.SOURCE_REF, enabled=enabled_now))
    else:
        row.enabled = enabled_now


def _row(session):
    return (
        session.query(LibrarySource)
        .filter_by(provider=provider.PROVIDER, ref=provider.SOURCE_REF)
        .one_or_none()
    )


def cards(host, *, query, task, license_id):
    """The Community cards a search finds, with the license policy applied. Empty when the source is not offered.

    A Publication carries no task, so a task filter finds none of them (provider.card)."""
    if not enabled(host):
        return []
    policy = policy_module.overrides(host.db)
    found = []
    for publication in _listings(host):
        if not provider.matches(publication, query):
            continue
        card = provider.card(publication)
        if task and task not in card["tasks"]:
            continue
        if license_id and card["license"] != license_id:
            continue
        license = LICENSES.get(card["license"], UNKNOWN)
        level = policy_module.level(license, policy)
        if level == "hide":
            continue
        found.append({**card, "license": license.at_level(level)})
    return found


def view(host, record_id, *, pin=None, fresh=False):
    """One Community dataset as the library shows it: its card, its license at the platform's level, whether it can be
    imported (with the reason when it cannot), and what its preview says.

    Zenodo is asked about the record: through the record cache of a minute, or live when `fresh`, as a start or a retry
    asks it. The preview is read from its cache and never from Zenodo here: a file not cached is queued for the worker,
    and the answer is `preparing` (#122, fix 1). Raises NotInLibrary, LicenseHidden, or ProviderError when Zenodo cannot
    be asked, or the preview's read failed for Zenodo's sake."""
    publication = _listed(host, record_id, pin)
    card = provider.card(publication)
    policy = policy_module.overrides(host.db)
    license = LICENSES.get(card["license"], UNKNOWN)
    level = policy_module.level(license, policy)
    # Hidden before anything is asked of Zenodo: a license the admin hid is not read at all.
    if level == "hide":
        raise LicenseHidden()
    body = {
        **card,
        "license": license.at_level(level),
        "importable": False,
        "reasonCode": None,
        "reason": None,
        "file": None,
        "revision": None,
        "preparing": False,
        "preview": "unavailable",
        "columns": [],
        "rows": [],
        "rowCount": None,
        "configs": [],
        "config": None,
        "split": None,
        "proposed": NO_PROPOSAL,
    }
    try:
        entry, cap = _checked(host, publication, pin, fresh)
    except NotImportable as refused:
        return {**body, "reasonCode": refused.code, "reason": refused.message}
    located = {**body, "file": {"key": entry["key"], "size": entry.get("size")}, "revision": entry.get("checksum")}
    return {**located, **_previewed(host, publication, entry, cap)}


def bind(host, record_id, pin=None):
    """The Publication, origin and file a Community import of this record is bound to (#122, fix 3). A new import takes the
    Publication the library lists for the record now, on the configured origin. A retry keeps its import's: the Publication
    with that id, on the origin it was started on, and its file. It never takes another Publication for the same record id."""
    if pin is None:
        publication = _listed(host, record_id)
        return Pin(publication["id"], _origin(host))
    if pin.origin != _origin(host):
        raise NotInLibrary()
    _listed(host, record_id, pin)
    return pin


def _origin(host):
    return origin_of(host.community.base_url())


def _listings(host):
    """The open dataset Publications whose record is on the configured Zenodo host. A Publication whose record
    is on another host, the sandbox or production, is not that host's record, even with the same id."""
    configured = _origin(host)
    if configured is None:
        return []
    return [publication for publication in host.community.datasets() if origin_of(publication.get("origin")) == configured]


def _listed(host, record_id, pin=None):
    """The open dataset Publication with this record id, or, for a pinned import, the Publication it was started on (#122, fix 3).
    A record the host does not list is not in the library."""
    if not enabled(host):
        raise NotInLibrary()
    for publication in _listings(host):
        if publication.get("record_id") != record_id:
            continue
        if pin is None or publication.get("id") == pin.publication_id:
            return publication
    raise NotInLibrary()


def import_cap(host):
    """The most one Community file may be read to an import: the admin's cap, never more than the host's limit (#122)."""
    return limits_module.import_cap_bytes(host.db.session, host.max_fetch_bytes)


def _checked(host, publication, pin, fresh):
    """The table file to read and the import cap it is read under, or NotImportable with the reason. Zenodo is asked whether
    the record is open now and which of its files is the table. For a retry, the file it was started on must still be there."""
    record = _record(host, publication["record_id"], fresh=fresh)
    if pin is not None and pin.file_key is not None:
        entry = provider.pinned_file(record, pin.file_key)
        if entry is None:
            raise NotImportable("not_downloadable", provider.NOT_DOWNLOADABLE)
    else:
        problem = provider.refusal(record)
        if problem is not None:
            raise NotImportable(*problem)
        entry = provider.table_file(record["files"])
    cap = import_cap(host)
    if entry.get("size") is not None and entry["size"] > cap:
        raise NotImportable("too_large", provider.too_large(cap))
    return entry, cap


def _record(host, record_id, *, fresh):
    """The record as Zenodo says it now. A cached one is returned unless `fresh`; either way what is found is kept for a minute."""
    key = _key(host, "record", record_id)
    if not fresh:
        cached = host.cache.get(key)
        if cached is not None:
            return cached
    try:
        record = host.community.record(record_id)
    except NotDownloadable:
        raise NotImportable("not_downloadable", provider.NOT_DOWNLOADABLE)
    host.cache.set(key, record, RECORD_SECONDS)
    return record


def _previewed(host, publication, entry, cap):
    """What the preview says about the file, as keys of the dataset: its head once the worker has cached it, the reason it
    cannot be previewed, or `preparing` while the worker has it queued. The file is never read here (#122, fix 1)."""
    budget = host.community.preview_max_bytes
    size = entry.get("size")
    # A file over the budget is not read at all: the card, files and size stand, and it can still be imported within the cap.
    if size is not None and size > budget and budget < cap:
        return {"importable": True, "preview": "too_large"}
    max_bytes = min(budget, cap)
    key = _key(host, "head", publication["id"], publication["record_id"], entry["key"], entry.get("checksum"), max_bytes)
    cached = host.cache.get(key)
    if cached is None:
        return _queue(host, key, publication["record_id"], entry, max_bytes)
    return _shown(cached, budget, cap)


def _queue(host, key, record_id, entry, max_bytes):
    """Queue the worker's read of this file, once for its cache key (#122, fix 1). A request that finds a job already queued or
    running queues nothing and is told the preview is being prepared. The lock's value is random, and only the job holding it
    releases it."""
    token = secrets.token_urlsafe(16)
    lock = f"{key}:lock"
    if not host.cache.add(lock, token, LOCK_SECONDS):
        return dict(PREPARING)
    job = {
        "key": key, "lock": lock, "token": token, "record_id": record_id,
        "file_key": entry["key"], "checksum": entry.get("checksum"), "max_bytes": max_bytes,
    }
    try:
        host.community.enqueue_preview(json.dumps(job))
    except Exception as failed:  # noqa: BLE001 - whatever the queue raised, no worker will read this job
        host.cache.delete_if(lock, token)
        raise ProviderUnavailable("The preview could not be queued") from failed
    return dict(PREPARING)


def _shown(value, budget, cap):
    """The keys a cached preview shows. A Zenodo failure the worker cached is raised, so the page says Zenodo is unavailable (#122)."""
    if "head" in value:
        head = value["head"]
        return {
            "importable": True,
            "preview": "ok",
            "columns": [{"name": name, "type": "text"} for name in head["columns"]],
            "rows": head["rows"],
            "rowCount": head["rowCount"],
            "proposed": _proposal(head["columns"]),
        }
    if value.get("tooLarge"):
        # Too large to preview within the budget: it can still be imported, unless the import cap is the smaller limit.
        if budget < cap:
            return {"importable": True, "preview": "too_large"}
        return {"reasonCode": "too_large", "reason": provider.too_large(cap)}
    if "refused" in value:
        return {"reasonCode": value["refused"]["code"], "reason": value["refused"]["message"]}
    raise ProviderUnavailable("Zenodo did not answer")


def run_preview(job, *, cache, download):
    """The worker's share of a preview (#122, fix 1): read the head of one file for its cache key and keep what it says, then
    release the job's lock. The read is under the same parser limits as an import, and its bytes must match the checksum the
    request saw. A refusal is kept for CACHE_SECONDS. Zenodo's own failure is kept for a minute, so the page shows it briefly.

    `job` is the JSON the request queued (`enqueue_preview`), read back as a dict; `cache` has get, set, add and delete_if;
    `download` is the Zenodo seam. Only the lock is released when the job ends, however it ends."""
    key, lock, token = job["key"], job["lock"], job["token"]
    try:
        # A job queued again after its lock expired in a long queue finds the first one's answer: nothing to read again.
        if cache.get(key) is not None:
            return
        try:
            value, ttl = _read_preview(job, download), CACHE_SECONDS
        except ProviderError:
            value, ttl = {"busy": True}, BUSY_SECONDS
        cache.set(key, value, ttl)
    finally:
        cache.delete_if(lock, token)


def _read_preview(job, download):
    """What the worker finds in one file: its head, or the reason it cannot be previewed (a refusal, or too large)."""
    try:
        return {"head": _head(job, download)}
    except NotImportable as refused:
        return {"refused": {"code": refused.code, "message": refused.message}}
    except TooLarge:
        return {"tooLarge": True}


def _head(job, download):
    """The table's columns, its first rows and its row count, read from the file's bytes. The worker's parser is imported here,
    where pyarrow is available: a request never parses a file."""
    try:
        from . import tables  # lazy: pyarrow is the [import] extra, which only a Community read needs
    except ImportError as missing:
        raise NotImportable("unreadable", provider.NO_READER) from missing
    try:
        data = download(job["record_id"], job["file_key"], job["max_bytes"])
    except NotDownloadable:
        raise NotImportable("not_downloadable", provider.NOT_DOWNLOADABLE)
    if job.get("checksum") is not None and not provider.checksum_matches(data, job["checksum"]):
        raise NotImportable("changed", provider.CHANGED)
    try:
        table = tables.read(data, job["file_key"])
    except tables.Unreadable as unreadable:
        raise NotImportable("unreadable", unreadable.message)
    if table.num_rows == 0:
        raise NotImportable("no_rows", provider.NO_ROWS)
    names, rows = tables.head(table, ROWS)
    return {
        "columns": names,
        "rows": [[_cell(value) for value in row] for row in rows],
        "rowCount": table.num_rows,
    }


def _key(host, *parts):
    """A cache key bound to the Zenodo origin this platform reads from, so a sandbox answer never serves production."""
    return "library:community:" + ":".join(str(part) for part in (_origin(host), *parts))


def _cell(value):
    if isinstance(value, str) and len(value) > CELL_CHARACTERS:
        return value[: CELL_CHARACTERS - 1] + "…"
    return value


def _proposal(names):
    """The import's column step starts with BAIGLE's names when the file has them: a `label` column and a `text` column.
    No task is proposed, as preview.py proposes none, so a changed text column does not claim a task (#122)."""
    lower = {}
    for name in names:
        lower.setdefault(name.lower(), name)
    label = lower.get("label")
    text = lower.get("text")
    if text == label:
        text = None
    return {"taskType": None, "textColumn": text, "labelColumn": label}
