"""The Community source for a host (#122): the datasets it lists, what the library shows for one, and the
switch an admin sets. It reads only through the host's seams (ADR 0005), so a host that gave no seams does
not offer it at all.

Zenodo is slow sometimes, and a preview is the one thing a student waits on, so previews never hold a
request for long (#122, fix 1): the repository is asked on a helper thread that the request waits for up
to `PREVIEW_SECONDS`. A preview's file is read only within `PREVIEW_MAX_BYTES`; a larger one is shown without
its rows and stays importable. One fetch per cache key runs at a time: a request that finds one running gets
`Preparing` and the UI asks again.
"""
import threading
from dataclasses import dataclass
from typing import Callable

from . import limits as limits_module
from . import policy as policy_module
from .licenses import LICENSES, UNKNOWN
from .models import LibrarySource
from .preview import LicenseHidden, NotInLibrary
from .providers import community as provider
from .providers.community import NotDownloadable, TooLarge, origin_of

CACHE_SECONDS = 600
# A record as Zenodo said it, kept for previews only: an import asks again before it downloads (imports.py).
RECORD_SECONDS = 60
# The most of a file a preview reads: a larger one is not previewed, and stays importable within the import cap.
PREVIEW_MAX_BYTES = 16_000_000
# How long a preview request waits for Zenodo before it answers "preparing". Well under the web worker's 120 s.
PREVIEW_SECONDS = 20
# How long a fetch's lock lasts if the process running it dies. A fetch that finishes releases its lock at once.
LOCK_SECONDS = 60
ROWS = 8
CELL_CHARACTERS = 400
NO_PROPOSAL = {"taskType": None, "textColumn": None, "labelColumn": None}


class NotImportable(Exception):
    """The dataset is in the library but cannot be imported now; `code` and `message` say why."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


class Preparing(Exception):
    """Zenodo has not answered in time, or another request is already asking it. The preview is being prepared."""


@dataclass(frozen=True)
class Seams:
    """The host's functions (see `init_dataset_library`), and the Community settings it gives.

    datasets() -> the open dataset Publications, as fair-publishing's open_datasets returns them, each with the
    Zenodo origin its record is on (`origin`).
    record(record_id) -> the record's access, files and downloadability, as record_files returns them.
    download(record_id, key, max_bytes) -> one public file's bytes.
    base_url() -> the Zenodo base URL this platform reads records from. Only Publications on this origin are listed.
    Each function may raise NotDownloadable, TooLarge, or ProviderError."""

    datasets: Callable[[], list]
    record: Callable[[str], dict]
    download: Callable[[str, str, int], bytes]
    base_url: Callable[[], str]
    preview_seconds: float = PREVIEW_SECONDS
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


def view(host, record_id):
    """One Community dataset as the library shows it: its card, its license at the platform's level, and
    whether it can be imported, with the reason when it cannot.

    Zenodo is asked now, through a record cache of a minute. The first rows of the table file are kept for a
    while, by its checksum and the caps. A file larger than the preview budget is not read: the card, files
    and size are shown, and it is importable within the import cap. Raises NotInLibrary, LicenseHidden, or
    ProviderError when Zenodo cannot be asked."""
    publication = _listed(host, record_id)
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
        entry, cap = _checked(host, publication)
    except NotImportable as refused:
        return {**body, "reasonCode": refused.code, "reason": refused.message}
    except Preparing:
        return {**body, "preparing": True, "preview": "preparing"}
    located = {**body, "file": {"key": entry["key"], "size": entry.get("size")}, "revision": entry.get("checksum")}
    try:
        head = _preview(host, publication, entry, cap)
    except NotImportable as refused:
        return {**located, "reasonCode": refused.code, "reason": refused.message}
    except Preparing:
        return {**located, "preparing": True, "preview": "preparing"}
    if head is None:
        # Too large to preview within the budget: the card, files and size stand, and it can still be imported.
        return {**located, "importable": True, "preview": "too_large"}
    return {
        **located,
        "importable": True,
        "preview": "ok",
        "columns": [{"name": name, "type": "text"} for name in head["columns"]],
        "rows": head["rows"],
        "rowCount": head["rowCount"],
        "proposed": _proposal(head["columns"]),
    }


def bound(host, record_id):
    """The Publication id and the Zenodo origin a Community import of this record is bound to (#122)."""
    publication = _listed(host, record_id)
    return publication["id"], _origin(host)


def _origin(host):
    return origin_of(host.community.base_url())


def _listings(host):
    """The open dataset Publications whose record is on the configured Zenodo host. A Publication whose record
    is on another host, the sandbox or production, is not that host's record, even with the same id."""
    configured = _origin(host)
    if configured is None:
        return []
    return [publication for publication in host.community.datasets() if origin_of(publication.get("origin")) == configured]


def _listed(host, record_id):
    """The open dataset Publication with this record id. A record the host does not list is not in the library."""
    if not enabled(host):
        raise NotInLibrary()
    for publication in _listings(host):
        if publication.get("record_id") == record_id:
            return publication
    raise NotInLibrary()


def import_cap(host):
    """The most one Community file may be read to an import: the admin's cap, never more than the host's limit (#122)."""
    return limits_module.import_cap_bytes(host.db.session, host.max_fetch_bytes)


def _checked(host, publication):
    """The table file to read and the import cap it is read under, or NotImportable with the reason. Asks
    Zenodo whether the record is open now, through the record cache, and which of its files is the table."""
    record_id = publication["record_id"]
    try:
        record = _fetched(host, _key(host, "record", record_id), lambda: host.community.record(record_id), RECORD_SECONDS)
    except NotDownloadable:
        raise NotImportable("not_downloadable", provider.NOT_DOWNLOADABLE)
    problem = provider.refusal(record)
    if problem is not None:
        raise NotImportable(*problem)
    entry = provider.table_file(record["files"])
    cap = import_cap(host)
    if entry.get("size") is not None and entry["size"] > cap:
        raise NotImportable("too_large", provider.too_large(cap))
    return entry, cap


def _preview(host, publication, entry, cap):
    """The first rows of the table file, or None when the file is over the preview budget and so is not read.

    The budget is a preview's, not the import's: a file over it is still importable when it is within the cap."""
    record_id = publication["record_id"]
    budget = host.community.preview_max_bytes
    size = entry.get("size")
    if size is not None and size > budget and budget < cap:
        return None
    max_bytes = min(budget, cap)
    key = _key(host, "head", publication["id"], record_id, entry["key"], entry.get("checksum"), max_bytes)
    try:
        return _fetched(host, key, lambda: _read_head(host, record_id, entry, max_bytes), CACHE_SECONDS)
    except TooLarge:
        if budget < cap:
            return None
        raise NotImportable("too_large", provider.too_large(cap))


def _read_head(host, record_id, entry, max_bytes):
    """The table's columns, its first rows and its row count, read from the file's bytes."""
    try:
        from . import tables  # lazy: pyarrow is the [import] extra, which only a Community read needs
    except ImportError as missing:
        raise NotImportable("unreadable", provider.NO_READER) from missing
    try:
        data = host.community.download(record_id, entry["key"], max_bytes)
    except NotDownloadable:
        raise NotImportable("not_downloadable", provider.NOT_DOWNLOADABLE)
    try:
        table = tables.read(data, entry["key"])
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


def _fetched(host, key, fetch, ttl):
    """`fetch()`'s value, kept under `key` for `ttl` seconds.

    A cached value is returned at once. Otherwise `fetch` runs on a helper thread, and the request waits for it
    up to the preview deadline. A request that finds the key's fetch already running, or whose fetch runs past
    the deadline, raises Preparing; the fetch keeps running and caches its value for the next request. A failure
    of `fetch` is raised to the request that waits. The thread touches no database, only the seams and the cache."""
    cached = host.cache.get(key)
    if cached is not None:
        return cached
    lock = f"{key}:lock"
    if not host.cache.add(lock, True, LOCK_SECONDS):
        raise Preparing()
    outcome = {}

    def run():
        try:
            value = fetch()
            host.cache.set(key, value, ttl)
            outcome["value"] = value
        except Exception as failed:  # noqa: BLE001 - handed to the request that waits for it
            outcome["error"] = failed
        finally:
            host.cache.delete(lock)

    worker = threading.Thread(target=run, name="library-community-preview", daemon=True)
    worker.start()
    worker.join(host.community.preview_seconds)
    if worker.is_alive():
        raise Preparing()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _key(host, *parts):
    """A cache key bound to the Zenodo origin this platform reads from, so a sandbox answer never serves production."""
    return "library:community:" + ":".join(str(part) for part in (_origin(host), *parts))


def _cell(value):
    if isinstance(value, str) and len(value) > CELL_CHARACTERS:
        return value[: CELL_CHARACTERS - 1] + "…"
    return value


def _proposal(names):
    """The import's column step starts with BAIGLE's names when the file has them: a `label`
    column and a `text` column. Nothing when it does not, and the student picks."""
    lower = {}
    for name in names:
        lower.setdefault(name.lower(), name)
    label = lower.get("label")
    text = lower.get("text")
    if text == label:
        text = None
    return {"taskType": "classification" if label else None, "textColumn": text, "labelColumn": label}
