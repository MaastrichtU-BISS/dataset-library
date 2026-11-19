"""The Community source for a host (#122): the datasets it lists, what the library
shows for one, and the switch an admin sets. It reads only through the host's
seams (ADR 0005), so a host that gave no seams does not offer it at all."""
from dataclasses import dataclass
from typing import Callable

from . import limits as limits_module
from . import policy as policy_module
from . import tables
from .licenses import LICENSES, UNKNOWN
from .models import LibrarySource
from .preview import LicenseHidden, NotInLibrary
from .providers import community as provider
from .providers.community import NotDownloadable, TooLarge

CACHE_SECONDS = 600
ROWS = 8
CELL_CHARACTERS = 400
NO_PROPOSAL = {"taskType": None, "textColumn": None, "labelColumn": None}


class NotImportable(Exception):
    """The dataset is in the library but cannot be imported now; `code` and `message` say why."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


@dataclass(frozen=True)
class Seams:
    """The host's three functions (see `init_dataset_library`).

    datasets() -> the open dataset Publications, as fair-publishing's open_datasets returns them.
    record(record_id) -> the record's access, files and downloadability, as record_files returns them.
    download(record_id, key, max_bytes) -> one public file's bytes. Each may raise NotDownloadable,
    TooLarge, or ProviderError."""

    datasets: Callable[[], list]
    record: Callable[[str], dict]
    download: Callable[[str, str, int], bytes]


def enabled(host):
    """Whether the library offers Community datasets: the host gave its seams and an admin switched the source on."""
    return host.community is not None and switched_on(host.db)


def switched_on(db):
    row = _row(db)
    return row is not None and row.enabled


def set_enabled(db, enabled_now):
    row = _row(db)
    if row is None:
        db.session.add(
            LibrarySource(provider=provider.PROVIDER, kind=provider.SOURCE_KIND, ref=provider.SOURCE_REF, enabled=enabled_now)
        )
    else:
        row.enabled = enabled_now


def _row(db):
    return (
        db.session.query(LibrarySource)
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
    for publication in host.community.datasets():
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

    Zenodo is asked now, so a record closed since it was listed shows as closed. The first rows of its
    table file are kept for a while, by the file's checksum and the download cap. Raises NotInLibrary,
    LicenseHidden, or ProviderError when Zenodo cannot be asked."""
    publication = _listed(host, record_id)
    card = provider.card(publication)
    policy = policy_module.overrides(host.db)
    license = LICENSES.get(card["license"], UNKNOWN)
    level = policy_module.level(license, policy)
    # Hidden before anything is downloaded: a license the admin hid is not read at all.
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
        entry, cap = _checked(host, record_id)
        head = _head(host, record_id, entry, cap)
    except NotImportable as refused:
        return {**body, "reasonCode": refused.code, "reason": refused.message}
    return {
        **body,
        "importable": True,
        "file": {"key": entry["key"], "size": entry.get("size")},
        "revision": entry.get("checksum"),
        "preview": "ok",
        "columns": [{"name": name, "type": "text"} for name in head["columns"]],
        "rows": head["rows"],
        "rowCount": head["rowCount"],
        "proposed": _proposal(head["columns"]),
    }


def _listed(host, record_id):
    """The open dataset Publication with this record id. A record the host does not list is not in the library."""
    if not enabled(host):
        raise NotInLibrary()
    for publication in host.community.datasets():
        if publication.get("record_id") == record_id:
            return publication
    raise NotInLibrary()


def download_cap(host):
    """The most one Community file may be read to a preview or an import: the admin's cap, else the default."""
    return limits_module.download_cap_bytes(host.db.session) or limits_module.DEFAULT_MAX_DOWNLOAD_MB * 1_000_000


def _checked(host, record_id):
    """The table file to read and the cap it is read under, or NotImportable with the reason.
    Asks Zenodo whether the record is open now and which of its files is the table."""
    try:
        record = host.community.record(record_id)
    except NotDownloadable:
        raise NotImportable("not_downloadable", provider.NOT_DOWNLOADABLE)
    problem = provider.refusal(record)
    if problem is not None:
        raise NotImportable(*problem)
    entry = provider.table_file(record["files"])
    cap = download_cap(host)
    if entry.get("size") is not None and entry["size"] > cap:
        raise NotImportable("too_large", provider.too_large(cap))
    return entry, cap


def _head(host, record_id, entry, cap):
    """The table's columns, its first rows and its row count. Read once per file, checksum and cap."""
    key = f"library:community:head:{record_id}:{entry['key']}:{entry.get('checksum')}:{cap}"
    head = host.cache.get(key)
    if head is not None:
        return head
    try:
        data = host.community.download(record_id, entry["key"], cap)
    except NotDownloadable:
        raise NotImportable("not_downloadable", provider.NOT_DOWNLOADABLE)
    except TooLarge:
        raise NotImportable("too_large", provider.too_large(cap))
    try:
        table = tables.read(data, entry["key"])
    except tables.Unreadable as unreadable:
        raise NotImportable("unreadable", unreadable.message)
    if table.num_rows == 0:
        raise NotImportable("no_rows", provider.NO_ROWS)
    names, rows = tables.head(table, ROWS)
    head = {
        "columns": names,
        "rows": [[_cell(value) for value in row] for row in rows],
        "rowCount": table.num_rows,
    }
    host.cache.set(key, head, CACHE_SECONDS)
    return head


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
