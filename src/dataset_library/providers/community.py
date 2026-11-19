"""Datasets BAIGLE users published openly to Zenodo (#122, decision D8).

This package does not import fair-publishing (ADR 0005). The host passes its seams: the open
dataset Publications, one record's files, and one file's bytes. This module turns what they
return into library cards and citations, says which file of a record is the table and why a
record cannot be imported, and decides which Zenodo host a Publication's record is on. It
fetches nothing itself.
"""
import os
import re
from urllib.parse import urlsplit

from ..licenses import resolve_license

PROVIDER = "community"
# The search filter value for this source. An organisation named "community" is refused (admin.py).
SOURCE = "community"
# The one Community source row an admin switches on (library_sources: provider, kind, ref).
SOURCE_KIND = "community"
SOURCE_REF = "zenodo"
DOI_PREFIX = "https://doi.org/"
RECORD_ID = re.compile(r"^[0-9]{1,20}$")
# BAIGLE publishes an uploaded dataset as `data<ext>`; that file is read in preference.
PREFERRED_STEM = "data"
TABLE_EXTENSIONS = (".csv", ".tsv", ".json", ".jsonl", ".xlsx", ".parquet")
SUMMARY_CHARACTERS = 300
# The DOI prefixes Zenodo issues its records under, for a Publication whose record link is missing.
DOI_ORIGINS = {"10.5281": "https://zenodo.org", "10.5072": "https://sandbox.zenodo.org"}
DEFAULT_PORTS = {"http": 80, "https": 443}

# The codes a refusal carries, and the messages a student sees for them.
NOT_DOWNLOADABLE = "This record is no longer downloadable"
RESTRICTED = "This record is restricted on Zenodo, so it cannot be imported."
METADATA_ONLY = "Only this record's description is public on Zenodo, not its files, so it cannot be imported."
NO_TABLE = "This record has no data file the library can read: CSV, TSV, JSON, JSONL, Parquet or XLSX."
NO_ROWS = "This record's data file has no rows to import."
UNREADABLE = "We can't read this record's data file as a table, so it cannot be imported."
# A host without the [import] extra (pyarrow, openpyxl) cannot read a data file at all.
NO_READER = "This platform can't read data files here, so this record cannot be imported."
CHANGED = "This record's data file changed on Zenodo after you opened it. Open it again to import the new version."
# A preview that is still being prepared, and a file too large to preview within the budget (it may still be imported).
PREPARING = "The preview is still being prepared. Try again in a moment."
TOO_LARGE_PREVIEW = "Too large to preview here; it can still be imported"
# A queued import whose dataset has left the library since it was queued (#122, fix 4).
SOURCE_OFF = "The Community source is switched off, so this dataset can no longer be imported."
NOT_LISTED = "This dataset is no longer in the library, so it cannot be imported."
LICENSE_HIDDEN = "This dataset's license is not allowed on this platform, so it cannot be imported."
NEEDS_ACK = "Confirm the license terms to import this dataset."


class NotDownloadable(Exception):
    """The record, or the file, is gone or no longer public. A host raises this
    for the repository's own refusal (fair-publishing's RecordNotDownloadable)."""


class TooLarge(Exception):
    """The file is over the byte limit the caller gave. A host raises this for the
    repository's own error (fair-publishing's FileTooLarge)."""


def too_large(max_bytes):
    return f"Too large to sample here (limit {max_bytes // 1_000_000} MB)"


def doi_url(doi):
    return f"{DOI_PREFIX}{doi}"


def origin_of(url):
    """`scheme://host[:port]` of an absolute http(s) URL, lower-cased, with a default port left out.
    None for anything else, including a URL with credentials, since it names no origin a record lives on."""
    try:
        parts = urlsplit((url or "").strip())
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in DEFAULT_PORTS or not parts.hostname or parts.username or parts.password:
        return None
    if port is None or port == DEFAULT_PORTS[scheme]:
        return f"{scheme}://{parts.hostname.lower()}"
    return f"{scheme}://{parts.hostname.lower()}:{port}"


def origin_for(record_url, doi):
    """The Zenodo origin a Publication's record is on: the origin of its record link, else the one its DOI
    prefix names. None when neither says, or when the two disagree, so a record of another host is never
    taken for the configured host's record with the same id (#122)."""
    found = {origin_of(record_url)} - {None}
    prefix = str(doi or "").split("/", 1)[0]
    if prefix in DOI_ORIGINS:
        found.add(DOI_ORIGINS[prefix])
    return found.pop() if len(found) == 1 else None


def creators_of(publication):
    return [creator for creator in publication.get("creators") or [] if isinstance(creator, dict)]


def author(name):
    """A creator's name as a citation holds it (#120): `Family, Given` as
    {family, given}; anything else, such as an organisation, as {literal}."""
    text = " ".join(str(name or "").split())
    if not text:
        return None
    family, comma, given = text.partition(",")
    if comma and family.strip() and given.strip():
        return {"family": family.strip(), "given": given.strip()}
    return {"literal": family.strip().rstrip(",") or text}


def year(published_at):
    """The publication year from an ISO timestamp, or None."""
    head = str(published_at or "")[:4]
    return int(head) if re.fullmatch(r"[0-9]{4}", head) else None


def citation(publication):
    """The fields a Dataset is cited from. A Community record has no revision of its own: Zenodo gives
    each version its own DOI, so the citation names the DOI and not a checksum (#122). Its access date is
    set when it is imported (imports._pinned)."""
    names = [author(creator.get("name")) for creator in creators_of(publication)]
    return {
        "title": publication["title"],
        "authors": [name for name in names if name],
        "year": year(publication.get("published_at")),
        "url": doi_url(publication["doi"]),
        "revision": None,
        "accessedAt": None,
        "bibtex": None,
    }


def summary(text, limit=SUMMARY_CHARACTERS):
    """The description as plain text: whitespace collapsed, cut at a word."""
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"


def card(publication):
    """A library card for one open dataset Publication, as fair-publishing lists it.

    A Publication says nothing about its task, so `tasks` is empty: the card
    matches a task filter only when no task is picked (search.py)."""
    record_id = publication["record_id"]
    creators = creators_of(publication)
    lead = author(creators[0].get("name")) if creators else None
    if lead is None:
        lead_text = None
    elif "literal" in lead:
        lead_text = lead["literal"]
    else:
        lead_text = f"{lead['family']}, {lead['given']}"
    return {
        "id": f"{PROVIDER}:{record_id}",
        "provider": PROVIDER,
        "externalId": record_id,
        "author": lead_text,
        "title": publication.get("title") or f"Zenodo record {record_id}",
        "url": doi_url(publication["doi"]),
        "doi": publication["doi"],
        "tasks": [],
        "license": resolve_license(publication.get("license")).id,
        "size": None,
        "gated": False,
        "description": summary(publication.get("description")),
        "updated": str(publication.get("published_at") or "")[:10] or None,
        "revision": None,
        "citation": citation(publication),
    }


def matches(publication, query):
    """Whether a Publication's title, description, DOI, record id or a creator's name
    contains the query, ignoring case. An empty query matches everything."""
    needle = (query or "").strip().lower()
    if not needle:
        return True
    haystack = [publication.get("title"), publication.get("description"), publication.get("doi"), publication.get("record_id")]
    haystack += [creator.get("name") for creator in creators_of(publication)]
    return any(needle in str(text).lower() for text in haystack if text)


def is_table(key):
    return isinstance(key, str) and key.lower().endswith(TABLE_EXTENSIONS)


def table_file(files):
    """The file a dataset is read from: `data<ext>` when there is one, else the first
    table file the record lists. None when it lists none."""
    tables = [entry for entry in files or [] if is_table(entry.get("key"))]
    for entry in tables:
        if os.path.splitext(entry["key"])[0].lower() == PREFERRED_STEM:
            return entry
    return tables[0] if tables else None


def refusal(record):
    """Why a record cannot be imported, as (code, message), or None when it can. `record`
    is what the repository says now: its access levels, whether it is downloadable, and its files."""
    access = record.get("access") if isinstance(record.get("access"), dict) else {}
    if access.get("record") != "public":
        return "restricted", RESTRICTED
    if access.get("files") != "public":
        return "metadata_only", METADATA_ONLY
    if not record.get("downloadable"):
        return "not_downloadable", NOT_DOWNLOADABLE
    if table_file(record.get("files")) is None:
        return "no_table", NO_TABLE
    return None
