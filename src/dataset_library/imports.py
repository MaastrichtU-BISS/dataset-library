"""Importing a sample of a Hugging Face dataset as the host's own Dataset (#113, #114).

    queued ─► fetching ─► sampling ─► validating ─► saving ─► completed
       ├──────────┴───────────┴────────────┴──────────┴─► failed
       └──────────┴───────────┴────────────┴─► cancelled (the student)

The import row is the source of truth. Every move is a conditional update
from the status the mover last read, so of two movers (the worker finishing,
the student cancelling) exactly one wins, and a terminal status never
changes. The worker claims ``saving`` before it writes the Dataset: a cancel
that lands first wins and no Dataset is made; one that lands after is refused.

While it runs, the worker refreshes ``heartbeat_at``; ``sweep`` fails imports
whose worker went quiet, and queued imports nothing picked up.

The host supplies how shards are fetched (``fetch_shards``, which takes the
revision the import was queued at as the keyword ``expected_revision`` and
refuses a conversion at another revision before it downloads anything (#174))
and how a file becomes a Dataset (``create_dataset``, which raises
``ValueError`` with a student-readable message for every refusal).
"""
import csv
import json
import logging
import math
import os
import random
import shutil
import threading
from array import array
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import sqlalchemy as sa
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.orm.exc import ObjectDeletedError

from . import community
from . import policy as policy_module
from . import snapshots as snapshots_module
from .errors import ProviderError
from .licenses import LICENSES, UNKNOWN, resolve_license
from .limits import import_cap_bytes
from .models import LibraryImport
from .ner import spans, tokens_to_text
from .providers import community as community_provider

logger = logging.getLogger("dataset_library")

QUEUED, FETCHING, SAMPLING, VALIDATING, SAVING = "queued", "fetching", "sampling", "validating", "saving"
COMPLETED, FAILED, CANCELLED = "completed", "failed", "cancelled"
TERMINAL = {COMPLETED, FAILED, CANCELLED}
RUNNING = {QUEUED, FETCHING, SAMPLING, VALIDATING, SAVING}
# Past these, the Dataset may already be written: too late to cancel.
CANCELLABLE = {QUEUED, FETCHING, SAMPLING, VALIDATING}
# Where an import's sample was read: Hugging Face, the platform's copy (#121), or a Zenodo record's file (#122).
HUGGING_FACE, SNAPSHOT, ZENODO = "huggingface", "snapshot", "zenodo"
COMMUNITY = community_provider.PROVIDER

UNEXPECTED = "The import stopped on our side. Try again, and tell your course staff if it keeps happening."
BUSY = "Hugging Face is busy. Try again in a few minutes."
COMMUNITY_BUSY = "Zenodo is busy. Try again in a few minutes."
COMMUNITY_NOT_OFFERED = "Importing from Zenodo is not set up on this platform."
NOT_STARTED = "Import could not start. Try again."
STOPPED = "Import stopped. Try again."
CHANGED = "This dataset changed on Hugging Face after you opened it. Open it again to import the new version."
BEHIND = (
    "Hugging Face hasn't converted this dataset's latest version yet, so it can't be imported now. "
    "Try again later."
)
COPY_GONE = "This dataset's copy is no longer kept here, so it cannot be imported. Try again in a few minutes."
# Shown when a balanced import falls back to a random one (#115).
LABEL_MISSING = "The label column was missing, so this is a random sample instead of a balanced one."
LABEL_EMPTY = "The label column was empty, so this is a random sample instead of a balanced one."
LABEL_LISTS = "The label column holds lists, so this is a random sample instead of a balanced one."
LABEL_TOO_MANY = "This label column has too many different values to balance, so this is a random sample instead."
LABEL_ROUGH = "Some labels had too few rows with text, so the sample is only roughly balanced."
# An NER import whose tags never cover their words (sub-word tags, for example)
# completes with every row left to annotate; the note says why.
NER_UNTAGGED = (
    "None of these rows had a tag for every word, so their entities were not "
    "imported: mark them in the annotation queue."
)
# Only these tasks have one class per row, so only they can be balanced by label.
BALANCEABLE = {"classification", "document_classification", "judgment_prediction"}
# These tasks store a list of BIO tags per row, not one label.
NER_TASKS = {"ner", "named_entity_recognition"}
# The host's training columns for the label side of each task (#117).
LABEL_HEADERS = {"ner": "entities", "named_entity_recognition": "entities", "summarization": "summary"}
NESTED_NER = (
    "These columns hold records, maps or lists of lists, so they can't be read as "
    "words and their entity tags. Pick a column of words and a column of tags."
)
NO_ENTITY_NAMES = (
    "We can't read entity names: this dataset's tags are numbers without names, "
    "so they can't become entity types."
)
# How much of a dataset an import may download before it samples.
MAX_FETCH_BYTES = 200_000_000
# How often a running import says it is alive, and how long a quiet one, or a
# queued one nothing picked up, is left before the sweep fails it.
HEARTBEAT_SECONDS = 15
STALE_AFTER = timedelta(minutes=2)
QUEUED_AFTER = timedelta(minutes=30)
# How long a failed import stays on the student's list unless put away.
RECENT = timedelta(days=1)


@dataclass
class Fetched:
    """Downloaded Parquet shards of one split, at the revision they were read
    from (#168): a Hub import records the conversion's source revision, a copy
    its own, and None when neither can say which one the rows came from."""

    paths: list = field(default_factory=list)
    revision: str | None = None
    # The dataset's current commit on the Hub, when the fetch asked: it is never
    # recorded, only used to say why a conversion at another revision is refused.
    head: str | None = None


class ImportFailed(Exception):
    """A refusal the student can act on: `message` is shown as it is."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


class Refused(Exception):
    """An import that cannot start; nothing was queued."""

    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def refusal_for(expected, revision, head):
    """The ImportFailed refusing an import queued at `expected` whose fetch read
    `revision`: `conversion_behind` when the Hub's head is still `expected` (the
    conversion has not caught up with the dataset, so opening it again would not
    help), `changed` when the dataset itself moved on. None when nothing refuses:
    either could not name its revision, or both name the same one. Shared by
    ``fetch_shards``, which raises it before any shard is downloaded (#174), and
    by ``_fetch``, which checks what a host's own fetch returned."""
    if not (expected and revision) or expected == revision:
        return None
    if head == expected:
        return ImportFailed("conversion_behind", BEHIND)
    return ImportFailed("changed", CHANGED)


def _facts(host, provider, external_id, config, split, binding=None):
    """The dataset as an import checks it, and the copy it reads (or None). When Hugging
    Face cannot be asked, or the viewer did not answer, the platform's copy of this part
    stands in, if it has one (#121); otherwise the preview is used as it is, and a
    ProviderError is raised only when there is no copy to fall back on.

    A Community record has no copy: it is read from its own file on Zenodo, and one that
    is not importable is refused with its reason (#122). Its eligibility is decided on a live
    record, and its Publication and file are the ``binding`` the import is bound to (#122, fix 3)."""
    if provider == COMMUNITY:
        try:
            body = community.view(host, external_id, pin=binding, fresh=True)
        except ProviderError as down:
            raise Refused(503, "provider_busy", COMMUNITY_BUSY) from down
        if body["preparing"]:
            # The record passed its live checks; only the cached head is missing (it expired, or was never read). The
            # worker reads the file and checks the columns itself, as for a file too large to preview (#122 round 3).
            body = {**body, "importable": True, "preview": "unseen"}
        if not body["importable"]:
            # A record that has closed or gone since the preview is refused in the words a queued import gets (#122,
            # fix 5); the restricted and metadata-only explanations are the preview's.
            if body["reasonCode"] in ("restricted", "metadata_only", "not_downloadable"):
                raise Refused(409, "not_downloadable", community_provider.NOT_DOWNLOADABLE)
            raise Refused(409, body["reasonCode"], body["reason"])
        return body, None
    from . import preview as preview_module  # lazy, as start() imports it

    try:
        dataset = preview_module.preview(host, external_id, config, split)
    except ProviderError:
        copy = snapshots_module.copy_for_offline(host, external_id, config, split)
        if copy is None:
            raise
        return snapshots_module.offline_detail(host, copy, rows=False), copy
    if dataset.get(preview_module.VIEWER_DOWN):
        copy = snapshots_module.copy_for_offline(host, external_id, config, split)
        if copy is not None:
            return snapshots_module.offline_detail(host, copy, rows=False), copy
    return dataset, None


def start(host, owner_id, body, *, ack_at=None, pin=None):
    """Check the request against the dataset as previewed, store the import as
    queued, then hand it to the host's queue. Returns the import as JSON, and
    whether it is new: the same import (dataset, revision, mapping and size),
    still queued or running, is returned as it is (a double click, a second tab).

    ``body.revision``, when given, is the revision the student saw; an import
    of anything newer is refused. ``ack_at`` is the confirmation time a retry
    carries from the import it repeats, so a retry keeps when the student
    actually agreed rather than claiming they agreed again now. ``pin`` is the
    Publication, origin and file a retry keeps from its import (#122, fix 3); a
    new Community import takes them as the library lists them now."""
    from . import preview as preview_module
    from .providers.huggingface import DATASET_ID_PATTERN, PrivateDataset

    provider = body.get("provider") or HUGGING_FACE
    external_id = body.get("dataset")
    if provider == COMMUNITY:
        if not isinstance(external_id, str) or not community_provider.RECORD_ID.match(external_id):
            raise Refused(400, "bad_id", "Pick a Zenodo record to import.")
    elif provider != HUGGING_FACE:
        raise Refused(400, "bad_provider", f"Unknown library source: {provider}")
    elif not isinstance(external_id, str) or not DATASET_ID_PATTERN.match(external_id):
        raise Refused(400, "bad_id", "Pick a Hugging Face dataset to import.")
    task_type = body.get("taskType") or "classification"
    if task_type not in host.task_types:
        raise Refused(400, "bad_task", f"Unknown task type: {task_type}")
    name = body.get("name")
    name = name.strip() if isinstance(name, str) else ""
    if len(name) > 255:
        raise Refused(400, "bad_name", "Keep the name under 256 characters.")
    size = body.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise Refused(400, "bad_size", "Say how many rows to import, as a whole number above 0.")
    binding = None
    try:
        if provider == COMMUNITY:
            binding = community.bind(host, external_id, pin)
        dataset, copy = _facts(host, provider, external_id, body.get("config") or None, body.get("split") or None, binding)
    except preview_module.LicenseHidden:
        raise Refused(403, "license_hidden", "This dataset's license is not allowed on this platform, so it cannot be imported.")
    except preview_module.NotInLibrary:
        raise Refused(404, "not_found", "That dataset is not in the library.")
    except PrivateDataset:
        raise Refused(409, "private", "Its owner keeps this dataset private on Hugging Face, so it cannot be imported.")
    except preview_module.UnknownConfig:
        raise Refused(400, "bad_config", "That dataset has no such configuration or split.")
    if dataset["gated"]:
        raise Refused(409, "gated", "Its owner only shares it with people signed in on Hugging Face, so it cannot be imported.")
    if provider == HUGGING_FACE and dataset["preview"] != "ok":
        raise Refused(409, "not_importable", "Hugging Face cannot show this dataset's rows, so it cannot be imported.")
    columns = {c["name"] for c in dataset["columns"]}
    text_column, label_column = body.get("textColumn"), body.get("labelColumn") or None
    # A Community file too large to preview, or whose preview is not cached, has no columns to check here; the worker
    # checks the names it reads (#122).
    unseen = provider == COMMUNITY and dataset["preview"] in ("too_large", "unseen")
    if unseen:
        mapped = isinstance(text_column, str) and text_column.strip() != ""
        labelled = label_column is None or isinstance(label_column, str)
    else:
        mapped = isinstance(text_column, str) and text_column in columns
        labelled = label_column is None or (isinstance(label_column, str) and label_column in columns)
    if not mapped:
        raise Refused(400, "bad_mapping", "Pick the column that holds the text.")
    if not labelled:
        raise Refused(400, "bad_mapping", "Pick a label column from this dataset, or none.")
    if label_column is not None and label_column == text_column:
        raise Refused(400, "bad_mapping", "Pick different columns for the text and the label.")
    balance = body.get("balance")
    if balance is not None and not isinstance(balance, bool):
        raise Refused(400, "bad_balance", "Say whether to keep the label balance.")
    # A balance means nothing without a label, and only these tasks carry one
    # class per row; anything else stores no balance whatever was sent.
    balance = label_column is not None and task_type in BALANCEABLE and (balance is None or balance is True)
    if dataset["license"]["level"] == "warn" and body.get("licenseAck") is not True:
        raise Refused(409, "needs_ack", "Confirm the license terms to import this dataset.")
    # Only a license that asks for confirmation records a new time; an allow
    # license stores none even when the client sent `licenseAck: true`. A retry
    # keeps the time the student first confirmed, even if an admin has since
    # set the license to allow: the confirmation still happened.
    confirmed = dataset["license"]["level"] == "warn" and body.get("licenseAck") is True
    license_ack_at = ack_at or (_now() if confirmed else None)
    seen_revision = body.get("revision")
    if seen_revision and dataset.get("revision") and seen_revision != dataset["revision"]:
        raise Refused(409, "changed", community_provider.CHANGED if provider == COMMUNITY else CHANGED)

    session = host.db.session
    cap = host.row_cap()
    running = (
        session.query(LibraryImport)
        .filter(
            LibraryImport.owner_id == owner_id,
            LibraryImport.provider == provider,
            LibraryImport.external_id == external_id,
            LibraryImport.config.is_(None) if dataset["config"] is None else LibraryImport.config == dataset["config"],
            LibraryImport.split.is_(None) if dataset["split"] is None else LibraryImport.split == dataset["split"],
            LibraryImport.revision.is_(None) if dataset.get("revision") is None else LibraryImport.revision == dataset["revision"],
            LibraryImport.text_column == text_column,
            LibraryImport.label_column.is_(None) if label_column is None else LibraryImport.label_column == label_column,
            LibraryImport.task_type == task_type,
            LibraryImport.sample_size == min(size, cap),
            LibraryImport.balance == balance,
            LibraryImport.status.in_(RUNNING),
        )
        .order_by(LibraryImport.created_at.desc())
        .first()
    )
    if running is not None:
        _log(running, "repeated")
        return running.to_dict(), False
    earlier = session.query(LibraryImport).filter_by(owner_id=owner_id, provider=provider, external_id=external_id).count()
    record = LibraryImport(
        owner_id=owner_id, provider=provider, external_id=external_id,
        config=dataset["config"], split=dataset["split"], revision=dataset.get("revision"), url=dataset["url"],
        license=dataset["license"]["id"], license_ack_at=license_ack_at,
        text_column=text_column, label_column=label_column, task_type=task_type,
        sample_size=min(size, cap), max_rows=cap, balance=balance,
        name=name or (dataset["title"] if not earlier else f"{dataset['title']} ({earlier + 1})"),
        # A Community citation names its DOI, not a revision: Zenodo gives each version a DOI of its own (#122).
        citation=_pinned(dataset.get("citation"), None if provider == COMMUNITY else dataset.get("revision"), _now()),
        source=_source(provider, copy),
        snapshot_id=None if copy is None else copy.id,
        # The Publication, origin and file this import is bound to, so a queued import checks them again (#122).
        publication_id=binding.publication_id if binding else None,
        origin=binding.origin if binding else None,
        file_key=dataset["file"]["key"] if provider == COMMUNITY else None,
        status=QUEUED,
    )
    session.add(record)
    session.commit()
    _log(record, QUEUED, size=record.sample_size)
    try:
        host.enqueue(record.id)
    except Exception:  # noqa: BLE001 - whatever the queue raised, nothing will run it
        logger.exception(f"library.import.enqueue_failed import={record.id}")
        record.status, record.error_code, record.error_message = FAILED, "not_started", NOT_STARTED
        session.commit()
        raise Refused(503, "not_started", NOT_STARTED)
    return record.to_dict(), True


def _source(provider, copy):
    if provider == COMMUNITY:
        return ZENODO
    return HUGGING_FACE if copy is None else SNAPSHOT


def _owned(host, owner_id, import_ref):
    record = host.db.session.get(LibraryImport, import_ref)
    return record if record is not None and record.owner_id == owner_id else None


def get(host, owner_id, import_ref):
    record = _owned(host, owner_id, import_ref)
    return record.to_dict() if record is not None else None


def for_owner(host, owner_id):
    found = host.db.session.query(LibraryImport).filter_by(owner_id=owner_id).order_by(LibraryImport.created_at.desc())
    return [record.to_dict() for record in found]


def provenance(host, owner_id, dataset_ref):
    """The most recent completed import that made this Dataset, as its
    provenance: where it came from, under which license, and when the student
    confirmed that license. The license's ``level`` is the platform's level
    now, which an admin can have changed since the import. None when this
    owner has no such import. Reads the database only, so it works without an
    import queue."""
    record = (
        host.db.session.query(LibraryImport)
        .filter_by(owner_id=owner_id, dataset_ref=dataset_ref, status=COMPLETED)
        .order_by(sa.func.coalesce(LibraryImport.imported_at, LibraryImport.created_at).desc())
        .first()
    )
    if record is None:
        return None
    license = LICENSES.get(record.license, UNKNOWN)
    level = policy_module.level(license, policy_module.overrides(host.db))
    found = {
        "provider": record.provider,
        "externalId": record.external_id,
        "config": record.config,
        "split": record.split,
        "revision": record.revision,
        # "snapshot" when the platform's copy was read, at the copy's revision (#121); "zenodo" for a Community record.
        "source": record.source,
        "url": record.url,
        "importedAt": _iso(record.imported_at),
        "licenseAckAt": _iso(record.license_ack_at),
        "license": license.at_level(level),
        "balance": record.balance,
        "sampleMethod": record.sample_method,
        "sampleNote": record.sample_note,
        "citation": record.citation,
    }
    if record.provider == COMMUNITY:
        # The DOI is the record's URL, so it is not stored twice.
        found["doi"] = (record.url or "").removeprefix(community_provider.DOI_PREFIX) or None
    return found


def _iso(value):
    return value.isoformat() + "Z" if value else None


def _pinned(citation, revision, accessed):
    """A citation as an import holds it: at the revision the import read and the
    time it read it. None when there is no citation to pin, which is an import
    made before citations existed (a preview cached before them is rebuilt, so a
    new import always has one)."""
    if not citation:
        return None
    return {**citation, "revision": revision, "accessedAt": _iso(accessed)}


def current(host, owner_id, now=None):
    """What the student is still following: imports queued or running, and
    failed ones from the last day not put away. Newest first."""
    since = (now or _now()) - RECENT
    found = (
        host.db.session.query(LibraryImport)
        .filter(
            LibraryImport.owner_id == owner_id,
            sa.or_(
                LibraryImport.status.in_(RUNNING),
                sa.and_(
                    LibraryImport.status == FAILED,
                    LibraryImport.dismissed_at.is_(None),
                    LibraryImport.updated_at >= since,
                ),
            ),
        )
        .order_by(LibraryImport.created_at.desc())
    )
    return [record.to_dict() for record in found]


def _move(session, record, state, from_states, **values):
    """Move `record` to `state` if it is still in one of `from_states`, in one
    conditional update. Returns whether it moved; `record` is refreshed."""
    moved = session.execute(
        sa.update(LibraryImport)
        .where(LibraryImport.id == record.id, LibraryImport.status.in_(from_states))
        .values(status=state, updated_at=_now(), **values)
        .execution_options(synchronize_session=False)
    ).rowcount
    session.commit()
    try:
        session.refresh(record)
    except (ObjectDeletedError, InvalidRequestError):
        return False
    return bool(moved)


def cancel(host, owner_id, import_ref):
    """Cancel an import that has not started saving. Returns it as JSON, None
    when there is no such import; refuses one that is saving or done. Cancelling
    a failed or cancelled import changes nothing."""
    record = _owned(host, owner_id, import_ref)
    if record is None:
        return None
    if record.status in CANCELLABLE and _move(host.db.session, record, CANCELLED, CANCELLABLE):
        _log(record, CANCELLED)
        return record.to_dict()
    if record.status == SAVING:
        raise Refused(409, "too_late", "It is already saving the dataset, so it can no longer be cancelled.")
    if record.status == COMPLETED:
        raise Refused(409, "too_late", "This import already finished.")
    return record.to_dict()


def retry(host, owner_id, import_ref):
    """Start a failed or cancelled import again with the same settings, at the
    revision the student saw, and put the old one away. Returns the new import."""
    record = _owned(host, owner_id, import_ref)
    if record is None:
        return None
    if record.status not in (FAILED, CANCELLED):
        raise Refused(409, "not_retryable", "Only a failed or cancelled import can be tried again.")
    body = {
        "provider": record.provider,
        "dataset": record.external_id,
        "config": record.config,
        "split": record.split,
        "revision": record.revision,
        "textColumn": record.text_column,
        "labelColumn": record.label_column,
        "taskType": record.task_type,
        "size": record.sample_size,
        "balance": record.balance,
        "name": record.name,
        "licenseAck": record.license_ack_at is not None,
    }
    # A Community retry keeps the Publication, origin and file its import was started on, never a record id's other Publication (#122, fix 3).
    pin = community.Pin(record.publication_id, record.origin, record.file_key) if record.provider == COMMUNITY else None
    started, _ = start(host, owner_id, body, ack_at=record.license_ack_at, pin=pin)
    if record.dismissed_at is None:
        record.dismissed_at = _now()
        host.db.session.commit()
    _log(record, "retried", retry=started["id"])
    return started


def dismiss(host, owner_id, import_ref):
    """Put away a finished import, so it leaves the student's list. Returns it,
    or None when there is no such import; a running one must be cancelled first."""
    record = _owned(host, owner_id, import_ref)
    if record is None:
        return None
    if record.status in RUNNING:
        raise Refused(409, "running", "Cancel the import first.")
    if record.dismissed_at is None:
        record.dismissed_at = _now()
        host.db.session.commit()
    return record.to_dict()


def sweep(session, *, stale_after=STALE_AFTER, queued_after=QUEUED_AFTER, now=None):
    """Fail running imports whose worker went quiet ("Import stopped") and queued
    ones nothing picked up. Safe to run anywhere, any number of times; returns
    the refs it failed."""
    now = now or _now()
    quiet = sa.func.coalesce(LibraryImport.heartbeat_at, LibraryImport.updated_at) < now - stale_after
    candidates = session.query(LibraryImport).filter(
        sa.or_(
            sa.and_(LibraryImport.status.in_(RUNNING - {QUEUED}), quiet),
            sa.and_(LibraryImport.status == QUEUED, LibraryImport.created_at < now - queued_after),
        )
    )
    swept = []
    for record in candidates.all():
        code, message = ("not_started", NOT_STARTED) if record.status == QUEUED else ("stopped", STOPPED)
        if _move(session, record, FAILED, {record.status}, error_code=code, error_message=message):
            _log(record, "swept", code=code)
            swept.append(record.id)
    return swept


def touch(engine, import_ref):
    """Say the import's worker is alive. Its own connection, so it can run
    beside the worker's session on another thread."""
    with engine.begin() as connection:
        connection.execute(
            sa.update(LibraryImport.__table__)
            .where(LibraryImport.id == import_ref, LibraryImport.status.in_(RUNNING - {QUEUED}))
            .values(heartbeat_at=_now())
        )


class _Heartbeat:
    """Calls `touch` every `seconds` on a daemon thread until stopped."""

    def __init__(self, engine, import_ref, seconds):
        self.engine, self.import_ref, self.seconds = engine, import_ref, seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"library-import-{import_ref}", daemon=True)

    def _run(self):
        while not self._stop.wait(self.seconds):
            try:
                touch(self.engine, self.import_ref)
            except Exception:  # noqa: BLE001 - a missed beat is only a later sweep
                logger.warning(f"library.import.heartbeat_failed import={self.import_ref}", exc_info=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=self.seconds)


class _Stopped(Exception):
    """The import reached a terminal status elsewhere (cancelled, say)."""


def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _log(record, state, **fields):
    extra = "".join(f" {k}={v}" for k, v in fields.items())
    logger.info(f"library.import.{state} import={record.id} dataset={record.external_id}{extra}")


def run_import(
    import_ref,
    *,
    session,
    fetch_shards,
    create_dataset,
    data_dir,
    max_bytes=MAX_FETCH_BYTES,
    seed=None,
    open_parquet=None,
    heartbeat_seconds=HEARTBEAT_SECONDS,
    snapshot_dir=None,
    community_datasets=None,
    community_base_url=None,
    community_record=None,
    community_download=None,
):
    """Drive one queued import to completed or failed. Safe to call twice.
    ``max_bytes`` is the worker's download cap; the admin's cap applies only up to it (#122).
    ``heartbeat_seconds`` is how often the import says it is alive (None: never).
    ``snapshot_dir`` is where the platform's copies live; without it, no copy is read.
    ``community_datasets``, ``community_base_url``, ``community_record`` and
    ``community_download`` are the Community seams (see ``init_dataset_library``); a Community
    import needs all four, and checks its Publication, the source and the license again before it reads."""
    record = session.get(LibraryImport, import_ref)
    if record is None or record.status != QUEUED:
        return
    max_bytes = import_cap_bytes(session, max_bytes)
    seams = None
    if None not in (community_datasets, community_base_url, community_record, community_download):
        seams = community.Seams(
            datasets=community_datasets, base_url=community_base_url, record=community_record, download=community_download,
        )
    provider, external_id = record.provider, record.external_id
    work = os.path.join(data_dir, "library-imports", record.id)
    path = os.path.join(data_dir, f"library-{record.id}.csv")
    heartbeat = _Heartbeat(session.get_bind(), record.id, heartbeat_seconds) if heartbeat_seconds else None
    try:
        if heartbeat:
            heartbeat.__enter__()
        _advance(session, record, FETCHING)
        if provider == COMMUNITY:
            fetched, copy_updates = _fetch_community(session, record, seams, max_bytes, work), {}
        else:
            fetched, copy_updates = _fetch(session, record, fetch_shards, max_bytes, work, snapshot_dir)
        # The import records the revision its rows were read from, whatever the
        # fetch could name; one that could not say records none (#120, #168).
        _advance(session, record, SAMPLING, {"revision": fetched.revision, **copy_updates})
        rows, method, note = _sample(record, fetched.paths, random.Random(seed), open_parquet)
        _advance(session, record, VALIDATING, {"sample_method": method, "sample_note": note})
        _write_csv(path, record, rows)
        # The last check before the Dataset is written: a cancel before this wins.
        _advance(session, record, SAVING)
        try:
            dataset_ref = create_dataset(record.owner_id, path, record.name, _description(record), record.task_type)
        except ValueError as exc:
            raise ImportFailed("invalid", str(exc)) from exc
        imported = _now()
        _advance(session, record, COMPLETED, {
            "dataset_ref": str(dataset_ref), "rows": len(rows), "imported_at": imported,
            # Cited at the revision that was read and the day it was imported.
            "citation": _pinned(record.citation, None if provider == COMMUNITY else record.revision, imported),
        })
    except _Stopped:
        _remove(path)
    except ImportFailed as exc:
        _remove(path)
        _fail(session, record, exc.code, exc.message)
    except ProviderError:
        _fail(session, record, "provider_busy", BUSY if provider == HUGGING_FACE else COMMUNITY_BUSY)
    except Exception:  # noqa: BLE001 - an import must end, whatever broke
        logger.exception(f"library.import.error import={record.id}")
        _remove(path)
        _fail(session, record, "unexpected", UNEXPECTED)
    finally:
        if heartbeat:
            heartbeat.__exit__(None, None, None)
        shutil.rmtree(work, ignore_errors=True)
        if snapshot_dir is not None and provider == HUGGING_FACE:
            _release_copies(session, snapshot_dir, provider, external_id)


def _fetch(session, record, fetch_shards, max_bytes, work, snapshot_dir):
    """The sample's Parquet files, and what to record about where they came from.

    An import queued from a copy reads that copy. Otherwise Hugging Face is asked,
    and when it cannot be, a copy of the same part of the dataset stands in, at the
    copy's revision (#121). The fetch names the revision its rows were read from:
    the converted files' source revision (#168). The import was queued at the
    revision the student saw, so a conversion still at that revision imports; one
    that lags behind it, or was rebuilt from a different commit, is refused."""
    if record.source == SNAPSHOT:
        copy = snapshots_module.reading(session, record.snapshot_id)
        if copy is None or snapshot_dir is None:
            raise ImportFailed("copy_gone", COPY_GONE)
        return _from_copy(snapshot_dir, copy), {}
    try:
        fetched = fetch_shards(
            record.external_id, record.config, record.split, max_bytes, work,
            expected_revision=record.revision,
        )
    except ProviderError:
        copy = snapshots_module.fallback_for(session, record, snapshot_dir)
        if copy is None:
            raise
        return _from_copy(snapshot_dir, copy), {"source": SNAPSHOT, "snapshot_id": copy.id, "revision": copy.revision}
    # A safety net for a host's own fetch: this one has already refused before
    # it downloaded (#174).
    refusal = refusal_for(record.revision, fetched.revision, fetched.head)
    if refusal is not None:
        raise refusal
    return fetched, {}


def _fetch_community(session, record, seams, max_bytes, work):
    """The sample's table as one Parquet file, read from the record's own file on Zenodo (#122).

    Never read from the local Dataset. The queued import is checked first, against what the student agreed to: its
    Publication is still listed on this Zenodo host, the source is still on, its license is the one it started on, is
    not now hidden, and a license that warns was confirmed. Then the record is asked again, and exactly the file the
    import was started on is read. A record closed or gone, or one left without that file, fails with the same words as a
    file that is gone; the worker never picks another table. The bytes must match the checksum the import was started on."""
    if seams is None:
        raise ImportFailed("not_offered", COMMUNITY_NOT_OFFERED)
    _still_offered(session, record, seams)
    try:
        listing = seams.record(record.external_id)
    except community_provider.NotDownloadable as gone:
        raise ImportFailed("not_downloadable", community_provider.NOT_DOWNLOADABLE) from gone
    # The pinned file is looked for first: a record with only a README left has no file to read (#122, fix 3).
    entry = community_provider.pinned_file(listing, record.file_key)
    if entry is None:
        raise ImportFailed("not_downloadable", community_provider.NOT_DOWNLOADABLE)
    if record.revision and entry.get("checksum") != record.revision:
        raise ImportFailed("changed", community_provider.CHANGED)
    if entry.get("size") is not None and entry["size"] > max_bytes:
        raise ImportFailed("too_large", community_provider.too_large(max_bytes))
    try:
        from . import tables  # lazy: pyarrow is the [import] extra, which only an import needs
        import pyarrow.parquet as pq
    except ImportError as missing:
        raise ImportFailed("unreadable", community_provider.NO_READER) from missing
    try:
        data = seams.download(record.external_id, entry["key"], max_bytes)
    except community_provider.NotDownloadable as gone:
        raise ImportFailed("not_downloadable", community_provider.NOT_DOWNLOADABLE) from gone
    except community_provider.TooLarge as big:
        raise ImportFailed("too_large", community_provider.too_large(max_bytes)) from big
    if record.revision and not community_provider.checksum_matches(data, record.revision):
        raise ImportFailed("changed", community_provider.CHANGED)
    os.makedirs(work, exist_ok=True)
    path = os.path.join(work, "sample.parquet")
    try:
        tables.write_parquet(data, entry["key"], path)
    except tables.Unreadable as unreadable:
        raise ImportFailed("unreadable", unreadable.message) from unreadable
    names = pq.read_schema(path).names
    for column in (record.text_column, record.label_column):
        if column and column not in names:
            raise ImportFailed("bad_mapping", f"This file has no column named {column}, so it cannot be imported.")
    return Fetched(paths=[path], revision=entry.get("checksum") or record.revision)


def _still_offered(session, record, seams):
    """Refuse a queued Community import whose dataset has changed, or left the library, since it was started (#122, fix 3):
    the source is switched off; its Publication is no longer listed on the configured Zenodo host, or not on the origin it
    was started on; its license is now hidden, is not the license it was started on (the student must start it again), or
    warns with no confirmation on record. The license is the Publication's current one, as the library resolves it. Reads the
    database and the listing, never the local Dataset."""
    if not community.switched_on(session):
        raise ImportFailed("source_off", community_provider.SOURCE_OFF)
    configured = community_provider.origin_of(seams.base_url())
    publication = next((listed for listed in seams.datasets() if listed.get("id") == record.publication_id), None)
    if (
        publication is None
        or publication.get("record_id") != record.external_id
        or configured is None
        or community_provider.origin_of(publication.get("origin")) != configured
        or community_provider.origin_of(record.origin) != configured
    ):
        raise ImportFailed("not_in_library", community_provider.NOT_LISTED)
    license = resolve_license(publication.get("license"))
    if license.id != record.license:
        raise ImportFailed("license_changed", community_provider.LICENSE_CHANGED)
    level = policy_module.level(license, policy_module.for_session(session))
    if level == "hide":
        raise ImportFailed("license_hidden", community_provider.LICENSE_HIDDEN)
    if level == "warn" and record.license_ack_at is None:
        raise ImportFailed("needs_ack", community_provider.NEEDS_ACK)


def _from_copy(snapshot_dir, copy):
    return Fetched(paths=[snapshots_module.file_of(snapshot_dir, copy)], revision=copy.revision)


def _release_copies(session, snapshot_dir, provider, external_id):
    """Once an import ends, delete its dataset's removed copy if no other import still
    reads it. Never raises: the import has already ended, and a sweep retries."""
    try:
        session.rollback()
        snapshots_module.release(session, snapshot_dir, provider, external_id)
    except Exception:  # noqa: BLE001 - a copy left behind is removed by the sweep
        logger.exception(f"library.snapshot.release_failed dataset={external_id}")


def _remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _advance(session, record, state, updates=None):
    """Move to `state` from the status last read, saving `updates` with it; stop
    when the import ended elsewhere (cancelled, swept) or its row is gone
    (account deletion)."""
    try:
        session.refresh(record)
    except (ObjectDeletedError, InvalidRequestError):
        raise _Stopped()
    if record.status in TERMINAL or not _move(session, record, state, {record.status}, heartbeat_at=_now(), **(updates or {})):
        raise _Stopped()
    _log(record, state, **({"rows": record.rows} if state == COMPLETED else {}))


def _fail(session, record, code, message):
    session.rollback()
    if _move(session, record, FAILED, RUNNING, error_code=code, error_message=message):
        _log(record, FAILED, code=code)


def _label_names(parquet_file, column):
    """The class label names for `column`: a label, or a row's list of tags."""
    metadata = parquet_file.schema_arrow.metadata or {}
    try:
        features = json.loads(metadata[b"huggingface"])["info"]["features"]
    except (KeyError, ValueError, TypeError, AttributeError):
        return None
    if not isinstance(features, dict):
        return None
    return _class_label_names(features.get(column) or {})


def _class_label_names(feature):
    """Names from a ClassLabel, or from the ClassLabel inside a Sequence."""
    if isinstance(feature, list):
        for item in feature:
            names = _class_label_names(item)
            if names is not None:
                return names
        return None
    if not isinstance(feature, dict):
        return None
    if feature.get("_type") == "ClassLabel":
        return feature.get("names")
    if feature.get("_type") in ("Sequence", "List", "LargeList"):
        return _class_label_names(feature.get("feature"))
    return None


def _refuse_numbered_ner_tags(record, files):
    """Fail before any row is read when NER columns cannot become words and
    entity types: records, maps, lists of those or of lists, or numbered tags
    the files do not name."""
    if record.task_type not in NER_TASKS:
        return
    for parquet_file in files:
        schema = parquet_file.schema_arrow
        for column in (record.text_column, record.label_column):
            if column and column in schema.names and _not_flat(schema.field(column).type):
                raise ImportFailed("nested_columns", NESTED_NER)
        if not record.label_column or record.label_column not in schema.names:
            continue
        # A text that is already a string gets blank entities (its tags have
        # no tokens to sit on), so its tags' names do not matter.
        text_is_tokens = record.text_column in schema.names and _is_list(schema.field(record.text_column).type)
        if text_is_tokens and _integer_list(schema.field(record.label_column).type) and not _label_names(parquet_file, record.label_column):
            raise ImportFailed("no_entity_names", NO_ENTITY_NAMES)


def _not_flat(arrow_type):
    """Whether an NER column is something other than a string or a flat list
    of words or tags: a record (Hugging Face stores a sequence of span dicts
    as a record of lists), a map, or a list of lists, records or maps."""
    import pyarrow.types as pat

    if pat.is_struct(arrow_type) or pat.is_map(arrow_type):
        return True
    return _is_list(arrow_type) and pat.is_nested(arrow_type.value_type)


def _is_list(arrow_type):
    import pyarrow.types as pat

    return pat.is_list(arrow_type) or pat.is_large_list(arrow_type) or pat.is_fixed_size_list(arrow_type)


def _integer_list(arrow_type):
    """Whether an Arrow type is a list of integers."""
    import pyarrow.types as pat

    return _is_list(arrow_type) and pat.is_integer(arrow_type.value_type)


def _sample(record, paths, rng, open_parquet):
    """Pick `sample_size` rows with text, reading as little as possible; balanced
    on the label column when asked and it has labels.

    Returns the rows, the method used (``"stratified"`` or ``"random"``) and a
    note when a balance was asked for but could not be kept, or when no NER
    row had a tag for every word."""
    if open_parquet is None:
        import pyarrow.parquet as pq

        open_parquet = pq.ParquetFile
    files = [open_parquet(path) for path in paths]
    _refuse_numbered_ner_tags(record, files)
    if record.balance and record.label_column:
        rows, note = _sample_balanced(record, files, rng)
        if rows is not None:
            rng.shuffle(rows)
            return rows, "stratified", note
        rows = _sample_random(record, files, rng)
        rng.shuffle(rows)
        return rows, "random", note
    rows = _sample_random(record, files, rng)
    rng.shuffle(rows)
    untagged = record.task_type in NER_TASKS and record.label_column and all(row[1] == "" for row in rows)
    return rows, "random", NER_UNTAGGED if untagged else None


def _sample_random(record, files, rng):
    """Pick rows uniformly at random.

    Pass 1 reads only each row group's row count; pass 2 reads only the mapped
    columns of the row groups that hold a chosen row. A summarization row needs
    both a text and a summary, so one with a missing summary is replaced like
    one with empty text."""
    groups, total = _groups(files)
    columns = [record.text_column] + ([record.label_column] if record.label_column else [])
    names = [_label_names(f, record.label_column) if record.label_column else None for f in files]
    needs_summary = record.task_type == "summarization" and record.label_column is not None

    want = min(record.sample_size, total)
    kept, seen = [], set()
    while len(kept) < want and len(seen) < total:
        batch = _draw(rng, total, seen, want - len(kept))
        for _, values, f_index in _read(files, groups, batch, columns):
            if not _text(record.task_type, values[0]).strip():
                continue
            if needs_summary and _empty_label(values[1]):
                continue
            kept.append((values, f_index))
    if not kept:
        if needs_summary:
            raise ImportFailed(
                "no_usable_rows",
                "None of the rows we read had both a text and a summary, so there is nothing to import.",
            )
        raise ImportFailed("no_usable_rows", "None of the rows we read had any text, so there is nothing to import.")
    return [_mapped(values, names[f_index], record.task_type) for values, f_index in kept[:want]]


def _sample_balanced(record, files, rng):
    """Pick rows so each label keeps its share of the whole dataset.

    Pass 1 reads only the label column of every row group to count labels and
    keep each row's place. Largest remainder gives every label floor or ceil of
    its share, so their total is exactly the sample size. Pass 2 then reads
    only the mapped columns of the row groups holding chosen rows, replacing
    rows whose text is empty from the same label. When a label runs out of
    rows with text, the shortfall is topped up from the others' unused rows.
    Returns ``(None, note)`` when the files cannot be balanced and a random
    sample should take over."""
    import pyarrow.types as pat

    label = record.label_column
    if any(label not in f.schema_arrow.names for f in files):
        return None, LABEL_MISSING
    # A list or struct is not a class label; it is not even hashable.
    if any(pat.is_nested(f.schema_arrow.field(label).type) for f in files):
        return None, LABEL_LISTS
    groups, _ = _groups(files)
    names = [_label_names(f, label) for f in files]
    by_label = {}
    for f_index, g, first, _ in groups:
        values = files[f_index].read_row_group(g, columns=[label]).to_pydict()[label]
        for offset, value in enumerate(values):
            if _empty_label(value, names[f_index]):
                continue
            by_label.setdefault(value, array("q")).append(first + offset)
            # A free-text column has a value per row: give up before indexing
            # all of it, since it can never be balanced into this sample.
            if len(by_label) > record.sample_size:
                return None, LABEL_TOO_MANY
    if not by_label:
        return None, LABEL_EMPTY
    want = min(record.sample_size, sum(len(rows) for rows in by_label.values()))
    # A free-text or summary column would otherwise be "balanced" by alphabet.
    if len(by_label) > want:
        return None, LABEL_TOO_MANY
    allocations = _allocate({value: len(rows) for value, rows in by_label.items()}, want, rng)
    columns = [record.text_column, label]

    # The first draw per label; later draws come from a shuffled pointer into
    # the label's unused rows, so a whole bucket is never shuffled up front.
    pending = {}
    attempted = {}
    spare = {}
    spare_at = {}
    for value, rows in by_label.items():
        picked = rng.sample(range(len(rows)), allocations.get(value, 0))
        attempted[value] = set(picked)
        pending[value] = [rows[position] for position in picked]

    def take_spare(value):
        if value not in spare:
            spare[value] = array("q", (i for i in range(len(by_label[value])) if i not in attempted[value]))
            rng.shuffle(spare[value])
            spare_at[value] = 0
        positions = spare[value]
        while spare_at[value] < len(positions):
            position = positions[spare_at[value]]
            spare_at[value] += 1
            if position not in attempted[value]:
                attempted[value].add(position)
                return by_label[value][position]
        return None

    chosen = []
    while any(pending.values()):
        row_label = {}
        rows = []
        for value, picked in pending.items():
            for row in picked:
                row_label[row] = value
                rows.append(row)
        pending = {value: [] for value in pending}
        for row, values, f_index in _read(files, groups, sorted(rows), columns):
            value = row_label[row]
            if not _text(record.task_type, values[0]).strip():
                replacement = take_spare(value)
                if replacement is not None:
                    pending[value].append(replacement)
                continue
            chosen.append((values, f_index))
    # Some labels had too few rows with text: fill the exact size from the
    # labels that still have rows, largest first.
    topped = False
    if len(chosen) < want:
        remaining = {value: len(rows) - len(attempted[value]) for value, rows in by_label.items()}
        order = sorted(
            (value for value, left in remaining.items() if left > 0),
            key=lambda value: (-remaining[value], rng.random()),
        )
        for value in order:
            while len(chosen) < want:
                batch = []
                while len(batch) < want - len(chosen):
                    row = take_spare(value)
                    if row is None:
                        break
                    batch.append(row)
                if not batch:
                    break
                for _, values, f_index in _read(files, groups, sorted(batch), columns):
                    if not _text(record.task_type, values[0]).strip():
                        continue
                    chosen.append((values, f_index))
                    topped = True
            if len(chosen) >= want:
                break
    # Checked after the top-up: a label with no place of its own may still
    # hold the only rows with text.
    if not chosen:
        raise ImportFailed("no_usable_rows", "None of the rows we read had any text, so there is nothing to import.")
    # Short of the size, or filled from other labels: either way the
    # proportions are not the ones asked for, and the note says so.
    rough = topped or len(chosen) < want
    return [_mapped(values, names[f_index], record.task_type) for values, f_index in chosen], (LABEL_ROUGH if rough else None)


def _groups(files):
    """Every row group as (file index, group, first row number, row count)."""
    groups = []
    total = 0
    for f_index, parquet_file in enumerate(files):
        metadata = parquet_file.metadata
        for g in range(metadata.num_row_groups):
            count = metadata.row_group(g).num_rows
            groups.append((f_index, g, total, count))
            total += count
    return groups, total


def _empty_label(value, names=None):
    """Whether a label value says nothing: null, blank, NaN, or an HF `-1`."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    # A ClassLabel's -1 means "no label"; it is not a label of its own.
    return names is not None and isinstance(value, int) and value < 0


def _allocate(counts, want, rng):
    """Largest remainder: each label gets floor or ceil of its share, total `want`.
    Equal remainders are broken at random, never by the label's text."""
    total = sum(counts.values())
    allocated = {}
    remainders = {}
    for value, count in counts.items():
        shared, remainder = divmod(want * count, total)
        allocated[value] = shared
        remainders[value] = remainder
    left = want - sum(allocated.values())
    order = sorted(counts, key=lambda value: (-remainders[value], rng.random()))
    for value in order:
        if left == 0:
            break
        if allocated[value] < counts[value]:
            allocated[value] += 1
            left -= 1
    return allocated


def _draw(rng, total, seen, k):
    """`k` row numbers not drawn before."""
    left = total - len(seen)
    k = min(k, left)
    if len(seen) > total // 2:
        picked = rng.sample([i for i in range(total) if i not in seen], k)
    else:
        picked = set()
        while len(picked) < k:
            i = rng.randrange(total)
            if i not in seen:
                picked.add(i)
        picked = sorted(picked)
    seen.update(picked)
    return sorted(picked)


def _read(files, groups, rows, columns):
    """The chosen rows' mapped columns, one row group read at a time, with each
    row's number and file so its label can be placed and named."""
    out = []
    g_index = 0
    by_group = {}
    for row in rows:
        while not (groups[g_index][2] <= row < groups[g_index][2] + groups[g_index][3]):
            g_index += 1
        by_group.setdefault(g_index, []).append(row - groups[g_index][2])
    for g_index, offsets in by_group.items():
        f_index, g, first, _ = groups[g_index]
        # A file may lack a mapped column (a fallback random sample); read what
        # it has and give the rest an empty value, so the CSV keeps its columns.
        available = [c for c in columns if c in files[f_index].schema_arrow.names]
        table = files[f_index].read_row_group(g, columns=available).to_pydict()
        for offset in offsets:
            out.append((first + offset, [table[c][offset] if c in table else None for c in columns], f_index))
    return out


def _text(task_type, value):
    """A row's text as written: NER tokens joined, anything else as it is."""
    if task_type in NER_TASKS:
        return tokens_to_text(value)
    return "" if value is None else str(value)


def _numbered(tags):
    """Whether a tag list holds numbers, which only names can turn into types."""
    if isinstance(tags, int) and not isinstance(tags, bool):
        return True
    return isinstance(tags, (list, tuple)) and any(
        isinstance(tag, int) and not isinstance(tag, bool) for tag in tags
    )


def _complete_tags(tokens, tags):
    """Whether an NER row has a token list and a tag for each of its tokens."""
    return (
        isinstance(tokens, (list, tuple))
        and isinstance(tags, (list, tuple))
        and len(tags) == len(tokens)
        and len(tokens) > 0
        and all(tag is not None for tag in tags)
    )


def _mapped(values, names, task_type="classification"):
    """A row as written for the host: text, then the label by name when it has one.

    NER tags become the row's entity spans as JSON; numbered tags with no
    names cannot, so the import says that rather than writing nonsense."""
    text = _text(task_type, values[0])
    if len(values) == 1:
        return [text]
    label = values[1]
    if task_type in NER_TASKS:
        # Only a token list with a tag for every token says where entities
        # are. Anything less (a text that is already a string, missing,
        # scalar, short or partly empty tags) leaves the cell blank, so the
        # row waits to be annotated rather than training as "no entities", as
        # a partial annotation, or as one entity over the whole text.
        if not _complete_tags(values[0], label):
            return [text, ""]
        if _numbered(label) and not names:
            raise ImportFailed("no_entity_names", NO_ENTITY_NAMES)
        text, found = spans(values[0], label, names)
        return [text, json.dumps(found, ensure_ascii=False)]
    if names and isinstance(label, int) and 0 <= label < len(names):
        label = names[label]
    return [text, "" if label is None else str(label)]


def _write_csv(path, record, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ["text", LABEL_HEADERS.get(record.task_type, "label")] if record.label_column else ["text"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _description(record):
    if record.provider == COMMUNITY:
        method = "A sample that keeps the label balance" if record.sample_method == "stratified" else "A random sample"
        return f"{method}, from Zenodo: {record.url or record.external_id}"
    where = f"{record.config}/{record.split}" if record.config else record.split or ""
    if record.sample_method == "stratified":
        return f"A sample that keeps the label balance, from Hugging Face: {record.external_id} {where}".strip()
    return f"A random sample from Hugging Face: {record.external_id} {where}".strip()
