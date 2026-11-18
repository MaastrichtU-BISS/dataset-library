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

The host supplies how shards are fetched (``fetch_shards``) and how a file
becomes a Dataset (``create_dataset``, which raises ``ValueError`` with a
student-readable message for every refusal).
"""
import csv
import json
import logging
import os
import random
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import sqlalchemy as sa
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.orm.exc import ObjectDeletedError

from . import policy as policy_module
from .errors import ProviderError
from .licenses import LICENSES, UNKNOWN
from .models import LibraryImport

logger = logging.getLogger("dataset_library")

QUEUED, FETCHING, SAMPLING, VALIDATING, SAVING = "queued", "fetching", "sampling", "validating", "saving"
COMPLETED, FAILED, CANCELLED = "completed", "failed", "cancelled"
TERMINAL = {COMPLETED, FAILED, CANCELLED}
RUNNING = {QUEUED, FETCHING, SAMPLING, VALIDATING, SAVING}
# Past these, the Dataset may already be written: too late to cancel.
CANCELLABLE = {QUEUED, FETCHING, SAMPLING, VALIDATING}

UNEXPECTED = "The import stopped on our side. Try again, and tell your course staff if it keeps happening."
BUSY = "Hugging Face is busy. Try again in a few minutes."
NOT_STARTED = "Import could not start. Try again."
STOPPED = "Import stopped. Try again."
CHANGED = "This dataset changed on Hugging Face after you opened it. Open it again to import the new version."
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
    """Downloaded Parquet shards of one split, at the revision they were read from."""

    paths: list = field(default_factory=list)
    revision: str | None = None


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


def start(host, owner_id, body, *, ack_at=None):
    """Check the request against the dataset as previewed, store the import as
    queued, then hand it to the host's queue. Returns the import as JSON, and
    whether it is new: the same import (dataset, revision, mapping and size),
    still queued or running, is returned as it is (a double click, a second tab).

    ``body.revision``, when given, is the revision the student saw; an import
    of anything newer is refused. ``ack_at`` is the confirmation time a retry
    carries from the import it repeats, so a retry keeps when the student
    actually agreed rather than claiming they agreed again now."""
    from . import preview as preview_module
    from .providers.huggingface import DATASET_ID_PATTERN, PrivateDataset

    external_id = body.get("dataset")
    if not isinstance(external_id, str) or not DATASET_ID_PATTERN.match(external_id):
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
    try:
        dataset = preview_module.preview(host, external_id, body.get("config") or None, body.get("split") or None)
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
    if dataset["preview"] != "ok":
        raise Refused(409, "not_importable", "Hugging Face cannot show this dataset's rows, so it cannot be imported.")
    columns = {c["name"] for c in dataset["columns"]}
    text_column, label_column = body.get("textColumn"), body.get("labelColumn") or None
    if text_column not in columns:
        raise Refused(400, "bad_mapping", "Pick the column that holds the text.")
    if label_column is not None and label_column not in columns:
        raise Refused(400, "bad_mapping", "Pick a label column from this dataset, or none.")
    if dataset["license"]["level"] == "warn" and body.get("licenseAck") is not True:
        raise Refused(409, "needs_ack", "Confirm the license terms to import this dataset.")
    # Only a license that asks for confirmation records a time; an allow license
    # stores none even when the client sent `licenseAck: true`.
    confirmed = dataset["license"]["level"] == "warn" and body.get("licenseAck") is True
    license_ack_at = (ack_at or _now()) if confirmed else None
    seen_revision = body.get("revision")
    if seen_revision and dataset.get("revision") and seen_revision != dataset["revision"]:
        raise Refused(409, "changed", CHANGED)

    session = host.db.session
    cap = host.row_cap()
    running = (
        session.query(LibraryImport)
        .filter(
            LibraryImport.owner_id == owner_id,
            LibraryImport.provider == "huggingface",
            LibraryImport.external_id == external_id,
            LibraryImport.config.is_(None) if dataset["config"] is None else LibraryImport.config == dataset["config"],
            LibraryImport.split.is_(None) if dataset["split"] is None else LibraryImport.split == dataset["split"],
            LibraryImport.revision.is_(None) if dataset.get("revision") is None else LibraryImport.revision == dataset["revision"],
            LibraryImport.text_column == text_column,
            LibraryImport.label_column.is_(None) if label_column is None else LibraryImport.label_column == label_column,
            LibraryImport.task_type == task_type,
            LibraryImport.sample_size == min(size, cap),
            LibraryImport.status.in_(RUNNING),
        )
        .order_by(LibraryImport.created_at.desc())
        .first()
    )
    if running is not None:
        _log(running, "repeated")
        return running.to_dict(), False
    earlier = session.query(LibraryImport).filter_by(owner_id=owner_id, provider="huggingface", external_id=external_id).count()
    record = LibraryImport(
        owner_id=owner_id, provider="huggingface", external_id=external_id,
        config=dataset["config"], split=dataset["split"], revision=dataset.get("revision"), url=dataset["url"],
        license=dataset["license"]["id"], license_ack_at=license_ack_at,
        text_column=text_column, label_column=label_column, task_type=task_type,
        sample_size=min(size, cap), max_rows=cap,
        name=name or (dataset["title"] if not earlier else f"{dataset['title']} ({earlier + 1})"),
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
    return {
        "provider": record.provider,
        "externalId": record.external_id,
        "config": record.config,
        "split": record.split,
        "revision": record.revision,
        "url": record.url,
        "importedAt": _iso(record.imported_at),
        "licenseAckAt": _iso(record.license_ack_at),
        "license": license.at_level(level),
    }


def _iso(value):
    return value.isoformat() + "Z" if value else None


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
        "dataset": record.external_id,
        "config": record.config,
        "split": record.split,
        "revision": record.revision,
        "textColumn": record.text_column,
        "labelColumn": record.label_column,
        "taskType": record.task_type,
        "size": record.sample_size,
        "name": record.name,
        "licenseAck": record.license_ack_at is not None,
    }
    started, _ = start(host, owner_id, body, ack_at=record.license_ack_at)
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
):
    """Drive one queued import to completed or failed. Safe to call twice.
    ``max_bytes`` is the worker's download cap; an admin's cap replaces it.
    ``heartbeat_seconds`` is how often the import says it is alive (None: never)."""
    from .limits import download_cap_bytes

    record = session.get(LibraryImport, import_ref)
    if record is None or record.status != QUEUED:
        return
    max_bytes = download_cap_bytes(session) or max_bytes
    work = os.path.join(data_dir, "library-imports", record.id)
    path = os.path.join(data_dir, f"library-{record.id}.csv")
    heartbeat = _Heartbeat(session.get_bind(), record.id, heartbeat_seconds) if heartbeat_seconds else None
    try:
        if heartbeat:
            heartbeat.__enter__()
        _advance(session, record, FETCHING)
        fetched = fetch_shards(record.external_id, record.config, record.split, max_bytes, work)
        if record.revision and fetched.revision and fetched.revision != record.revision:
            raise ImportFailed("changed", CHANGED)
        _advance(session, record, SAMPLING, {"revision": fetched.revision})
        rows = _sample(record, fetched.paths, random.Random(seed), open_parquet)
        _advance(session, record, VALIDATING)
        _write_csv(path, record, rows)
        # The last check before the Dataset is written: a cancel before this wins.
        _advance(session, record, SAVING)
        try:
            dataset_ref = create_dataset(record.owner_id, path, record.name, _description(record), record.task_type)
        except ValueError as exc:
            raise ImportFailed("invalid", str(exc)) from exc
        _advance(session, record, COMPLETED, {"dataset_ref": str(dataset_ref), "rows": len(rows), "imported_at": _now()})
    except _Stopped:
        _remove(path)
    except ImportFailed as exc:
        _remove(path)
        _fail(session, record, exc.code, exc.message)
    except ProviderError:
        _fail(session, record, "provider_busy", BUSY)
    except Exception:  # noqa: BLE001 - an import must end, whatever broke
        logger.exception(f"library.import.error import={record.id}")
        _remove(path)
        _fail(session, record, "unexpected", UNEXPECTED)
    finally:
        if heartbeat:
            heartbeat.__exit__(None, None, None)
        shutil.rmtree(work, ignore_errors=True)


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
    metadata = parquet_file.schema_arrow.metadata or {}
    try:
        features = json.loads(metadata[b"huggingface"])["info"]["features"]
    except (KeyError, ValueError, TypeError):
        return None
    feature = features.get(column) or {}
    return feature.get("names") if feature.get("_type") == "ClassLabel" else None


def _sample(record, paths, rng, open_parquet):
    """Pick `sample_size` rows with text, uniformly at random, reading as little as possible.

    Pass 1 reads only each row group's row count; pass 2 reads only the mapped
    columns of the row groups that hold a chosen row."""
    if open_parquet is None:
        import pyarrow.parquet as pq

        open_parquet = pq.ParquetFile
    files = [open_parquet(path) for path in paths]
    groups = []  # (file index, row group, first row, row count)
    total = 0
    for f_index, parquet_file in enumerate(files):
        metadata = parquet_file.metadata
        for g in range(metadata.num_row_groups):
            count = metadata.row_group(g).num_rows
            groups.append((f_index, g, total, count))
            total += count
    columns = [record.text_column] + ([record.label_column] if record.label_column else [])
    names = [_label_names(f, record.label_column) if record.label_column else None for f in files]

    want = min(record.sample_size, total)
    kept, seen = [], set()
    while len(kept) < want and len(seen) < total:
        batch = _draw(rng, total, seen, want - len(kept))
        for values, f_index in _read(files, groups, batch, columns):
            text = values[0]
            if text is None or not str(text).strip():
                continue
            kept.append((values, f_index))
    if not kept:
        raise ImportFailed("no_usable_rows", "None of the rows we read had any text, so there is nothing to import.")
    return [_mapped(values, names[f_index]) for values, f_index in kept[:want]]


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
    """The chosen rows' mapped columns, one row group read at a time, with the
    row's file so its label names can be found."""
    out = []
    g_index = 0
    by_group = {}
    for row in rows:
        while not (groups[g_index][2] <= row < groups[g_index][2] + groups[g_index][3]):
            g_index += 1
        by_group.setdefault(g_index, []).append(row - groups[g_index][2])
    for g_index, offsets in by_group.items():
        f_index, g, _, _ = groups[g_index]
        table = files[f_index].read_row_group(g, columns=columns).to_pydict()
        for offset in offsets:
            out.append(([table[c][offset] for c in columns], f_index))
    return out


def _mapped(values, names):
    """A row as written for the host: text, then the label by name when it has one."""
    text = str(values[0])
    if len(values) == 1:
        return [text]
    label = values[1]
    if names and isinstance(label, int) and 0 <= label < len(names):
        label = names[label]
    return [text, "" if label is None else str(label)]


def _write_csv(path, record, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ["text", "label"] if record.label_column else ["text"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _description(record):
    where = f"{record.config}/{record.split}" if record.config else record.split or ""
    return f"A random sample from Hugging Face: {record.external_id} {where}".strip()
