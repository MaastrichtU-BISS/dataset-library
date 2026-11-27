"""The platform's copies of pinned datasets, so a pin can be imported while Hugging
Face is down (#121).

A pin whose license is Allow gets a copy: the parts of the dataset an import would
download, up to the same byte cap, joined into one Parquet file in the host's
snapshot directory. A worker makes it, off the request path, from its own queue.
It writes a temporary file beside the copy and renames it into place, so a reader
never sees a part-written file; a write that fails is recorded as `failed`.

Each try at a copy is a new row with a new id, and that id names everything the try
writes: the file `<id>.parquet` and the work directory `.incoming/<id>`. So a try that
is still running after its copy was failed and tried again cannot finish or delete the
newer try's files. A running try refreshes its `updated_at`; the sweep fails a copy
whose worker has gone quiet, and removes what its build left behind.

While Hugging Face cannot be asked, an import reads the copy at the copy's revision,
and only while the copy's pin exists and its license is still Allow. A copy goes when
its pin does, or when its license stops being Allow. A copy that a running import may
still read stays, as `removing` and not served, until that import finishes, so nothing
is deleted under a reader.
"""
import errno
import logging
import os
import re
import shutil
import threading
from datetime import datetime, timedelta, timezone

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from . import policy as policy_module
from .errors import NotParquet, ProviderError, ProviderRateLimited
from .licenses import LICENSES, UNKNOWN
from .limits import download_cap_bytes
from .models import LibraryImport, LibraryPin, LibrarySnapshot
from .preview import LicenseHidden, ROWS, _cell, _propose, _type, ner_part
from .providers import huggingface

logger = logging.getLogger("dataset_library")

PROVIDER = "huggingface"
QUEUED, BUILDING, READY, FAILED, REMOVING = "queued", "building", "ready", "failed", "removing"
# A building copy left this long is failed by the sweep. It refreshes its updated_at every
# HEARTBEAT_SECONDS, so only a worker that has really stopped is failed. A queued copy waits
# behind imports on the same worker, so it gets far longer before it counts as lost.
STALE_AFTER = timedelta(minutes=30)
QUEUED_AFTER = timedelta(hours=6)
# Copies and their work areas are named by a row id, a UUID; nothing else in the directory is touched.
_OWN_NAME = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
HEARTBEAT_SECONDS = 15
# How often the Hub is asked whether a copy's revision still exists, and how many copies per round.
CHECK_AFTER = timedelta(hours=1)
CHECK_LIMIT = 20
# Where a copy's parts download to, inside the snapshot directory: the same disk, so the rename is atomic.
INCOMING = ".incoming"

BUSY = "Hugging Face is busy, so no copy was kept. Try again in a few minutes."
NO_ROWS = "Hugging Face cannot show this dataset's rows, so there is nothing to copy."
NO_REVISION = "Hugging Face did not say which version this is, so no copy was kept."
FULL = "There is no room left to keep a copy of this dataset. Your platform admin can free space; see the library runbook."
NOT_STARTED = "The copy could not be queued. Try again."
STOPPED = "The copy stopped before it was kept. Try again."
UNEXPECTED = "The copy could not be made. Try again, and tell your course staff if it keeps happening."


class CopyFailed(Exception):
    """A copy that cannot be kept; `message` is shown to the admin as it is."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


def request_copy(host, pin, facts):
    """Queue a copy of `pin`, from `facts` (its dataset as previewed), when this
    platform keeps copies and the dataset's license is Allow. Returns the copy, or
    None when none is queued (no copies here, the license is not Allow, or the pin
    has gone). A copy that is ready or on its way is returned as it is. A failed copy
    is tried again as a new copy, with a new id. A queue that refuses the job fails
    the copy at once, as it does an import."""
    if host.snapshot_dir is None or host.enqueue_snapshot is None:
        return None
    if facts["license"]["level"] != "allow" or facts.get("gated"):
        return None
    session = host.db.session
    pin_id = pin.id
    copy = for_pin(session, pin_id)
    if copy is not None and copy.status in (QUEUED, BUILDING, READY):
        return copy
    replaced = None
    if copy is not None:
        # The id names this try's files, so the failed try's row goes and a new id takes its place.
        replaced = copy.id
        session.delete(copy)
        session.flush()
    copy = LibrarySnapshot(pin_id=pin_id, provider=pin.provider, external_id=pin.external_id)
    session.add(copy)
    copy.config, copy.split, copy.license = facts.get("config"), facts.get("split"), facts["license"]["id"]
    if facts.get("viewerDown"):
        # The viewer did not answer, so the part is not known: the admin can try again.
        copy.status, copy.error_code, copy.error_message = FAILED, "provider_busy", BUSY
    elif facts.get("preview") != "ok" or not copy.config:
        copy.status, copy.error_code, copy.error_message = FAILED, "no_rows", NO_ROWS
    else:
        copy.status = QUEUED
    try:
        session.flush()
        copy_id, external_id, status = copy.id, copy.external_id, copy.status
        if not _pinned(session, pin_id):
            # The pin was removed while this copy was prepared: no copy may outlive its pin.
            session.rollback()
            return None
        session.commit()
    except IntegrityError:
        # Another request queued a copy of this pin first: that copy is the answer.
        session.rollback()
        return for_pin(session, pin_id)
    if replaced is not None:
        _purge(host.snapshot_dir, replaced)
    if status != QUEUED:
        return copy
    try:
        host.enqueue_snapshot(copy_id)
    except Exception:  # noqa: BLE001 - whatever the queue raised, no worker will make it
        logger.exception(f"library.snapshot.enqueue_failed snapshot={copy_id}")
        if _move(session, copy_id, FAILED, {QUEUED}, error_code="not_started", error_message=NOT_STARTED):
            _log(copy_id, external_id, FAILED, code="not_started")
    return copy


def run_copy(copy_ref, *, session, fetch_shards, snapshot_dir, max_bytes, heartbeat_seconds=HEARTBEAT_SECONDS):
    """Make one queued copy: fetch the dataset's parts up to the byte cap (an admin's
    cap replaces `max_bytes`), write them as one Parquet file, and keep it as ready.
    The copy's id names this try's files; while the try runs it refreshes the copy's
    `updated_at` every `heartbeat_seconds` (None: it does not). Stops, and keeps
    nothing, if the copy was removed, or its pin or license went, meanwhile. Safe to
    call twice."""
    from .imports import ImportFailed  # lazy: imports imports this module

    copy = session.get(LibrarySnapshot, copy_ref)
    if copy is None or copy.status != QUEUED:
        return
    # Read once, so nothing below touches the instance after a commit.
    copy_id, external_id, config, split = copy.id, copy.external_id, copy.config, copy.split
    pin_id, license_id = copy.pin_id, copy.license
    max_bytes = download_cap_bytes(session) or max_bytes
    work = _work_dir(snapshot_dir, copy_id)
    name = _file_name(copy_id)
    # Claimed before anything is written, and only then cleaned up: a second worker
    # that pops the same copy must not clean up the first one's work.
    try:
        claimed = _move(session, copy_id, BUILDING, {QUEUED})
    except Exception:  # noqa: BLE001 - the copy stays queued, and the sweep fails it if no one takes it
        logger.exception(f"library.snapshot.claim_failed snapshot={copy_id}")
        session.rollback()
        return
    if not claimed:
        return
    heartbeat, kept = None, False
    try:
        if heartbeat_seconds:
            heartbeat = _Heartbeat(session.get_bind(), copy_id, heartbeat_seconds)
            heartbeat.__enter__()
        _log(copy_id, external_id, BUILDING)
        fetched = fetch_shards(external_id, config, split, max_bytes, work)
        if not fetched.revision:
            raise CopyFailed("no_revision", NO_REVISION)
        os.makedirs(work, exist_ok=True)
        staged = os.path.join(work, name)
        rows, columns = _concatenate(fetched.paths, staged)
        os.replace(staged, os.path.join(snapshot_dir, name))
        # The last check before the copy is kept: its pin must still exist, and its license must still be Allow.
        level = _level_now(session, pin_id, license_id)
        if level != "allow":
            _log(copy_id, external_id, "dropped", reason="pin" if level is None else "license")
            _drop_building(session, copy_id)
            return
        kept = _move(session, copy_id, READY, {BUILDING}, file_path=name, revision=fetched.revision, row_count=rows, columns=columns)
        if kept:
            _log(copy_id, external_id, READY, rows=rows)
    except (ImportFailed, CopyFailed) as exc:
        _fail(session, copy_id, external_id, exc.code, exc.message)
    except NotParquet:
        # A shard that cannot be read: asking again cannot help, so it is not busy.
        _fail(session, copy_id, external_id, "no_rows", NO_ROWS)
    except ProviderError:
        _fail(session, copy_id, external_id, "provider_busy", BUSY)
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            _fail(session, copy_id, external_id, "storage_full", FULL)
        else:
            logger.exception(f"library.snapshot.write_failed snapshot={copy_id}")
            _fail(session, copy_id, external_id, "write_failed", UNEXPECTED)
    except Exception:  # noqa: BLE001 - a copy must end, whatever broke
        logger.exception(f"library.snapshot.error snapshot={copy_id}")
        _fail(session, copy_id, external_id, "unexpected", UNEXPECTED)
    finally:
        if heartbeat is not None:
            heartbeat.__exit__(None, None, None)
        shutil.rmtree(work, ignore_errors=True)
        if not kept:
            # Nothing of a try that did not keep its copy may stay: its file is named by its id.
            _unlink(snapshot_dir, name)


def _concatenate(paths, dest):
    """Every row of the Parquet files at `paths`, in order, in one file at `dest`.
    Batch by batch, so a copy never holds a whole shard in memory. Returns the row
    count and the columns as {name, type}."""
    import pyarrow.parquet as pq

    if not paths:
        raise CopyFailed("no_rows", NO_ROWS)
    writer, rows, columns = None, 0, []
    try:
        for path in paths:
            parquet_file = pq.ParquetFile(path)
            if writer is None:
                schema = parquet_file.schema_arrow
                columns = [{"name": field.name, "type": str(field.type)} for field in schema]
                writer = pq.ParquetWriter(dest, schema)
            for batch in parquet_file.iter_batches():
                writer.write_batch(batch)
                rows += batch.num_rows
    finally:
        if writer is not None:
            writer.close()
    return rows, columns


def for_pin(session, pin_id):
    return session.query(LibrarySnapshot).filter_by(pin_id=pin_id).first()


def by_pin(session):
    """Every pin's copy, by pin id. A pin with none is absent."""
    return {copy.pin_id: copy for copy in session.query(LibrarySnapshot).filter(LibrarySnapshot.pin_id.is_not(None))}


def summary(copy):
    """A pin's copy as the settings page and the library page show it."""
    if copy is None:
        return {"status": "none", "error": None, "revision": None, "rows": None}
    return {"status": copy.status, "error": copy.error_message, "revision": copy.revision, "rows": copy.row_count}


def ready_for(session, provider, external_id):
    """The kept copy of a dataset, or None. Only a ready copy is served."""
    return session.query(LibrarySnapshot).filter_by(provider=provider, external_id=external_id, status=READY).first()


def reading(session, copy_id):
    """A copy an import was queued from, while it is still kept or being removed for
    that import's sake. None when it has gone."""
    copy = session.get(LibrarySnapshot, copy_id) if copy_id else None
    return copy if copy is not None and copy.status in (READY, REMOVING) else None


def fallback_for(session, record, snapshot_dir):
    """The ready copy an import may read when Hugging Face cannot be asked: the same
    part of the dataset, while its pin exists and its license is still Allow. The copy's
    revision is the one the import reads, as its provenance says; None when no copy holds
    that part."""
    if snapshot_dir is None:
        return None
    copy = ready_for(session, record.provider, record.external_id)
    if copy is None or copy.config != record.config or copy.split != record.split:
        return None
    if _level_now(session, copy.pin_id, copy.license) != "allow":
        return None
    return copy


def copy_for_offline(host, external_id, config, split):
    """The ready copy that stands in for Hugging Face for this part of a dataset, or
    None. A part the copy does not hold (another config or split) is not offered, nor is
    a copy whose pin has gone or whose license is no longer Allow. A copy whose license is
    now hidden raises LicenseHidden, as the preview does: the dataset is not in the library."""
    session = host.db.session
    copy = ready_for(session, PROVIDER, external_id)
    if copy is None or (config and config != copy.config) or (split and split != copy.split):
        return None
    level = _level_now(session, copy.pin_id, copy.license)
    if level == "hide":
        raise LicenseHidden()
    return copy if level == "allow" else None


def offline_detail(host, copy, *, rows=True, tag_names=None):
    """A dataset as the library shows it while Hugging Face cannot be asked: the card of
    the pin the copy was made for, the copy's columns with the types its Hugging Face
    features give, its first rows, and the task the live preview would propose for those
    columns. An import takes the same facts (`rows=False` skips the first rows). The
    license is the platform's level now, as preview applies it; LicenseHidden when it is
    hidden. `rowCount` is the copy's rows, not the split's. `tag_names` name numbered NER
    tags as on the live preview (#180)."""
    pin = host.db.session.get(LibraryPin, copy.pin_id)
    if pin is None:
        # Unpinned since the copy was found: it no longer stands in for anything.
        raise LicenseHidden()
    license = LICENSES.get(copy.license or "", UNKNOWN)
    level = policy_module.level(license, policy_module.overrides(host.db))
    if level == "hide":
        raise LicenseHidden()
    card = pin.card
    columns = copy.columns or []
    names = [column["name"] for column in columns]
    path = file_of(host.snapshot_dir, copy) if host.snapshot_dir is not None and copy.file_path else None
    features, sample = _read_copy(path, names, rows)
    named = [{"name": n, "type": features.get(n)} for n in names]
    viewer_rows = [{"row": raw} for raw in sample]
    proposed = _propose(card.get("tasks") or [], named, viewer_rows)
    return {
        "id": f"{copy.provider}:{copy.external_id}",
        "provider": copy.provider,
        "externalId": copy.external_id,
        "author": copy.external_id.split("/")[0],
        "title": card.get("title") or copy.external_id.split("/")[-1],
        "url": card.get("url"),
        "tasks": card.get("tasks") or [],
        "license": license.at_level(level),
        "size": card.get("size"),
        "gated": bool(card.get("gated")),
        "description": card.get("description"),
        "updated": card.get("updated"),
        "revision": copy.revision,
        "downloads": card.get("downloads") or 0,
        "citation": {**card["citation"], "revision": copy.revision} if card.get("citation") else None,
        "labels": pin.labelled,
        "preview": "ok",
        "configs": [{"name": copy.config, "splits": [copy.split]}],
        "config": copy.config,
        "split": copy.split,
        "columns": [{"name": column["name"], "type": _column_type(column, features)} for column in columns],
        "rows": [[_cell(raw.get(name), features.get(name)) for name in names] for raw in sample],
        "rowCount": copy.row_count,
        "proposed": proposed,
        "ner": ner_part(proposed, named, viewer_rows, tag_names),
        # The copy stands in for Hugging Face: the page says so, and it can still be imported.
        "offline": True,
    }


def _column_type(column, features):
    """A column's type as a student reads it: the Hugging Face feature's when the copy has
    one, else from the copy's Arrow type."""
    feature = features.get(column["name"])
    return _type(feature) if feature is not None else _readable_type(column.get("type", ""))


def _readable_type(arrow_type):
    """A column's type as a student reads it, from the copy's Arrow type."""
    if arrow_type in ("string", "large_string"):
        return "text"
    if arrow_type.startswith(("int", "uint", "float", "double", "halffloat")):
        return "number"
    if arrow_type == "bool":
        return "yes/no"
    if arrow_type.startswith(("list", "large_list", "fixed_size_list")):
        return "list"
    return arrow_type


def _read_copy(path, names, rows):
    """The copy's Hugging Face features, from its schema metadata, and when `rows`, its
    first rows as raw values keyed by column. Nothing when the file cannot be read: the
    copy still opens, it just shows no rows (#121). A corrupt file is no file."""
    import pyarrow.parquet as pq
    from pyarrow.lib import ArrowInvalid

    features = {}
    if path is None:
        return features, []
    try:
        parquet_file = pq.ParquetFile(path)
        features = _features(parquet_file.schema_arrow.metadata)
        if not rows or not names:
            return features, []
        batch = next(parquet_file.iter_batches(batch_size=ROWS, columns=names), None)
    except (OSError, ValueError, KeyError, ArrowInvalid) as exc:
        logger.warning(f"library.snapshot.preview_unreadable file={os.path.basename(path)} error={exc}")
        return features, []
    return features, ([] if batch is None else batch.to_pylist())


def _features(metadata):
    """The Hugging Face feature map a shard carries in its schema metadata, or {}."""
    import json

    try:
        return json.loads((metadata or {})[b"huggingface"])["info"]["features"] or {}
    except (KeyError, ValueError, TypeError):
        return {}


def file_of(snapshot_dir, copy):
    return os.path.join(snapshot_dir, copy.file_path)


def take_down(session, copy):
    """Stop keeping a copy. Its row goes at once, unless a queued or running import of its
    dataset may still read the file: then the row stays as `removing`, not served, until
    `release` finds no such import. Returns whether the row went, so the caller removes
    the copy's files once it has committed."""
    if _being_read(session, copy.provider, copy.external_id):
        copy.pin_id, copy.status = None, REMOVING
        return False
    session.delete(copy)
    return True


def discard(session, snapshot_dir, copies):
    """Take down the copies, then delete the files no import can still read. Rows and
    pins change in one commit, so a copy is never served after its pin is gone; files
    go after the commit, so a failed delete leaves an orphan file, never a served copy."""
    gone = [copy.id for copy in copies if take_down(session, copy)]
    session.commit()
    for copy_id in gone:
        _purge(snapshot_dir, copy_id)


def release(session, snapshot_dir, provider, external_id):
    """Delete a dataset's removing copies, once no running import of it remains."""
    if _being_read(session, provider, external_id):
        return
    copies = session.query(LibrarySnapshot).filter_by(provider=provider, external_id=external_id, status=REMOVING).all()
    if copies:
        discard(session, snapshot_dir, copies)


def reconcile(session, snapshot_dir, policy):
    """Take down the copies whose license the admin's policy no longer allows.
    Returns how many."""
    copies = session.query(LibrarySnapshot).filter(LibrarySnapshot.pin_id.is_not(None)).all()
    gone = [copy for copy in copies if policy_module.level(LICENSES.get(copy.license or "", UNKNOWN), policy) != "allow"]
    discard(session, snapshot_dir, gone)
    return len(gone)


def sweep(session, snapshot_dir, *, stale_after=STALE_AFTER, queued_after=QUEUED_AFTER, now=None):
    """Fail copies a worker left queued or building, release removing copies no import
    reads any more, and delete what no kept copy owns: the work directory or file a dead
    build left, and anything of a failed copy. Safe to run anywhere, any number of times;
    returns the ids it failed."""
    now = now or _now()
    # Listed before any row is read: a build that starts after this listing is not in it, so it is never removed.
    files, work = _listing(snapshot_dir)
    stuck = [
        (copy.id, copy.external_id, copy.status)
        for copy in session.query(LibrarySnapshot).filter(sa.or_(
            sa.and_(LibrarySnapshot.status == BUILDING, LibrarySnapshot.updated_at < now - stale_after),
            sa.and_(LibrarySnapshot.status == QUEUED, LibrarySnapshot.updated_at < now - queued_after),
        ))
    ]
    failed = []
    for copy_id, external_id, status in stuck:
        code, message = ("not_started", NOT_STARTED) if status == QUEUED else ("stopped", STOPPED)
        if _move(session, copy_id, FAILED, {status}, error_code=code, error_message=message):
            _log(copy_id, external_id, FAILED, code=code)
            _purge(snapshot_dir, copy_id)
            failed.append(copy_id)
    datasets = {(c.provider, c.external_id) for c in session.query(LibrarySnapshot).filter_by(status=REMOVING)}
    for provider, external_id in datasets:
        release(session, snapshot_dir, provider, external_id)
    _remove_orphans(session, snapshot_dir, files, work)
    return failed


def check_upstream(session, *, revision_exists=None, now=None, after=CHECK_AFTER, limit=CHECK_LIMIT):
    """Ask the Hub whether each ready copy's revision still exists. One it no longer has
    sets `upstream_changed`; the copy is kept either way. Each answer is saved on its own,
    and a copy unpinned meanwhile is left alone. The round ends at the first check the Hub
    cannot answer twice in a row (it is down), or when it limits requests; the rest are
    asked in a later round. A copy whose own check fails is put to the back, its flag
    unchanged, so one dataset the Hub always refuses cannot stop the others being checked.
    Returns how many copies were recorded."""
    revision_exists = revision_exists or huggingface.revision_exists
    now = now or _now()
    due = [
        (copy.id, copy.external_id, copy.revision)
        for copy in session.query(LibrarySnapshot).filter(
            LibrarySnapshot.status == READY,
            sa.or_(LibrarySnapshot.upstream_checked_at.is_(None), LibrarySnapshot.upstream_checked_at < now - after),
        )
        # Never-checked copies first, then the least recently checked.
        .order_by(LibrarySnapshot.upstream_checked_at.is_not(None), LibrarySnapshot.upstream_checked_at)
        .limit(limit)
    ]
    checked, failures = 0, 0
    for copy_id, external_id, revision in due:
        try:
            exists = revision_exists(external_id, revision)
        except ProviderRateLimited:
            break
        except ProviderError:
            failures += 1
            # To the back of the order, flag unchanged: asked again in a later round.
            session.execute(
                sa.update(LibrarySnapshot)
                .where(LibrarySnapshot.id == copy_id, LibrarySnapshot.status == READY)
                .values(upstream_checked_at=now)
                .execution_options(synchronize_session=False)
            )
            session.commit()
            if failures >= 2:
                break
            continue
        failures = 0
        recorded = session.execute(
            sa.update(LibrarySnapshot)
            .where(LibrarySnapshot.id == copy_id, LibrarySnapshot.status == READY)
            .values(upstream_changed=not exists, upstream_checked_at=now)
            .execution_options(synchronize_session=False)
        ).rowcount
        session.commit()
        if recorded:
            checked += 1
            if not exists:
                _log(copy_id, external_id, "upstream_changed", revision=revision)
    return checked


def touch(engine, copy_id):
    """Say a copy's worker is alive: its `updated_at`, while the copy is still building.
    Its own connection, so it can run beside the worker's session on another thread."""
    with engine.begin() as connection:
        connection.execute(
            sa.update(LibrarySnapshot.__table__)
            .where(LibrarySnapshot.id == copy_id, LibrarySnapshot.status == BUILDING)
            .values(updated_at=_now())
        )


class _Heartbeat:
    """Calls `touch` every `seconds` on a daemon thread until stopped (as imports do)."""

    def __init__(self, engine, copy_id, seconds):
        self.engine, self.copy_id, self.seconds = engine, copy_id, seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"library-copy-{copy_id}", daemon=True)

    def _run(self):
        while not self._stop.wait(self.seconds):
            try:
                touch(self.engine, self.copy_id)
            except Exception:  # noqa: BLE001 - a missed beat is only a later sweep
                logger.warning(f"library.snapshot.heartbeat_failed snapshot={self.copy_id}", exc_info=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=self.seconds)


def _being_read(session, provider, external_id):
    """Whether a queued or running import of this dataset may still read its copy."""
    from .imports import RUNNING  # lazy: imports imports this module

    return session.query(LibraryImport.id).filter(
        LibraryImport.provider == provider,
        LibraryImport.external_id == external_id,
        LibraryImport.status.in_(RUNNING),
    ).first() is not None


def _pinned(session, pin_id):
    """Whether the pin exists, read now."""
    return session.query(LibraryPin.id).filter_by(id=pin_id).first() is not None


def _level_now(session, pin_id, license_id):
    """The level a copy of this pin is served at now, or None when its pin is gone: a copy
    is kept or served only for a pin that exists."""
    if pin_id is None or not _pinned(session, pin_id):
        return None
    return policy_module.level(LICENSES.get(license_id or "", UNKNOWN), policy_module.for_session(session))


def _drop_building(session, copy_id):
    """Delete a copy whose try is building, if it still is: its pin or its license went
    while it was made. The file goes with the caller's cleanup."""
    session.execute(sa.delete(LibrarySnapshot).where(LibrarySnapshot.id == copy_id, LibrarySnapshot.status == BUILDING))
    session.commit()


def _move(session, copy_id, state, from_states, **values):
    """Move a copy to `state` if it is still in one of `from_states`, in one conditional
    update, so of two movers (the worker, a removal) exactly one wins. The copy's id is
    its build's, so a worker's moves touch only its own build. Returns whether it moved."""
    moved = session.execute(
        sa.update(LibrarySnapshot)
        .where(LibrarySnapshot.id == copy_id, LibrarySnapshot.status.in_(from_states))
        .values(status=state, updated_at=_now(), **values)
        .execution_options(synchronize_session=False)
    ).rowcount
    session.commit()
    return bool(moved)


def _fail(session, copy_id, external_id, code, message):
    # Whatever the failed step left pending is dropped before the status is written.
    session.rollback()
    if _move(session, copy_id, FAILED, {QUEUED, BUILDING}, error_code=code, error_message=message):
        _log(copy_id, external_id, FAILED, code=code)


def _file_name(copy_id):
    return f"{copy_id}.parquet"


def _work_dir(snapshot_dir, copy_id):
    return os.path.join(snapshot_dir, INCOMING, copy_id)


def _purge(snapshot_dir, copy_id):
    """Delete a copy's file and work directory, whatever its row says: both are named by
    the copy's id, so nothing else is touched."""
    if snapshot_dir is None:
        return
    _unlink(snapshot_dir, _file_name(copy_id))
    shutil.rmtree(_work_dir(snapshot_dir, copy_id), ignore_errors=True)


def _listing(snapshot_dir):
    """The copy files (`<id>.parquet`) and work directories (`.incoming/<id>`) in the
    snapshot directory, as their names."""
    if snapshot_dir is None or not os.path.isdir(snapshot_dir):
        return [], []
    files = [
        name for name in os.listdir(snapshot_dir)
        if name.endswith(".parquet") and _OWN_NAME.fullmatch(name[: -len(".parquet")])
    ]
    incoming = os.path.join(snapshot_dir, INCOMING)
    work = [name for name in os.listdir(incoming) if _OWN_NAME.fullmatch(name)] if os.path.isdir(incoming) else []
    return files, work


def _remove_orphans(session, snapshot_dir, files, work):
    """Delete the files and work directories no kept copy owns. A file is kept only for a
    building copy, or as a ready or removing copy's own file; a work directory only for a
    building copy. Everything else is left by a build that died, or by a copy that is gone."""
    if snapshot_dir is None:
        return
    rows = {copy_id: (status, file_path) for copy_id, status, file_path in session.query(
        LibrarySnapshot.id, LibrarySnapshot.status, LibrarySnapshot.file_path
    )}
    for name in files:
        status, file_path = rows.get(name[: -len(".parquet")], (None, None))
        if status == BUILDING or (status in (READY, REMOVING) and name == file_path):
            continue
        _unlink(snapshot_dir, name)
    for name in work:
        status, _ = rows.get(name, (None, None))
        if status != BUILDING:
            shutil.rmtree(os.path.join(snapshot_dir, INCOMING, name), ignore_errors=True)


def _unlink(snapshot_dir, name):
    if not name or snapshot_dir is None:
        return
    try:
        os.remove(os.path.join(snapshot_dir, name))
    except FileNotFoundError:
        pass
    except OSError:
        logger.exception(f"library.snapshot.delete_failed file={name}")


def _log(copy_id, external_id, state, **fields):
    extra = "".join(f" {k}={v}" for k, v in fields.items())
    logger.info(f"library.snapshot.{state} snapshot={copy_id} dataset={external_id}{extra}")


def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)
