"""The platform's copies of pinned datasets, so a pin can be imported while Hugging
Face is down (#121).

A pin whose license is Allow gets a copy: the parts of the dataset an import would
download, up to the same byte cap, joined into one Parquet file in the host's
snapshot directory. A worker makes it, off the request path, from its own queue.
It writes a temporary file beside the copy and renames it into place, so a reader
never sees a part-written file; a write that fails is recorded as `failed`.

While Hugging Face cannot be asked, an import reads the copy at the copy's revision.
A copy goes when its pin does, or when its license stops being Allow. A copy that a
running import may still read stays, as `removing` and not served, until that
import finishes, so nothing is deleted under a reader.
"""
import errno
import logging
import os
import shutil
from datetime import datetime, timedelta, timezone

import sqlalchemy as sa

from . import policy as policy_module
from .errors import ProviderError
from .licenses import LICENSES, UNKNOWN
from .limits import download_cap_bytes
from .models import LibraryImport, LibraryPin, LibrarySnapshot
from .preview import LicenseHidden
from .providers import huggingface

logger = logging.getLogger("dataset_library")

PROVIDER = "huggingface"
QUEUED, BUILDING, READY, FAILED, REMOVING = "queued", "building", "ready", "failed", "removing"
# A copy a worker left queued or building this long is failed by the sweep.
STALE_AFTER = timedelta(minutes=30)
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
    None when none is queued. A copy that is ready or on its way is returned as it
    is. A queue that refuses the job fails the copy at once, as it does an import."""
    if host.snapshot_dir is None or host.enqueue_snapshot is None:
        return None
    if facts["license"]["level"] != "allow" or facts.get("gated"):
        return None
    session = host.db.session
    copy = for_pin(session, pin.id)
    if copy is not None and copy.status in (QUEUED, BUILDING, READY):
        return copy
    if copy is None:
        copy = LibrarySnapshot(pin_id=pin.id, provider=pin.provider, external_id=pin.external_id)
        session.add(copy)
    # A copy tried again starts from nothing: its earlier answer does not carry over.
    copy.config, copy.split, copy.license = facts.get("config"), facts.get("split"), facts["license"]["id"]
    copy.revision, copy.columns, copy.file_path, copy.row_count = None, None, None, None
    copy.upstream_changed, copy.upstream_checked_at = False, None
    copy.error_code, copy.error_message = None, None
    if facts.get("preview") != "ok" or not copy.config:
        copy.status, copy.error_code, copy.error_message = FAILED, "no_rows", NO_ROWS
        session.commit()
        return copy
    copy.status = QUEUED
    session.commit()
    try:
        host.enqueue_snapshot(copy.id)
    except Exception:  # noqa: BLE001 - whatever the queue raised, no worker will make it
        logger.exception(f"library.snapshot.enqueue_failed snapshot={copy.id}")
        copy.status, copy.error_code, copy.error_message = FAILED, "not_started", NOT_STARTED
        session.commit()
    return copy


def run_copy(copy_ref, *, session, fetch_shards, snapshot_dir, max_bytes):
    """Make one queued copy: fetch the dataset's parts up to the byte cap (an admin's
    cap replaces `max_bytes`), write them as one Parquet file, and keep it as ready.
    Stops if the copy was removed meanwhile. Safe to call twice."""
    from .imports import ImportFailed  # lazy: imports imports this module

    copy = session.get(LibrarySnapshot, copy_ref)
    if copy is None or copy.status != QUEUED:
        return
    # Read once, so nothing below touches the instance after a commit.
    copy_id, external_id, config, split = copy.id, copy.external_id, copy.config, copy.split
    max_bytes = download_cap_bytes(session) or max_bytes
    work = os.path.join(snapshot_dir, INCOMING, copy_id)
    name = f"{copy_id}.parquet"
    try:
        if not _move(session, copy_id, BUILDING, {QUEUED}):
            return
        _log(copy_id, external_id, BUILDING)
        fetched = fetch_shards(external_id, config, split, max_bytes, work)
        if not fetched.revision:
            raise CopyFailed("no_revision", NO_REVISION)
        os.makedirs(work, exist_ok=True)
        staged = os.path.join(work, name)
        rows, columns = _concatenate(fetched.paths, staged)
        os.replace(staged, os.path.join(snapshot_dir, name))
        if _move(session, copy_id, READY, {BUILDING}, file_path=name, revision=fetched.revision, row_count=rows, columns=columns):
            _log(copy_id, external_id, READY, rows=rows)
        else:
            # Removed while it was made: nothing may keep the file.
            _unlink(snapshot_dir, name)
    except (ImportFailed, CopyFailed) as exc:
        _fail(session, copy_id, external_id, exc.code, exc.message)
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
        shutil.rmtree(work, ignore_errors=True)


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
    part of the dataset. The copy's revision is the one the import reads, as its
    provenance says; None when no copy holds that part."""
    if snapshot_dir is None:
        return None
    copy = ready_for(session, record.provider, record.external_id)
    if copy is None or copy.config != record.config or copy.split != record.split:
        return None
    return copy


def copy_for_offline(host, external_id, config, split):
    """The ready copy that stands in for Hugging Face for this part of a dataset, or
    None. A part the copy does not hold (another config or split) is not offered, and
    neither is a copy whose pin has gone."""
    session = host.db.session
    copy = ready_for(session, PROVIDER, external_id)
    if copy is None or (config and config != copy.config) or (split and split != copy.split):
        return None
    if session.get(LibraryPin, copy.pin_id) is None:
        return None
    return copy


def offline_preview(host, copy):
    """What a preview says about a copy's dataset, taken from the pin it was made for,
    for an import while Hugging Face cannot be asked. The license is the platform's
    level now, as preview applies it. Raises LicenseHidden when it is hidden."""
    pin = host.db.session.get(LibraryPin, copy.pin_id)
    license = LICENSES.get(copy.license or "", UNKNOWN)
    level = policy_module.level(license, policy_module.overrides(host.db))
    if level == "hide":
        raise LicenseHidden()
    card = pin.card
    return {
        "externalId": copy.external_id,
        "title": card.get("title") or copy.external_id.split("/")[-1],
        "url": card.get("url"),
        "citation": card.get("citation"),
        "gated": bool(card.get("gated")),
        "preview": "ok",
        "license": license.at_level(level),
        "config": copy.config,
        "split": copy.split,
        "columns": copy.columns or [],
        "revision": copy.revision,
    }


def file_of(snapshot_dir, copy):
    return os.path.join(snapshot_dir, copy.file_path)


def take_down(session, copy):
    """Stop keeping a copy. Its row goes at once, unless a running import of its
    dataset may still read the file: then the row stays as `removing`, not served,
    until `release` finds no such import. Returns the file's name to delete once the
    caller has committed, or None."""
    if _being_read(session, copy.provider, copy.external_id):
        copy.pin_id, copy.status = None, REMOVING
        return None
    name = copy.file_path
    session.delete(copy)
    return name


def discard(session, snapshot_dir, copies):
    """Take down the copies, then delete the files no import can still read. Rows and
    pins change in one commit, so a copy is never served after its pin is gone; files
    go after the commit, so a failed delete leaves an orphan file, never a served copy."""
    names = [take_down(session, copy) for copy in copies]
    session.commit()
    for name in names:
        _unlink(snapshot_dir, name)


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


def sweep(session, snapshot_dir, *, stale_after=STALE_AFTER, now=None):
    """Fail copies a worker left queued or building, and release removing copies no
    import reads any more. Safe to run anywhere, any number of times; returns the ids
    it failed."""
    now = now or _now()
    stuck = session.query(LibrarySnapshot).filter(
        LibrarySnapshot.status.in_([QUEUED, BUILDING]), LibrarySnapshot.updated_at < now - stale_after
    ).all()
    failed = []
    for copy in stuck:
        copy_id, external_id, status = copy.id, copy.external_id, copy.status
        code, message = ("not_started", NOT_STARTED) if status == QUEUED else ("stopped", STOPPED)
        if _move(session, copy_id, FAILED, {status}, error_code=code, error_message=message):
            _log(copy_id, external_id, FAILED, code=code)
            failed.append(copy_id)
    datasets = {(c.provider, c.external_id) for c in session.query(LibrarySnapshot).filter_by(status=REMOVING)}
    for provider, external_id in datasets:
        release(session, snapshot_dir, provider, external_id)
    return failed


def check_upstream(session, *, revision_exists=None, now=None, after=CHECK_AFTER, limit=CHECK_LIMIT):
    """Ask the Hub whether each ready copy's revision still exists. One it no longer has
    sets `upstream_changed`; the copy is kept either way, and a check the Hub cannot
    answer is tried again in a later round. Returns how many copies were checked."""
    revision_exists = revision_exists or huggingface.revision_exists
    now = now or _now()
    due = (
        session.query(LibrarySnapshot)
        .filter(
            LibrarySnapshot.status == READY,
            sa.or_(LibrarySnapshot.upstream_checked_at.is_(None), LibrarySnapshot.upstream_checked_at < now - after),
        )
        # Never-checked copies first, then the least recently checked.
        .order_by(LibrarySnapshot.upstream_checked_at.is_not(None), LibrarySnapshot.upstream_checked_at)
        .limit(limit)
        .all()
    )
    checked = 0
    for copy in due:
        try:
            exists = revision_exists(copy.external_id, copy.revision)
        except ProviderError:
            continue
        copy.upstream_changed, copy.upstream_checked_at = not exists, now
        if not exists:
            _log(copy.id, copy.external_id, "upstream_changed", revision=copy.revision)
        checked += 1
    session.commit()
    return checked


def _being_read(session, provider, external_id):
    """Whether a queued or running import of this dataset may still read its copy."""
    from .imports import RUNNING  # lazy: imports imports this module

    return session.query(LibraryImport.id).filter(
        LibraryImport.provider == provider,
        LibraryImport.external_id == external_id,
        LibraryImport.status.in_(RUNNING),
    ).first() is not None


def _move(session, copy_id, state, from_states, **values):
    """Move a copy to `state` if it is still in one of `from_states`, in one conditional
    update, so of two movers (the worker, a removal) exactly one wins. Returns whether
    it moved."""
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
